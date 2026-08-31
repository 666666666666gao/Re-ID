#!/usr/bin/env python3
"""Fail-closed MFRNet RGBNT201 checkpoint reproduction runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path("/root/mmreid-trifusion/baselines/MFRNet")
PYTHON = Path("/root/miniconda3/envs/mfrnet/bin/python")
CHECKPOINT = Path(
    "/root/mmreid-trifusion/checkpoints/MFRNet/RGBNT201_MFRNetbest.pth"
)
CLIP = Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt")
DATASET = Path("/root/mmreid-trifusion/data/RGBNT201")
CONFIG = SOURCE / "configs/RGBNT201/MFRNet.yml"
ENTRYPOINT = SOURCE / "test_net.py"

EXPECTED = {
    "source_commit": "ec54a1302321cda4b5fad9ca1c0878dabf0b46b6",
    "checkpoint_sha256": "f0c2df33f3901738051a917e728c73d9b494113e1e327361bc8f1acf4711126e",
    "checkpoint_bytes": 407_297_967,
    "clip_sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "config_sha256": "fecef5e40461930b84f4313aca86f33617be25c37fa3459c39367e6b18caa43e",
    "entrypoint_sha256": "d0136fbe91dd7d7e9b4044637798ed0e26db44038b290a6beffbd3cefd1b77b3",
    "dataset_receipt_sha256": "ec36309921a3dd7c12d46bb60a83406440ba316f171e419a67ad2cc83bf24318",
}
DATASET_RECEIPT = PROJECT / "evidence/rgbnt201_audit_20260831.json"
REQUIREMENTS_LOCK = PROJECT / "environment/mfrnet_requirements-lock.txt"
TUTEL_SOURCE = PROJECT / "environment/mfrnet_tutel_source.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_packages() -> dict[str, str]:
    required: dict[str, str] = {}
    for path in (REQUIREMENTS_LOCK, TUTEL_SOURCE):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "--")):
                continue
            if " @ " in line:
                name = line.split(" @ ", 1)[0]
                if _normalise_distribution(name) == "tutel":
                    required["tutel"] = "0.3"
                continue
            if "==" in line:
                name, version = line.split("==", 1)
                required[_normalise_distribution(name)] = version
    return required


def _package_check() -> dict[str, Any]:
    completed = _run([str(PYTHON), "-m", "pip", "list", "--format=json"])
    if completed.returncode != 0:
        return {"passed": False, "error": completed.stderr.strip()}
    installed = {
        _normalise_distribution(item["name"]): item["version"]
        for item in json.loads(completed.stdout)
    }
    required = _required_packages()
    mismatches = {
        name: {"required": version, "installed": installed.get(name)}
        for name, version in required.items()
        if installed.get(name) != version
    }
    pip_check = _run([str(PYTHON), "-m", "pip", "check"])
    return {
        "passed": not mismatches and pip_check.returncode == 0,
        "required_count": len(required),
        "mismatches": mismatches,
        "pip_check": pip_check.stdout.strip(),
        "python_no_user_site": True,
    }


def _gpu_state() -> dict[str, Any]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        return {"query_passed": False, "error": completed.stderr.strip()}
    first_line = completed.stdout.strip().splitlines()[0]
    fields = [field.strip() for field in first_line.split(",")]
    if len(fields) != 3:
        return {"query_passed": False, "error": f"unexpected nvidia-smi row: {first_line}"}
    return {
        "query_passed": True,
        "name": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
    }


def _protocol_from_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "dataset": str(config["DATASETS"]["NAMES"]).strip("()'\" "),
        "feature_norm": str(config["TEST"]["FEAT_NORM"]),
        "missing_modality": str(config["TEST"]["MISS"]),
        "neck_feature": str(config["TEST"]["NECK_FEAT"]),
        "query_items": len(list((DATASET / "test/RGB").glob("*.jpg"))),
        "gallery_items": len(list((DATASET / "test/RGB").glob("*.jpg"))),
        "reranking": str(config["TEST"]["RE_RANKING"]),
        "return_pattern": 3,
        "test_batch_size": int(config["TEST"]["IMS_PER_BATCH"]),
        "test_size": [int(value) for value in config["INPUT"]["SIZE_TEST"]],
    }


def _resolved_command(output_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        "test_net.py",
        "--config_file",
        "configs/RGBNT201/MFRNet.yml",
        "--model_path",
        str(CHECKPOINT),
        "MODEL.DEVICE_ID",
        "0",
        "MODEL.PRETRAIN_PATH_T",
        str(CLIP),
        "DATASETS.ROOT_DIR",
        "/root/mmreid-trifusion/data",
        "DATALOADER.NUM_WORKERS",
        "0",
        "OUTPUT_DIR",
        str(output_dir / "upstream"),
    ]


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    map_matches = re.findall(r"\bmAP:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if map_matches:
        metrics["mAP"] = float(map_matches[-1])
    for rank in (1, 5, 10):
        matches = re.findall(
            rf"Rank-{rank}\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text
        )
        if matches:
            metrics[f"Rank-{rank}"] = float(matches[-1])
    return metrics


def _parse_batch_count(text: str) -> int | None:
    complete = [
        int(done)
        for done, total in re.findall(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", text)
        if done == total
    ]
    return complete[-1] if complete else None


def _classify_failure(returncode: int, text: str) -> str:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "OOM"
    if "unknown error" in lowered or "driver shutting down" in lowered:
        return "DRIVER_RESET"
    if "no kernel image" in lowered or "cuda error: invalid device function" in lowered:
        return "CUDA_INCOMPATIBLE"
    return "FAILED" if returncode else "PARITY_MISMATCH"


def _execute_official(receipt: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], int]:
    if not receipt["launch_allowed"]:
        receipt["mode"] = "official128"
        receipt["status"] = "BLOCKED"
        return receipt, 3

    official_command = list(receipt["resolved_command"])
    test_executable = os.environ.get("TRIFUSION_MFRNET_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        receipt["mode"] = "official128"
        receipt["status"] = "FAILED"
        receipt["blockers"].append("test_executable_without_contract_testing")
        return receipt, 2
    executed_command = (
        [test_executable, *official_command[1:]] if test_executable else official_command
    )
    upstream_dir = output_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined.log"
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    stop_monitor = threading.Event()
    peak_memory = int(receipt["gpu"]["memory_used_mib"])

    def monitor() -> None:
        nonlocal peak_memory
        while not stop_monitor.wait(0.2):
            state = _gpu_state()
            if state.get("query_passed"):
                peak_memory = max(peak_memory, int(state["memory_used_mib"]))

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    started_at = time.time()
    monitor_thread.start()
    returncode = 127
    with combined_path.open("w", encoding="utf-8") as combined:
        try:
            process = subprocess.Popen(
                executed_command,
                cwd=SOURCE,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                combined.write(line)
                combined.flush()
            returncode = process.wait()
        except Exception as error:
            combined.write(f"RUNNER_EXCEPTION: {type(error).__name__}: {error}\n")
            combined.flush()
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=2.0)
    final_gpu = _gpu_state()
    if final_gpu.get("query_passed"):
        peak_memory = max(peak_memory, int(final_gpu["memory_used_mib"]))

    combined_text = combined_path.read_text(encoding="utf-8", errors="replace")
    upstream_log = upstream_dir / "test_log.txt"
    upstream_text = (
        upstream_log.read_text(encoding="utf-8", errors="replace")
        if upstream_log.is_file()
        else ""
    )
    metrics = _parse_metrics(upstream_text + "\n" + combined_text)
    required_metrics = {"mAP", "Rank-1", "Rank-5", "Rank-10"}
    metrics_complete = set(metrics) == required_metrics
    if returncode == 0 and metrics_complete:
        status = (
            "PASS"
            if round(metrics["mAP"], 1) == 80.7
            and round(metrics["Rank-1"], 1) == 83.6
            else "PARITY_MISMATCH"
        )
    else:
        status = _classify_failure(returncode, upstream_text + "\n" + combined_text)
    batch_count = _parse_batch_count(upstream_text + "\n" + combined_text)
    receipt.update(
        {
            "mode": "official128",
            "status": status,
            "upstream_command_executed": True,
            "executed_command": executed_command,
            "test_override_used": bool(test_executable),
            "official_test_evaluation_count": 1,
            "returncode": returncode,
            "elapsed_seconds": time.time() - started_at,
            "peak_gpu_memory_mib": peak_memory,
            "final_gpu": final_gpu,
            "metrics_percent": metrics if metrics_complete else None,
            "batch_count": batch_count,
            "expected_batch_count": 14,
            "query_items": 836,
            "gallery_items": 836,
            "combined_log": str(combined_path),
            "combined_log_sha256": _sha256(combined_path),
            "upstream_log": str(upstream_log) if upstream_log.is_file() else None,
            "upstream_log_sha256": _sha256(upstream_log) if upstream_log.is_file() else None,
            "fatal_scan": {
                "oom": "out of memory" in combined_text.lower(),
                "driver_reset": "unknown error" in combined_text.lower(),
                "traceback": "traceback (most recent call last)" in combined_text.lower(),
            },
            "result_label": "local parity of released test-selected checkpoint"
            if status == "PASS"
            else None,
            "sota_claim_supported": False,
        }
    )
    return receipt, 0 if status == "PASS" else 4


def _preflight(output_dir: Path) -> tuple[dict[str, Any], bool]:
    blockers: list[str] = []
    file_checks: dict[str, Any] = {}
    config_path = CONFIG
    if os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1" and os.environ.get(
        "TRIFUSION_MFRNET_TEST_CONFIG"
    ):
        config_path = Path(os.environ["TRIFUSION_MFRNET_TEST_CONFIG"])
    immutable_files = {
        "checkpoint": (CHECKPOINT, EXPECTED["checkpoint_sha256"]),
        "clip": (CLIP, EXPECTED["clip_sha256"]),
        "config": (config_path, EXPECTED["config_sha256"]),
        "entrypoint": (ENTRYPOINT, EXPECTED["entrypoint_sha256"]),
        "dataset_receipt": (DATASET_RECEIPT, EXPECTED["dataset_receipt_sha256"]),
    }
    for label, (path, expected_hash) in immutable_files.items():
        if not path.is_file():
            file_checks[label] = {"path": str(path), "exists": False}
            blockers.append(f"missing:{label}")
            continue
        actual_hash = _sha256(path)
        file_checks[label] = {
            "path": str(path),
            "exists": True,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_match": actual_hash == expected_hash,
            "bytes": path.stat().st_size,
        }
        if actual_hash != expected_hash:
            blockers.append(f"hash_drift:{label}")
    if CHECKPOINT.is_file() and CHECKPOINT.stat().st_size != EXPECTED["checkpoint_bytes"]:
        blockers.append("checkpoint_size_drift")

    commit = _run(["git", "rev-parse", "HEAD"], cwd=SOURCE)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SOURCE)
    source_check = {
        "commit": commit.stdout.strip(),
        "expected_commit": EXPECTED["source_commit"],
        "tracked_worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }
    if commit.returncode != 0 or commit.stdout.strip() != EXPECTED["source_commit"]:
        blockers.append("source_commit_drift")
    if not source_check["tracked_worktree_clean"]:
        blockers.append("source_tracked_worktree_dirty")

    package_check = _package_check() if PYTHON.is_file() else {"passed": False, "error": "missing python"}
    if not package_check["passed"]:
        blockers.append("environment_lock_mismatch")

    try:
        protocol = _protocol_from_config(config_path)
    except Exception as error:  # fail-closed receipt, not a metric path
        protocol = {"error": f"{type(error).__name__}: {error}"}
        blockers.append("protocol_parse_failed")
    expected_protocol = {
        "dataset": "RGBNT201",
        "feature_norm": "yes",
        "missing_modality": "nothing",
        "neck_feature": "before",
        "query_items": 836,
        "gallery_items": 836,
        "reranking": "no",
        "return_pattern": 3,
        "test_batch_size": 128,
        "test_size": [256, 128],
    }
    if protocol != expected_protocol:
        blockers.append("scientific_protocol_drift")

    gpu = _gpu_state()
    gpu_eligible = bool(gpu.get("query_passed")) and int(gpu["memory_used_mib"]) < 500
    if not gpu.get("query_passed"):
        blockers.append("gpu_query_failed")
    elif not gpu_eligible:
        blockers.append("gpu_memory_gate")
    launch_allowed = not blockers
    receipt = {
        "schema_version": "1.0",
        "mode": "preflight",
        "status": "READY" if launch_allowed else "BLOCKED",
        "launch_allowed": launch_allowed,
        "required_memory_used_strictly_below_mib": 500,
        "gpu": gpu,
        "blockers": blockers,
        "file_checks": file_checks,
        "source_check": source_check,
        "package_check": package_check,
        "scientific_protocol": protocol,
        "resolved_command": _resolved_command(output_dir),
        "resolved_environment": {
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "runner_sha256": _sha256(Path(__file__)),
        "upstream_command_executed": False,
        "metric_result": None,
        "claim_boundary": "preflight only; no MFRNet model import, GPU forward, metric, parity, or SOTA claim",
    }
    return receipt, launch_allowed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "official128"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt, _ = _preflight(args.output_dir)
    if args.mode == "official128":
        receipt, returncode = _execute_official(receipt, args.output_dir)
    else:
        returncode = 0
    _atomic_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output_dir / "receipt.json")}))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
