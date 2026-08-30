#!/usr/bin/env python3
"""Capture a bounded post-launch provenance receipt for a live DeMo process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEMO_COMMIT = "b4f323a430b32e3a1637c3e7acb25868cb52e9cd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/baselines/DeMo"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/root/mmreid-trifusion/runs/demo_rgbnt201_seed42_b32k4_tb64"
        ),
    )
    parser.add_argument(
        "--clip-weight",
        type=Path,
        default=Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt"),
    )
    parser.add_argument(
        "--dataset-audit",
        type=Path,
        default=Path("/root/mmreid-trifusion/artifacts/rgbnt201_audit_20260831.json"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    proc_root = Path("/proc") / str(args.pid)
    if not proc_root.is_dir():
        raise ProcessLookupError(args.pid)
    baseline_root = args.baseline_root.resolve()
    run_dir = args.run_dir.resolve()
    config_path = baseline_root / "configs/RGBNT201/DeMo.yml"
    train_entry = baseline_root / "tools/train.py"
    train_log = run_dir / "train_log.txt"
    driver_stdout = run_dir / "driver.stdout.log"
    checkpoint = run_dir / "DeMobest.pth"
    for required in (
        baseline_root,
        run_dir,
        config_path,
        train_entry,
        train_log,
        driver_stdout,
        args.clip_weight,
        args.dataset_audit,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    command = (proc_root / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    command = [token.decode("utf-8", errors="strict") for token in command]
    process_cwd = Path(os.readlink(proc_root / "cwd")).resolve()
    process_executable = os.readlink(proc_root / "exe")
    process_started = subprocess.run(
        ["ps", "-p", str(args.pid), "-o", "lstart="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    baseline_commit = git_output(baseline_root, "rev-parse", "HEAD")
    tracked_status = git_output(
        baseline_root, "status", "--porcelain", "--untracked-files=no"
    )
    expected_command = [
        "/root/miniconda3/envs/tri_reid/bin/python",
        "tools/run_demo_baseline.py",
        "--output-dir",
        str(run_dir),
        "--max-epochs",
        "50",
        "--eval-period",
        "1",
        "--checkpoint-period",
        "10",
        "--batch-size",
        "32",
        "--test-batch-size",
        "64",
        "--num-instances",
        "4",
        "--workers",
        "4",
        "--seed",
        "42",
    ]
    log_text = driver_stdout.read_text(encoding="utf-8", errors="strict")
    effective_fragments = (
        "'SOLVER.IMS_PER_BATCH', '32'",
        "'TEST.IMS_PER_BATCH', '64'",
        "'DATALOADER.NUM_INSTANCE', '4'",
        "'SOLVER.MAX_EPOCHS', '50'",
        "'SOLVER.SEED', '42'",
    )
    checks = {
        "process_command_exact": command == expected_command,
        "process_cwd_is_frozen_checkout": process_cwd == baseline_root,
        "baseline_commit_is_frozen": baseline_commit == DEMO_COMMIT,
        "baseline_tracked_tree_clean_at_observation": not tracked_status,
        "effective_overrides_present_in_driver_log": all(
            fragment in log_text for fragment in effective_fragments
        ),
        "checkpoint_present": checkpoint.is_file(),
    }
    report = {
        "valid": all(checks.values()),
        "scope": "post_launch_live_observation",
        "launch_attestation": False,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "The process started before the launcher commit/GPU guards were added.",
            "This receipt verifies the live command, effective logged overrides, and exact clean upstream checkout at observation time; it does not claim those later guards executed at launch.",
            "Mutable log and checkpoint hashes bind only the bytes present at observation time.",
        ],
        "checks": checks,
        "process": {
            "pid": args.pid,
            "started_local": process_started,
            "executable": process_executable,
            "cwd": str(process_cwd),
            "command": command,
        },
        "baseline": {
            "root": str(baseline_root),
            "commit": baseline_commit,
            "expected_commit": DEMO_COMMIT,
            "tracked_status": tracked_status,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "train_entry": str(train_entry),
            "train_entry_sha256": sha256(train_entry),
        },
        "inputs": {
            "clip_weight": str(args.clip_weight.resolve()),
            "clip_weight_sha256": sha256(args.clip_weight),
            "dataset_audit": str(args.dataset_audit.resolve()),
            "dataset_audit_sha256": sha256(args.dataset_audit),
        },
        "mutable_run_snapshot": {
            "run_dir": str(run_dir),
            "train_log_sha256": sha256(train_log),
            "driver_stdout_sha256": sha256(driver_stdout),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
