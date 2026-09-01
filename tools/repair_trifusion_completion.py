#!/usr/bin/env python3
"""Repair one completed TriFusion run whose post-training audit missed an import.

The repair is deliberately narrower than a resume: it accepts only an epoch-60
``phase=complete`` recovery endpoint, forbids every optimizer step, preserves
the original failed launch ledger, and creates an append-only repair ledger.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
VARIANT = "trifusion_circ_urgc"
MISSING_LOADER_SYMBOL = "build_rgbnt201_record_eval_loader"
MISSING_LOADER_ERROR = (
    "NameError: name 'build_rgbnt201_record_eval_loader' is not defined"
)
EVIDENCE_SCOPE = "mean_only_training_input"
EXPECTED_RUNNER_SHA256 = (
    "50540f112d99b55e761be91eaa36a273444c0318c9929929cb8a62d8cb25897c"
)
EXPECTED_CONFIG_SHA256 = (
    "2ec65df9fd71e6b8dc2754a741266557fc72ba93ab13d1644dd62dacd2353fb7"
)
EXPECTED_RUNNER = PROJECT / "tools/run_trifusion_experiment.py"
EXPECTED_CONFIG = (
    PROJECT
    / "configs/RGBNT201/TriFusion-circ-urgc-shared-semantic-rtx3090-v3.yml"
)
EXPECTED_OUTPUT = Path(
    "/root/autodl-tmp/trifusion-v2/artifacts/"
    "trifusion_shared_semantic_circ_urgc_v3_amp_safe_dev_seed42"
)
EXPECTED_LEDGER = Path(
    "/root/autodl-tmp/trifusion-v2/artifacts/"
    "trifusion_shared_semantic_circ_urgc_v3_amp_safe_dev_seed42_launch_ledger"
)
EXPECTED_FAILED_ENTRY = EXPECTED_LEDGER / "launch-0002"
FAILED_SNAPSHOT_BINDINGS = {
    "dev_worker_result.json": "failed_worker_result",
    "dev_worker.log": "failed_worker_log",
    "run_summary.json": "failed_run_summary",
    "best_dev_receipt.json": "best_receipt",
    "latest.json": "recovery_manifest",
    "launch_completion_receipt.json": "failed_completion_receipt",
    "launch_prelaunch_receipt.json": "failed_prelaunch_receipt",
    "launch_runner.log": "failed_runner_log",
}


class CompletionRepairInterrupted(RuntimeError):
    """Raised when a catchable process signal requests transactional rollback."""


def _install_repair_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise CompletionRepairInterrupted(
            f"completion repair interrupted by signal {signum}"
        )

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json(dict(payload)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"completion repair expected a JSON object: {path}")
    return value


def _contained_file(root: Path, value: object, label: str) -> Path:
    path = (root / str(value)).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"completion repair rejected unsafe or absent {label}")
    return path


def _require_zero_official_access(*documents: Mapping[str, Any]) -> None:
    for document in documents:
        values = [
            value
            for key, value in document.items()
            if "official_test_access" in str(key)
        ]
        if any(value not in (0, False) for value in values):
            raise ValueError("completion repair rejected nonzero official test access")


def _validate_canonical_bindings(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    ledger_dir: Path,
    runner_path: Path,
    config_path: Path,
) -> None:
    resolved = {
        "output": output_dir.expanduser().resolve(),
        "failed": failed_ledger_entry.expanduser().resolve(),
        "ledger": ledger_dir.expanduser().resolve(),
        "runner": runner_path.expanduser().resolve(),
        "config": config_path.expanduser().resolve(),
    }
    expected = {
        "output": EXPECTED_OUTPUT.resolve(),
        "failed": EXPECTED_FAILED_ENTRY.resolve(),
        "ledger": EXPECTED_LEDGER.resolve(),
        "runner": EXPECTED_RUNNER.resolve(),
        "config": EXPECTED_CONFIG.resolve(),
    }
    if resolved != expected:
        raise ValueError("completion repair rejected noncanonical paths")
    if (
        resolved["failed"].parent != resolved["ledger"]
        or resolved["output"] == resolved["ledger"]
        or resolved["output"] in resolved["ledger"].parents
        or resolved["ledger"] in resolved["output"].parents
        or _sha256(resolved["runner"]) != EXPECTED_RUNNER_SHA256
        or _sha256(resolved["config"]) != EXPECTED_CONFIG_SHA256
    ):
        raise ValueError("completion repair rejected canonical binding drift")


def validate_failed_state(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    runner_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate the exact safe failure seam before any artifact is changed."""

    output_dir = output_dir.expanduser().resolve()
    failed_ledger_entry = failed_ledger_entry.expanduser().resolve()
    runner_path = runner_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    required_files = {
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": output_dir / ".resume/latest.json",
        "best_receipt": output_dir / "best_dev_receipt.json",
        "best_checkpoint": output_dir / "best_dev_model.pth",
        "failed_worker_result": output_dir / "dev_worker_result.json",
        "failed_worker_log": output_dir / "dev_worker.log",
        "failed_run_summary": output_dir / "run_summary.json",
        "failed_completion_receipt": failed_ledger_entry
        / "completion_receipt.json",
        "failed_prelaunch_receipt": failed_ledger_entry / "prelaunch_receipt.json",
        "failed_runner_log": failed_ledger_entry / "runner.log",
        "runner": runner_path,
        "config": config_path,
    }
    missing = sorted(name for name, path in required_files.items() if not path.is_file())
    if missing:
        raise ValueError(f"completion repair missing required files: {missing}")

    identity = _load_json(required_files["run_identity"])
    manifest = _load_json(required_files["recovery_manifest"])
    best = _load_json(required_files["best_receipt"])
    worker = _load_json(required_files["failed_worker_result"])
    summary = _load_json(required_files["failed_run_summary"])
    failed_completion = _load_json(required_files["failed_completion_receipt"])
    _require_zero_official_access(identity, manifest, best, worker, summary, failed_completion)

    if (
        identity.get("data_mode") != "development"
        or identity.get("variant") != VARIANT
        or identity.get("runner_sha256") != _sha256(runner_path)
        or identity.get("config_sha256") != _sha256(config_path)
        or identity.get("scientific_evidence_eligible") is not True
        or identity.get("contract_testing") is not False
        or identity.get("official_test_access_during_development") is not False
    ):
        raise ValueError("completion repair rejected run identity drift")
    if (
        int(manifest.get("epoch", -1)) != 60
        or manifest.get("phase") != "complete"
        or manifest.get("run_identity_sha256")
        != _sha256(required_files["run_identity"])
    ):
        raise ValueError("completion repair requires the complete recovery endpoint")
    current = manifest.get("current", {})
    current_path = _contained_file(
        output_dir, current.get("path", ""), "current recovery generation"
    )
    if current.get("sha256") != _sha256(current_path):
        raise ValueError("completion repair rejected recovery generation drift")
    import torch

    saved = torch.load(current_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("completion repair rejected a non-dictionary recovery generation")
    if (
        int(saved.get("epoch", -1)) != 60
        or saved.get("phase") != "complete"
        or int(saved.get("dev_evaluation_count", -1)) != 60
        or saved.get("run_identity_sha256")
        != _sha256(required_files["run_identity"])
        or int(saved.get("best_epoch", -1)) != int(best.get("epoch", -2))
        or float(saved.get("best_map", float("nan")))
        != float(best.get("dev_selection_mAP", float("inf")))
        or saved.get("best_checkpoint_sha256")
        != _sha256(required_files["best_checkpoint"])
    ):
        raise ValueError(
            "completion repair requires generation-internal epoch-60 complete state"
        )
    evidence = manifest.get("completion_evidence", {})
    if (
        int(evidence.get("epoch", -1)) != 60
        or evidence.get("phase") != "complete"
        or evidence.get("scientific_evidence_eligible") is not True
        or evidence.get("best_checkpoint_sha256")
        != _sha256(required_files["best_checkpoint"])
    ):
        raise ValueError("completion repair requires intact completion evidence")
    checkpoint_path = Path(str(best.get("checkpoint", ""))).expanduser().resolve()
    if (
        checkpoint_path != required_files["best_checkpoint"].resolve()
        or best.get("checkpoint_sha256") != _sha256(checkpoint_path)
        or best.get("scientific_evidence_eligible") is not True
        or int(best.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("completion repair rejected best-checkpoint drift")
    if (
        worker.get("status") != "FAILED"
        or worker.get("error") != MISSING_LOADER_ERROR
        or int(worker.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("completion repair rejected a different worker failure")
    if (
        summary.get("status") != "FAILED"
        or "dev_worker_failed" not in summary.get("blockers", [])
        or int(summary.get("worker_returncode", -1)) != 4
        or int(summary.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("completion repair rejected the failed run summary")
    if (
        failed_completion.get("status") != "FAIL"
        or int(failed_completion.get("returncode", -1)) != 2
        or int(failed_completion.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("completion repair rejected the failed launch receipt")

    return {
        "schema_version": "trifusion-completion-repair-input-v1",
        "failed_symbol": MISSING_LOADER_SYMBOL,
        "failure_error": MISSING_LOADER_ERROR,
        "recovery_epoch": 60,
        "recovery_phase": "complete",
        "best_epoch": int(best["epoch"]),
        "best_map": float(best["dev_selection_mAP"]),
        "training_reexecution_allowed": False,
        "official_test_access_count": 0,
        "artifacts_sha256": {
            name: _sha256(path) for name, path in required_files.items()
        },
    }


def _reserve_repair_entry(ledger_dir: Path) -> Path:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for sequence in range(1, 10000):
        entry = ledger_dir / f"repair-{sequence:04d}"
        try:
            entry.mkdir()
            return entry
        except FileExistsError:
            continue
    raise RuntimeError("completion repair ledger exhausted")


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


class AuditPathRowCollate:
    """Adapt the legacy val-collate path strings to the frozen audit contract."""

    def __init__(self, original_collate: Any) -> None:
        self.original_collate = original_collate

    def __call__(self, batch: object) -> tuple[Any, ...]:
        collated = self.original_collate(batch)
        if not isinstance(collated, tuple) or len(collated) != 6:
            raise ValueError("completion repair rejected audit collate structure")
        *prefix, paths = collated
        if not isinstance(paths, tuple) or any(
            not isinstance(path, str) for path in paths
        ):
            raise ValueError("completion repair expected tuple[str] audit paths")
        return (*prefix, tuple((path,) for path in paths))


def _adapt_audit_loader_paths(loader: Any) -> Any:
    loader.collate_fn = AuditPathRowCollate(loader.collate_fn)
    return loader


def _load_frozen_runner(runner_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "trifusion_frozen_completion_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("completion repair could not load the frozen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from modeling.trifusion.data import (
        build_rgbnt201_record_eval_loader as original_record_eval_loader,
    )

    def build_audit_record_eval_loader(*args: object, **kwargs: object) -> Any:
        return _adapt_audit_loader_paths(
            original_record_eval_loader(*args, **kwargs)
        )

    setattr(module, MISSING_LOADER_SYMBOL, build_audit_record_eval_loader)
    return module


def _is_worker_command(command: object) -> bool:
    return isinstance(command, (list, tuple)) and "--_worker" in command


def _worker_failure_message(result_path: Path, returncode: int) -> str:
    if not result_path.is_file():
        return f"completion repair audit worker returned {returncode} without a result"
    result = _load_json(result_path)
    return (
        f"completion repair audit worker returned {returncode}: "
        f"{result.get('error', 'unknown worker failure')}"
    )


def _success_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "run_summary": _artifact_evidence(output_dir / "run_summary.json"),
        "run_identity": _artifact_evidence(output_dir / "run_identity.json"),
        "recovery_manifest": _artifact_evidence(output_dir / ".resume/latest.json"),
        "worker_result": _artifact_evidence(output_dir / "dev_worker_result.json"),
        "best_receipt": _artifact_evidence(output_dir / "best_dev_receipt.json"),
        "best_checkpoint": _artifact_evidence(output_dir / "best_dev_model.pth"),
        "router_calibration_receipt": _artifact_evidence(
            output_dir / "router_calibration_receipt.json"
        ),
    }


def _atomic_restore(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.completion-repair-restore-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _restore_failed_output(output_dir: Path, entry: Path) -> None:
    snapshots = entry / "failed_snapshots"
    destinations = {
        "dev_worker_result.json": output_dir / "dev_worker_result.json",
        "dev_worker.log": output_dir / "dev_worker.log",
        "run_summary.json": output_dir / "run_summary.json",
        "best_dev_receipt.json": output_dir / "best_dev_receipt.json",
        "latest.json": output_dir / ".resume/latest.json",
    }
    for name, destination in destinations.items():
        _atomic_restore(snapshots / name, destination)
    generated = output_dir / "router_calibration_receipt.json"
    if generated.exists():
        rollback_dir = entry / "rolled_back_generated"
        rollback_dir.mkdir(exist_ok=True)
        rollback_target = rollback_dir / generated.name
        if rollback_target.exists():
            raise RuntimeError("completion repair rollback target already exists")
        os.replace(generated, rollback_target)


def _verify_repair_document(entry: Path, completion_name: str) -> dict[str, Any]:
    entry = entry.expanduser().resolve()
    prelaunch_path = entry / "pre_repair_receipt.json"
    completion_path = entry / completion_name
    log_path = entry / "repair.log"
    if not all(path.is_file() for path in (prelaunch_path, completion_path, log_path)):
        raise ValueError("completion repair verification found an incomplete ledger")
    prelaunch = _load_json(prelaunch_path)
    completion = _load_json(completion_path)
    tool_path = Path(__file__).resolve()
    runner_path = EXPECTED_RUNNER.resolve()
    config_path = EXPECTED_CONFIG.resolve()
    output_dir = EXPECTED_OUTPUT.resolve()
    failed_ledger_entry = EXPECTED_FAILED_ENTRY.resolve()
    _validate_canonical_bindings(
        output_dir=output_dir,
        failed_ledger_entry=failed_ledger_entry,
        ledger_dir=EXPECTED_LEDGER,
        runner_path=runner_path,
        config_path=config_path,
    )
    expected_prelaunch_keys = {
        "schema_version",
        "status",
        "variant",
        "started_at_utc",
        "repair_tool",
        "repair_tool_sha256",
        "runner",
        "runner_sha256",
        "config",
        "config_sha256",
        "output_dir",
        "failed_ledger_entry",
        "failed_state",
        "training_reexecution_allowed",
        "official_test_access_count",
    }
    expected_completion_keys = {
        "schema_version",
        "status",
        "variant",
        "completed_at_utc",
        "pre_repair_receipt_sha256",
        "repair_log_sha256",
        "artifacts",
        "scientific_evidence_scope",
        "training_reexecuted",
        "optimizer_steps",
        "calibration_claim_eligible",
        "concentration_claim_eligible",
        "official_test_access_count",
    }
    if (
        entry.parent != EXPECTED_LEDGER.resolve()
        or not entry.name.startswith("repair-")
        or set(prelaunch) != expected_prelaunch_keys
        or prelaunch.get("schema_version") != "trifusion-completion-pre-repair-v1"
        or prelaunch.get("status") != "AUTHORIZED"
        or prelaunch.get("variant") != VARIANT
        or prelaunch.get("training_reexecution_allowed") is not False
        or int(prelaunch.get("official_test_access_count", -1)) != 0
        or prelaunch.get("repair_tool") != str(tool_path)
        or prelaunch.get("repair_tool_sha256") != _sha256(tool_path)
        or prelaunch.get("runner") != str(runner_path)
        or prelaunch.get("runner_sha256") != _sha256(runner_path)
        or prelaunch.get("config") != str(config_path)
        or prelaunch.get("config_sha256") != _sha256(config_path)
        or prelaunch.get("output_dir") != str(output_dir)
        or prelaunch.get("failed_ledger_entry") != str(failed_ledger_entry)
    ):
        raise ValueError("completion repair verification found pre-repair drift")
    failed_state = prelaunch.get("failed_state", {})
    original_hashes = failed_state.get("artifacts_sha256", {})
    snapshots = entry / "failed_snapshots"
    if (
        not isinstance(failed_state, dict)
        or not snapshots.is_dir()
        or set(path.name for path in snapshots.iterdir())
        != set(FAILED_SNAPSHOT_BINDINGS)
        or any(
            _sha256(snapshots / name) != original_hashes.get(binding)
            for name, binding in FAILED_SNAPSHOT_BINDINGS.items()
        )
        or _sha256(failed_ledger_entry / "completion_receipt.json")
        != original_hashes.get("failed_completion_receipt")
        or _sha256(failed_ledger_entry / "prelaunch_receipt.json")
        != original_hashes.get("failed_prelaunch_receipt")
        or _sha256(failed_ledger_entry / "runner.log")
        != original_hashes.get("failed_runner_log")
        or _sha256(output_dir / "run_identity.json")
        != original_hashes.get("run_identity")
        or _sha256(output_dir / "best_dev_model.pth")
        != original_hashes.get("best_checkpoint")
    ):
        raise ValueError("completion repair verification found original evidence drift")
    expected_artifacts = _success_artifacts(output_dir)
    if (
        set(completion) != expected_completion_keys
        or completion.get("schema_version") != "trifusion-completion-repair-v1"
        or completion.get("status") != "PASS"
        or completion.get("variant") != VARIANT
        or completion.get("scientific_evidence_scope") != EVIDENCE_SCOPE
        or completion.get("training_reexecuted") is not False
        or int(completion.get("optimizer_steps", -1)) != 0
        or int(completion.get("official_test_access_count", -1)) != 0
        or completion.get("pre_repair_receipt_sha256") != _sha256(prelaunch_path)
        or completion.get("repair_log_sha256") != _sha256(log_path)
        or completion.get("artifacts") != expected_artifacts
        or any(not item["exists"] for item in expected_artifacts.values())
    ):
        raise ValueError("completion repair verification found completion drift")
    summary = _load_json(output_dir / "run_summary.json")
    identity = _load_json(output_dir / "run_identity.json")
    manifest = _load_json(output_dir / ".resume/latest.json")
    worker = _load_json(output_dir / "dev_worker_result.json")
    best = _load_json(output_dir / "best_dev_receipt.json")
    calibration = _load_json(output_dir / "router_calibration_receipt.json")
    current = manifest.get("current", {})
    current_path = _contained_file(
        output_dir, current.get("path", ""), "current recovery generation"
    )
    if (
        summary.get("status") != "PASS"
        or summary.get("phase") != "complete"
        or summary.get("scientific_evidence_eligible") is not True
        or int(summary.get("official_test_access_count", -1)) != 0
        or summary.get("completion_repair", {}).get("repair_ledger_entry")
        != str(entry)
        or identity.get("runner_sha256") != prelaunch.get("runner_sha256")
        or identity.get("config_sha256") != prelaunch.get("config_sha256")
        or manifest.get("phase") != "complete"
        or int(manifest.get("epoch", -1)) != 60
        or current.get("sha256") != _sha256(current_path)
        or worker.get("status") != "COMPLETE"
        or worker.get("phase") != "complete"
        or int(worker.get("official_test_access_count", -1)) != 0
        or best.get("phase") != "complete"
        or int(best.get("official_test_access_count", -1)) != 0
        or calibration.get("official_test_access_count") != 0
        or calibration.get("causal_calibration_claim_eligible") is not False
    ):
        raise ValueError("completion repair verification found an invalid result chain")
    return completion


def verify_repair(entry: Path) -> dict[str, Any]:
    """Verify an append-only repair receipt and all result artifacts."""

    entry = entry.expanduser().resolve()
    if (entry / "failure_receipt.json").exists():
        raise ValueError("completion repair verification found a failure receipt")
    return _verify_repair_document(entry, "completion_receipt.json")


def run_repair(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    ledger_dir: Path,
    runner_path: Path,
    config_path: Path,
) -> Path:
    """Run only the missing audit and construct a separately receipted result."""

    output_dir = output_dir.expanduser().resolve()
    failed_ledger_entry = failed_ledger_entry.expanduser().resolve()
    ledger_dir = ledger_dir.expanduser().resolve()
    runner_path = runner_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    _validate_canonical_bindings(
        output_dir=output_dir,
        failed_ledger_entry=failed_ledger_entry,
        ledger_dir=ledger_dir,
        runner_path=runner_path,
        config_path=config_path,
    )
    snapshot = validate_failed_state(
        output_dir=output_dir,
        failed_ledger_entry=failed_ledger_entry,
        runner_path=runner_path,
        config_path=config_path,
    )
    entry = _reserve_repair_entry(ledger_dir)
    prelaunch = {
        "schema_version": "trifusion-completion-pre-repair-v1",
        "status": "AUTHORIZED",
        "variant": VARIANT,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_tool": str(Path(__file__).resolve()),
        "repair_tool_sha256": _sha256(Path(__file__).resolve()),
        "runner": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_dir": str(output_dir),
        "failed_ledger_entry": str(failed_ledger_entry),
        "failed_state": snapshot,
        "training_reexecution_allowed": False,
        "official_test_access_count": 0,
    }
    _exclusive_json(entry / "pre_repair_receipt.json", prelaunch)
    snapshots = entry / "failed_snapshots"
    snapshots.mkdir()
    snapshot_sources = {
        "dev_worker_result.json": output_dir / "dev_worker_result.json",
        "dev_worker.log": output_dir / "dev_worker.log",
        "run_summary.json": output_dir / "run_summary.json",
        "best_dev_receipt.json": output_dir / "best_dev_receipt.json",
        "latest.json": output_dir / ".resume/latest.json",
        "launch_completion_receipt.json": failed_ledger_entry
        / "completion_receipt.json",
        "launch_prelaunch_receipt.json": failed_ledger_entry
        / "prelaunch_receipt.json",
        "launch_runner.log": failed_ledger_entry / "runner.log",
    }
    for name, source in snapshot_sources.items():
        shutil.copyfile(source, snapshots / name)
    copied_hashes = {
        name: _sha256(snapshots / name) for name in snapshot_sources
    }
    expected_copied_hashes = {
        name: snapshot["artifacts_sha256"][FAILED_SNAPSHOT_BINDINGS[name]]
        for name in snapshot_sources
    }
    if copied_hashes != expected_copied_hashes:
        raise RuntimeError("completion repair snapshot changed during capture")
    second_snapshot = validate_failed_state(
        output_dir=output_dir,
        failed_ledger_entry=failed_ledger_entry,
        runner_path=runner_path,
        config_path=config_path,
    )
    if second_snapshot != snapshot:
        raise RuntimeError("completion repair source changed after snapshot capture")
    repair_log = entry / "repair.log"
    previous_signal_handlers = _install_repair_signal_handlers()
    try:
        with repair_log.open("x", encoding="utf-8") as log_handle, redirect_stdout(
            log_handle
        ), redirect_stderr(log_handle):
            module = _load_frozen_runner(runner_path)
            import torch

            original_optimizer_step = torch.optim.AdamW.step

            def forbidden_optimizer_step(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("completion repair forbids optimizer steps")

            torch.optim.AdamW.step = forbidden_optimizer_step
            try:
                worker_returncode = module._worker_dev(
                    config_path,
                    VARIANT,
                    output_dir,
                    data_mode="development",
                )
            finally:
                torch.optim.AdamW.step = original_optimizer_step
            if worker_returncode != 0:
                raise RuntimeError(
                    _worker_failure_message(
                        output_dir / "dev_worker_result.json", worker_returncode
                    )
                )
            gc.collect()
            torch.cuda.empty_cache()

            original_subprocess_run = module.subprocess.run

            def patched_subprocess_run(
                command: object, *args: object, **kwargs: object
            ) -> Any:
                if _is_worker_command(command):
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "completion repair reused the verified post-training "
                            "audit result; optimizer steps were forbidden\n"
                        ),
                        stderr="",
                    )
                return original_subprocess_run(command, *args, **kwargs)

            module.subprocess.run = patched_subprocess_run
            try:
                summary, summary_returncode = module._dev(
                    config_path,
                    VARIANT,
                    output_dir,
                    data_mode="development",
                )
            finally:
                module.subprocess.run = original_subprocess_run
            if summary_returncode != 0 or summary.get("status") != "PASS":
                raise RuntimeError("completion repair could not qualify the result summary")
            summary["completion_repair"] = {
                "schema_version": "trifusion-completion-repair-link-v1",
                "repair_ledger_entry": str(entry),
                "repair_tool_sha256": prelaunch["repair_tool_sha256"],
                "failed_ledger_entry": str(failed_ledger_entry),
                "failed_completion_receipt_sha256": snapshot["artifacts_sha256"][
                    "failed_completion_receipt"
                ],
                "failure_error": MISSING_LOADER_ERROR,
                "action": (
                    "post_training_router_calibration_audit_only_with_"
                    "frozen_val_collate_path_shape_adapter"
                ),
                "training_reexecuted": False,
                "optimizer_steps": 0,
                "official_test_access_count": 0,
            }
            module._atomic_json(output_dir / "run_summary.json", summary)
        with repair_log.open("rb") as handle:
            os.fsync(handle.fileno())
        completion = {
            "schema_version": "trifusion-completion-repair-v1",
            "status": "PASS",
            "variant": VARIANT,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pre_repair_receipt_sha256": _sha256(
                entry / "pre_repair_receipt.json"
            ),
            "repair_log_sha256": _sha256(repair_log),
            "artifacts": _success_artifacts(output_dir),
            "scientific_evidence_scope": EVIDENCE_SCOPE,
            "training_reexecuted": False,
            "optimizer_steps": 0,
            "calibration_claim_eligible": False,
            "concentration_claim_eligible": False,
            "official_test_access_count": 0,
        }
        candidate_path = entry / "completion_candidate.json"
        _exclusive_json(candidate_path, completion)
        _verify_repair_document(entry, candidate_path.name)
        os.replace(candidate_path, entry / "completion_receipt.json")
        return entry
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        for signum in previous_signal_handlers:
            signal.signal(signum, signal.SIG_IGN)
        with repair_log.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMPLETION_REPAIR_EXCEPTION\n")
            handle.write(traceback.format_exc())
            handle.flush()
            os.fsync(handle.fileno())
        rollback_verified = False
        rollback_error = None
        try:
            _restore_failed_output(output_dir, entry)
            validate_failed_state(
                output_dir=output_dir,
                failed_ledger_entry=failed_ledger_entry,
                runner_path=runner_path,
                config_path=config_path,
            )
            rollback_verified = True
        except Exception as restore_error:
            rollback_error = f"{type(restore_error).__name__}: {restore_error}"
        failure_path = entry / "failure_receipt.json"
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema_version": "trifusion-completion-repair-failure-v1",
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "rollback_verified": rollback_verified,
                    "rollback_error": rollback_error,
                    "training_reexecuted": False,
                    "optimizer_steps": 0,
                    "official_test_access_count": 0,
                },
            )
        raise
    finally:
        _restore_signal_handlers(previous_signal_handlers)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--failed-ledger-entry", type=Path)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    repair_fields = (
        args.output_dir,
        args.failed_ledger_entry,
        args.ledger_dir,
        args.runner,
        args.config,
    )
    if args.verify:
        if any(value is not None for value in repair_fields):
            parser.error("--verify cannot be combined with repair arguments")
    elif any(value is None for value in repair_fields):
        parser.error("all repair arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        receipt = verify_repair(args.verify)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    entry = run_repair(
        output_dir=args.output_dir,
        failed_ledger_entry=args.failed_ledger_entry,
        ledger_dir=args.ledger_dir,
        runner_path=args.runner,
        config_path=args.config,
    )
    print(json.dumps({"status": "PASS", "repair_ledger_entry": str(entry)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
