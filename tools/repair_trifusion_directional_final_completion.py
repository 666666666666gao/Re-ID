#!/usr/bin/env python3
"""Repair the post-official-audit seam of one frozen TriFusion final run.

This is intentionally not a resume utility.  It accepts only the exact seed-42
epoch-60 failure caused by the missing record-eval-loader import, forbids both
optimizer steps and official-test re-evaluation, preserves the failed launch
ledger, and publishes a separate append-only repair ledger.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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


VARIANT = "trifusion_circ_urgc"
MISSING_LOADER_SYMBOL = "build_rgbnt201_record_eval_loader"
MISSING_LOADER_ERROR = (
    "NameError: name 'build_rgbnt201_record_eval_loader' is not defined"
)
EVIDENCE_SCOPE = "calibrated_directional_training_input"
EXPECTED_RUNNER_SHA256 = (
    "50540f112d99b55e761be91eaa36a273444c0318c9929929cb8a62d8cb25897c"
)
EXPECTED_CONFIG_SHA256 = (
    "24fc81f984d1f4c6094a22edb0a4969d249467c0a5c75af052e952be1d3478ae"
)
EXPECTED_REPOSITORY_HEAD = "22c3beeced01e2f450a6b9bee8416d007283c7e4"
EXPECTED_PROJECT = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
EXPECTED_RUNNER = EXPECTED_PROJECT / "tools/run_trifusion_experiment.py"
EXPECTED_DIRECTIONAL_LAUNCHER = (
    EXPECTED_PROJECT / "tools/run_trifusion_directional_final.py"
)
EXPECTED_AUTHORIZATION = (
    EXPECTED_PROJECT / "protocols/circ_directional_final_authorization_v1.json"
)
EXPECTED_CONFIG = (
    EXPECTED_PROJECT
    / "configs/RGBNT201/"
    "TriFusion-circ-urgc-postfreeze-final-shared-semantic-rtx3090.yml"
)
EXPECTED_OUTPUT = Path(
    "/root/autodl-tmp/trifusion-v2/artifacts/"
    "trifusion_shared_semantic_circ_urgc_directional_final_seed42"
)
EXPECTED_LEDGER = Path(
    "/root/autodl-tmp/trifusion-v2/artifacts/"
    "trifusion_shared_semantic_circ_urgc_directional_final_seed42_launch_ledger"
)
EXPECTED_FAILED_ENTRY = EXPECTED_LEDGER / "launch-0001"
PROTECTED_NAMES = (
    "run_identity",
    "recovery_manifest",
    "recovery_generation",
    "fixed_checkpoint",
    "official_metrics",
    "official_guard",
    "failed_launch_receipt",
)
MUTABLE_SNAPSHOT_NAMES = {
    "final_worker_result.json": "failed_worker_result",
    "final_worker.log": "failed_worker_log",
    "run_summary.json": "failed_run_summary",
}


class FinalRepairInterrupted(RuntimeError):
    """Raised when a catchable signal requests transactional rollback."""


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"final repair expected a JSON object: {path}")
    return value


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json(dict(payload)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _contained_file(root: Path, value: object, label: str) -> Path:
    root = root.resolve()
    candidate = (root / str(value)).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"final repair rejected unsafe or absent {label}")
    return candidate


def _canonical_bindings(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    ledger_dir: Path,
    runner_path: Path,
    config_path: Path,
) -> None:
    observed = {
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
    if observed != expected or observed["failed"].parent != observed["ledger"]:
        raise ValueError("final repair rejected noncanonical paths")
    if (
        _sha256(observed["runner"]) != EXPECTED_RUNNER_SHA256
        or _sha256(observed["config"]) != EXPECTED_CONFIG_SHA256
    ):
        raise ValueError("final repair rejected frozen runner/config drift")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=EXPECTED_PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=EXPECTED_PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != EXPECTED_REPOSITORY_HEAD or dirty:
        raise ValueError("final repair requires the original clean repository identity")


def validate_failed_state(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    runner_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate the exact safe post-official failure seam without mutation."""

    output_dir = output_dir.expanduser().resolve()
    failed_ledger_entry = failed_ledger_entry.expanduser().resolve()
    runner_path = runner_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    paths = {
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": output_dir / ".resume/latest.json",
        "fixed_checkpoint": output_dir / "fixed_final_model.pth",
        "official_metrics": output_dir / "official_test_metrics.json",
        "official_guard": output_dir / "official_test_access_guard.json",
        "failed_worker_result": output_dir / "final_worker_result.json",
        "failed_worker_log": output_dir / "final_worker.log",
        "failed_run_summary": output_dir / "run_summary.json",
        "failed_launch_receipt": failed_ledger_entry / "failure_receipt.json",
        "failed_prelaunch_receipt": failed_ledger_entry / "prelaunch_receipt.json",
        "failed_launcher_log": failed_ledger_entry / "launcher.log",
        "runner": runner_path,
        "config": config_path,
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise ValueError(f"final repair missing required files: {missing}")
    if (
        (output_dir / "fixed_final_receipt.json").exists()
        or (output_dir / "router_calibration_receipt.json").exists()
        or (failed_ledger_entry / "completion_receipt.json").exists()
    ):
        raise ValueError("final repair requires the exact pre-audit-publication seam")

    identity = _load_json(paths["run_identity"])
    manifest = _load_json(paths["recovery_manifest"])
    metrics = _load_json(paths["official_metrics"])
    guard = _load_json(paths["official_guard"])
    worker = _load_json(paths["failed_worker_result"])
    summary = _load_json(paths["failed_run_summary"])
    launch_failure = _load_json(paths["failed_launch_receipt"])
    prelaunch = _load_json(paths["failed_prelaunch_receipt"])

    if (
        identity.get("data_mode") != "postfreeze-final"
        or identity.get("variant") != VARIANT
        or identity.get("runner_sha256") != _sha256(runner_path)
        or identity.get("config_sha256") != _sha256(config_path)
        or identity.get("scientific_evidence_eligible") is not True
        or identity.get("contract_testing") is not False
    ):
        raise ValueError("final repair rejected run identity drift")
    identity_sha = _sha256(paths["run_identity"])
    checkpoint_sha = _sha256(paths["fixed_checkpoint"])
    metrics_sha = _sha256(paths["official_metrics"])
    guard_sha = _sha256(paths["official_guard"])

    if (
        metrics.get("schema_version") != "trifusion-official-fixed-v1"
        or int(metrics.get("fixed_epoch", -1)) != 60
        or int(metrics.get("official_test_access_count", -1)) != 1
        or int(metrics.get("official_test_evaluation_count", -1)) != 1
        or metrics.get("further_model_selection") is not False
        or metrics.get("checkpoint_sha256") != checkpoint_sha
        or metrics.get("run_identity_sha256") != identity_sha
        or set(metrics.get("metrics_percent", {}))
        != {"fused", "cnn", "transformer", "mamba"}
    ):
        raise ValueError("final repair rejected official metrics drift")
    if (
        guard.get("schema_version") != "trifusion-official-access-guard-v1"
        or guard.get("status") != "COMPLETE"
        or int(guard.get("fixed_epoch", -1)) != 60
        or int(guard.get("official_test_access_count", -1)) != 1
        or guard.get("metrics_sha256") != metrics_sha
        or guard.get("checkpoint_sha256") != checkpoint_sha
        or guard.get("run_identity_sha256") != identity_sha
    ):
        raise ValueError("final repair rejected official access guard drift")

    if (
        int(manifest.get("epoch", -1)) != 60
        or manifest.get("phase") != "complete"
        or manifest.get("run_identity_sha256") != identity_sha
    ):
        raise ValueError("final repair requires the complete recovery endpoint")
    current = manifest.get("current", {})
    generation = _contained_file(
        output_dir, current.get("path", ""), "current recovery generation"
    )
    if current.get("sha256") != _sha256(generation):
        raise ValueError("final repair rejected recovery generation drift")
    import torch

    saved = torch.load(generation, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("final repair rejected non-dictionary recovery state")
    if (
        int(saved.get("epoch", -1)) != 60
        or saved.get("phase") != "complete"
        or int(saved.get("best_epoch", -1)) != 60
        or int(saved.get("dev_evaluation_count", -1)) != 0
        or int(saved.get("eval_loader_iteration_count", -1)) != 1
        or saved.get("best_metrics") != metrics.get("metrics_percent")
        or saved.get("best_checkpoint_sha256") != checkpoint_sha
        or saved.get("run_identity_sha256") != identity_sha
    ):
        raise ValueError("final repair requires generation-internal final evidence")
    evidence = manifest.get("completion_evidence", {})
    if (
        evidence.get("kind") != "postfreeze-final-fixed"
        or int(evidence.get("epoch", -1)) != 60
        or evidence.get("phase") != "complete"
        or int(evidence.get("official_test_evaluation_count", -1)) != 1
        or evidence.get("further_model_selection") is not False
        or evidence.get("fixed_metrics") != metrics.get("metrics_percent")
        or evidence.get("fixed_checkpoint_sha256") != checkpoint_sha
        or evidence.get("official_metrics_receipt_sha256") != metrics_sha
        or evidence.get("official_access_guard_sha256") != guard_sha
        or evidence.get("scientific_evidence_eligible") is not True
    ):
        raise ValueError("final repair rejected completion evidence drift")
    if (
        worker.get("status") != "FAILED"
        or worker.get("error") != MISSING_LOADER_ERROR
        or int(worker.get("official_test_access_count", -1)) != 1
    ):
        raise ValueError("final repair requires the exact missing-import failure")
    if (
        summary.get("status") != "FAILED"
        or "dev_worker_failed" not in summary.get("blockers", [])
        or int(summary.get("worker_returncode", -1)) != 4
    ):
        raise ValueError("final repair rejected failed run summary")
    worker_evidence = launch_failure.get("final_worker_result", {})
    metric_evidence = launch_failure.get("official_test_metrics", {})
    guard_evidence = launch_failure.get("official_test_guard", {})
    if (
        launch_failure.get("status") != "FAIL"
        or launch_failure.get("error")
        != "frozen final runner failed with return code 2"
        or int(launch_failure.get("official_test_access_count", -1)) != 1
        or launch_failure.get("official_test_access_ambiguous") is not False
        or worker_evidence.get("sha256") != _sha256(paths["failed_worker_result"])
        or metric_evidence.get("sha256") != metrics_sha
        or guard_evidence.get("sha256") != guard_sha
        or prelaunch.get("status") != "AUTHORIZED"
    ):
        raise ValueError("final repair rejected failed launch evidence")

    paths["recovery_generation"] = generation
    return {
        "schema_version": "trifusion-directional-final-repair-input-v1",
        "failed_symbol": MISSING_LOADER_SYMBOL,
        "failure_error": MISSING_LOADER_ERROR,
        "recovery_epoch": 60,
        "recovery_phase": "complete",
        "official_test_access_count": 1,
        "official_test_evaluation_count": 1,
        "training_reexecution_allowed": False,
        "official_test_reexecution_allowed": False,
        "metrics_percent": metrics["metrics_percent"],
        "artifacts_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


class AuditPathRowCollate:
    def __init__(self, original_collate: Any) -> None:
        self.original_collate = original_collate

    def __call__(self, batch: object) -> tuple[Any, ...]:
        collated = self.original_collate(batch)
        if not isinstance(collated, tuple) or len(collated) != 6:
            raise ValueError("final repair rejected audit collate structure")
        *prefix, paths = collated
        if not isinstance(paths, tuple) or any(
            not isinstance(path, str) for path in paths
        ):
            raise ValueError("final repair expected tuple[str] audit paths")
        return (*prefix, tuple((path,) for path in paths))


def _load_frozen_runner(runner_path: Path) -> Any:
    project = runner_path.resolve().parents[1]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location(
        "trifusion_frozen_directional_final_repair_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("final repair could not load the frozen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from modeling.trifusion.data import build_rgbnt201_record_eval_loader

    def build_audit_loader(*args: object, **kwargs: object) -> Any:
        loader = build_rgbnt201_record_eval_loader(*args, **kwargs)
        loader.collate_fn = AuditPathRowCollate(loader.collate_fn)
        return loader

    setattr(module, MISSING_LOADER_SYMBOL, build_audit_loader)
    return module


def _load_directional_launcher() -> Any:
    path = EXPECTED_DIRECTIONAL_LAUNCHER.resolve()
    spec = importlib.util.spec_from_file_location(
        "trifusion_frozen_directional_authorization_for_repair", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("final repair could not load the directional launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _directional_summary_authorization(
    runner: Any,
    authorized_preflight: Mapping[str, Any],
    evidence: Mapping[str, Any],
):
    """Apply the same narrow in-memory authorization used by the frozen launch."""

    original_preflight = runner._preflight
    original_identity = runner._run_identity

    def authorized_identity(preflight: dict[str, Any], variant: str) -> dict[str, Any]:
        identity = original_identity(preflight, variant)
        identity.update(
            {
                "scientific_evidence_scope": EVIDENCE_SCOPE,
                "directional_authorization": copy.deepcopy(dict(evidence)),
                "query_gallery_symmetry_claim_eligible": False,
                "calibration_claim_eligible": True,
            }
        )
        return identity

    runner._preflight = lambda *_args, **_kwargs: copy.deepcopy(
        dict(authorized_preflight)
    )
    runner._run_identity = authorized_identity
    try:
        yield
    finally:
        runner._preflight = original_preflight
        runner._run_identity = original_identity


def _forbidden_official_evaluation(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("final completion repair forbids official test re-evaluation")


def _reserve_entry(ledger_dir: Path) -> Path:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for sequence in range(1, 10000):
        entry = ledger_dir / f"repair-{sequence:04d}"
        try:
            entry.mkdir()
            return entry
        except FileExistsError:
            continue
    raise RuntimeError("final repair ledger exhausted")


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise FinalRepairInterrupted(f"final repair interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _atomic_restore(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.final-repair-restore-", dir=destination.parent
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
    for name in MUTABLE_SNAPSHOT_NAMES:
        _atomic_restore(snapshots / name, output_dir / name)
    for name in ("fixed_final_receipt.json", "router_calibration_receipt.json"):
        generated = output_dir / name
        if generated.exists():
            rollback_dir = entry / "rolled_back_generated"
            rollback_dir.mkdir(exist_ok=True)
            target = rollback_dir / name
            if target.exists():
                raise RuntimeError("final repair rollback target already exists")
            os.replace(generated, target)


def _success_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_json(output_dir / ".resume/latest.json")
    generation = _contained_file(
        output_dir, manifest.get("current", {}).get("path", ""), "recovery generation"
    )
    return {
        "run_summary": _artifact(output_dir / "run_summary.json"),
        "run_identity": _artifact(output_dir / "run_identity.json"),
        "recovery_manifest": _artifact(output_dir / ".resume/latest.json"),
        "recovery_generation": _artifact(generation),
        "fixed_receipt": _artifact(output_dir / "fixed_final_receipt.json"),
        "fixed_checkpoint": _artifact(output_dir / "fixed_final_model.pth"),
        "official_metrics": _artifact(output_dir / "official_test_metrics.json"),
        "official_guard": _artifact(output_dir / "official_test_access_guard.json"),
        "worker_result": _artifact(output_dir / "final_worker_result.json"),
        "worker_log": _artifact(output_dir / "final_worker.log"),
        "router_calibration_receipt": _artifact(
            output_dir / "router_calibration_receipt.json"
        ),
    }


def _verify_document(entry: Path, completion_name: str) -> dict[str, Any]:
    prelaunch_path = entry / "pre_repair_receipt.json"
    completion_path = entry / completion_name
    repair_log = entry / "repair.log"
    tool_snapshot = entry / "repair_tool.py"
    if not all(
        path.is_file()
        for path in (prelaunch_path, completion_path, repair_log, tool_snapshot)
    ):
        raise ValueError("final repair verification found an incomplete ledger")
    prelaunch = _load_json(prelaunch_path)
    completion = _load_json(completion_path)
    output_dir = Path(str(prelaunch.get("output_dir", ""))).resolve()
    failed_entry = Path(str(prelaunch.get("failed_ledger_entry", ""))).resolve()
    if (
        output_dir != EXPECTED_OUTPUT.resolve()
        or failed_entry != EXPECTED_FAILED_ENTRY.resolve()
        or entry.parent.resolve() != EXPECTED_LEDGER.resolve()
        or not entry.name.startswith("repair-")
        or prelaunch.get("status") != "AUTHORIZED"
        or prelaunch.get("repair_tool_snapshot_sha256") != _sha256(tool_snapshot)
        or prelaunch.get("training_reexecution_allowed") is not False
        or prelaunch.get("official_test_reexecution_allowed") is not False
        or int(prelaunch.get("official_test_access_count", -1)) != 1
    ):
        raise ValueError("final repair verification found pre-repair drift")
    failed_hashes = prelaunch.get("failed_state", {}).get("artifacts_sha256", {})
    manifest = _load_json(output_dir / ".resume/latest.json")
    generation = _contained_file(
        output_dir, manifest.get("current", {}).get("path", ""), "recovery generation"
    )
    protected_paths = {
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": output_dir / ".resume/latest.json",
        "recovery_generation": generation,
        "fixed_checkpoint": output_dir / "fixed_final_model.pth",
        "official_metrics": output_dir / "official_test_metrics.json",
        "official_guard": output_dir / "official_test_access_guard.json",
        "failed_launch_receipt": failed_entry / "failure_receipt.json",
    }
    if any(_sha256(path) != failed_hashes.get(name) for name, path in protected_paths.items()):
        raise ValueError("final repair verification found protected evidence drift")
    snapshots = entry / "failed_snapshots"
    if any(
        not (snapshots / name).is_file()
        or _sha256(snapshots / name) != failed_hashes.get(binding)
        for name, binding in MUTABLE_SNAPSHOT_NAMES.items()
    ):
        raise ValueError("final repair verification found failed snapshot drift")
    artifacts = _success_artifacts(output_dir)
    if (
        completion.get("status") != "PASS"
        or completion.get("schema_version")
        != "trifusion-directional-final-completion-repair-v1"
        or completion.get("scientific_evidence_scope") != EVIDENCE_SCOPE
        or completion.get("training_reexecuted") is not False
        or completion.get("official_test_reexecuted") is not False
        or int(completion.get("optimizer_steps", -1)) != 0
        or int(completion.get("official_test_access_count", -1)) != 1
        or int(completion.get("official_test_evaluation_count", -1)) != 1
        or completion.get("pre_repair_receipt_sha256") != _sha256(prelaunch_path)
        or completion.get("repair_log_sha256") != _sha256(repair_log)
        or completion.get("artifacts") != artifacts
        or any(not value["exists"] for value in artifacts.values())
    ):
        raise ValueError("final repair verification found completion drift")
    summary = _load_json(output_dir / "run_summary.json")
    worker = _load_json(output_dir / "final_worker_result.json")
    fixed = _load_json(output_dir / "fixed_final_receipt.json")
    metrics = _load_json(output_dir / "official_test_metrics.json")
    guard = _load_json(output_dir / "official_test_access_guard.json")
    calibration = _load_json(output_dir / "router_calibration_receipt.json")
    if (
        summary.get("status") != "PASS"
        or summary.get("phase") != "complete"
        or summary.get("completion_repair", {}).get("repair_ledger_entry")
        != str(entry)
        or summary.get("metric_result") != metrics.get("metrics_percent")
        or worker.get("status") != "COMPLETE"
        or worker.get("phase") != "complete"
        or int(worker.get("official_test_access_count", -1)) != 1
        or int(worker.get("official_test_evaluation_count", -1)) != 1
        or fixed.get("metrics_percent") != metrics.get("metrics_percent")
        or int(fixed.get("official_test_access_count", -1)) != 1
        or int(fixed.get("official_test_evaluation_count", -1)) != 1
        or int(guard.get("official_test_access_count", -1)) != 1
        or int(calibration.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("final repair verification found an invalid result chain")
    return completion


def verify_repair(entry: Path) -> dict[str, Any]:
    entry = entry.expanduser().resolve()
    if (entry / "failure_receipt.json").exists():
        raise ValueError("final repair verification found a failure receipt")
    return _verify_document(entry, "completion_receipt.json")


def run_repair(
    *,
    output_dir: Path,
    failed_ledger_entry: Path,
    ledger_dir: Path,
    runner_path: Path,
    config_path: Path,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    failed_ledger_entry = failed_ledger_entry.expanduser().resolve()
    ledger_dir = ledger_dir.expanduser().resolve()
    runner_path = runner_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    _canonical_bindings(
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
    entry = _reserve_entry(ledger_dir)
    tool_snapshot = entry / "repair_tool.py"
    shutil.copyfile(Path(__file__).resolve(), tool_snapshot)
    with tool_snapshot.open("rb") as handle:
        os.fsync(handle.fileno())
    prelaunch = {
        "schema_version": "trifusion-directional-final-pre-repair-v1",
        "status": "AUTHORIZED",
        "variant": VARIANT,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_tool": str(Path(__file__).resolve()),
        "repair_tool_sha256": _sha256(Path(__file__).resolve()),
        "repair_tool_snapshot_sha256": _sha256(tool_snapshot),
        "runner": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "output_dir": str(output_dir),
        "failed_ledger_entry": str(failed_ledger_entry),
        "failed_state": snapshot,
        "training_reexecution_allowed": False,
        "official_test_reexecution_allowed": False,
        "official_test_access_count": 1,
        "official_test_evaluation_count": 1,
    }
    _exclusive_json(entry / "pre_repair_receipt.json", prelaunch)
    snapshots = entry / "failed_snapshots"
    snapshots.mkdir()
    for name in MUTABLE_SNAPSHOT_NAMES:
        shutil.copyfile(output_dir / name, snapshots / name)
    if any(
        _sha256(snapshots / name)
        != snapshot["artifacts_sha256"][binding]
        for name, binding in MUTABLE_SNAPSHOT_NAMES.items()
    ):
        raise RuntimeError("final repair snapshot changed during capture")
    if validate_failed_state(
        output_dir=output_dir,
        failed_ledger_entry=failed_ledger_entry,
        runner_path=runner_path,
        config_path=config_path,
    ) != snapshot:
        raise RuntimeError("final repair source changed after snapshot capture")

    repair_log = entry / "repair.log"
    previous_handlers = _install_signal_handlers()
    try:
        with repair_log.open("x", encoding="utf-8") as log_handle, redirect_stdout(
            log_handle
        ), redirect_stderr(log_handle):
            module = _load_frozen_runner(runner_path)
            module._evaluate_official_fixed_endpoint_once = _forbidden_official_evaluation
            directional_launcher = _load_directional_launcher()
            authorized, evidence = directional_launcher._validate_authorization(
                authorization_path=EXPECTED_AUTHORIZATION,
                config_path=config_path,
                output_dir=output_dir,
                ledger_dir=ledger_dir,
            )
            identity = _load_json(output_dir / "run_identity.json")
            if identity.get("directional_authorization") != evidence:
                raise RuntimeError(
                    "final repair authorization differs from the frozen run identity"
                )
            import torch

            original_optimizer_step = torch.optim.AdamW.step

            def forbidden_optimizer_step(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("final completion repair forbids optimizer steps")

            torch.optim.AdamW.step = forbidden_optimizer_step
            try:
                worker_returncode = module._worker_dev(
                    config_path,
                    VARIANT,
                    output_dir,
                    data_mode="postfreeze-final",
                )
            finally:
                torch.optim.AdamW.step = original_optimizer_step
            if worker_returncode != 0:
                result = _load_json(output_dir / "final_worker_result.json")
                raise RuntimeError(
                    "final repair audit worker failed: "
                    f"{result.get('error', 'unknown worker failure')}"
                )
            for name in PROTECTED_NAMES:
                protected_path = {
                    "run_identity": output_dir / "run_identity.json",
                    "recovery_manifest": output_dir / ".resume/latest.json",
                    "recovery_generation": _contained_file(
                        output_dir,
                        _load_json(output_dir / ".resume/latest.json")["current"]["path"],
                        "recovery generation",
                    ),
                    "fixed_checkpoint": output_dir / "fixed_final_model.pth",
                    "official_metrics": output_dir / "official_test_metrics.json",
                    "official_guard": output_dir / "official_test_access_guard.json",
                    "failed_launch_receipt": failed_ledger_entry / "failure_receipt.json",
                }[name]
                if _sha256(protected_path) != snapshot["artifacts_sha256"][name]:
                    raise RuntimeError(f"final repair changed protected artifact: {name}")
            gc.collect()
            torch.cuda.empty_cache()

            original_subprocess_run = module.subprocess.run

            def patched_subprocess_run(
                command: object, *args: object, **kwargs: object
            ) -> Any:
                if isinstance(command, (list, tuple)) and "--_worker" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout="final repair reused verified audit-only worker result\n",
                        stderr="",
                    )
                return original_subprocess_run(command, *args, **kwargs)

            module.subprocess.run = patched_subprocess_run
            try:
                with _directional_summary_authorization(module, authorized, evidence):
                    summary, summary_returncode = module._dev(
                        config_path,
                        VARIANT,
                        output_dir,
                        data_mode="postfreeze-final",
                    )
            finally:
                module.subprocess.run = original_subprocess_run
            if summary_returncode != 0 or summary.get("status") != "PASS":
                print(
                    "FINAL_REPAIR_SUMMARY_REJECTED",
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                )
                raise RuntimeError("final repair could not qualify the repaired summary")
            summary["completion_repair"] = {
                "schema_version": "trifusion-directional-final-repair-link-v1",
                "repair_ledger_entry": str(entry),
                "failed_ledger_entry": str(failed_ledger_entry),
                "failed_launch_receipt_sha256": snapshot["artifacts_sha256"][
                    "failed_launch_receipt"
                ],
                "failure_error": MISSING_LOADER_ERROR,
                "action": "post_official_training_set_router_calibration_audit_only",
                "training_reexecuted": False,
                "optimizer_steps": 0,
                "official_test_reexecuted": False,
                "official_test_access_count": 1,
                "official_test_evaluation_count": 1,
            }
            module._atomic_json(output_dir / "run_summary.json", summary)
        with repair_log.open("rb") as handle:
            os.fsync(handle.fileno())
        completion = {
            "schema_version": "trifusion-directional-final-completion-repair-v1",
            "status": "PASS",
            "variant": VARIANT,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pre_repair_receipt_sha256": _sha256(entry / "pre_repair_receipt.json"),
            "repair_log_sha256": _sha256(repair_log),
            "artifacts": _success_artifacts(output_dir),
            "scientific_evidence_scope": EVIDENCE_SCOPE,
            "training_reexecuted": False,
            "optimizer_steps": 0,
            "official_test_reexecuted": False,
            "official_test_access_count": 1,
            "official_test_evaluation_count": 1,
            "query_gallery_symmetry_claim_eligible": False,
            "sota_claim_supported": False,
        }
        candidate = entry / "completion_candidate.json"
        _exclusive_json(candidate, completion)
        _verify_document(entry, candidate.name)
        os.replace(candidate, entry / "completion_receipt.json")
        return entry
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        with repair_log.open("a", encoding="utf-8") as handle:
            handle.write("\nFINAL_COMPLETION_REPAIR_EXCEPTION\n")
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
        if not (entry / "failure_receipt.json").exists():
            _exclusive_json(
                entry / "failure_receipt.json",
                {
                    "schema_version": (
                        "trifusion-directional-final-completion-repair-failure-v1"
                    ),
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "rollback_verified": rollback_verified,
                    "rollback_error": rollback_error,
                    "training_reexecuted": False,
                    "optimizer_steps": 0,
                    "official_test_reexecuted": False,
                    "official_test_access_count": 1,
                    "official_test_evaluation_count": 1,
                },
            )
        raise
    finally:
        _restore_signal_handlers(previous_handlers)


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
        print(json.dumps(verify_repair(args.verify), ensure_ascii=False, sort_keys=True))
        return 0
    entry = run_repair(
        output_dir=args.output_dir,
        failed_ledger_entry=args.failed_ledger_entry,
        ledger_dir=args.ledger_dir,
        runner_path=args.runner,
        config_path=args.config,
    )
    print(json.dumps(verify_repair(entry), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
