#!/usr/bin/env python3
"""Fail-closed launcher for a calibrated directional-only TriFusion final run.

This launcher never promotes the failed query/gallery symmetry audit.  It permits
the frozen final worker only when an in-memory counterfactual proves symmetry is
the sole parent preflight blocker and every registered artifact remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import traceback
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_trifusion_experiment as frozen_runner  # noqa: E402


FROZEN_RUNNER = PROJECT / "tools/run_trifusion_experiment.py"
FROZEN_RUNNER_SHA256 = (
    "50540f112d99b55e761be91eaa36a273444c0318c9929929cb8a62d8cb25897c"
)
AMP_SAFE_SITECUSTOMIZE = PROJECT / "tools/runtime_amp_safe/sitecustomize.py"
VARIANT = "trifusion_circ_urgc"
SCIENTIFIC_EVIDENCE_SCOPE = "calibrated_directional_training_input"
SOLE_BLOCKER = "invalid_circ_target_cache_evidence"
REQUIRED_PARENT_ARTIFACTS = (
    "scoring_receipt.json",
    "run_identity.json",
    "scored_source.json",
    "cache/receipt.json",
    "cache/calibration_receipt.json",
    "cache/targets.jsonl",
    "cache/symmetry_receipt.json",
    "cache/target_transfer_receipt.json",
)
REQUIRED_COMPLETION_ARTIFACTS = (
    "run_summary",
    "run_identity",
    "recovery_manifest",
    "recovery_checkpoint",
    "fixed_final_receipt",
    "fixed_checkpoint",
    "official_test_metrics",
    "official_test_guard",
    "final_worker_result",
    "router_calibration_receipt",
)
STATIC_AUTHORIZATION_EVIDENCE_KEYS = {
    "authorization_path",
    "authorization_sha256",
    "parent_root",
    "parent_artifacts_sha256",
    "parent_scoring_receipt_sha256",
    "parent_targets_sha256",
    "scientific_evidence_scope",
    "query_gallery_symmetry_claim_eligible",
    "calibration_claim_eligible",
    "official_test_access_count",
}


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    frozen_runner._atomic_json(path, dict(payload))


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(dict(payload)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _authorize_preflight(
    *,
    actual: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    symmetry: Mapping[str, Any],
    calibration: Mapping[str, Any],
    authorization_sha256: str,
) -> dict[str, Any]:
    """Narrow a symmetry-only blocker without changing its failed claim."""

    if (
        actual.get("status") != "BLOCKED"
        or actual.get("launch_allowed") is not False
        or list(actual.get("blockers", [])) != [SOLE_BLOCKER]
        or actual.get("model_constructed") is not False
        or actual.get("training_started") is not False
        or int(actual.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("directional authorization requires symmetry as the sole preflight blocker")
    if (
        counterfactual.get("status") != "READY"
        or counterfactual.get("launch_allowed") is not True
        or list(counterfactual.get("blockers", []))
        or counterfactual.get("model_constructed") is not False
        or counterfactual.get("training_started") is not False
        or int(counterfactual.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("directional authorization counterfactual preflight is not READY")

    sign_agreement = float(symmetry.get("sign_agreement", float("nan")))
    minimum_sign = float(symmetry.get("minimum_sign_agreement", float("nan")))
    spearman = float(symmetry.get("spearman", float("nan")))
    minimum_spearman = float(symmetry.get("minimum_spearman", float("nan")))
    if (
        symmetry.get("status") != "FAIL"
        or symmetry.get("claim_eligible") is not False
        or int(symmetry.get("sample_rows", 0)) < 2
        or not all(
            math.isfinite(value)
            for value in (sign_agreement, minimum_sign, spearman, minimum_spearman)
        )
        or (sign_agreement >= minimum_sign and spearman >= minimum_spearman)
    ):
        raise ValueError("directional authorization requires the registered failure to remain intact")

    coverage = calibration.get("empirical_concentration_coverage", {})
    if (
        calibration.get("status") != "COMPLETE"
        or not isinstance(coverage, dict)
        or not coverage
        or any(
            not isinstance(item, dict) or item.get("claim_eligible") is not True
            for item in coverage.values()
        )
    ):
        raise ValueError("directional authorization requires eligible calibration in every condition")
    if not isinstance(authorization_sha256, str) or len(authorization_sha256) != 64:
        raise ValueError("directional authorization SHA-256 is invalid")

    authorized = copy.deepcopy(dict(actual))
    authorized.update(
        {
            "status": "READY",
            "launch_allowed": True,
            "blockers": [],
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "directional_authorization_sha256": authorization_sha256,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "observed_failed_symmetry": copy.deepcopy(dict(symmetry)),
            "official_test_access_count": 0,
            "sota_claim_supported": False,
            "claim_boundary": (
                "calibrated directional CIRC training input only; query/gallery "
                "symmetry claim remains failed; no metric before fixed final endpoint"
            ),
        }
    )
    return authorized


def _counterfactual_preflight(config_path: Path, scoring_path: Path) -> dict[str, Any]:
    """Prove in memory that the registered symmetry result is the sole blocker."""

    original_read_text = Path.read_text
    scoring_path = scoring_path.resolve()

    def patched_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        text = original_read_text(path, *args, **kwargs)
        if path.resolve() == scoring_path:
            payload = json.loads(text)
            symmetry = dict(payload["symmetry_audit"])
            symmetry.update({"status": "PASS", "claim_eligible": True})
            payload["symmetry_audit"] = symmetry
            return json.dumps(payload, sort_keys=True)
        return text

    with patch.object(Path, "read_text", patched_read_text):
        return frozen_runner._preflight(
            config_path,
            VARIANT,
            data_mode="postfreeze-final",
        )


def _stable_authorization_evidence(
    *,
    authorization_path: Path,
    authorization_sha256: str,
    parent_root: Path,
    parent_artifacts_sha256: Mapping[str, str],
    parent_scoring_receipt_sha256: str,
    parent_targets_sha256: str,
) -> dict[str, Any]:
    """Return only immutable evidence suitable for the resumable run identity."""

    return {
        "authorization_path": str(authorization_path.resolve()),
        "authorization_sha256": authorization_sha256,
        "parent_root": str(parent_root.resolve()),
        "parent_artifacts_sha256": dict(parent_artifacts_sha256),
        "parent_scoring_receipt_sha256": parent_scoring_receipt_sha256,
        "parent_targets_sha256": parent_targets_sha256,
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "official_test_access_count": 0,
    }


def _validate_authorization(
    *,
    authorization_path: Path,
    config_path: Path,
    output_dir: Path,
    ledger_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization_path = authorization_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ledger_dir = ledger_dir.expanduser().resolve()
    authorization = _load_json(authorization_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("directional final configuration is invalid")

    parent_root = Path(str(authorization.get("parent_root", ""))).expanduser().resolve()
    expected_hashes = authorization.get("parent_artifacts_sha256", {})
    execution = authorization.get("execution", {})
    policy = authorization.get("policy", {})
    forbidden_environment = sorted(
        key for key, value in os.environ.items() if key.startswith("TRIFUSION_") and value
    )
    if (
        authorization.get("schema_version") != "circ-directional-final-authorization-v1"
        or authorization.get("scientific_evidence_scope") != SCIENTIFIC_EVIDENCE_SCOPE
        or int(authorization.get("official_test_access_count", -1)) != 0
        or set(expected_hashes) != set(REQUIRED_PARENT_ARTIFACTS)
        or policy.get("calibrated_uncertainty_control_allowed") is not True
        or policy.get("query_gallery_symmetry_claim_allowed") is not False
        or policy.get("parent_symmetry_failure_must_be_preserved") is not True
        or policy.get("further_model_selection") is not False
        or policy.get("official_test_exactly_once_after_fixed_endpoint") is not True
        or Path(str(execution.get("config", ""))).expanduser().resolve() != config_path
        or execution.get("config_sha256") != _sha256(config_path)
        or Path(str(execution.get("runner", ""))).expanduser().resolve()
        != FROZEN_RUNNER.resolve()
        or execution.get("runner_sha256") != FROZEN_RUNNER_SHA256
        or _sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256
        or Path(str(execution.get("launcher", ""))).expanduser().resolve()
        != Path(__file__).resolve()
        or execution.get("launcher_sha256") != _sha256(Path(__file__).resolve())
        or Path(str(execution.get("amp_safe_sitecustomize", ""))).expanduser().resolve()
        != AMP_SAFE_SITECUSTOMIZE.resolve()
        or execution.get("amp_safe_sitecustomize_sha256")
        != _sha256(AMP_SAFE_SITECUSTOMIZE)
        or Path(str(execution.get("output_dir", ""))).expanduser().resolve() != output_dir
        or Path(str(execution.get("ledger_dir", ""))).expanduser().resolve() != ledger_dir
        or forbidden_environment
        or output_dir == ledger_dir
        or output_dir in ledger_dir.parents
        or ledger_dir in output_dir.parents
    ):
        raise ValueError("directional final authorization or execution binding drifted")

    artifact_paths = {name: parent_root / name for name in REQUIRED_PARENT_ARTIFACTS}
    for name, path in artifact_paths.items():
        if not path.is_file() or _sha256(path) != expected_hashes[name]:
            raise ValueError(f"directional final parent artifact changed: {name}")
    scoring = _load_json(artifact_paths["scoring_receipt.json"])
    identity = _load_json(artifact_paths["run_identity.json"])
    cache = _load_json(artifact_paths["cache/receipt.json"])
    calibration = _load_json(artifact_paths["cache/calibration_receipt.json"])
    symmetry = _load_json(artifact_paths["cache/symmetry_receipt.json"])
    targets_sha256 = _sha256(artifact_paths["cache/targets.jsonl"])

    experiment = dict(config.get("EXPERIMENT", {}))
    optimization = dict(config.get("OPTIMIZATION", {}))
    loss = dict(config.get("LOSS", {}))
    circ = dict(config.get("CIRC", {}))
    protocol = dict(config.get("PROTOCOL", {}))
    if (
        experiment.get("VARIANT") != VARIANT
        or int(experiment.get("SEED", -1)) != 42
        or config.get("MODEL", {}).get("ARCHITECTURE") != "shared_semantic_residual"
        or int(optimization.get("MAX_EPOCHS", -1)) != 60
        or optimization.get("AMP") is not True
        or float(loss.get("EVIDENCE_WEIGHT", -1.0)) != 0.1
        or Path(str(circ.get("TARGET_CACHE", ""))).expanduser().resolve()
        != (parent_root / "cache").resolve()
        or Path(str(circ.get("SCORING_RECEIPT", ""))).expanduser().resolve()
        != (parent_root / "scoring_receipt.json").resolve()
        or protocol.get("MODEL_SELECTION") != "none_fixed_endpoint"
        or int(protocol.get("OFFICIAL_TEST_EVALUATIONS_AFTER_FIXED_ENDPOINT", -1)) != 1
        or protocol.get("OFFICIAL_TEST_DURING_DEVELOPMENT") is not False
    ):
        raise ValueError("directional final model, loss, or fixed endpoint configuration drifted")

    coverage = calibration.get("empirical_concentration_coverage", {})
    if (
        scoring.get("status") != "COMPLETE"
        or scoring.get("mode") != "postfreeze-final"
        or scoring.get("calibration_claim_eligible") is not True
        or scoring.get("scientific_evidence_eligible") is not True
        or scoring.get("contract_testing") is not False
        or int(scoring.get("official_test_access_count", -1)) != 0
        or scoring.get("symmetry_audit") != {
            key: value
            for key, value in symmetry.items()
            if key not in {"schema_version", "protocol_hash", "targets_sha256"}
        }
        or cache.get("query_gallery_symmetry_audit") != scoring.get("symmetry_audit")
        or cache.get("targets_sha256") != targets_sha256
        or scoring.get("targets_sha256") != targets_sha256
        or calibration.get("targets_sha256") != targets_sha256
        or symmetry.get("targets_sha256") != targets_sha256
        or cache.get("mode") != "postfreeze-final"
        or int(cache.get("row_count", 0)) <= 0
        or int(cache.get("cross_camera_primary_rows", -1)) != int(cache.get("row_count", -2))
        or int(cache.get("same_camera_only_rows", -1)) != 0
        or int(cache.get("invalid_support_rows", -1)) != 0
        or cache.get("zero_identity_overlap") is not True
        or int(cache.get("official_test_access_count", -1)) != 0
        or calibration.get("status") != "COMPLETE"
        or not isinstance(coverage, dict)
        or not coverage
        or any(item.get("claim_eligible") is not True for item in coverage.values())
        or symmetry.get("status") != "FAIL"
        or symmetry.get("claim_eligible") is not False
        or int(identity.get("official_test_access_count", -1)) != 0
        or identity.get("scientific_evidence_eligible") is not True
        or identity.get("contract_testing") is not False
    ):
        raise ValueError("directional final evidence policy is not satisfied")

    actual = frozen_runner._preflight(
        config_path,
        VARIANT,
        data_mode="postfreeze-final",
    )
    counterfactual = _counterfactual_preflight(
        config_path,
        artifact_paths["scoring_receipt.json"],
    )
    authorization_sha256 = _sha256(authorization_path)
    authorized = _authorize_preflight(
        actual=actual,
        counterfactual=counterfactual,
        symmetry=scoring["symmetry_audit"],
        calibration=calibration,
        authorization_sha256=authorization_sha256,
    )
    evidence = _stable_authorization_evidence(
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        parent_root=parent_root,
        parent_artifacts_sha256=expected_hashes,
        parent_scoring_receipt_sha256=_sha256(
            artifact_paths["scoring_receipt.json"]
        ),
        parent_targets_sha256=targets_sha256,
    )
    authorized["directional_authorization"] = evidence
    return authorized, evidence


def _reserve_ledger_entry(ledger_dir: Path) -> Path:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        entry = ledger_dir / f"launch-{index:04d}"
        try:
            entry.mkdir()
            return entry
        except FileExistsError:
            continue
    raise RuntimeError("directional final launch ledger exhausted")


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _path_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"completion artifact escapes {label}: {resolved}")
    return resolved


def _completion_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / ".resume/latest.json"
    recovery_checkpoint = output_dir / ".resume/__missing_current_generation__.pt"
    if manifest_path.is_file():
        try:
            manifest = _load_json(manifest_path)
            current = manifest.get("current", {})
            relative = Path(str(current.get("path", "")))
            if relative.is_absolute() or not str(relative):
                raise ValueError("invalid current recovery path")
            recovery_checkpoint = _path_within(
                output_dir / relative,
                output_dir,
                label="output directory",
            )
        except Exception:
            recovery_checkpoint = output_dir / ".resume/__invalid_current_generation__.pt"
    paths = {
        "run_summary": output_dir / "run_summary.json",
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": manifest_path,
        "recovery_checkpoint": recovery_checkpoint,
        "fixed_final_receipt": output_dir / "fixed_final_receipt.json",
        "fixed_checkpoint": output_dir / "fixed_final_model.pth",
        "official_test_metrics": output_dir / "official_test_metrics.json",
        "official_test_guard": output_dir / "official_test_access_guard.json",
        "final_worker_result": output_dir / "final_worker_result.json",
        "router_calibration_receipt": output_dir / "router_calibration_receipt.json",
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _completion_payload(
    *,
    entry: Path,
    output_dir: Path,
    returncode: int,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    summary_path = output_dir / "run_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    artifacts = _completion_artifacts(output_dir)
    successful = (
        returncode == 0
        and summary.get("status") == "PASS"
        and summary.get("mode") == "postfreeze-final"
        and int(summary.get("official_test_access_count", -1)) == 1
        and int(summary.get("official_test_evaluation_count", -1)) == 1
        and summary.get("further_model_selection") is False
        and summary.get("query_gallery_symmetry_claim_eligible") is False
        and summary.get("scientific_evidence_scope") == SCIENTIFIC_EVIDENCE_SCOPE
        and all(item["exists"] for item in artifacts.values())
        and (entry / "prelaunch_receipt.json").is_file()
        and (entry / "launcher.log").is_file()
    )
    return {
        "schema_version": "trifusion-directional-final-completion-v1",
        "status": "PASS" if successful else "FAIL",
        "returncode": returncode,
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "official_test_access_count": int(summary.get("official_test_access_count", 0)),
        "official_test_evaluation_count": int(
            summary.get("official_test_evaluation_count", 0)
        ),
        "output_dir": str(output_dir),
        "prelaunch_receipt_sha256": (
            _sha256(entry / "prelaunch_receipt.json")
            if (entry / "prelaunch_receipt.json").is_file()
            else None
        ),
        "runner_log_sha256": (
            _sha256(entry / "launcher.log")
            if (entry / "launcher.log").is_file()
            else None
        ),
        "artifacts": artifacts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _require_fields(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"completion artifact {label} field mismatch: {mismatches}")


def _validate_metrics(metrics: object) -> dict[str, Any]:
    if not isinstance(metrics, dict) or set(metrics) != {
        "fused",
        "cnn",
        "transformer",
        "mamba",
    }:
        raise ValueError("completion artifact metrics outputs are incomplete")
    required = {"mAP", "Rank-1", "Rank-5", "Rank-10"}
    for name, values in metrics.items():
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError(f"completion artifact metrics are incomplete for {name}")
        for metric, value in values.items():
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                raise ValueError(
                    f"completion artifact metric is invalid: {name}.{metric}={value}"
                )
    return metrics


def _verify_static_authorization_root(
    *,
    entry: Path,
    output_dir: Path,
    prelaunch: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-anchor a completed run without consulting live GPU/preflight state."""

    if set(evidence) != STATIC_AUTHORIZATION_EVIDENCE_KEYS:
        raise ValueError("completion authorization evidence keys are not exact")
    authorization_path = Path(str(evidence.get("authorization_path", ""))).resolve()
    if not authorization_path.is_file():
        raise ValueError("completion authorization file is missing")
    authorization_sha = _sha256(authorization_path)
    if authorization_sha != evidence.get("authorization_sha256"):
        raise ValueError("completion authorization file hash changed")
    authorization = _load_json(authorization_path)
    parent_root = Path(str(authorization.get("parent_root", ""))).resolve()
    parent_hashes = authorization.get("parent_artifacts_sha256")
    execution = authorization.get("execution")
    policy = authorization.get("policy")
    if (
        authorization.get("schema_version")
        != "circ-directional-final-authorization-v1"
        or authorization.get("scientific_evidence_scope")
        != SCIENTIFIC_EVIDENCE_SCOPE
        or int(authorization.get("official_test_access_count", -1)) != 0
        or not isinstance(parent_hashes, dict)
        or set(parent_hashes) != set(REQUIRED_PARENT_ARTIFACTS)
        or not isinstance(execution, dict)
        or not isinstance(policy, dict)
        or policy.get("calibrated_uncertainty_control_allowed") is not True
        or policy.get("query_gallery_symmetry_claim_allowed") is not False
        or policy.get("parent_symmetry_failure_must_be_preserved") is not True
        or policy.get("target_cache_must_remain_byte_identical") is not True
        or policy.get("further_model_selection") is not False
        or policy.get("official_test_exactly_once_after_fixed_endpoint") is not True
    ):
        raise ValueError("completion authorization policy or schema changed")
    if (
        str(authorization_path) != evidence.get("authorization_path")
        or str(parent_root) != evidence.get("parent_root")
        or parent_hashes != evidence.get("parent_artifacts_sha256")
    ):
        raise ValueError("completion authorization evidence differs from its trust root")
    for name in REQUIRED_PARENT_ARTIFACTS:
        path = parent_root / name
        if not path.is_file() or _sha256(path) != parent_hashes[name]:
            raise ValueError(f"completion parent artifact changed: {name}")
    if (
        parent_hashes["scoring_receipt.json"]
        != evidence.get("parent_scoring_receipt_sha256")
        or parent_hashes["cache/targets.jsonl"]
        != evidence.get("parent_targets_sha256")
    ):
        raise ValueError("completion parent artifact evidence is internally inconsistent")

    config_path = Path(str(execution.get("config", ""))).resolve()
    runner_path = Path(str(execution.get("runner", ""))).resolve()
    launcher_path = Path(str(execution.get("launcher", ""))).resolve()
    amp_path = Path(str(execution.get("amp_safe_sitecustomize", ""))).resolve()
    ledger_dir = Path(str(execution.get("ledger_dir", ""))).resolve()
    if (
        config_path != Path(str(prelaunch.get("config", ""))).resolve()
        or not config_path.is_file()
        or _sha256(config_path) != execution.get("config_sha256")
        or prelaunch.get("config_sha256") != execution.get("config_sha256")
        or runner_path != FROZEN_RUNNER.resolve()
        or not runner_path.is_file()
        or _sha256(runner_path) != FROZEN_RUNNER_SHA256
        or execution.get("runner_sha256") != FROZEN_RUNNER_SHA256
        or prelaunch.get("runner") != str(runner_path)
        or prelaunch.get("runner_sha256") != FROZEN_RUNNER_SHA256
        or launcher_path != Path(__file__).resolve()
        or not launcher_path.is_file()
        or _sha256(launcher_path) != execution.get("launcher_sha256")
        or prelaunch.get("launcher") != str(launcher_path)
        or prelaunch.get("launcher_sha256") != execution.get("launcher_sha256")
        or amp_path != AMP_SAFE_SITECUSTOMIZE.resolve()
        or not amp_path.is_file()
        or _sha256(amp_path) != execution.get("amp_safe_sitecustomize_sha256")
        or prelaunch.get("amp_safe_sitecustomize") != str(amp_path)
        or prelaunch.get("amp_safe_sitecustomize_sha256")
        != execution.get("amp_safe_sitecustomize_sha256")
        or Path(str(execution.get("output_dir", ""))).resolve() != output_dir
        or Path(str(prelaunch.get("output_dir", ""))).resolve() != output_dir
        or Path(str(prelaunch.get("ledger_dir", ""))).resolve() != ledger_dir
        or entry.parent.resolve() != ledger_dir
        or not entry.name.startswith("launch-")
        or len(entry.name) != len("launch-0001")
        or not entry.name.removeprefix("launch-").isdigit()
    ):
        raise ValueError("completion authorization execution binding changed")
    _require_fields(
        prelaunch,
        {
            "schema_version": "trifusion-directional-final-prelaunch-v1",
            "status": "AUTHORIZED",
            "variant": VARIANT,
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "official_test_access_count": 0,
        },
        label="prelaunch authorization",
    )
    recovery_before = prelaunch.get("recovery_before")
    if not isinstance(recovery_before, dict) or recovery_before.get("valid") is not True:
        raise ValueError("completion authorization prelaunch recovery is invalid")
    return authorization


def verify_completion(
    entry: Path,
    *,
    receipt_name: str = "completion_receipt.json",
) -> dict[str, Any]:
    """Revalidate the complete fixed-endpoint evidence chain from durable bytes."""

    if receipt_name not in {"completion_candidate.json", "completion_receipt.json"}:
        raise ValueError("completion artifact receipt name is not allowed")
    entry = entry.expanduser().resolve()
    if (entry / "failure_receipt.json").exists():
        raise ValueError("completion failure receipt conflicts with PASS evidence")
    receipt_path = entry / receipt_name
    receipt = _load_json(receipt_path)
    _require_fields(
        receipt,
        {
            "schema_version": "trifusion-directional-final-completion-v1",
            "status": "PASS",
            "returncode": 0,
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "official_test_access_count": 1,
            "official_test_evaluation_count": 1,
        },
        label="completion receipt",
    )
    output_dir = Path(str(receipt.get("output_dir", ""))).expanduser().resolve()
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        REQUIRED_COMPLETION_ARTIFACTS
    ):
        raise ValueError("completion artifact set is incomplete")
    for name in REQUIRED_COMPLETION_ARTIFACTS:
        record = artifacts[name]
        if not isinstance(record, dict) or record.get("exists") is not True:
            raise ValueError(f"completion artifact is missing: {name}")
        path = _path_within(
            Path(str(record.get("path", ""))),
            output_dir,
            label="output directory",
        )
        expected_sha = record.get("sha256")
        if (
            not path.is_file()
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or _sha256(path) != expected_sha
        ):
            raise ValueError(f"completion artifact changed or is corrupt: {name}")

    prelaunch_path = entry / "prelaunch_receipt.json"
    log_path = entry / "launcher.log"
    if (
        not prelaunch_path.is_file()
        or _sha256(prelaunch_path) != receipt.get("prelaunch_receipt_sha256")
        or not log_path.is_file()
        or _sha256(log_path) != receipt.get("runner_log_sha256")
    ):
        raise ValueError("completion artifact prelaunch or launcher log changed")

    paths = {name: Path(artifacts[name]["path"]).resolve() for name in artifacts}
    fixed_paths = {
        "run_summary": output_dir / "run_summary.json",
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": output_dir / ".resume/latest.json",
        "fixed_final_receipt": output_dir / "fixed_final_receipt.json",
        "fixed_checkpoint": output_dir / "fixed_final_model.pth",
        "official_test_metrics": output_dir / "official_test_metrics.json",
        "official_test_guard": output_dir / "official_test_access_guard.json",
        "final_worker_result": output_dir / "final_worker_result.json",
        "router_calibration_receipt": output_dir / "router_calibration_receipt.json",
    }
    if any(paths[name] != expected.resolve() for name, expected in fixed_paths.items()):
        raise ValueError("completion artifact path binding changed")

    prelaunch = _load_json(prelaunch_path)
    identity = _load_json(paths["run_identity"])
    manifest = _load_json(paths["recovery_manifest"])
    fixed = _load_json(paths["fixed_final_receipt"])
    official = _load_json(paths["official_test_metrics"])
    guard = _load_json(paths["official_test_guard"])
    worker = _load_json(paths["final_worker_result"])
    summary = _load_json(paths["run_summary"])
    router = _load_json(paths["router_calibration_receipt"])

    identity_sha = artifacts["run_identity"]["sha256"]
    checkpoint_sha = artifacts["fixed_checkpoint"]["sha256"]
    metrics_sha = artifacts["official_test_metrics"]["sha256"]
    guard_sha = artifacts["official_test_guard"]["sha256"]
    router_sha = artifacts["router_calibration_receipt"]["sha256"]
    manifest_sha = artifacts["recovery_manifest"]["sha256"]
    worker_sha = artifacts["final_worker_result"]["sha256"]
    metrics = _validate_metrics(official.get("metrics_percent"))

    authorization_evidence = identity.get("directional_authorization")
    if not isinstance(authorization_evidence, dict):
        raise ValueError("completion artifact run identity lacks authorization")
    _require_fields(
        authorization_evidence,
        {
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "official_test_access_count": 0,
        },
        label="directional authorization",
    )
    authorization_sha = authorization_evidence.get("authorization_sha256")
    if not isinstance(authorization_sha, str) or len(authorization_sha) != 64:
        raise ValueError("completion artifact authorization SHA-256 is invalid")
    if (
        prelaunch.get("directional_authorization") != authorization_evidence
        or summary.get("directional_authorization") != authorization_evidence
    ):
        raise ValueError("completion artifact authorization chain changed")
    _verify_static_authorization_root(
        entry=entry,
        output_dir=output_dir,
        prelaunch=prelaunch,
        evidence=authorization_evidence,
    )
    _require_fields(
        identity,
        {
            "data_mode": "postfreeze-final",
            "variant": VARIANT,
            "runner_sha256": FROZEN_RUNNER_SHA256,
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "all_171_training_identities": True,
            "former_dev_identities_training_only": True,
            "further_model_selection": False,
            "official_test_evaluations_before_fixed_endpoint": 0,
            "official_test_evaluations_after_fixed_endpoint": 1,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
        label="run identity",
    )
    optimization = identity.get("optimization", {})
    if not isinstance(optimization, dict) or int(optimization.get("max_epochs", -1)) != 60:
        raise ValueError("completion artifact run identity is not fixed at 60 epochs")
    config_sha = identity.get("config_sha256")
    if (
        not isinstance(config_sha, str)
        or len(config_sha) != 64
        or prelaunch.get("config_sha256") != config_sha
        or fixed.get("config_sha256") != config_sha
        or prelaunch.get("runner_sha256") != FROZEN_RUNNER_SHA256
    ):
        raise ValueError("completion artifact config or runner binding changed")

    _require_fields(
        official,
        {
            "schema_version": "trifusion-official-fixed-v1",
            "fixed_epoch": 60,
            "query_records": 836,
            "gallery_records": 836,
            "official_test_evaluation_count": 1,
            "official_test_access_count": 1,
            "further_model_selection": False,
            "checkpoint_sha256": checkpoint_sha,
            "run_identity_sha256": identity_sha,
        },
        label="official metrics",
    )
    _require_fields(
        guard,
        {
            "schema_version": "trifusion-official-access-guard-v1",
            "fixed_epoch": 60,
            "checkpoint_sha256": checkpoint_sha,
            "run_identity_sha256": identity_sha,
            "official_test_access_count": 1,
            "status": "COMPLETE",
            "metrics_sha256": metrics_sha,
        },
        label="official access guard",
    )

    _require_fields(
        manifest,
        {
            "schema_version": "1.0",
            "epoch": 60,
            "phase": "complete",
            "run_identity_sha256": identity_sha,
        },
        label="recovery manifest",
    )
    current = manifest.get("current", {})
    if not isinstance(current, dict):
        raise ValueError("completion artifact recovery current entry is invalid")
    current_relative = Path(str(current.get("path", "")))
    if current_relative.is_absolute():
        raise ValueError("completion artifact recovery path is absolute")
    current_path = _path_within(
        output_dir / current_relative,
        output_dir,
        label="output directory",
    )
    if (
        current_path != paths["recovery_checkpoint"]
        or current.get("sha256") != artifacts["recovery_checkpoint"]["sha256"]
    ):
        raise ValueError("completion artifact recovery checkpoint binding changed")
    recovery_evidence = manifest.get("completion_evidence", {})
    if not isinstance(recovery_evidence, dict):
        raise ValueError("completion artifact recovery evidence is missing")
    _require_fields(
        recovery_evidence,
        {
            "kind": "postfreeze-final-fixed",
            "epoch": 60,
            "phase": "complete",
            "fixed_epoch": 60,
            "official_test_evaluation_count": 1,
            "further_model_selection": False,
            "fixed_metrics": metrics,
            "fixed_checkpoint_sha256": checkpoint_sha,
            "official_metrics_receipt_sha256": metrics_sha,
            "official_access_guard_sha256": guard_sha,
            "run_identity_sha256": identity_sha,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
        label="recovery completion evidence",
    )

    _require_fields(
        fixed,
        {
            "schema_version": "trifusion-postfreeze-final-v1",
            "mode": "postfreeze-final",
            "epoch": 60,
            "phase": "complete",
            "metrics_percent": metrics,
            "checkpoint": str(paths["fixed_checkpoint"]),
            "checkpoint_sha256": checkpoint_sha,
            "router_calibration_receipt": str(paths["router_calibration_receipt"]),
            "router_calibration_receipt_sha256": router_sha,
            "training_split": "RGBNT201/train_171 all identities",
            "evaluation_split": "RGBNT201 official test",
            "further_model_selection": False,
            "official_test_evaluation_count": 1,
            "official_test_access_count": 1,
            "run_identity": str(paths["run_identity"]),
            "run_identity_sha256": identity_sha,
            "recovery_manifest": str(paths["recovery_manifest"]),
            "recovery_manifest_sha256": manifest_sha,
            "model_constructed": True,
            "training_started": True,
            "fatal_or_nonfinite_detected": False,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
        label="fixed final receipt",
    )
    _require_fields(
        worker,
        {
            "status": "COMPLETE",
            "mode": "postfreeze-final",
            "epoch": 60,
            "phase": "complete",
            "metrics_percent": metrics,
            "last_metrics_percent": metrics,
            "dev_evaluation_count": 0,
            "official_test_evaluation_count": 1,
            "official_test_access_count": 1,
            "further_model_selection": False,
            "query_records": 836,
            "gallery_records": 836,
            "train_records": 3951,
            "fixed_checkpoint": str(paths["fixed_checkpoint"]),
            "fixed_checkpoint_sha256": checkpoint_sha,
            "router_calibration_receipt": str(paths["router_calibration_receipt"]),
            "router_calibration_receipt_sha256": router_sha,
            "fatal_or_nonfinite_detected": False,
            "model_constructed": True,
            "training_started": True,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
        label="final worker result",
    )
    _require_fields(
        summary,
        {
            "status": "PASS",
            "mode": "postfreeze-final",
            "epoch": 60,
            "phase": "complete",
            "metrics_percent": metrics,
            "last_metrics_percent": metrics,
            "metric_result": metrics,
            "dev_evaluation_count": 0,
            "official_test_evaluation_count": 1,
            "official_test_access_count": 1,
            "further_model_selection": False,
            "query_records": 836,
            "gallery_records": 836,
            "train_records": 3951,
            "worker_result_sha256": worker_sha,
            "run_identity_sha256": identity_sha,
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "query_gallery_symmetry_claim_eligible": False,
            "calibration_claim_eligible": True,
            "sota_claim_supported": False,
            "fatal_or_nonfinite_detected": False,
            "model_constructed": True,
            "training_started": True,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
        label="run summary",
    )
    _require_fields(
        router,
        {
            "model_checkpoint": str(paths["fixed_checkpoint"]),
            "model_checkpoint_sha256": checkpoint_sha,
            "official_test_access_count": 0,
        },
        label="router calibration receipt",
    )

    verified = dict(receipt)
    verified["verified"] = True
    verified["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    return verified


def _failure_payload(
    *,
    entry: Path,
    output_dir: Path,
    error: BaseException,
) -> dict[str, Any]:
    guard_path = output_dir / "official_test_access_guard.json"
    metrics_path = output_dir / "official_test_metrics.json"
    worker_path = output_dir / "final_worker_result.json"
    try:
        guard = _load_json(guard_path) if guard_path.is_file() else {}
    except Exception:
        guard = {"status": "CORRUPT"}
    try:
        worker = _load_json(worker_path) if worker_path.is_file() else {}
    except Exception:
        worker = {"official_test_access_count": 1}
    access_count = 1 if (
        guard_path.is_file()
        or metrics_path.is_file()
        or int(worker.get("official_test_access_count", 0)) > 0
    ) else 0
    ambiguous = bool(
        access_count
        and (
            guard.get("status") != "COMPLETE"
            or not metrics_path.is_file()
        )
    )
    prelaunch_path = entry / "prelaunch_receipt.json"
    return {
        "schema_version": "trifusion-directional-final-failure-v1",
        "status": "FAIL",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "official_test_access_count": access_count,
        "official_test_access_ambiguous": ambiguous,
        "prelaunch_receipt_sha256": (
            _sha256(prelaunch_path) if prelaunch_path.is_file() else None
        ),
        "runner_log": _artifact(entry / "launcher.log"),
        "official_test_guard": _artifact(guard_path),
        "official_test_metrics": _artifact(metrics_path),
        "final_worker_result": _artifact(worker_path),
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@contextmanager
def _temporary_worker_environment(sitecustomize_root: Path):
    sentinel = object()
    prior = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", sentinel),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", sentinel),
    }
    os.environ["PYTHONPATH"] = str(sitecustomize_root.resolve())
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


class _LaunchInterrupted(RuntimeError):
    pass


@contextmanager
def _termination_as_exception():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        raise _LaunchInterrupted(f"received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _ignore_termination_signals():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def launch(
    *,
    authorization_path: Path,
    config_path: Path,
    output_dir: Path,
    ledger_dir: Path,
    preflight_only: bool = False,
) -> int:
    authorized, evidence = _validate_authorization(
        authorization_path=authorization_path,
        config_path=config_path,
        output_dir=output_dir,
        ledger_dir=ledger_dir,
    )
    if preflight_only:
        print(json.dumps(authorized, ensure_ascii=False, sort_keys=True))
        return 0

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = frozen_runner._validate_recovery(output_dir)
    if not recovery.get("valid"):
        raise ValueError("directional final output recovery is invalid")
    if (output_dir / "official_test_access_guard.json").exists() and recovery.get("phase") != "complete":
        raise ValueError("ambiguous official-test access forbids directional final retry")

    entry = _reserve_ledger_entry(ledger_dir.expanduser().resolve())
    prelaunch = {
        "schema_version": "trifusion-directional-final-prelaunch-v1",
        "status": "AUTHORIZED",
        "variant": VARIANT,
        "config": str(config_path.expanduser().resolve()),
        "config_sha256": _sha256(config_path.expanduser().resolve()),
        "runner": str(FROZEN_RUNNER.resolve()),
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "amp_safe_sitecustomize": str(AMP_SAFE_SITECUSTOMIZE.resolve()),
        "amp_safe_sitecustomize_sha256": _sha256(AMP_SAFE_SITECUSTOMIZE),
        "output_dir": str(output_dir),
        "ledger_dir": str(ledger_dir.expanduser().resolve()),
        "recovery_before": recovery,
        "directional_authorization": evidence,
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "official_test_access_count": 0,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive_json(entry / "prelaunch_receipt.json", prelaunch)

    original_identity = frozen_runner._run_identity

    def authorized_identity(preflight: dict[str, Any], variant: str) -> dict[str, Any]:
        identity = original_identity(preflight, variant)
        identity.update(
            {
                "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
                "directional_authorization": evidence,
                "query_gallery_symmetry_claim_eligible": False,
                "calibration_claim_eligible": True,
            }
        )
        return identity

    candidate_path = entry / "completion_candidate.json"
    completion_path = entry / "completion_receipt.json"
    try:
        with _termination_as_exception(), _temporary_worker_environment(
            AMP_SAFE_SITECUSTOMIZE.parent
        ):
            with (entry / "launcher.log").open("xb") as log_handle:
                with patch.object(
                    frozen_runner,
                    "_preflight",
                    lambda *_args, **_kwargs: copy.deepcopy(authorized),
                ), patch.object(frozen_runner, "_run_identity", authorized_identity):
                    summary, returncode = frozen_runner._dev(
                        config_path.expanduser().resolve(),
                        VARIANT,
                        output_dir,
                        data_mode="postfreeze-final",
                    )
                _atomic_json(output_dir / "run_summary.json", summary)
                log_handle.write(
                    _canonical_json(
                        {
                            "returncode": returncode,
                            "status": summary.get("status"),
                            "official_test_access_count": summary.get(
                                "official_test_access_count", 0
                            ),
                        }
                    )
                    + b"\n"
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
        if returncode != 0:
            raise RuntimeError(f"frozen final runner failed with return code {returncode}")
        completion = _completion_payload(
            entry=entry,
            output_dir=output_dir,
            returncode=returncode,
        )
        if completion["status"] != "PASS":
            raise RuntimeError("fixed final completion artifact set is incomplete")
        _exclusive_json(candidate_path, completion)
        verify_completion(entry, receipt_name=candidate_path.name)
        if completion_path.exists():
            raise RuntimeError("formal completion receipt already exists")
        os.replace(candidate_path, completion_path)
        directory_fd = os.open(entry, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        completion = verify_completion(entry)
    except BaseException as error:
        with _ignore_termination_signals():
            if not completion_path.exists() and not (entry / "failure_receipt.json").exists():
                try:
                    _exclusive_json(
                        entry / "failure_receipt.json",
                        _failure_payload(entry=entry, output_dir=output_dir, error=error),
                    )
                except Exception as receipt_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            "failure receipt publication also failed: "
                            f"{type(receipt_error).__name__}: {receipt_error}"
                        )
        raise

    print(
        json.dumps(
            {
                "status": completion["status"],
                "completion_receipt": str(completion_path),
                "run_summary": str(output_dir / "run_summary.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return launch(
        authorization_path=args.authorization,
        config_path=args.config,
        output_dir=args.output_dir,
        ledger_dir=args.ledger_dir,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
