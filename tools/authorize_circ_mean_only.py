#!/usr/bin/env python3
"""Authorize an immutable failed-calibration CIRC cache for mean-only training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


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
IMMUTABLE_OUTPUT_ARTIFACTS = (
    "run_identity.json",
    "cache/receipt.json",
    "cache/calibration_receipt.json",
    "cache/targets.jsonl",
    "cache/symmetry_receipt.json",
    "cache/target_transfer_receipt.json",
)
SCIENTIFIC_EVIDENCE_SCOPE = "mean_only_training_input"


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json(dict(payload)) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_zero_official_access(label: str, payload: Mapping[str, Any]) -> None:
    if int(payload.get("official_test_access_count", -1)) != 0:
        raise ValueError(f"{label} is not official-test-zero evidence")


def _load_context(
    *,
    parent_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    parent_root = parent_root.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "circ-mean-only-overlay-protocol-v1":
        raise ValueError("unsupported mean-only overlay protocol")
    _require_zero_official_access("mean-only protocol", protocol)

    policy = dict(protocol.get("policy", {}))
    required_policy = {
        "posterior_mean_control_only": True,
        "uncertainty_control_allowed": False,
        "concentration_claim_allowed": False,
        "parent_calibration_failure_must_be_preserved": True,
    }
    if any(policy.get(key) is not value for key, value in required_policy.items()):
        raise ValueError("mean-only overlay policy is not fail-closed")

    expected_hashes = dict(protocol.get("parent_artifacts_sha256", {}))
    if set(expected_hashes) != set(REQUIRED_PARENT_ARTIFACTS):
        raise ValueError("mean-only protocol does not bind every parent artifact")
    parent_paths = {name: parent_root / name for name in REQUIRED_PARENT_ARTIFACTS}
    for name, path in parent_paths.items():
        if not path.is_file() or _sha256(path) != expected_hashes[name]:
            raise ValueError(f"parent artifact is missing or changed: {name}")

    scoring = _load_json(parent_paths["scoring_receipt.json"])
    identity = _load_json(parent_paths["run_identity.json"])
    cache = _load_json(parent_paths["cache/receipt.json"])
    calibration = _load_json(parent_paths["cache/calibration_receipt.json"])
    target_sha256 = _sha256(parent_paths["cache/targets.jsonl"])
    calibration_audit = scoring.get("calibration_audit")
    concentration = (
        calibration_audit.get("empirical_concentration_coverage", {})
        if isinstance(calibration_audit, dict)
        else {}
    )
    failed_conditions = [
        key
        for key, result in concentration.items()
        if isinstance(result, dict) and result.get("claim_eligible") is False
    ]
    symmetry = scoring.get("symmetry_audit", {})
    if (
        scoring.get("status") != "COMPLETE"
        or scoring.get("contract_testing") is not False
        or scoring.get("scientific_evidence_eligible") is not False
        or scoring.get("calibration_claim_eligible") is not False
        or cache.get("targets_sha256") != target_sha256
        or scoring.get("targets_sha256") != target_sha256
        or cache.get("protocol_hash")
        != protocol.get("parent_circ_protocol_sha256")
        or not isinstance(calibration_audit, dict)
        or cache.get("calibration_audit") != calibration_audit
        or calibration_audit.get("status") != "COMPLETE"
        or not failed_conditions
        or symmetry.get("status") != "PASS"
        or symmetry.get("claim_eligible") is not True
        or int(cache.get("row_count", 0)) <= 0
        or int(cache.get("cross_camera_primary_rows", -1))
        != int(cache.get("row_count", 0))
        or int(cache.get("same_camera_only_rows", -1)) != 0
        or int(cache.get("invalid_support_rows", -1)) != 0
        or cache.get("zero_identity_overlap") is not True
    ):
        raise ValueError(
            "parent CIRC evidence is not the registered failed-calibration run"
        )
    _require_zero_official_access("parent scoring receipt", scoring)
    _require_zero_official_access("parent run identity", identity)
    _require_zero_official_access("parent cache receipt", cache)
    if calibration.get("protocol_hash") != protocol.get("parent_circ_protocol_sha256"):
        raise ValueError("parent calibration receipt has the wrong protocol")

    return {
        "parent_root": parent_root,
        "protocol_path": protocol_path,
        "protocol": protocol,
        "expected_hashes": expected_hashes,
        "parent_paths": parent_paths,
        "parent_scoring": scoring,
        "failed_conditions": sorted(failed_conditions),
    }


def _provenance(
    context: Mapping[str, Any],
    config_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "circ-mean-only-overlay-v1",
        "parent_artifacts_sha256": dict(context["expected_hashes"]),
        "parent_circ_protocol_sha256": context["protocol"][
            "parent_circ_protocol_sha256"
        ],
        "overlay_protocol_sha256": _sha256(context["protocol_path"]),
        "overlay_implementation_sha256": _sha256(Path(__file__).resolve()),
        "training_config_sha256": config_evidence["config_sha256"],
        "training_seed": config_evidence["seed"],
        "training_evidence_weight": config_evidence["evidence_weight"],
        "cache_changed": False,
        "parent_calibration_failure_preserved": True,
        "authorized_control": "posterior_mean_r_only",
        "official_test_access_count": 0,
    }


def _expected_scoring(
    parent_scoring: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    scoring = dict(parent_scoring)
    scoring.update(
        {
            "schema_version": "circ-scoring-receipt-v2-mean-only",
            "scientific_evidence_eligible": True,
            "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
            "mean_only_training_eligible": True,
            "calibration_claim_eligible": False,
            "concentration_claim_eligible": False,
            "uncertainty_control_allowed": False,
            "mean_only_overlay": dict(provenance),
        }
    )
    return scoring


def _expected_overlay_receipt(
    *,
    context: Mapping[str, Any],
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    copied_hashes = {
        name: _sha256(output_root / name) for name in IMMUTABLE_OUTPUT_ARTIFACTS
    }
    return {
        "schema_version": "circ-mean-only-overlay-receipt-v1",
        "status": "COMPLETE",
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "mean_only_training_eligible": True,
        "calibration_claim_eligible": False,
        "concentration_claim_eligible": False,
        "uncertainty_control_allowed": False,
        "failed_calibration_conditions": list(context["failed_conditions"]),
        "immutable_output_artifacts_sha256": copied_hashes,
        "output_scoring_receipt_sha256": _sha256(
            output_root / "scoring_receipt.json"
        ),
        "mean_only_overlay": dict(provenance),
        "official_test_access_count": 0,
    }


def build_overlay(
    *,
    parent_root: Path,
    protocol_path: Path,
    output_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a new overlay without changing any target or calibration bytes."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("mean-only overlay output must be new")
    context = _load_context(parent_root=parent_root, protocol_path=protocol_path)
    config_evidence = verify_training_config(
        config_path=config_path,
        protocol_path=protocol_path,
        output_root=output_root,
    )
    for name in IMMUTABLE_OUTPUT_ARTIFACTS:
        _atomic_bytes(
            output_root / name,
            context["parent_paths"][name].read_bytes(),
        )
    provenance = _provenance(context, config_evidence)
    scoring = _expected_scoring(context["parent_scoring"], provenance)
    _atomic_json(output_root / "scoring_receipt.json", scoring)
    receipt = _expected_overlay_receipt(
        context=context,
        output_root=output_root,
        provenance=provenance,
    )
    _atomic_json(output_root / "mean_only_overlay_receipt.json", receipt)
    return receipt


def verify_overlay(
    *,
    parent_root: Path,
    protocol_path: Path,
    output_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Derive every authorization field again and reject any output drift."""

    output_root = output_root.expanduser().resolve()
    context = _load_context(parent_root=parent_root, protocol_path=protocol_path)
    config_evidence = verify_training_config(
        config_path=config_path,
        protocol_path=protocol_path,
        output_root=output_root,
    )
    required_output = (
        *IMMUTABLE_OUTPUT_ARTIFACTS,
        "scoring_receipt.json",
        "mean_only_overlay_receipt.json",
    )
    if any(not (output_root / name).is_file() for name in required_output):
        raise ValueError("mean-only overlay verification failed: incomplete output")
    for name in IMMUTABLE_OUTPUT_ARTIFACTS:
        if (
            context["parent_paths"][name].read_bytes()
            != (output_root / name).read_bytes()
        ):
            raise ValueError(
                f"mean-only overlay verification failed: immutable drift in {name}"
            )

    provenance = _provenance(context, config_evidence)
    expected_scoring = _expected_scoring(context["parent_scoring"], provenance)
    actual_scoring = _load_json(output_root / "scoring_receipt.json")
    if actual_scoring != expected_scoring:
        raise ValueError(
            "mean-only overlay verification failed: scoring receipt mismatch"
        )
    expected_receipt = _expected_overlay_receipt(
        context=context,
        output_root=output_root,
        provenance=provenance,
    )
    actual_receipt = _load_json(output_root / "mean_only_overlay_receipt.json")
    if actual_receipt != expected_receipt:
        raise ValueError(
            "mean-only overlay verification failed: overlay receipt mismatch"
        )
    return actual_receipt


def verify_training_config(
    *,
    config_path: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Bind the mean-only authorization to the exact training configuration."""

    import yaml

    project = Path(__file__).resolve().parents[1]
    config_path = config_path.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("mean-only configuration verification failed: invalid config")
    experiment = dict(config.get("EXPERIMENT", {}))
    loss = dict(config.get("LOSS", {}))
    circ = dict(config.get("CIRC", {}))
    registered = dict(config.get("PROTOCOL", {}))

    configured_protocol = Path(
        str(registered.get("CIRC_MEAN_ONLY_OVERLAY", ""))
    ).expanduser()
    if not configured_protocol.is_absolute():
        configured_protocol = project / configured_protocol
    configured_cache = Path(str(circ.get("TARGET_CACHE", ""))).expanduser().resolve()
    configured_scoring = Path(
        str(circ.get("SCORING_RECEIPT", ""))
    ).expanduser().resolve()
    evidence_weight = float(loss.get("EVIDENCE_WEIGHT", -1.0))
    seed = int(experiment.get("SEED", -1))
    if (
        configured_protocol.resolve() != protocol_path
        or registered.get("CIRC_MEAN_ONLY_OVERLAY_SHA256")
        != _sha256(protocol_path)
        or registered.get("CIRC_SCIENTIFIC_EVIDENCE_SCOPE")
        != SCIENTIFIC_EVIDENCE_SCOPE
        or registered.get("CIRC_UNCERTAINTY_CONTROL_ALLOWED") is not False
        or registered.get("OFFICIAL_TEST_DURING_DEVELOPMENT") is not False
        or configured_cache != (output_root / "cache").resolve()
        or configured_scoring != (output_root / "scoring_receipt.json").resolve()
        or evidence_weight != 0.0
        or seed != 42
    ):
        raise ValueError(
            "mean-only configuration verification failed: protocol, paths, "
            "seed, or uncertainty controls drifted"
        )
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "overlay_protocol_sha256": _sha256(protocol_path),
        "target_cache": str(configured_cache),
        "scoring_receipt": str(configured_scoring),
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "uncertainty_control_allowed": False,
        "evidence_weight": evidence_weight,
        "seed": seed,
        "official_test_access_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing:
        receipt = verify_overlay(
            parent_root=args.parent_root,
            protocol_path=args.protocol,
            output_root=args.output_root,
            config_path=args.config,
        )
    else:
        build_overlay(
            parent_root=args.parent_root,
            protocol_path=args.protocol,
            output_root=args.output_root,
            config_path=args.config,
        )
        receipt = verify_overlay(
            parent_root=args.parent_root,
            protocol_path=args.protocol,
            output_root=args.output_root,
            config_path=args.config,
        )
    config_evidence = verify_training_config(
        config_path=args.config,
        protocol_path=args.protocol,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {"overlay_receipt": receipt, "training_config": config_evidence},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
