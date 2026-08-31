#!/usr/bin/env python3
"""Fail-closed launcher for the frozen TriFusion mean-only development run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

PROJECT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = PROJECT / "tools/run_trifusion_experiment.py"
FROZEN_RUNNER_SHA256 = (
    "50540f112d99b55e761be91eaa36a273444c0318c9929929cb8a62d8cb25897c"
)
VARIANT = "trifusion_circ_urgc"
EVIDENCE_SCOPE = "mean_only_training_input"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.authorize_circ_mean_only import (  # noqa: E402
    verify_overlay,
    verify_training_config,
)


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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mean-only launch validation failed: invalid config")
    return value


def _resolve_registered_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    return path.resolve()


def _recovery_before(mode: str, output_dir: Path) -> dict[str, Any]:
    if not output_dir.exists():
        return {"kind": "absent", "output_exists": False}
    if not output_dir.is_dir():
        raise ValueError("mean-only launch validation failed: output is not a directory")
    children = sorted(path.name for path in output_dir.iterdir())
    if not children:
        return {"kind": "fresh", "output_exists": True, "children": []}
    if mode == "preflight":
        return {
            "kind": "preflight_refresh",
            "output_exists": True,
            "children": children,
            "preflight_sha256": (
                _sha256(output_dir / "preflight.json")
                if (output_dir / "preflight.json").is_file()
                else None
            ),
        }
    identity = output_dir / "run_identity.json"
    latest = output_dir / ".resume/latest.json"
    if not identity.is_file() or not latest.is_file():
        raise ValueError(
            "mean-only launch validation failed: foreign nonempty development output"
        )
    manifest = _load_json(latest)
    if manifest.get("run_identity_sha256") != _sha256(identity):
        raise ValueError(
            "mean-only launch validation failed: recovery identity hash mismatch"
        )
    return {
        "kind": "resume",
        "output_exists": True,
        "children": children,
        "run_identity_sha256": _sha256(identity),
        "recovery_manifest_sha256": _sha256(latest),
        "recovery_epoch": int(manifest.get("epoch", -1)),
        "recovery_phase": str(manifest.get("phase", "")),
    }


def _validate_static_launch_request(
    *,
    mode: str,
    config_path: Path,
    parent_root: Path,
    protocol_path: Path,
    overlay_root: Path,
    output_dir: Path,
    ledger_dir: Path,
) -> dict[str, Any]:

    if mode not in {"preflight", "dev"}:
        raise ValueError("mean-only launch validation failed: mode must be preflight or dev")
    config_path = config_path.expanduser().resolve()
    parent_root = parent_root.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    overlay_root = overlay_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ledger_dir = ledger_dir.expanduser().resolve()

    overlay_receipt = verify_overlay(
        parent_root=parent_root,
        protocol_path=protocol_path,
        output_root=overlay_root,
        config_path=config_path,
    )
    config_evidence = verify_training_config(
        config_path=config_path,
        protocol_path=protocol_path,
        output_root=overlay_root,
    )
    config = _load_config(config_path)
    execution = dict(config.get("EXECUTION", {}))
    launcher = Path(__file__).resolve()
    registered_output_key = (
        "MEAN_ONLY_DEV_OUTPUT_DIR"
        if mode == "dev"
        else "MEAN_ONLY_PREFLIGHT_OUTPUT_DIR"
    )
    forbidden_environment = sorted(
        key for key, value in os.environ.items() if key.startswith("TRIFUSION_") and value
    )
    if (
        _resolve_registered_path(execution.get("MEAN_ONLY_LAUNCHER", ""))
        != launcher
        or execution.get("MEAN_ONLY_LAUNCHER_SHA256") != _sha256(launcher)
        or _resolve_registered_path(execution.get("FROZEN_RUNNER", ""))
        != FROZEN_RUNNER.resolve()
        or execution.get("FROZEN_RUNNER_SHA256") != FROZEN_RUNNER_SHA256
        or not FROZEN_RUNNER.is_file()
        or _sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256
        or _resolve_registered_path(execution.get(registered_output_key, ""))
        != output_dir
        or _resolve_registered_path(execution.get("MEAN_ONLY_LEDGER_DIR", ""))
        != ledger_dir
        or execution.get("REQUIRE_COMPLETION_RECEIPT") is not True
        or execution.get("RESULT_QUALIFICATION_GATE")
        != "mean_only_completion_receipt"
        or forbidden_environment
        or output_dir == ledger_dir
        or output_dir in ledger_dir.parents
        or ledger_dir in output_dir.parents
    ):
        raise ValueError(
            "mean-only launch validation failed: launcher, frozen runner, paths, "
            "or environment overrides drifted"
        )

    runner_argv = [
        sys.executable,
        str(FROZEN_RUNNER.resolve()),
        "--mode",
        mode,
        "--variant",
        VARIANT,
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
    ]
    return {
        "schema_version": "trifusion-mean-only-launch-validation-v1",
        "mode": mode,
        "variant": VARIANT,
        "config_path": str(config_path),
        "config_sha256": config_evidence["config_sha256"],
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256(launcher),
        "runner_path": str(FROZEN_RUNNER.resolve()),
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "parent_root": str(parent_root),
        "overlay_protocol_path": str(protocol_path),
        "overlay_protocol_sha256": _sha256(protocol_path),
        "overlay_root": str(overlay_root),
        "overlay_receipt_sha256": _sha256(
            overlay_root / "mean_only_overlay_receipt.json"
        ),
        "scoring_receipt_sha256": _sha256(
            overlay_root / "scoring_receipt.json"
        ),
        "output_dir": str(output_dir),
        "ledger_dir": str(ledger_dir),
        "runner_argv": runner_argv,
        "scientific_evidence_scope": EVIDENCE_SCOPE,
        "result_qualification_gate": "mean_only_completion_receipt",
        "completion_receipt_required": True,
        "mean_only_training_eligible": overlay_receipt[
            "mean_only_training_eligible"
        ],
        "uncertainty_control_allowed": False,
        "official_test_access_count": 0,
    }


def validate_launch_request(
    *,
    mode: str,
    config_path: Path,
    parent_root: Path,
    protocol_path: Path,
    overlay_root: Path,
    output_dir: Path,
    ledger_dir: Path,
) -> dict[str, Any]:
    """Validate an exact, non-extensible launch request without writing state."""

    static = _validate_static_launch_request(
        mode=mode,
        config_path=config_path,
        parent_root=parent_root,
        protocol_path=protocol_path,
        overlay_root=overlay_root,
        output_dir=output_dir,
        ledger_dir=ledger_dir,
    )
    return {
        **static,
        "recovery_before": _recovery_before(mode, Path(static["output_dir"])),
    }


def _reserve_ledger_entry(ledger_dir: Path) -> Path:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for sequence in range(1, 10000):
        entry = ledger_dir / f"launch-{sequence:04d}"
        try:
            entry.mkdir()
            return entry
        except FileExistsError:
            continue
    raise RuntimeError("mean-only launch ledger exhausted")


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _completion_payload(
    *,
    validation: Mapping[str, Any],
    entry: Path,
    returncode: int,
) -> dict[str, Any]:
    output_dir = Path(str(validation["output_dir"]))
    mode = str(validation["mode"])
    primary = output_dir / (
        "run_summary.json" if mode == "dev" else "preflight.json"
    )
    artifacts = {
        "primary_receipt": _artifact_evidence(primary),
        "run_identity": _artifact_evidence(output_dir / "run_identity.json"),
        "recovery_manifest": _artifact_evidence(output_dir / ".resume/latest.json"),
    }
    primary_payload = _load_json(primary) if primary.is_file() else {}
    if mode == "dev":
        successful = (
            returncode == 0
            and primary_payload.get("status") == "PASS"
            and primary_payload.get("phase") == "complete"
            and primary_payload.get("scientific_evidence_eligible") is True
            and int(primary_payload.get("official_test_access_count", -1)) == 0
            and artifacts["run_identity"]["exists"]
            and artifacts["recovery_manifest"]["exists"]
        )
    else:
        successful = (
            returncode == 0
            and primary_payload.get("status") == "READY"
            and primary_payload.get("launch_allowed") is True
            and int(primary_payload.get("official_test_access_count", -1)) == 0
        )
    scientific = successful and mode == "dev"
    return {
        "schema_version": "trifusion-mean-only-completion-v1",
        "status": "PASS" if successful else "FAIL",
        "mode": mode,
        "variant": VARIANT,
        "returncode": returncode,
        "prelaunch_receipt_sha256": _sha256(entry / "prelaunch_receipt.json"),
        "runner_log_sha256": _sha256(entry / "runner.log"),
        "artifacts": artifacts,
        "scientific_evidence_scope": EVIDENCE_SCOPE,
        "result_qualification_gate": "mean_only_completion_receipt",
        "scientific_evidence_eligible": scientific,
        "calibration_claim_eligible": False,
        "concentration_claim_eligible": False,
        "uncertainty_control_allowed": False,
        "official_test_access_count": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_completion(entry: Path) -> dict[str, Any]:
    """Qualification gate for every result emitted by the mean-only launcher."""

    entry = entry.expanduser().resolve()
    prelaunch_path = entry / "prelaunch_receipt.json"
    completion_path = entry / "completion_receipt.json"
    log_path = entry / "runner.log"
    if not prelaunch_path.is_file() or not completion_path.is_file() or not log_path.is_file():
        raise ValueError("mean-only completion verification failed: incomplete ledger")
    prelaunch = _load_json(prelaunch_path)
    completion = _load_json(completion_path)
    if (
        prelaunch.get("schema_version") != "trifusion-mean-only-prelaunch-v1"
        or prelaunch.get("status") != "AUTHORIZED"
        or prelaunch.get("mode") != "dev"
        or prelaunch.get("variant") != VARIANT
        or prelaunch.get("result_qualification_gate")
        != "mean_only_completion_receipt"
        or prelaunch.get("completion_receipt_required") is not True
        or prelaunch.get("scientific_evidence_scope") != EVIDENCE_SCOPE
        or prelaunch.get("mean_only_training_eligible") is not True
        or prelaunch.get("uncertainty_control_allowed") is not False
        or int(prelaunch.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("mean-only completion verification failed: invalid prelaunch")
    try:
        static = _validate_static_launch_request(
            mode="dev",
            config_path=Path(str(prelaunch["config_path"])),
            parent_root=Path(str(prelaunch["parent_root"])),
            protocol_path=Path(str(prelaunch["overlay_protocol_path"])),
            overlay_root=Path(str(prelaunch["overlay_root"])),
            output_dir=Path(str(prelaunch["output_dir"])),
            ledger_dir=Path(str(prelaunch["ledger_dir"])),
        )
    except Exception as error:
        raise ValueError(
            "mean-only completion verification failed: static launch binding"
        ) from error
    for key, expected in static.items():
        if key != "schema_version" and prelaunch.get(key) != expected:
            raise ValueError(
                f"mean-only completion verification failed: prelaunch drift in {key}"
            )
    expected_prelaunch_keys = set(static) | {
        "status",
        "started_at_utc",
        "recovery_before",
    }
    if set(prelaunch) != expected_prelaunch_keys:
        raise ValueError("mean-only completion verification failed: prelaunch fields")
    recovery_before = prelaunch.get("recovery_before", {})
    if (
        not isinstance(recovery_before, dict)
        or recovery_before.get("kind") not in {"absent", "fresh", "resume"}
        or (
            recovery_before.get("kind") == "resume"
            and (
                not recovery_before.get("run_identity_sha256")
                or not recovery_before.get("recovery_manifest_sha256")
                or int(recovery_before.get("recovery_epoch", -1)) < 0
            )
        )
        or entry.parent != Path(str(static["ledger_dir"])).resolve()
    ):
        raise ValueError("mean-only completion verification failed: recovery ledger")

    expected_completion_keys = {
        "schema_version",
        "status",
        "mode",
        "variant",
        "returncode",
        "prelaunch_receipt_sha256",
        "runner_log_sha256",
        "artifacts",
        "scientific_evidence_scope",
        "result_qualification_gate",
        "scientific_evidence_eligible",
        "calibration_claim_eligible",
        "concentration_claim_eligible",
        "uncertainty_control_allowed",
        "official_test_access_count",
        "completed_at_utc",
    }
    if (
        set(completion) != expected_completion_keys
        or completion.get("schema_version")
        != "trifusion-mean-only-completion-v1"
        or completion.get("status") != "PASS"
        or completion.get("mode") != "dev"
        or completion.get("variant") != VARIANT
        or int(completion.get("returncode", -1)) != 0
        or completion.get("scientific_evidence_eligible") is not True
        or completion.get("scientific_evidence_scope") != EVIDENCE_SCOPE
        or completion.get("result_qualification_gate")
        != "mean_only_completion_receipt"
        or completion.get("calibration_claim_eligible") is not False
        or completion.get("concentration_claim_eligible") is not False
        or completion.get("uncertainty_control_allowed") is not False
        or int(completion.get("official_test_access_count", -1)) != 0
        or completion.get("prelaunch_receipt_sha256")
        != _sha256(prelaunch_path)
        or completion.get("runner_log_sha256") != _sha256(log_path)
    ):
        raise ValueError("mean-only completion verification failed: receipt drift")

    output_dir = Path(str(static["output_dir"])).resolve()
    artifact_paths = {
        "primary_receipt": output_dir / "run_summary.json",
        "run_identity": output_dir / "run_identity.json",
        "recovery_manifest": output_dir / ".resume/latest.json",
    }
    expected_artifacts = {
        key: _artifact_evidence(path) for key, path in artifact_paths.items()
    }
    if (
        set(completion.get("artifacts", {})) != set(artifact_paths)
        or any(not evidence["exists"] for evidence in expected_artifacts.values())
        or completion.get("artifacts") != expected_artifacts
    ):
        raise ValueError("mean-only completion verification failed: artifact drift")

    primary = _load_json(artifact_paths["primary_receipt"])
    identity = _load_json(artifact_paths["run_identity"])
    manifest = _load_json(artifact_paths["recovery_manifest"])
    current = manifest.get("current", {})
    current_path = (output_dir / str(current.get("path", ""))).resolve()
    if (
        primary.get("status") != "PASS"
        or primary.get("phase") != "complete"
        or primary.get("scientific_evidence_eligible") is not True
        or int(primary.get("official_test_access_count", -1)) != 0
        or identity.get("scientific_evidence_eligible") is not True
        or identity.get("official_test_access_during_development") is not False
        or identity.get("contract_testing") is not False
        or identity.get("data_mode") != "development"
        or identity.get("variant") != VARIANT
        or identity.get("runner_sha256") != FROZEN_RUNNER_SHA256
        or identity.get("config_sha256") != static["config_sha256"]
        or manifest.get("phase") != "complete"
        or manifest.get("run_identity_sha256")
        != expected_artifacts["run_identity"]["sha256"]
        or output_dir not in current_path.parents
        or not current_path.is_file()
        or current.get("sha256") != _sha256(current_path)
    ):
        raise ValueError("mean-only completion verification failed: invalid result chain")
    previous = manifest.get("previous")
    if previous:
        previous_path = (output_dir / str(previous.get("path", ""))).resolve()
        if (
            output_dir not in previous_path.parents
            or not previous_path.is_file()
            or previous.get("sha256") != _sha256(previous_path)
        ):
            raise ValueError(
                "mean-only completion verification failed: previous recovery drift"
            )
    return completion


def launch(validation: Mapping[str, Any]) -> tuple[Path, int]:
    ledger_dir = Path(str(validation["ledger_dir"]))
    entry = _reserve_ledger_entry(ledger_dir)
    prelaunch = {
        **dict(validation),
        "schema_version": "trifusion-mean-only-prelaunch-v1",
        "status": "AUTHORIZED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive_json(entry / "prelaunch_receipt.json", prelaunch)
    with (entry / "runner.log").open("xb") as log_handle:
        process = subprocess.run(
            list(validation["runner_argv"]),
            cwd=PROJECT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    completion = _completion_payload(
        validation=validation,
        entry=entry,
        returncode=process.returncode,
    )
    _exclusive_json(entry / "completion_receipt.json", completion)
    if validation["mode"] == "dev" and completion["status"] == "PASS":
        verify_completion(entry)
    return entry, process.returncode


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "dev"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--verify-completion", type=Path)
    args = parser.parse_args(argv)
    launch_fields = (
        args.mode,
        args.config,
        args.parent_root,
        args.protocol,
        args.overlay_root,
        args.output_dir,
        args.ledger_dir,
    )
    if args.verify_completion:
        if any(value is not None for value in launch_fields):
            parser.error("--verify-completion cannot be combined with launch arguments")
    elif any(value is None for value in launch_fields):
        parser.error("all fixed launch arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_completion:
        receipt = verify_completion(args.verify_completion)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    validation = validate_launch_request(
        mode=args.mode,
        config_path=args.config,
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        overlay_root=args.overlay_root,
        output_dir=args.output_dir,
        ledger_dir=args.ledger_dir,
    )
    entry, returncode = launch(validation)
    print(json.dumps({"ledger_entry": str(entry), "returncode": returncode}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
