#!/usr/bin/env python3
"""Compile externally scored full-network CIRC interventions into a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from modeling.trifusion.intervention_targets import (
    compile_circ_targets,
    write_circ_target_cache,
)
from modeling.trifusion.protocol import (
    load_trusted_circ_protocol,
    trifusion_source_hashes,
)
from modeling.trifusion.variants import resolve_variant, variant_sha256


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        loaded = yaml.safe_load(payload.decode("utf-8"))
    else:
        loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("CIRC config root must be a mapping")
    return loaded, config_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _resolve_registered_file(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_complete_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    expected_epoch: int,
    run_identity_sha256: str,
) -> str:
    expected_manifest = output_root / ".resume/latest.json"
    if manifest_path.resolve() != expected_manifest.resolve() or not manifest_path.is_file():
        raise ValueError("recovery manifest path is not contained in its run output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("phase") != "complete"
        or int(manifest.get("epoch", -1)) != expected_epoch
        or manifest.get("run_identity_sha256") != run_identity_sha256
    ):
        raise ValueError("recovery manifest is not the expected complete endpoint")
    resume_root = (output_root / ".resume").resolve()
    for label in ("current", "previous"):
        generation = manifest.get(label)
        if generation is None:
            continue
        generation_path = (output_root / str(generation["path"])).resolve()
        if (
            generation_path.parent != resume_root
            or not generation_path.is_file()
            or _sha256(generation_path) != generation.get("sha256")
        ):
            raise ValueError(f"{label} recovery generation is missing or foreign")
    return _sha256(manifest_path)


def _orchestrate_oof_generators(
    config: dict[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    mode: str,
    output: Path,
) -> int:
    if mode != "development":
        raise ValueError("OOF generator selection is development-only")
    if config.get("schema_version") != "circ-generator-orchestration-v1":
        raise ValueError("unsupported CIRC generator orchestration schema")
    protocol_path = _resolve_registered_file(config_path, config["circ_protocol"])
    generator_config_path = _resolve_registered_file(
        config_path, config["generator_config"]
    )
    endpoint_receipt_path = _resolve_registered_file(
        config_path, config["endpoint_receipt"]
    )
    protocol, protocol_sha256 = load_trusted_circ_protocol(protocol_path)
    endpoint = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    folds = dict(protocol.get("folds", {}))
    selection = dict(protocol.get("generator_selection", {}))
    contract = dict(resolve_variant("hfer_uniform_generator"))
    contract_sha256 = variant_sha256(contract)
    source_sha256 = trifusion_source_hashes()
    if (
        protocol.get("schema_version") != "circ-protocol-v1"
        or int(protocol.get("official_test_access_count", -1)) != 0
        or int(folds.get("count", 0)) != 3
        or folds.get("identity_canonicalization") != "unsigned-decimal"
    ):
        raise ValueError("invalid frozen three-fold CIRC protocol")
    schedule_horizon = int(selection.get("schedule_horizon_epochs", 0))
    if (
        endpoint.get("schema_version") != "trifusion-dev-selection-v1"
        or endpoint.get("variant") != "hfer_uniform_generator"
        or endpoint.get("selection_output") != "fused"
        or endpoint.get("variant_contract_sha256") != contract_sha256
        or endpoint.get("phase") != "complete"
        or int(endpoint.get("schedule_horizon_epochs", -1)) != schedule_horizon
        or int(endpoint.get("dev_evaluation_count", -1)) != schedule_horizon
        or endpoint.get("model_constructed") is not True
        or endpoint.get("training_started") is not True
        or endpoint.get("fatal_or_nonfinite_detected") is not False
        or int(endpoint.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("invalid HFER-uniform endpoint receipt")
    generator_config_sha256 = _sha256(generator_config_path)
    registered_generator_config = (
        Path(__file__).resolve().parents[1] / str(selection.get("config", ""))
    ).resolve()
    if (
        generator_config_path != registered_generator_config
        or generator_config_sha256 != selection.get("config_sha256")
    ):
        raise ValueError("generator config differs from the frozen protocol")
    if endpoint.get("circ_protocol_sha256") != protocol_sha256:
        raise ValueError("endpoint receipt is not bound to the CIRC protocol")
    if endpoint.get("config_sha256") != generator_config_sha256:
        raise ValueError("endpoint receipt is not bound to the generator config")
    fixed_endpoint = int(endpoint["epoch"])
    if not 1 <= fixed_endpoint <= schedule_horizon or schedule_horizon != 60:
        raise ValueError("fixed endpoint is outside the registered 60-epoch schedule")
    metrics = dict(endpoint.get("metrics_percent", {}))
    if set(metrics) != set(contract["evaluation_outputs"]):
        raise ValueError("endpoint receipt lacks complete branch and fused metrics")
    if float(endpoint.get("dev_selection_mAP", float("nan"))) != float(
        metrics["fused"]["mAP"]
    ):
        raise ValueError("endpoint selected mAP differs from its fused metric")
    selector_root = endpoint_receipt_path.parent.resolve()
    checkpoint = Path(str(endpoint["checkpoint"])).expanduser().resolve()
    if (
        checkpoint != selector_root / "best_dev_model.pth"
        or not checkpoint.is_file()
        or _sha256(checkpoint) != endpoint.get("checkpoint_sha256")
    ):
        raise ValueError("selector checkpoint is missing or hash-mismatched")
    selector_identity_path = Path(str(endpoint.get("run_identity", ""))).resolve()
    if (
        selector_identity_path != selector_root / "run_identity.json"
        or not selector_identity_path.is_file()
        or _sha256(selector_identity_path) != endpoint.get("run_identity_sha256")
    ):
        raise ValueError("selector run identity is missing or hash-mismatched")
    selector_identity = json.loads(
        selector_identity_path.read_text(encoding="utf-8")
    )
    if (
        selector_identity.get("variant") != "hfer_uniform_generator"
        or selector_identity.get("variant_contract_sha256") != contract_sha256
        or selector_identity.get("config_sha256") != generator_config_sha256
        or selector_identity.get("circ_protocol_sha256") != protocol_sha256
        or selector_identity.get("source_sha256") != source_sha256
        or selector_identity.get("official_test_access_during_development") is not False
        or int(selector_identity.get("optimization", {}).get("max_epochs", -1))
        != schedule_horizon
    ):
        raise ValueError("selector run identity differs from the frozen generator")
    selector_manifest = Path(str(endpoint.get("recovery_manifest", ""))).resolve()
    manifest_sha256 = _validate_complete_manifest(
        manifest_path=selector_manifest,
        output_root=selector_root,
        expected_epoch=schedule_horizon,
        run_identity_sha256=endpoint["run_identity_sha256"],
    )
    if manifest_sha256 != endpoint.get("recovery_manifest_sha256"):
        raise ValueError("selector recovery manifest hash mismatch")

    test_executable = os.environ.get("TRIFUSION_CIRC_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        raise ValueError("test executable requires contract-testing mode")
    worker_executable = test_executable or sys.executable
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_receipts = []
    for target_fold in range(3):
        fold_output = output / f"fold-{target_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema_version": "circ-generator-run-v1",
            "operation": "train-oof-generator",
            "orchestration_config_sha256": config_sha256,
            "circ_protocol_sha256": protocol_sha256,
            "generator_config_sha256": generator_config_sha256,
            "endpoint_receipt_sha256": _sha256(endpoint_receipt_path),
            "selector_checkpoint_sha256": endpoint["checkpoint_sha256"],
            "source_sha256": source_sha256,
            "variant": "hfer_uniform_generator",
            "variant_contract_sha256": contract_sha256,
            "target_fold": target_fold,
            "fixed_endpoint": fixed_endpoint,
            "schedule_horizon_epochs": schedule_horizon,
            "official_test_access_count": 0,
        }
        identity_path = fold_output / "run_identity.json"
        if identity_path.is_file():
            actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if actual_identity != identity:
                raise ValueError(f"foreign fold-{target_fold} recovery identity")
        elif any(fold_output.iterdir()):
            raise ValueError(f"nonempty fold-{target_fold} output lacks run identity")
        else:
            _atomic_json(identity_path, identity)

        receipt_path = fold_output / "generator_receipt.json"
        receipt_complete = False
        if receipt_path.is_file():
            previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_complete = previous_receipt.get("status") == "COMPLETE"
        if not receipt_complete:
            command = [
                worker_executable,
                *([] if test_executable else [str(Path(__file__).resolve())]),
                "--config",
                str(config_path),
                "--mode",
                mode,
                "--output",
                str(fold_output),
                "--_worker-fold",
                str(target_fold),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=dict(os.environ),
                check=False,
                capture_output=True,
                text=True,
            )
            attempt = len(tuple(fold_output.glob("worker-attempt-*.log"))) + 1
            (fold_output / f"worker-attempt-{attempt:04d}.log").write_text(
                completed.stdout + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0 or not receipt_path.is_file():
                raise RuntimeError(f"fold-{target_fold} generator worker failed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required = {
            "status": "COMPLETE",
            "phase": "complete",
            "epoch": fixed_endpoint,
            "target_fold": target_fold,
            "fixed_endpoint": fixed_endpoint,
            "schedule_horizon_epochs": schedule_horizon,
            "generator_target_identity_overlap": 0,
            "dev_evaluation_count": 0,
            "target_loader_iteration_count": 0,
            "official_test_access_count": 0,
            "variant": "hfer_uniform_generator",
            "variant_contract_sha256": contract_sha256,
            "circ_protocol_sha256": protocol_sha256,
            "config_sha256": generator_config_sha256,
            "source_sha256": source_sha256,
            "run_identity_sha256": _sha256(identity_path),
            "model_constructed": True,
            "training_started": True,
            "fatal_or_nonfinite_detected": False,
            "parameter_budget_pass": True,
        }
        mismatches = {
            key: {"expected": value, "actual": receipt.get(key)}
            for key, value in required.items()
            if receipt.get(key) != value
        }
        checkpoint_path = Path(
            str(receipt.get("checkpoint", ""))
        ).expanduser().resolve()
        if (
            checkpoint_path != (fold_output / "generator.pth").resolve()
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            mismatches["checkpoint"] = "missing or hash-mismatched"
        train_history = dict(receipt.get("train_history", {}))
        if set(train_history) != {str(epoch) for epoch in range(1, fixed_endpoint + 1)}:
            mismatches["train_history"] = "does not cover every fixed-endpoint epoch"
        provenance = dict(receipt.get("data_provenance", {}))
        if (
            provenance.get("target_fold") != target_fold
            or provenance.get("generator_target_identity_overlap") != 0
            or provenance.get("target_forbidden_dev_identity_overlap") != 0
            or provenance.get("official_test_records") != 0
        ):
            mismatches["data_provenance"] = "fold isolation contract failed"
        manifest_path = Path(
            str(receipt.get("recovery_manifest", ""))
        ).expanduser().resolve()
        try:
            manifest_sha256 = _validate_complete_manifest(
                manifest_path=manifest_path,
                output_root=fold_output,
                expected_epoch=fixed_endpoint,
                run_identity_sha256=_sha256(identity_path),
            )
            if manifest_sha256 != receipt.get("recovery_manifest_sha256"):
                mismatches["recovery_manifest"] = "hash mismatch"
        except (KeyError, TypeError, ValueError) as error:
            mismatches["recovery_manifest"] = str(error)
        if mismatches:
            raise ValueError(f"fold-{target_fold} receipt contract failed: {mismatches}")
        fold_receipts.append(receipt)

    aggregate = {
        "schema_version": "circ-generators-receipt-v1",
        "status": "COMPLETE",
        "mode": mode,
        "completed_folds": [receipt["target_fold"] for receipt in fold_receipts],
        "fixed_endpoint": fixed_endpoint,
        "schedule_horizon_epochs": schedule_horizon,
        "zero_identity_overlap": all(
            receipt["generator_target_identity_overlap"] == 0
            for receipt in fold_receipts
        ),
        "dev_evaluation_count": sum(
            receipt["dev_evaluation_count"] for receipt in fold_receipts
        ),
        "target_loader_iteration_count": sum(
            receipt["target_loader_iteration_count"] for receipt in fold_receipts
        ),
        "official_test_access_count": sum(
            receipt["official_test_access_count"] for receipt in fold_receipts
        ),
        "circ_protocol": str(protocol_path),
        "circ_protocol_sha256": protocol_sha256,
        "generator_config_sha256": generator_config_sha256,
        "endpoint_receipt_sha256": _sha256(endpoint_receipt_path),
        "fold_receipt_sha256": {
            str(index): _sha256(output / f"fold-{index}" / "generator_receipt.json")
            for index in range(3)
        },
    }
    _atomic_json(output / "generators_receipt.json", aggregate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("development", "postfreeze-final"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--_worker-fold", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config_path = arguments.config.expanduser().resolve()
    config, config_sha256 = _load_config(config_path)
    if config.get("operation") == "train-oof-generators":
        if arguments._worker_fold is not None:
            target_fold = int(arguments._worker_fold)
            if target_fold not in (0, 1, 2):
                raise ValueError("OOF worker fold must be 0, 1 or 2")
            protocol_path = _resolve_registered_file(
                config_path, config["circ_protocol"]
            )
            generator_config_path = _resolve_registered_file(
                config_path, config["generator_config"]
            )
            endpoint_receipt_path = _resolve_registered_file(
                config_path, config["endpoint_receipt"]
            )
            endpoint = json.loads(
                endpoint_receipt_path.read_text(encoding="utf-8")
            )
            if int(endpoint.get("official_test_access_count", -1)) != 0:
                raise ValueError("OOF worker endpoint accessed official test")
            from tools.run_trifusion_experiment import _preflight, _worker_dev

            preflight = _preflight(
                generator_config_path,
                "hfer_uniform_generator",
            )
            if not preflight["launch_allowed"]:
                _atomic_json(
                    arguments.output.expanduser().resolve()
                    / "generator_receipt.json",
                    {
                        "status": "BLOCKED",
                        "target_fold": target_fold,
                        "official_test_access_count": 0,
                        "preflight": preflight,
                    },
                )
                return 3

            return _worker_dev(
                generator_config_path,
                "hfer_uniform_generator",
                arguments.output.expanduser().resolve(),
                oof_target_fold=target_fold,
                circ_protocol_path=protocol_path,
                fixed_endpoint=int(endpoint["epoch"]),
            )
        return _orchestrate_oof_generators(
            config,
            config_path=config_path,
            config_sha256=config_sha256,
            mode=arguments.mode,
            output=arguments.output,
        )
    rows, receipt = compile_circ_targets(
        config, mode=arguments.mode, config_sha256=config_sha256
    )
    write_circ_target_cache(
        rows, receipt, output_directory=arguments.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
