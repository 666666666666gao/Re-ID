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
from modeling.trifusion.protocol import load_trusted_circ_protocol


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
    if (
        protocol.get("schema_version") != "circ-protocol-v1"
        or int(protocol.get("official_test_access_count", -1)) != 0
        or int(folds.get("count", 0)) != 3
        or folds.get("identity_canonicalization") != "unsigned-decimal"
    ):
        raise ValueError("invalid frozen three-fold CIRC protocol")
    if (
        endpoint.get("schema_version") != "trifusion-dev-selection-v1"
        or endpoint.get("variant") != "hfer_uniform_generator"
        or endpoint.get("selection_output") != "fused"
        or int(endpoint.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("invalid HFER-uniform endpoint receipt")
    generator_config_sha256 = _sha256(generator_config_path)
    if endpoint.get("circ_protocol_sha256") != protocol_sha256:
        raise ValueError("endpoint receipt is not bound to the CIRC protocol")
    if endpoint.get("config_sha256") != generator_config_sha256:
        raise ValueError("endpoint receipt is not bound to the generator config")
    fixed_endpoint = int(endpoint["epoch"])
    schedule_horizon = int(selection.get("schedule_horizon_epochs", 0))
    if not 1 <= fixed_endpoint <= schedule_horizon or schedule_horizon != 60:
        raise ValueError("fixed endpoint is outside the registered 60-epoch schedule")
    checkpoint = Path(str(endpoint["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != endpoint.get(
        "checkpoint_sha256"
    ):
        raise ValueError("selector checkpoint is missing or hash-mismatched")

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
        if not receipt_path.is_file():
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
            (fold_output / "worker.log").write_text(
                completed.stdout + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0 or not receipt_path.is_file():
                raise RuntimeError(f"fold-{target_fold} generator worker failed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required = {
            "status": "COMPLETE",
            "target_fold": target_fold,
            "fixed_endpoint": fixed_endpoint,
            "schedule_horizon_epochs": schedule_horizon,
            "generator_target_identity_overlap": 0,
            "dev_evaluation_count": 0,
            "target_loader_iteration_count": 0,
            "official_test_access_count": 0,
        }
        mismatches = {
            key: {"expected": value, "actual": receipt.get(key)}
            for key, value in required.items()
            if receipt.get(key) != value
        }
        checkpoint_path = Path(str(receipt.get("checkpoint", ""))).expanduser()
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            mismatches["checkpoint"] = "missing or hash-mismatched"
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
            from tools.run_trifusion_experiment import _worker_dev

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
