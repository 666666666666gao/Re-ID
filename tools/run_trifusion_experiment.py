#!/usr/bin/env python3
"""Fail-closed TriFusion RGBNT201 experiment entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from modeling.trifusion.variants import (  # noqa: E402
    resolve_variant,
    variant_names,
    variant_sha256,
)
from modeling.trifusion.protocol import (  # noqa: E402
    CIRC_PROTOCOL_PATH,
    CIRC_PROTOCOL_SHA256,
    load_trusted_circ_protocol,
    trifusion_source_hashes,
)

DEFAULT_CONFIG = PROJECT / "configs/RGBNT201/TriFusion.yml"
DEV_PROTOCOL = PROJECT / "protocols/rgbnt201_dev_v1.json"
DATASET_RECEIPT = PROJECT / "evidence/rgbnt201_audit_20260831.json"
EXPECTED = {
    "clip_sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "dev_protocol_sha256": "d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946",
    "dataset_receipt_sha256": "ec36309921a3dd7c12d46bb60a83406440ba316f171e419a67ad2cc83bf24318",
}
RESOURCE_PROFILES = {
    "standard_b16k4": {
        "train_batch_size": 16,
        "num_instances": 4,
        "eval_batch_size": 64,
        "num_workers": 0,
        "gradient_accumulation": 1,
        "amp": True,
        "amp_init_scale": 65536.0,
        "max_epochs": 60,
    },
    "low_vram_b8k4": {
        "train_batch_size": 8,
        "num_instances": 4,
        "eval_batch_size": 32,
        "num_workers": 0,
        "gradient_accumulation": 1,
        "amp": True,
        "amp_init_scale": 1024.0,
        "max_epochs": 60,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _collaborative_reliability_mode(contract: dict[str, Any]) -> str:
    reliability = contract.get("reliability")
    if reliability == "uniform":
        return "uniform"
    if reliability in ("joint_beta_observational", "joint_beta_circ"):
        return "joint_beta"
    raise ValueError(f"unsupported collaborative reliability mode: {reliability}")


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


def _gpu_state() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return {"query_passed": False, "error": completed.stderr.strip()}
    fields = [field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 3:
        return {"query_passed": False, "error": "unexpected nvidia-smi output"}
    return {
        "query_passed": True,
        "name": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
    }


def _gpu_gate(gpu: dict[str, Any], resource_profile: str) -> dict[str, Any]:
    if not gpu.get("query_passed"):
        return {
            "policy": "gpu_query_required",
            "error": gpu.get("error"),
            "passed": False,
        }
    if resource_profile == "low_vram_b8k4":
        observed_free_mib = int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"])
        required_free_mib = 6144
        return {
            "policy": "minimum_free_memory",
            "required_free_mib": required_free_mib,
            "observed_free_mib": observed_free_mib,
            "passed": observed_free_mib >= required_free_mib,
        }
    required_used_mib = 500
    return {
        "policy": "maximum_used_memory",
        "required_used_strictly_below_mib": required_used_mib,
        "observed_used_mib": int(gpu["memory_used_mib"]),
        "passed": int(gpu["memory_used_mib"]) < required_used_mib,
    }


def _complete_selection_chain_valid(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    identity_path: Path,
    checkpoint_path: Path,
) -> bool:
    """Fail closed unless a selector export is bound to its full recovery state."""

    try:
        output_root = receipt_path.parent.resolve()
        manifest_path = Path(str(receipt["recovery_manifest"])).expanduser().resolve()
        if manifest_path != output_root / ".resume/latest.json":
            return False
        if (
            not manifest_path.is_file()
            or receipt.get("recovery_manifest_sha256") != _sha256(manifest_path)
        ):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(manifest.get("epoch", -1)) != 60
            or manifest.get("phase") != "complete"
            or manifest.get("run_identity_sha256") != _sha256(identity_path)
        ):
            return False
        current = dict(manifest.get("current", {}))
        generation = (output_root / str(current.get("path", ""))).resolve()
        if (
            not generation.is_relative_to(output_root)
            or not generation.is_file()
            or current.get("sha256") != _sha256(generation)
        ):
            return False
        evidence = dict(manifest.get("completion_evidence", {}))
        return (
            evidence.get("kind") == "selector"
            and int(evidence.get("epoch", -1)) == 60
            and evidence.get("phase") == "complete"
            and evidence.get("run_identity_sha256") == _sha256(identity_path)
            and int(evidence.get("best_epoch", -1)) == int(receipt.get("epoch", -2))
            and float(evidence.get("best_map", float("nan")))
            == float(receipt.get("dev_selection_mAP", float("inf")))
            and evidence.get("best_metrics") == receipt.get("metrics_percent")
            and evidence.get("best_checkpoint_sha256") == _sha256(checkpoint_path)
            and evidence.get("contract_testing") is False
            and evidence.get("scientific_evidence_eligible") is True
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _preflight(
    config_path: Path,
    variant: str,
    *,
    data_mode: str = "development",
) -> dict[str, Any]:
    blockers: list[str] = []
    if data_mode not in ("development", "postfreeze-final"):
        raise ValueError("preflight data mode is not registered")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("SCHEMA_VERSION") != 1:
        blockers.append("config_schema_drift")
    if variant != config.get("EXPERIMENT", {}).get("VARIANT"):
        blockers.append("variant_config_mismatch")
    contract = dict(resolve_variant(variant))
    current_source_sha256 = trifusion_source_hashes()
    clip = Path(config["MODEL"]["CLIP_CHECKPOINT"]).expanduser().resolve()
    immutable = {
        "clip": (clip, EXPECTED["clip_sha256"]),
        "dev_protocol": (DEV_PROTOCOL, EXPECTED["dev_protocol_sha256"]),
        "dataset_receipt": (DATASET_RECEIPT, EXPECTED["dataset_receipt_sha256"]),
    }
    if variant == "hfer_uniform_generator" or contract.get(
        "circ_targets_required"
    ):
        configured_circ = (
            PROJECT / str(config.get("PROTOCOL", {}).get("CIRC_PROTOCOL", ""))
        ).resolve()
        if configured_circ != CIRC_PROTOCOL_PATH.resolve():
            blockers.append("circ_protocol_path_drift")
        immutable["circ_protocol"] = (
            CIRC_PROTOCOL_PATH,
            CIRC_PROTOCOL_SHA256,
        )
    file_checks: dict[str, Any] = {}
    for label, (path, expected_hash) in immutable.items():
        if not path.is_file():
            blockers.append(f"missing:{label}")
            file_checks[label] = {"path": str(path), "exists": False}
            continue
        actual_hash = _sha256(path)
        file_checks[label] = {
            "path": str(path),
            "exists": True,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_match": actual_hash == expected_hash,
        }
        if actual_hash != expected_hash:
            blockers.append(f"hash_drift:{label}")
    circ_assets: dict[str, Any] = {}
    if contract.get("circ_targets_required"):
        circ_config = dict(config.get("CIRC", {}))
        target_cache = Path(str(circ_config.get("TARGET_CACHE", ""))).expanduser().resolve()
        scoring_receipt_path = Path(
            str(circ_config.get("SCORING_RECEIPT", ""))
        ).expanduser().resolve()
        warm_checkpoint = Path(
            str(circ_config.get("WARM_START_CHECKPOINT", ""))
        ).expanduser().resolve()
        warm_receipt_path = Path(
            str(circ_config.get("WARM_START_RECEIPT", ""))
        ).expanduser().resolve()
        required_paths = {
            "target_cache_receipt": target_cache / "receipt.json",
            "target_cache_rows": target_cache / "targets.jsonl",
            "target_cache_calibration": target_cache / "calibration_receipt.json",
            "scoring_receipt": scoring_receipt_path,
            "scoring_run_identity": scoring_receipt_path.parent / "run_identity.json",
            "warm_start_checkpoint": warm_checkpoint,
            "warm_start_receipt": warm_receipt_path,
        }
        if any(not path.is_file() for path in required_paths.values()):
            blockers.append("missing_circ_training_assets")
            circ_assets = {
                label: {"path": str(path), "exists": path.is_file()}
                for label, path in required_paths.items()
            }
        else:
            cache_receipt = json.loads(
                required_paths["target_cache_receipt"].read_text(encoding="utf-8")
            )
            scoring_receipt = json.loads(
                scoring_receipt_path.read_text(encoding="utf-8")
            )
            warm_receipt = json.loads(warm_receipt_path.read_text(encoding="utf-8"))
            calibration_receipt = json.loads(
                required_paths["target_cache_calibration"].read_text(encoding="utf-8")
            )
            scoring_identity = json.loads(
                required_paths["scoring_run_identity"].read_text(encoding="utf-8")
            )
            targets_sha256 = _sha256(required_paths["target_cache_rows"])
            circ_protocol_payload = json.loads(
                CIRC_PROTOCOL_PATH.read_text(encoding="utf-8")
            )
            expected_condition_count = sum(
                len(item["seeds"])
                for item in circ_protocol_payload.get("conditions", [])
            )
            target_rows = int(cache_receipt.get("row_count", -1))
            symmetry_audit = dict(scoring_receipt.get("symmetry_audit", {}))
            calibration_audit = dict(cache_receipt.get("calibration_audit", {}))
            expected_group_axes = list(
                circ_protocol_payload.get("audits", {}).get("group_axes", [])
            )
            if (
                cache_receipt.get("protocol_hash") != CIRC_PROTOCOL_SHA256
                or cache_receipt.get("targets_sha256") != targets_sha256
                or cache_receipt.get("mode") != data_mode
                or target_rows <= 0
                or expected_condition_count <= 0
                or target_rows % expected_condition_count != 0
                or target_rows // expected_condition_count
                > (3126 if data_mode == "development" else 3951)
                or int(cache_receipt.get("cross_camera_primary_rows", -1))
                != target_rows
                or int(cache_receipt.get("same_camera_only_rows", -1)) != 0
                or int(cache_receipt.get("invalid_support_rows", -1)) != 0
                or int(cache_receipt.get("official_test_access_count", -1)) != 0
                or cache_receipt.get("zero_identity_overlap") is not True
                or scoring_receipt.get("status") != "COMPLETE"
                or scoring_receipt.get("mode", "development") != data_mode
                or scoring_receipt.get("scientific_evidence_eligible") is not True
                or scoring_receipt.get("contract_testing") is not False
                or int(scoring_receipt.get("official_test_access_count", -1)) != 0
                or scoring_receipt.get("circ_protocol_sha256")
                != CIRC_PROTOCOL_SHA256
                or scoring_receipt.get("targets_sha256") != targets_sha256
                or scoring_receipt.get("cache_receipt_sha256")
                != _sha256(required_paths["target_cache_receipt"])
                or scoring_receipt.get("calibration_receipt_sha256")
                != _sha256(required_paths["target_cache_calibration"])
                or scoring_receipt.get("run_identity_sha256")
                != _sha256(required_paths["scoring_run_identity"])
                or scoring_receipt.get("generator_aggregate_sha256")
                != scoring_identity.get("generator_aggregate_sha256")
                or scoring_receipt.get("generator_fold_receipt_sha256")
                != scoring_identity.get("generator_fold_receipt_sha256")
                or int(scoring_receipt.get("rows", -1)) != target_rows
                or int(scoring_receipt.get("conditions", -1))
                != expected_condition_count
                or symmetry_audit.get("status") != "PASS"
                or symmetry_audit.get("claim_eligible") is not True
                or calibration_receipt.get("protocol_hash")
                != CIRC_PROTOCOL_SHA256
                or calibration_receipt.get("targets_sha256") != targets_sha256
                or calibration_receipt.get("status") != "COMPLETE"
                or calibration_receipt.get("group_axes") != expected_group_axes
                or calibration_receipt.get("effective_sample_size", {}).get("unit")
                != circ_protocol_payload.get("audits", {}).get(
                    "effective_sample_size_unit"
                )
                or len(calibration_receipt.get("per_condition", {}))
                != expected_condition_count
                or calibration_audit.get("audit_sha256")
                != calibration_receipt.get("audit_sha256")
                or scoring_receipt.get("calibration_audit") != calibration_audit
                or scoring_identity.get("circ_protocol_sha256")
                != CIRC_PROTOCOL_SHA256
                or scoring_identity.get("source_sha256") != current_source_sha256
                or scoring_identity.get("scientific_evidence_eligible") is not True
                or scoring_identity.get("contract_testing") is not False
                or int(scoring_identity.get("official_test_access_count", -1)) != 0
            ):
                blockers.append("invalid_circ_target_cache_evidence")
            warm_identity_path = Path(
                str(warm_receipt.get("run_identity", ""))
            ).expanduser().resolve()
            warm_identity = (
                json.loads(warm_identity_path.read_text(encoding="utf-8"))
                if warm_identity_path.is_file()
                else {}
            )
            if (
                warm_receipt.get("schema_version")
                != "trifusion-dev-selection-v1"
                or warm_receipt.get("variant") != "hfer_uniform_generator"
                or warm_receipt.get("phase") != "complete"
                or warm_receipt.get("scientific_evidence_eligible") is not True
                or warm_receipt.get("contract_testing") is not False
                or int(warm_receipt.get("official_test_access_count", -1)) != 0
                or warm_receipt.get("circ_protocol_sha256")
                != CIRC_PROTOCOL_SHA256
                or int(warm_receipt.get("schedule_horizon_epochs", -1)) != 60
                or int(warm_receipt.get("dev_evaluation_count", -1)) != 60
                or Path(str(warm_receipt.get("checkpoint", ""))).resolve()
                != warm_checkpoint
                or warm_receipt.get("checkpoint_sha256") != _sha256(warm_checkpoint)
                or not warm_identity
                or warm_receipt.get("run_identity_sha256")
                != _sha256(warm_identity_path)
                or warm_identity.get("circ_protocol_sha256")
                != CIRC_PROTOCOL_SHA256
                or warm_identity.get("source_sha256") != current_source_sha256
                or warm_identity.get("scientific_evidence_eligible") is not True
                or warm_identity.get("contract_testing") is not False
                or not _complete_selection_chain_valid(
                    receipt=warm_receipt,
                    receipt_path=warm_receipt_path,
                    identity_path=warm_identity_path,
                    checkpoint_path=warm_checkpoint,
                )
            ):
                blockers.append("invalid_circ_warm_start_evidence")
            circ_assets = {
                label: {
                    "path": str(path),
                    "exists": True,
                    "sha256": _sha256(path),
                }
                for label, path in required_paths.items()
            }
    protocol = json.loads(DEV_PROTOCOL.read_text(encoding="utf-8"))
    if data_mode == "development":
        data_protocol = {
            "fit_identities": len(protocol["train_ids"]),
            "fit_records": int(protocol["counts"]["train_triplets"]),
            "dev_identities": len(protocol["dev_ids"]),
            "query_records": int(protocol["evaluation"]["query_triplets"]),
            "gallery_records": int(protocol["evaluation"]["gallery_triplets"]),
            "identity_overlap": len(
                set(protocol["train_ids"]) & set(protocol["dev_ids"])
            ),
            "official_test_records": 0,
            "uses_test_labels": bool(protocol["selection"]["uses_test_labels"]),
        }
        expected_data_protocol = {
            "fit_identities": 141,
            "fit_records": 3126,
            "dev_identities": 30,
            "query_records": 825,
            "gallery_records": 825,
            "identity_overlap": 0,
            "official_test_records": 0,
            "uses_test_labels": False,
        }
    else:
        data_protocol = {
            "fit_identities": len(protocol["train_ids"])
            + len(protocol["dev_ids"]),
            "fit_records": 3951,
            "dev_identities": 0,
            "query_records": 836,
            "gallery_records": 836,
            "identity_overlap": 0,
            "official_test_records": 836,
            "uses_test_labels_for_selection": False,
            "further_model_selection": False,
        }
        expected_data_protocol = {
            "fit_identities": 171,
            "fit_records": 3951,
            "dev_identities": 0,
            "query_records": 836,
            "gallery_records": 836,
            "identity_overlap": 0,
            "official_test_records": 836,
            "uses_test_labels_for_selection": False,
            "further_model_selection": False,
        }
    if data_protocol != expected_data_protocol:
        blockers.append(f"{data_mode}_protocol_drift")
    optimization = {
        "train_batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "eval_batch_size": int(config["DATA"]["EVAL_BATCH_SIZE"]),
        "num_workers": int(config["DATA"]["NUM_WORKERS"]),
        "gradient_accumulation": int(config["OPTIMIZATION"]["GRADIENT_ACCUMULATION"]),
        "amp": bool(config["OPTIMIZATION"]["AMP"]),
        "amp_init_scale": float(config["OPTIMIZATION"].get("AMP_INIT_SCALE", 65536.0)),
        "max_epochs": int(config["OPTIMIZATION"]["MAX_EPOCHS"]),
        "router_warm_epochs": int(
            config["OPTIMIZATION"].get("ROUTER_WARM_EPOCHS", 0)
        ),
    }
    if contract.get("circ_targets_required") and not (
        0
        < optimization["router_warm_epochs"]
        < optimization["max_epochs"]
    ):
        blockers.append("invalid_circ_router_warm_schedule")
    if not contract.get("circ_targets_required") and optimization["router_warm_epochs"]:
        blockers.append("router_warm_schedule_without_circ")
    resource_profile = str(
        config.get("EXPERIMENT", {}).get("RESOURCE_PROFILE", "standard_b16k4")
    )
    registered_profile = RESOURCE_PROFILES.get(resource_profile)
    expected_optimization = (
        None
        if registered_profile is None
        else {
            **registered_profile,
            "router_warm_epochs": (
                7 if contract.get("circ_targets_required") else 0
            ),
        }
    )
    if expected_optimization is None:
        blockers.append("unknown_resource_profile")
    elif optimization != expected_optimization:
        blockers.append("resource_profile_drift")
    if (
        optimization["num_instances"] < 2
        or optimization["train_batch_size"] % optimization["num_instances"] != 0
        or optimization["train_batch_size"] // optimization["num_instances"] < 2
    ):
        blockers.append("invalid_pk_batch")
    if optimization["gradient_accumulation"] != 1:
        blockers.append("batch_hard_accumulation_forbidden")
    gpu = _gpu_state()
    gpu_gate = _gpu_gate(gpu, resource_profile)
    if not gpu.get("query_passed"):
        blockers.append("gpu_query_failed")
    elif not gpu_gate["passed"]:
        blockers.append("gpu_memory_gate")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "schema_version": "1.0",
        "mode": "preflight",
        "data_mode": data_mode,
        "variant": variant,
        "variant_contract": contract,
        "variant_contract_sha256": variant_sha256(contract),
        "resource_profile": resource_profile,
        "status": "READY" if not blockers else "BLOCKED",
        "launch_allowed": not blockers,
        "required_memory_used_strictly_below_mib": (
            500 if resource_profile != "low_vram_b8k4" else None
        ),
        "blockers": blockers,
        "gpu": gpu,
        "gpu_gate": gpu_gate,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "repository_head": head.stdout.strip() if head.returncode == 0 else None,
        "runner_sha256": _sha256(Path(__file__)),
        "source_sha256": current_source_sha256,
        "file_checks": file_checks,
        "circ_assets": circ_assets,
        "data_protocol": data_protocol,
        "optimization": optimization,
        "model_constructed": False,
        "training_started": False,
        "official_test_access_count": 0,
        "metric_result": None,
        "sota_claim_supported": False,
        "claim_boundary": "preflight only; no model construction, CUDA forward, training metric or SOTA claim",
    }


def _capacity(
    config_path: Path, variant: str, output_dir: Path
) -> tuple[dict[str, Any], int]:
    receipt = _preflight(config_path, variant)
    receipt["mode"] = "capacity"
    receipt["worker_executed"] = False
    if not receipt["launch_allowed"]:
        receipt["claim_boundary"] = "capacity blocked before model construction; no metric"
        return receipt, 0
    test_executable = os.environ.get("TRIFUSION_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("test_executable_without_contract_testing")
        return receipt, 2
    command = (
        [
            test_executable,
            "--_worker",
            "capacity",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if test_executable
        else [
            sys.executable,
            str(Path(__file__)),
            "--_worker",
            "capacity",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    worker_log = output_dir / "capacity_worker.log"
    worker_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    result_path = output_dir / "worker_result.json"
    receipt.update(
        {
            "worker_executed": True,
            "worker_command": command,
            "worker_returncode": completed.returncode,
            "worker_log_sha256": _sha256(worker_log),
            "test_override_used": bool(test_executable),
        }
    )
    if not result_path.is_file():
        receipt["status"] = "FAILED"
        receipt["blockers"].append("capacity_worker_result_missing")
        return receipt, 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt.update(result)
    receipt["worker_result_sha256"] = _sha256(result_path)
    required = {
        "status": "PASS",
        "steps": 8,
        "batch_size": receipt["optimization"]["train_batch_size"],
        "num_instances": receipt["optimization"]["num_instances"],
        "finite_losses": True,
        "gradient_safety_pass": True,
        "model_parameters_finite": True,
        "gradient_parameter_coverage": 1.0,
        "official_test_access_count": 0,
        "dev_loader_iterations": 0,
        "parameter_budget_pass": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": result.get(key)}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    if completed.returncode or mismatches:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("capacity_contract_failed")
        receipt["capacity_contract_mismatches"] = mismatches
        return receipt, 2
    receipt["status"] = "PASS"
    receipt["claim_boundary"] = "eight train-only steps; no dev/test metric and no SOTA claim"
    return receipt, 0


def _overfit(
    config_path: Path, variant: str, output_dir: Path
) -> tuple[dict[str, Any], int]:
    receipt = _preflight(config_path, variant)
    receipt["mode"] = "overfit"
    receipt["worker_executed"] = False
    if not receipt["launch_allowed"]:
        receipt["claim_boundary"] = "overfit blocked before model construction; no metric"
        return receipt, 0
    test_executable = os.environ.get("TRIFUSION_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("test_executable_without_contract_testing")
        return receipt, 2
    command = (
        [
            test_executable,
            "--_worker",
            "overfit",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if test_executable
        else [
            sys.executable,
            str(Path(__file__)),
            "--_worker",
            "overfit",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    worker_log = output_dir / "overfit_worker.log"
    worker_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    result_path = output_dir / "worker_result.json"
    receipt.update(
        {
            "worker_executed": True,
            "worker_command": command,
            "worker_returncode": completed.returncode,
            "worker_log_sha256": _sha256(worker_log),
            "test_override_used": bool(test_executable),
        }
    )
    if not result_path.is_file():
        receipt["status"] = "FAILED"
        receipt["blockers"].append("overfit_worker_result_missing")
        return receipt, 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt.update(result)
    receipt["worker_result_sha256"] = _sha256(result_path)
    required = {
        "status": "PASS",
        "steps": 100,
        "batch_size": receipt["optimization"]["train_batch_size"],
        "num_instances": receipt["optimization"]["num_instances"],
        "finite_losses": True,
        "gradient_safety_pass": True,
        "model_parameters_finite": True,
        "official_test_access_count": 0,
        "dev_loader_iterations": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in required.items()
        if result.get(key) != value
    }
    if not isinstance(result.get("fixed_batch_sha256"), str) or len(
        result.get("fixed_batch_sha256", "")
    ) != 64:
        mismatches["fixed_batch_sha256"] = {"expected": "64 hex chars", "actual": result.get("fixed_batch_sha256")}
    if float(result.get("loss_ratio", float("inf"))) > 0.2:
        mismatches["loss_ratio"] = {"expected": "<=0.2", "actual": result.get("loss_ratio")}
    if completed.returncode or mismatches:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("overfit_contract_failed")
        receipt["overfit_contract_mismatches"] = mismatches
        return receipt, 2
    receipt["status"] = "PASS"
    receipt["claim_boundary"] = "one fixed train-only batch; no dev/test metric and no SOTA claim"
    return receipt, 0


def _run_identity(preflight: dict[str, Any], variant: str) -> dict[str, Any]:
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    data_mode = str(preflight.get("data_mode", "development"))
    identity = {
        "schema_version": "1.0",
        "run_type": (
            "TriFusion/RGBNT201/train-only-dev"
            if data_mode == "development"
            else "TriFusion/RGBNT201/postfreeze-final-fixed"
        ),
        "data_mode": data_mode,
        "variant": variant,
        "variant_contract": preflight["variant_contract"],
        "variant_contract_sha256": preflight["variant_contract_sha256"],
        "repository_head": preflight["repository_head"],
        "runner_sha256": _sha256(Path(__file__)),
        "source_sha256": preflight["source_sha256"],
        "config_sha256": preflight["config_sha256"],
        "clip_sha256": EXPECTED["clip_sha256"],
        "dev_protocol_sha256": EXPECTED["dev_protocol_sha256"],
        "dataset_receipt_sha256": EXPECTED["dataset_receipt_sha256"],
        "data_protocol": preflight["data_protocol"],
        "optimization": preflight["optimization"],
        "official_test_access_during_development": False,
        "contract_testing": contract_testing,
        "scientific_evidence_eligible": not contract_testing,
    }
    if data_mode == "postfreeze-final":
        identity.update(
            {
                "all_171_training_identities": True,
                "former_dev_identities_training_only": True,
                "further_model_selection": False,
                "official_test_evaluations_before_fixed_endpoint": 0,
                "official_test_evaluations_after_fixed_endpoint": 1,
            }
        )
    if variant == "hfer_uniform_generator" or preflight["variant_contract"].get(
        "circ_targets_required"
    ):
        identity["circ_protocol_sha256"] = preflight["file_checks"][
            "circ_protocol"
        ]["sha256"]
    if preflight["variant_contract"].get("circ_targets_required"):
        identity["circ_assets"] = preflight["circ_assets"]
    return identity


def _validate_recovery(output_dir: Path) -> dict[str, Any]:
    children = list(output_dir.iterdir())
    if not children:
        return {"valid": True, "kind": "fresh", "epoch": 0, "phase": "initial"}
    identity_path = output_dir / "run_identity.json"
    latest_path = output_dir / ".resume/latest.json"
    if not identity_path.is_file() or not latest_path.is_file():
        return {"valid": False, "error": "nonempty_output_without_recovery"}
    try:
        manifest = json.loads(latest_path.read_text(encoding="utf-8"))
        if manifest.get("run_identity_sha256") != _sha256(identity_path):
            raise ValueError("run identity hash mismatch")
        epoch = int(manifest["epoch"])
        phase = str(manifest["phase"])
        if epoch < 0 or epoch > 60 or phase not in {
            "epoch_boundary",
            "post_train",
            "post_eval",
            "complete",
        }:
            raise ValueError("invalid epoch or phase")
        current = manifest["current"]
        current_path = output_dir / current["path"]
        if not current_path.is_file() or _sha256(current_path) != current["sha256"]:
            raise ValueError("current generation missing or corrupt")
        previous = manifest.get("previous")
        if previous:
            previous_path = output_dir / previous["path"]
            if not previous_path.is_file() or _sha256(previous_path) != previous["sha256"]:
                raise ValueError("previous generation missing or corrupt")
        return {
            "valid": True,
            "kind": "resume",
            "epoch": epoch,
            "phase": phase,
            "manifest_sha256": _sha256(latest_path),
        }
    except Exception as error:
        return {"valid": False, "error": f"invalid_recovery:{type(error).__name__}:{error}"}


def _dev(
    config_path: Path,
    variant: str,
    output_dir: Path,
    *,
    data_mode: str = "development",
) -> tuple[dict[str, Any], int]:
    recovery = _validate_recovery(output_dir)
    preflight = _preflight(config_path, variant, data_mode=data_mode)
    preflight["mode"] = "dev" if data_mode == "development" else "final"
    preflight["recovery"] = recovery
    preflight["worker_executed"] = False
    if not recovery["valid"]:
        preflight["status"] = "RECOVERY_REJECTED"
        preflight["launch_allowed"] = False
        preflight["blockers"].append("invalid_or_foreign_recovery")
        return preflight, 2
    expected_identity = _run_identity(preflight, variant)
    identity_path = output_dir / "run_identity.json"
    if recovery["kind"] == "fresh":
        if not preflight["launch_allowed"]:
            preflight["claim_boundary"] = (
                f"{data_mode} run blocked before model construction"
            )
            return preflight, 0
        _atomic_json(identity_path, expected_identity)
    else:
        actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if actual_identity != expected_identity:
            preflight["status"] = "RECOVERY_REJECTED"
            preflight["launch_allowed"] = False
            preflight["blockers"].append("foreign_run_identity")
            return preflight, 2
        if recovery["phase"] == "complete":
            summary_path = output_dir / "run_summary.json"
            if not summary_path.is_file():
                preflight["status"] = "RECOVERY_REJECTED"
                preflight["blockers"].append("complete_without_summary")
                return preflight, 2
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["complete_resume_no_work"] = True
            summary["worker_executed"] = False
            return summary, 0
        if not preflight["launch_allowed"]:
            preflight["claim_boundary"] = (
                f"{data_mode} resume blocked before model construction"
            )
            return preflight, 0
    test_executable = os.environ.get("TRIFUSION_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        preflight["status"] = "FAILED"
        preflight["blockers"].append("test_executable_without_contract_testing")
        return preflight, 2
    command = (
        [
            test_executable,
            "--_worker",
            "dev" if data_mode == "development" else "final",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if test_executable
        else [
            sys.executable,
            str(Path(__file__)),
            "--_worker",
            "dev" if data_mode == "development" else "final",
            "--variant",
            variant,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    worker_log = output_dir / (
        "dev_worker.log" if data_mode == "development" else "final_worker.log"
    )
    worker_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    result_path = output_dir / (
        "dev_worker_result.json"
        if data_mode == "development"
        else "final_worker_result.json"
    )
    preflight.update(
        {
            "worker_executed": True,
            "worker_command": command,
            "worker_returncode": completed.returncode,
            "worker_log_sha256": _sha256(worker_log),
            "test_override_used": bool(test_executable),
            "run_identity_sha256": _sha256(identity_path),
        }
    )
    if completed.returncode or not result_path.is_file():
        preflight["status"] = "FAILED"
        preflight["blockers"].append("dev_worker_failed")
        return preflight, 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recovery_after = _validate_recovery(output_dir)
    required = (
        {
            "status": "COMPLETE",
            "epoch": 60,
            "phase": "complete",
            "official_test_access_count": 0,
            "dev_evaluation_count": 60,
            "query_records": 825,
            "gallery_records": 825,
            "train_records": 3126,
            "model_constructed": True,
            "training_started": True,
        }
        if data_mode == "development"
        else {
            "status": "COMPLETE",
            "mode": "postfreeze-final",
            "epoch": 60,
            "phase": "complete",
            "official_test_access_count": 1,
            "official_test_evaluation_count": 1,
            "dev_evaluation_count": 0,
            "query_records": 836,
            "gallery_records": 836,
            "train_records": 3951,
            "further_model_selection": False,
            "model_constructed": True,
            "training_started": True,
        }
    )
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in required.items()
        if result.get(key) != value
    }
    metrics = result.get("metrics_percent", {})
    expected_outputs = set(preflight["variant_contract"]["evaluation_outputs"])
    if set(metrics) != expected_outputs:
        mismatches["metrics_percent"] = {
            "expected": sorted(expected_outputs),
            "actual": sorted(metrics),
        }
    expected_selection = (
        "fused"
        if "fused" in expected_outputs
        else preflight["variant_contract"]["active_experts"][0]
    )
    actual_selection = result.get("selection_output", expected_selection)
    if actual_selection != expected_selection:
        mismatches["selection_output"] = {
            "expected": expected_selection,
            "actual": actual_selection,
        }
    result["selection_output"] = actual_selection
    if not recovery_after["valid"] or recovery_after.get("phase") != "complete":
        mismatches["recovery"] = {"expected": "valid complete", "actual": recovery_after}
    preflight.update(result)
    preflight["worker_result_sha256"] = _sha256(result_path)
    preflight["recovery"] = recovery_after
    preflight["claim_scope"] = (
        "train-only development result"
        if data_mode == "development"
        else "single-seed postfreeze-final official result"
    )
    preflight["metric_result"] = metrics
    preflight["claim_boundary"] = (
        "train-only development metrics; no official-test metric and no SOTA claim"
        if data_mode == "development"
        else (
            "one frozen seed-42 official evaluation; target-exceedance may be reported, "
            "but no multi-seed SOTA claim"
        )
    )
    if data_mode == "postfreeze-final":
        fused = dict(metrics.get("fused", {}))
        preflight["registered_public_target_percent"] = {
            "mAP": 85.3,
            "Rank-1": 87.9,
        }
        preflight["single_seed_target_exceeded"] = (
            float(fused.get("mAP", float("-inf"))) > 85.3
            and float(fused.get("Rank-1", float("-inf"))) > 87.9
        )
    preflight["sota_claim_supported"] = False
    preflight["complete_resume_no_work"] = False
    if mismatches:
        preflight["status"] = "FAILED"
        preflight["blockers"].append("dev_contract_failed")
        preflight["dev_contract_mismatches"] = mismatches
        return preflight, 2
    preflight["status"] = "PASS"
    return preflight, 0


def _worker_capacity(
    config_path: Path,
    variant: str,
    output_dir: Path,
    *,
    overfit: bool = False,
) -> int:
    result_path = output_dir / "worker_result.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resource_profile = str(
        config.get("EXPERIMENT", {}).get("RESOURCE_PROFILE", "standard_b16k4")
    )
    second_gpu = _gpu_state()
    second_gpu_gate = _gpu_gate(second_gpu, resource_profile)
    if not second_gpu_gate["passed"]:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "steps": 0,
                "official_test_access_count": 0,
                "dev_loader_iterations": 0,
                "second_gpu_gate": second_gpu_gate,
            },
        )
        return 3
    try:
        import random

        import numpy as np
        import torch

        from modeling.trifusion.builder import (
            build_single_branch_from_clip,
            build_trifusion_from_clip,
        )
        from modeling.trifusion.criterion import (
            SingleBranchCriterion,
            TriFusionCriterion,
        )
        from modeling.trifusion.data import build_rgbnt201_dev_loaders

        contract = dict(resolve_variant(variant))
        seed = int(config["EXPERIMENT"]["SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        data = build_rgbnt201_dev_loaders(
            dataset_root=Path(config["DATA"]["DATASET_ROOT"]),
            protocol_path=PROJECT / config["DATA"]["DEV_PROTOCOL"],
            train_batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
            num_instances=int(config["DATA"]["NUM_INSTANCES"]),
            eval_batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
            num_workers=int(config["DATA"]["NUM_WORKERS"]),
        )
        build_kwargs = {
            "num_classes": data.num_classes,
            "image_size": tuple(config["MODEL"]["IMAGE_SIZE"]),
            "patch_size": int(config["MODEL"]["PATCH_SIZE"]),
            "cnn_width": int(config["MODEL"]["CNN_WIDTH"]),
            "mamba_width": int(config["MODEL"]["MAMBA_WIDTH"]),
            "embedding_width": int(config["MODEL"]["EMBEDDING_WIDTH"]),
            "private_width": int(config["MODEL"]["PRIVATE_WIDTH"]),
        }
        if contract["family"] == "standalone":
            built = build_single_branch_from_clip(
                config["MODEL"]["CLIP_CHECKPOINT"],
                expert_name=contract["active_experts"][0],
                **build_kwargs,
            )
        else:
            built = build_trifusion_from_clip(
                config["MODEL"]["CLIP_CHECKPOINT"],
                relay_rank=int(config["MODEL"]["RELAY_RANK"]),
                reliability_mode=_collaborative_reliability_mode(contract),
                **build_kwargs,
            )
        model = built.model.cuda()
        for name, parameter in model.named_parameters():
            if "private_projection" in name:
                parameter.requires_grad_(False)
        optimizer_parameter_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        pretrained_tokens = (
            (
                "tokenizer.patch_projection",
                "tokenizer.positional_embedding",
                "expert.blocks",
                "expert.class_embedding",
                "expert.class_position",
                "expert.pre_norm",
                "expert.post_norm",
            )
            if contract["family"] == "standalone"
            else (
                "encoder.tokenizer.patch_projection",
                "encoder.tokenizer.positional_embedding",
                "encoder.experts.transformer.blocks",
                "encoder.experts.transformer.class_embedding",
                "encoder.experts.transformer.class_position",
                "encoder.experts.transformer.pre_norm",
                "encoder.experts.transformer.post_norm",
            )
        )
        pretrained_parameters = []
        new_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                pretrained_parameters
                if any(token in name for token in pretrained_tokens)
                else new_parameters
            )
            target.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": pretrained_parameters,
                    "lr": float(config["OPTIMIZATION"]["PRETRAINED_LR"]),
                },
                {
                    "params": new_parameters,
                    "lr": float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
                },
            ],
            weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
        )
        if contract["family"] == "standalone":
            criterion = SingleBranchCriterion(
                triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"])
            ).cuda()
        else:
            criterion = TriFusionCriterion(
                target_cache=None,
                triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
            ).cuda()
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(config["OPTIMIZATION"]["AMP"]),
            init_scale=float(config["OPTIMIZATION"].get("AMP_INIT_SCALE", 65536.0)),
        )
        if contract["family"] == "standalone":
            expert_name = contract["active_experts"][0]
            loss_weights = {
                f"id_{expert_name}": float(config["LOSS"]["ID_BRANCH"]),
                f"triplet_{expert_name}": float(config["LOSS"]["TRIPLET_BRANCH"]),
            }
        else:
            loss_weights = {
                "id_fused": float(config["LOSS"]["ID_FUSED"]),
                "triplet_fused": float(config["LOSS"]["TRIPLET_FUSED"]),
                "id_cnn": float(config["LOSS"]["ID_BRANCH"]),
                "id_transformer": float(config["LOSS"]["ID_BRANCH"]),
                "id_mamba": float(config["LOSS"]["ID_BRANCH"]),
                "triplet_cnn": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "triplet_transformer": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "triplet_mamba": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "reliability": float(config["LOSS"]["RELIABILITY"]),
                "peer_logits": float(config["LOSS"]["PEER_LOGITS"]),
                "peer_role": float(config["LOSS"]["PEER_ROLE"]),
                "private_diversity": float(config["LOSS"]["PRIVATE_DIVERSITY"]),
            }
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        finite_gradient_names: set[str] = set()
        losses_by_step = []
        all_losses_finite = True
        all_gradients_finite = True
        nonfinite_gradients_by_step = []
        last_step_gradients_finite = True
        iterator = iter(data.train_loader)
        fixed_batch = next(iterator) if overfit else None
        fixed_batch_sha256 = None
        if fixed_batch is not None:
            batch_digest = hashlib.sha256()
            batch_images, batch_labels, batch_cameras, batch_views, batch_keys = fixed_batch
            for name in ("RGB", "NI", "TI"):
                batch_digest.update(name.encode("utf-8"))
                batch_digest.update(batch_images[name].contiguous().numpy().tobytes())
            for tensor in (batch_labels, batch_cameras, batch_views):
                batch_digest.update(tensor.contiguous().numpy().tobytes())
            for key in batch_keys:
                batch_digest.update(str(key).encode("utf-8"))
                batch_digest.update(b"\0")
            fixed_batch_sha256 = batch_digest.hexdigest()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        steps = 100 if overfit else 8
        for _step in range(steps):
            raw_batch = fixed_batch if fixed_batch is not None else next(iterator)
            images, labels, _camera_ids, _view_ids, _sample_keys = raw_batch
            images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
            labels = labels.cuda(non_blocking=False)
            modality_mask = torch.ones(labels.shape[0], 3, dtype=torch.bool, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(config["OPTIMIZATION"]["AMP"])):
                output = model(
                    {"images": images, "modality_mask": modality_mask},
                    targets=labels,
                    return_aux=True,
                )
                named_losses = criterion(output, labels)
                total_loss = sum(
                    named_losses[name] * weight for name, weight in loss_weights.items()
                )
            all_losses_finite = all_losses_finite and bool(torch.isfinite(total_loss).item())
            losses_by_step.append(
                {
                    "total": float(total_loss.detach().cpu()),
                    **{name: float(value.detach().cpu()) for name, value in named_losses.items()},
                }
            )
            scale_before = float(scaler.get_scale())
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nonfinite_names = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                finite = bool(torch.isfinite(parameter.grad).all().item())
                all_gradients_finite = all_gradients_finite and finite
                if finite:
                    finite_gradient_names.add(name)
                else:
                    nonfinite_names.append(name)
            scaler.step(optimizer)
            scaler.update()
            last_step_gradients_finite = not nonfinite_names
            if nonfinite_names:
                nonfinite_gradients_by_step.append(
                    {
                        "step": _step + 1,
                        "scale_before": scale_before,
                        "scale_after": float(scaler.get_scale()),
                        "parameter_names": nonfinite_names,
                    }
                )
            torch.cuda.synchronize()
        coverage = len(finite_gradient_names) / len(trainable_names)
        model_parameters_finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in model.parameters()
        )
        amp_overflow_recovered = bool(nonfinite_gradients_by_step) and all(
            event["scale_after"] < event["scale_before"]
            for event in nonfinite_gradients_by_step
        ) and (
            nonfinite_gradients_by_step[-1]["step"] < steps
            and last_step_gradients_finite
        )
        gradient_safety_pass = (
            coverage == 1.0
            and model_parameters_finite
            and (all_gradients_finite or amp_overflow_recovered)
        )
        initial_loss = losses_by_step[0]["total"]
        final_loss = losses_by_step[-1]["total"]
        loss_ratio = final_loss / max(abs(initial_loss), 1e-12)
        gate_pass = all_losses_finite and gradient_safety_pass
        if overfit:
            gate_pass = gate_pass and loss_ratio <= 0.2
        status = "PASS" if gate_pass else "FAILED"
        _atomic_json(
            result_path,
            {
                "status": status,
                "steps": steps,
                "batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
                "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
                "finite_losses": all_losses_finite,
                "finite_gradients": all_gradients_finite,
                "gradient_safety_pass": gradient_safety_pass,
                "model_parameters_finite": model_parameters_finite,
                "amp_overflow_events": len(nonfinite_gradients_by_step),
                "amp_overflow_recovered": amp_overflow_recovered,
                "last_step_gradients_finite": last_step_gradients_finite,
                "gradient_parameter_coverage": coverage,
                "trainable_parameter_tensors": len(trainable_names),
                "finite_gradient_parameter_tensors": len(finite_gradient_names),
                "missing_gradient_parameters": sorted(trainable_names - finite_gradient_names),
                "nonfinite_gradients_by_step": nonfinite_gradients_by_step,
                "fixed_batch_sha256": fixed_batch_sha256,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_ratio": loss_ratio,
                "official_test_access_count": 0,
                "dev_loader_iterations": 0,
                "parameter_budget_pass": bool(built.provenance["parameter_budget_pass"]),
                "total_parameters": int(built.provenance["total_parameters"]),
                "losses_by_step": losses_by_step,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "data_provenance": dict(data.provenance),
                "model_provenance": dict(built.provenance),
                "variant_contract": contract,
                "variant_contract_sha256": variant_sha256(contract),
                "second_gpu_gate": second_gpu_gate,
            },
        )
        return 0 if status == "PASS" else 4
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        _atomic_json(
            result_path,
            {
                "status": "OOM" if "out of memory" in message.lower() else "FAILED",
                "steps": 0,
                "official_test_access_count": 0,
                "dev_loader_iterations": 0,
                "error": message,
                "traceback": traceback.format_exc(),
                "second_gpu_gate": second_gpu_gate,
            },
        )
        return 4


def _atomic_torch_save(path: Path, payload: Any, torch_module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch_module.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _capture_rng(random_module: Any, numpy_module: Any, torch_module: Any) -> dict[str, Any]:
    return {
        "python": random_module.getstate(),
        "numpy": numpy_module.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": torch_module.cuda.get_rng_state_all(),
    }


def _restore_rng(
    rng: dict[str, Any], random_module: Any, numpy_module: Any, torch_module: Any
) -> None:
    random_module.setstate(rng["python"])
    numpy_module.random.set_state(rng["numpy"])
    torch_module.set_rng_state(rng["torch_cpu"])
    torch_module.cuda.set_rng_state_all(rng["torch_cuda"])


def _save_dev_generation(
    output_dir: Path,
    *,
    epoch: int,
    phase: str,
    payload: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    resume_dir = output_dir / ".resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    latest_path = resume_dir / "latest.json"
    old = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.is_file() else None
    generation = resume_dir / f"generation-{epoch:04d}-{phase}.pt"
    _atomic_torch_save(generation, payload, torch_module)
    current = {
        "path": str(generation.relative_to(output_dir)),
        "sha256": _sha256(generation),
        "bytes": generation.stat().st_size,
    }
    previous = old.get("current") if old else None
    if previous and previous["path"] == current["path"]:
        previous = old.get("previous")
    manifest = {
        "schema_version": "1.0",
        "epoch": epoch,
        "phase": phase,
        "run_identity_sha256": _sha256(output_dir / "run_identity.json"),
        "current": current,
        "previous": previous,
    }
    completion_evidence = payload.get("completion_evidence")
    if phase == "complete":
        if not isinstance(completion_evidence, dict):
            raise ValueError("complete recovery requires JSON completion evidence")
        manifest["completion_evidence"] = completion_evidence
    elif completion_evidence is not None:
        raise ValueError("completion evidence is valid only at the complete endpoint")
    _atomic_json(latest_path, manifest)
    keep = {current["path"]}
    if previous:
        keep.add(previous["path"])
    resume_root = resume_dir.resolve()
    for candidate in resume_dir.glob("generation-*.pt"):
        if str(candidate.relative_to(output_dir)) in keep:
            continue
        if candidate.resolve().parent != resume_root:
            raise RuntimeError(f"unsafe recovery cleanup path: {candidate}")
        candidate.unlink()
    return manifest


def _worker_dev(
    config_path: Path,
    variant: str,
    output_dir: Path,
    *,
    oof_target_fold: int | None = None,
    circ_protocol_path: Path | None = None,
    fixed_endpoint: int | None = None,
    data_mode: str = "development",
) -> int:
    oof_mode = oof_target_fold is not None
    if oof_mode != (circ_protocol_path is not None and fixed_endpoint is not None):
        raise ValueError("OOF target fold, protocol and fixed endpoint are all-or-none")
    final_mode = not oof_mode and data_mode == "postfreeze-final"
    result_path = output_dir / (
        "generator_receipt.json"
        if oof_mode
        else ("final_worker_result.json" if final_mode else "dev_worker_result.json")
    )
    eval_loader_iteration_count = 0
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resource_profile = str(
        config.get("EXPERIMENT", {}).get("RESOURCE_PROFILE", "standard_b16k4")
    )
    second_gpu = _gpu_state()
    second_gpu_gate = _gpu_gate(second_gpu, resource_profile)
    if not second_gpu_gate["passed"]:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "epoch": 0,
                "phase": "worker_gpu_recheck",
                "official_test_access_count": 0,
                "dev_evaluation_count": 0,
                "second_gpu_gate": second_gpu_gate,
            },
        )
        return 3
    try:
        import math
        import random

        import numpy as np
        import torch
        import torch.nn.functional as functional

        from modeling.trifusion.builder import (
            build_single_branch_from_clip,
            build_trifusion_from_clip,
        )
        from modeling.trifusion.criterion import (
            SingleBranchCriterion,
            TriFusionCriterion,
        )
        from modeling.trifusion.circ_scoring import (
            apply_registered_condition_batch,
            expand_registered_conditions,
            select_training_conditions,
        )
        from modeling.trifusion.data import (
            build_rgbnt201_dev_loaders,
            build_rgbnt201_final_loaders,
            build_rgbnt201_oof_loaders,
        )
        from modeling.trifusion.intervention_targets import (
            CIRCTargetCache,
            compute_calibration_audit,
        )
        from modeling.trifusion.training_phases import (
            active_loss_weights,
            parameter_trainable_in_phase,
            resolve_training_phase,
        )
        from modeling.trifusion.warm_start import load_hfer_uniform_warm_start
        from utils.reid_evaluation import evaluate_reid

        contract = dict(resolve_variant(variant))
        registered_circ_protocol_path: Path | None = None
        registered_circ_protocol_sha256: str | None = None
        registered_circ_protocol: dict[str, Any] | None = None
        if variant == "hfer_uniform_generator" or contract.get(
            "circ_targets_required"
        ):
            registered_value = config.get("PROTOCOL", {}).get("CIRC_PROTOCOL")
            if not registered_value:
                raise ValueError("HFER-uniform generator requires a frozen CIRC protocol")
            registered_circ_protocol_path = (PROJECT / str(registered_value)).resolve()
            registered_circ_protocol, registered_circ_protocol_sha256 = load_trusted_circ_protocol(
                registered_circ_protocol_path
            )
            if (
                oof_mode
                and Path(circ_protocol_path).resolve()
                != registered_circ_protocol_path
            ):
                raise ValueError("OOF worker protocol differs from generator config")
        seed = int(config["EXPERIMENT"]["SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader_kwargs = {
            "dataset_root": Path(config["DATA"]["DATASET_ROOT"]),
            "protocol_path": PROJECT / config["DATA"]["DEV_PROTOCOL"],
            "train_batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
            "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
            "eval_batch_size": int(config["DATA"]["EVAL_BATCH_SIZE"]),
            "num_workers": int(config["DATA"]["NUM_WORKERS"]),
        }
        if oof_mode:
            if variant != "hfer_uniform_generator":
                raise ValueError("OOF target generators must use HFER-uniform")
            data = build_rgbnt201_oof_loaders(
                **loader_kwargs,
                circ_protocol_path=circ_protocol_path,
                target_fold=int(oof_target_fold),
                mode=data_mode,
            )
        elif data_mode == "postfreeze-final":
            if registered_circ_protocol_path is None:
                raise ValueError("postfreeze-final training requires a CIRC protocol")
            data = build_rgbnt201_final_loaders(
                **loader_kwargs,
                circ_protocol_path=registered_circ_protocol_path,
            )
        else:
            if data_mode != "development":
                raise ValueError("training data mode is not registered")
            data = build_rgbnt201_dev_loaders(**loader_kwargs)
        build_kwargs = {
            "num_classes": data.num_classes,
            "image_size": tuple(config["MODEL"]["IMAGE_SIZE"]),
            "patch_size": int(config["MODEL"]["PATCH_SIZE"]),
            "cnn_width": int(config["MODEL"]["CNN_WIDTH"]),
            "mamba_width": int(config["MODEL"]["MAMBA_WIDTH"]),
            "embedding_width": int(config["MODEL"]["EMBEDDING_WIDTH"]),
            "private_width": int(config["MODEL"]["PRIVATE_WIDTH"]),
        }
        if contract["family"] == "standalone":
            built = build_single_branch_from_clip(
                config["MODEL"]["CLIP_CHECKPOINT"],
                expert_name=contract["active_experts"][0],
                **build_kwargs,
            )
        else:
            built = build_trifusion_from_clip(
                config["MODEL"]["CLIP_CHECKPOINT"],
                relay_rank=int(config["MODEL"]["RELAY_RANK"]),
                reliability_mode=_collaborative_reliability_mode(contract),
                **build_kwargs,
            )
        model = built.model
        circ_target_cache: CIRCTargetCache | None = None
        circ_conditions: tuple[dict[str, int | str], ...] = ()
        circ_operators: dict[str, dict[str, object]] = {}
        circ_training_evidence: dict[str, Any] = {}
        if contract.get("circ_targets_required"):
            if registered_circ_protocol is None or registered_circ_protocol_sha256 is None:
                raise RuntimeError("full CIRC training lacks its trusted protocol")
            circ_config = dict(config["CIRC"])
            target_cache_path = Path(circ_config["TARGET_CACHE"]).expanduser().resolve()
            circ_target_cache = CIRCTargetCache.from_directory(target_cache_path)
            circ_conditions = expand_registered_conditions(registered_circ_protocol)
            circ_operators = {
                str(name): dict(specification)
                for name, specification in registered_circ_protocol[
                    "condition_operators"
                ].items()
            }
            if set(circ_operators) != {
                str(condition["family"]) for condition in circ_conditions
            }:
                raise ValueError("CIRC training operators do not cover the frozen suite")
            train_sample_keys = tuple(
                Path(record[0][0]).name for record in data.train_records
            )
            if len(set(train_sample_keys)) != len(train_sample_keys):
                raise ValueError("RGBNT201 fit sample keys are not unique")
            cache_sample_keys = {
                str(row["sample_key"]) for row in circ_target_cache.rows
            }
            expected_row_count = len(cache_sample_keys) * len(circ_conditions)
            if (
                circ_target_cache.receipt.get("protocol_hash")
                != registered_circ_protocol_sha256
                or circ_target_cache.receipt.get("mode") != data_mode
                or int(circ_target_cache.receipt.get("row_count", -1))
                != expected_row_count
                or len(circ_target_cache.rows) != expected_row_count
                or int(circ_target_cache.receipt.get("official_test_access_count", -1))
                != 0
                or not cache_sample_keys
                or not cache_sample_keys.issubset(set(train_sample_keys))
            ):
                raise ValueError("CIRC cache coverage is not a valid fit-record subset")
            for condition in circ_conditions:
                ordered_supported_keys = tuple(sorted(cache_sample_keys))
                for start in range(0, len(ordered_supported_keys), 512):
                    keys = ordered_supported_keys[start : start + 512]
                    circ_target_cache.lookup(
                        keys,
                        tuple(dict(condition) for _ in keys),
                        device="cpu",
                    )
            warm_checkpoint = Path(
                circ_config["WARM_START_CHECKPOINT"]
            ).expanduser().resolve()
            warm_start = load_hfer_uniform_warm_start(
                model,
                warm_checkpoint,
                allow_classifier_reinitialization=(
                    data_mode == "postfreeze-final"
                ),
            )
            circ_training_evidence = {
                "schedule": "sample-hash offset plus epoch modulo frozen conditions",
                "cycle_epochs": len(circ_conditions),
                "conditions": list(circ_conditions),
                "condition_operators_sha256": _canonical_json_sha256(
                    circ_operators
                ),
                "target_cache": str(target_cache_path),
                "target_cache_rows": len(circ_target_cache.rows),
                "cross_camera_supported_train_samples": len(cache_sample_keys),
                "excluded_no_cross_camera_train_samples": (
                    len(train_sample_keys) - len(cache_sample_keys)
                ),
                "target_cache_targets_sha256": circ_target_cache.receipt[
                    "targets_sha256"
                ],
                "warm_start": {
                    **warm_start,
                    "checkpoint_sha256": _sha256(warm_checkpoint),
                },
                "router_only_warm_epochs": int(
                    config["OPTIMIZATION"]["ROUTER_WARM_EPOCHS"]
                ),
                "router_only_loss_scope": "immutable_circ_reliability_only",
                "joint_phase_starts_epoch": int(
                    config["OPTIMIZATION"]["ROUTER_WARM_EPOCHS"]
                )
                + 1,
                "official_test_access_count": 0,
            }
        model = model.cuda()
        for name, parameter in model.named_parameters():
            if "private_projection" in name:
                parameter.requires_grad_(False)
        pretrained_tokens = (
            (
                "tokenizer.patch_projection",
                "tokenizer.positional_embedding",
                "expert.blocks",
                "expert.class_embedding",
                "expert.class_position",
                "expert.pre_norm",
                "expert.post_norm",
            )
            if contract["family"] == "standalone"
            else (
                "encoder.tokenizer.patch_projection",
                "encoder.tokenizer.positional_embedding",
                "encoder.experts.transformer.blocks",
                "encoder.experts.transformer.class_embedding",
                "encoder.experts.transformer.class_position",
                "encoder.experts.transformer.pre_norm",
                "encoder.experts.transformer.post_norm",
            )
        )
        pretrained_parameters = []
        new_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(token in name for token in pretrained_tokens):
                pretrained_parameters.append(parameter)
            else:
                new_parameters.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": pretrained_parameters,
                    "lr": float(config["OPTIMIZATION"]["PRETRAINED_LR"]),
                },
                {
                    "params": new_parameters,
                    "lr": float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
                },
            ],
            weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
        )
        schedule_horizon_epochs = int(config["OPTIMIZATION"]["MAX_EPOCHS"])
        router_warm_epochs = int(
            config["OPTIMIZATION"].get("ROUTER_WARM_EPOCHS", 0)
        )
        max_epochs = (
            int(fixed_endpoint) if oof_mode else schedule_horizon_epochs
        )
        if max_epochs < 1 or max_epochs > schedule_horizon_epochs:
            raise ValueError("fixed endpoint is outside the configured schedule horizon")
        warmup_epochs = int(config["OPTIMIZATION"]["WARMUP_EPOCHS"])
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(1, warmup_epochs),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, schedule_horizon_epochs - warmup_epochs),
            eta_min=float(config["OPTIMIZATION"]["PRETRAINED_LR"]) * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        if contract["family"] == "standalone":
            criterion = SingleBranchCriterion(
                triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"])
            ).cuda()
        else:
            criterion = TriFusionCriterion(
                target_cache=circ_target_cache,
                triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
                brier_weight=float(config["LOSS"].get("BRIER_WEIGHT", 1.0)),
                evidence_weight=float(
                    config["LOSS"].get("EVIDENCE_WEIGHT", 0.1)
                ),
            ).cuda()
        amp_enabled = bool(config["OPTIMIZATION"]["AMP"])
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
            init_scale=float(config["OPTIMIZATION"].get("AMP_INIT_SCALE", 65536.0)),
        )
        if contract["family"] == "standalone":
            expert_name = contract["active_experts"][0]
            loss_weights = {
                f"id_{expert_name}": float(config["LOSS"]["ID_BRANCH"]),
                f"triplet_{expert_name}": float(config["LOSS"]["TRIPLET_BRANCH"]),
            }
        else:
            loss_weights = {
                "id_fused": float(config["LOSS"]["ID_FUSED"]),
                "triplet_fused": float(config["LOSS"]["TRIPLET_FUSED"]),
                "id_cnn": float(config["LOSS"]["ID_BRANCH"]),
                "id_transformer": float(config["LOSS"]["ID_BRANCH"]),
                "id_mamba": float(config["LOSS"]["ID_BRANCH"]),
                "triplet_cnn": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "triplet_transformer": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "triplet_mamba": float(config["LOSS"]["TRIPLET_BRANCH"]),
                "reliability": float(config["LOSS"]["RELIABILITY"]),
                "peer_logits": float(config["LOSS"]["PEER_LOGITS"]),
                "peer_role": float(config["LOSS"]["PEER_ROLE"]),
                "private_diversity": float(config["LOSS"]["PRIVATE_DIVERSITY"]),
            }
        identity_hash = _sha256(output_dir / "run_identity.json")
        latest_path = output_dir / ".resume/latest.json"
        current_epoch = 0
        phase = "epoch_boundary"
        best_epoch = 0
        best_map = float("-inf")
        best_metrics: dict[str, Any] | None = None
        best_checkpoint_sha256: str | None = None
        generator_checkpoint_sha256: str | None = None
        generator_model_state_sha256: str | None = None
        dev_evaluation_count = 0
        train_history: dict[str, Any] = {}
        resume_history: list[dict[str, Any]] = []

        if latest_path.is_file():
            manifest = json.loads(latest_path.read_text(encoding="utf-8"))
            generation = output_dir / manifest["current"]["path"]
            if manifest["run_identity_sha256"] != identity_hash:
                raise RuntimeError("worker recovery identity mismatch")
            if _sha256(generation) != manifest["current"]["sha256"]:
                raise RuntimeError("worker recovery generation hash mismatch")
            saved = torch.load(generation, map_location="cpu")
            required = {
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "rng",
                "epoch",
                "phase",
                "best_epoch",
                "best_map",
                "best_checkpoint_sha256",
                "generator_checkpoint_sha256",
                "generator_model_state_sha256",
                "dev_evaluation_count",
                "run_identity_sha256",
            }
            missing = sorted(required - set(saved))
            if missing or saved["run_identity_sha256"] != identity_hash:
                raise RuntimeError(f"incomplete or foreign recovery: {missing}")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            _restore_rng(saved["rng"], random, np, torch)
            current_epoch = int(saved["epoch"])
            phase = str(saved["phase"])
            best_epoch = int(saved["best_epoch"])
            best_map = float(saved["best_map"])
            best_metrics = saved.get("best_metrics")
            best_checkpoint_sha256 = saved["best_checkpoint_sha256"]
            generator_checkpoint_sha256 = saved["generator_checkpoint_sha256"]
            generator_model_state_sha256 = saved["generator_model_state_sha256"]
            dev_evaluation_count = int(saved["dev_evaluation_count"])
            train_history = dict(saved.get("train_history", {}))
            resume_history = list(saved.get("resume_history", []))
            resume_history.append(
                {
                    "epoch": current_epoch,
                    "phase": phase,
                    "generation_sha256": manifest["current"]["sha256"],
                }
            )
        else:
            initial = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _capture_rng(random, np, torch),
                "epoch": 0,
                "phase": "epoch_boundary",
                "best_epoch": 0,
                "best_map": best_map,
                "best_metrics": None,
                "best_checkpoint_sha256": None,
                "generator_checkpoint_sha256": None,
                "generator_model_state_sha256": None,
                "dev_evaluation_count": 0,
                "run_identity_sha256": identity_hash,
                "train_history": {},
                "resume_history": [],
            }
            _save_dev_generation(
                output_dir,
                epoch=0,
                phase="epoch_boundary",
                payload=initial,
                torch_module=torch,
            )

        def state_payload(epoch: int, state_phase: str) -> dict[str, Any]:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _capture_rng(random, np, torch),
                "epoch": epoch,
                "phase": state_phase,
                "best_epoch": best_epoch,
                "best_map": best_map,
                "best_metrics": best_metrics,
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "generator_checkpoint_sha256": generator_checkpoint_sha256,
                "generator_model_state_sha256": generator_model_state_sha256,
                "dev_evaluation_count": dev_evaluation_count,
                "run_identity_sha256": identity_hash,
                "train_history": train_history,
                "resume_history": resume_history,
                "data_provenance": dict(data.provenance),
                "circ_training_evidence": circ_training_evidence,
            }
            if state_phase == "complete":
                common_evidence = {
                    "epoch": epoch,
                    "phase": state_phase,
                    "run_identity_sha256": identity_hash,
                    "train_history_sha256": _canonical_json_sha256(train_history),
                    "data_provenance_sha256": _canonical_json_sha256(
                        dict(data.provenance)
                    ),
                    "contract_testing": os.environ.get(
                        "TRIFUSION_CONTRACT_TESTING"
                    )
                    == "1",
                    "scientific_evidence_eligible": os.environ.get(
                        "TRIFUSION_CONTRACT_TESTING"
                    )
                    != "1",
                }
                if oof_mode:
                    common_evidence.update(
                        {
                            "kind": "oof-generator",
                            "target_fold": int(oof_target_fold),
                            "generator_checkpoint_sha256": (
                                generator_checkpoint_sha256
                            ),
                            "generator_model_state_sha256": (
                                generator_model_state_sha256
                            ),
                        }
                    )
                elif final_mode:
                    common_evidence.update(
                        {
                            "kind": "postfreeze-final-fixed",
                            "fixed_epoch": max_epochs,
                            "official_test_evaluation_count": 1,
                            "further_model_selection": False,
                            "fixed_metrics": best_metrics,
                            "fixed_checkpoint_sha256": best_checkpoint_sha256,
                        }
                    )
                else:
                    common_evidence.update(
                        {
                            "kind": "selector",
                            "best_epoch": best_epoch,
                            "best_map": best_map,
                            "best_metrics": best_metrics,
                            "best_checkpoint_sha256": best_checkpoint_sha256,
                        }
                    )
                payload["completion_evidence"] = common_evidence
            return payload

        def evaluate() -> dict[str, dict[str, float]]:
            nonlocal eval_loader_iteration_count
            if oof_mode:
                raise RuntimeError("OOF target loader iteration is forbidden")
            eval_loader_iteration_count += 1
            model.eval()
            features: dict[str, list[Any]] = {
                name: [] for name in contract["evaluation_outputs"]
            }
            identities = []
            cameras = []
            for images, pids, camids, _camids_batch, _viewids, _paths in data.eval_loader:
                images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
                mask = torch.ones(len(pids), 3, dtype=torch.bool, device="cuda")
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(
                        {"images": images, "modality_mask": mask},
                        return_aux=True,
                    )
                if contract["family"] == "standalone":
                    features[output.expert].append(
                        output.embedding.detach().float().cpu()
                    )
                else:
                    features["fused"].append(
                        output.fused_embedding.detach().float().cpu()
                    )
                    for expert in ("cnn", "transformer", "mamba"):
                        features[expert].append(
                            output.branch_embeddings[expert].detach().float().cpu()
                        )
                identities.extend(int(pid) for pid in pids)
                cameras.extend(int(camid) for camid in camids.tolist())
            pid_array = np.asarray(identities)
            camera_array = np.asarray(cameras)
            result = {}
            for name, chunks in features.items():
                feature = functional.normalize(torch.cat(chunks), dim=1)
                distances = torch.cdist(
                    feature[: data.num_query],
                    feature[data.num_query :],
                    p=2,
                ).numpy()
                cmc, mean_ap = evaluate_reid(
                    distances,
                    pid_array[: data.num_query],
                    pid_array[data.num_query :],
                    camera_array[: data.num_query],
                    camera_array[data.num_query :],
                    max_rank=50,
                )
                result[name] = {
                    "mAP": float(mean_ap * 100.0),
                    "Rank-1": float(cmc[0] * 100.0),
                    "Rank-5": float(cmc[4] * 100.0),
                    "Rank-10": float(cmc[9] * 100.0),
                }
            return result

        def audit_router_calibration(checkpoint_path: Path) -> dict[str, Any]:
            if circ_target_cache is None or registered_circ_protocol is None:
                return {}
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            supported_keys = {
                str(row["sample_key"]) for row in circ_target_cache.rows
            }
            calibration_records = tuple(
                record
                for record in data.train_records
                if Path(record[0][0]).name in supported_keys
            )
            if {
                Path(record[0][0]).name for record in calibration_records
            } != supported_keys:
                raise RuntimeError("router calibration records do not cover the CIRC cache")
            calibration_loader = build_rgbnt201_record_eval_loader(
                calibration_records,
                batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
                num_workers=int(config["DATA"]["NUM_WORKERS"]),
            )
            predictions: dict[tuple[str, str, str], float] = {}
            for condition in circ_conditions:
                condition_key = json.dumps(
                    condition,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for images, _pids, _camids, _camids_batch, _viewids, paths in calibration_loader:
                    sample_keys = tuple(Path(path_row[0]).name for path_row in paths)
                    conditioned, mask = apply_registered_condition_batch(
                        images,
                        sample_keys,
                        tuple(dict(condition) for _ in sample_keys),
                        operators=circ_operators,
                    )
                    conditioned = {
                        name: tensor.cuda(non_blocking=False)
                        for name, tensor in conditioned.items()
                    }
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                        output = model(
                            {
                                "images": conditioned,
                                "modality_mask": mask.cuda(non_blocking=False),
                            },
                            return_aux=True,
                        )
                    scores = output.reliability.r.detach().float().cpu()
                    for row_index, sample_key in enumerate(sample_keys):
                        for expert_index, expert in enumerate(
                            ("cnn", "transformer", "mamba")
                        ):
                            for modality_index, modality in enumerate(
                                ("RGB", "NI", "TI")
                            ):
                                predictions[
                                    (sample_key, condition_key, f"{expert}.{modality}")
                                ] = float(scores[row_index, expert_index, modality_index])
            audit = compute_calibration_audit(
                circ_target_cache.rows,
                router_probabilities=predictions,
            )
            receipt = {
                **audit,
                "schema_version": "circ-router-calibration-receipt-v1",
                "model_checkpoint": str(checkpoint_path),
                "model_checkpoint_sha256": _sha256(checkpoint_path),
                "targets_sha256": circ_target_cache.receipt["targets_sha256"],
                "circ_protocol_sha256": registered_circ_protocol_sha256,
                "evaluation_rows_are_training_targets": True,
                "causal_calibration_claim_eligible": False,
                "claim_boundary": (
                    "descriptive endpoint calibration on immutable training targets; "
                    "identity-held-out router calibration remains required for a claim"
                ),
                "official_test_access_count": 0,
            }
            _atomic_json(output_dir / "router_calibration_receipt.json", receipt)
            return receipt

        torch.cuda.reset_peak_memory_stats()
        last_metrics = best_metrics
        router_calibration_evidence: dict[str, Any] = {}
        while current_epoch < max_epochs or phase == "post_train":
            if phase in {"epoch_boundary", "post_eval"}:
                epoch = current_epoch + 1
                training_phase = resolve_training_phase(
                    epoch=epoch,
                    circ_enabled=circ_target_cache is not None,
                    router_warm_epochs=router_warm_epochs,
                    schedule_horizon_epochs=schedule_horizon_epochs,
                )
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        name in optimizer_parameter_names
                        and parameter_trainable_in_phase(name, training_phase)
                    )
                if training_phase.name == "router_only":
                    model.eval()
                    model.encoder.reliability_gate.train()
                else:
                    model.train()
                epoch_loss_weights = active_loss_weights(
                    loss_weights, training_phase
                )
                trainable_names = tuple(
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                )
                total_sum = 0.0
                sample_count = 0
                named_sums: dict[str, float] = {name: 0.0 for name in loss_weights}
                condition_counts: dict[str, int] = {}
                circ_supervised_samples = 0
                circ_excluded_samples = 0
                for images, labels, _camera_ids, _view_ids, sample_keys in data.train_loader:
                    labels = labels.cuda(non_blocking=False)
                    batch_conditions = None
                    if circ_target_cache is None:
                        images = {
                            name: tensor.cuda(non_blocking=False)
                            for name, tensor in images.items()
                        }
                        mask = torch.ones(
                            labels.shape[0], 3, dtype=torch.bool, device="cuda"
                        )
                    else:
                        if registered_circ_protocol is None:
                            raise RuntimeError("CIRC condition schedule lost its protocol")
                        sample_keys = tuple(str(value) for value in sample_keys)
                        batch_conditions = select_training_conditions(
                            sample_keys,
                            epoch=epoch,
                            protocol=registered_circ_protocol,
                        )
                        images, mask = apply_registered_condition_batch(
                            images,
                            sample_keys,
                            batch_conditions,
                            operators=circ_operators,
                        )
                        images = {
                            name: tensor.cuda(non_blocking=False)
                            for name, tensor in images.items()
                        }
                        mask = mask.cuda(non_blocking=False)
                        for condition in batch_conditions:
                            condition_key = (
                                f"{condition['family']}:s{condition['severity']}:"
                                f"seed{condition['seed']}"
                            )
                            condition_counts[condition_key] = (
                                condition_counts.get(condition_key, 0) + 1
                            )
                        target_presence = [
                            circ_target_cache.contains(sample_key, condition)
                            for sample_key, condition in zip(
                                sample_keys, batch_conditions
                            )
                        ]
                        circ_supervised_samples += sum(target_presence)
                        circ_excluded_samples += len(target_presence) - sum(
                            target_presence
                        )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        output = model(
                            {"images": images, "modality_mask": mask},
                            targets=labels,
                            return_aux=True,
                        )
                        if circ_target_cache is None:
                            named_losses = criterion(output, labels)
                        else:
                            named_losses = criterion(
                                output,
                                labels,
                                sample_keys=sample_keys,
                                conditions=batch_conditions,
                            )
                        total_loss = sum(
                            named_losses[name] * weight
                            for name, weight in epoch_loss_weights.items()
                        )
                    if not bool(torch.isfinite(total_loss).item()):
                        raise FloatingPointError(f"nonfinite train loss at epoch {epoch}")
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    batch_size = int(labels.shape[0])
                    sample_count += batch_size
                    total_sum += float(total_loss.detach().cpu()) * batch_size
                    for name, value in named_losses.items():
                        named_sums[name] += float(value.detach().cpu()) * batch_size
                scheduler.step()
                if circ_target_cache is not None and circ_supervised_samples == 0:
                    raise RuntimeError(
                        "epoch contained no cross-camera-supported CIRC supervision"
                    )
                train_history[str(epoch)] = {
                    "training_phase": training_phase.name,
                    "parameter_scope": training_phase.parameter_scope,
                    "loss_scope": training_phase.loss_scope,
                    "active_loss_weights": epoch_loss_weights,
                    "trainable_parameter_tensors": len(trainable_names),
                    "trainable_parameter_names_sha256": _canonical_json_sha256(
                        trainable_names
                    ),
                    "total": total_sum / sample_count,
                    "named": {name: value / sample_count for name, value in named_sums.items()},
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "circ_condition_counts": condition_counts,
                    "circ_supervised_samples": circ_supervised_samples,
                    "circ_excluded_no_cross_camera_samples": circ_excluded_samples,
                }
                current_epoch = epoch
                phase = "post_train"
                _save_dev_generation(
                    output_dir,
                    epoch=current_epoch,
                    phase=phase,
                    payload=state_payload(current_epoch, phase),
                    torch_module=torch,
                )

            if phase == "post_train":
                if oof_mode:
                    if current_epoch == max_epochs:
                        checkpoint_path = output_dir / "generator.pth"
                        generator_state = model.state_dict()
                        _atomic_torch_save(checkpoint_path, generator_state, torch)
                        generator_checkpoint_sha256 = _sha256(checkpoint_path)
                        generator_model_state_sha256 = _state_dict_sha256(
                            generator_state
                        )
                        phase = "complete"
                    else:
                        phase = "post_eval"
                    _save_dev_generation(
                        output_dir,
                        epoch=current_epoch,
                        phase=phase,
                        payload=state_payload(current_epoch, phase),
                        torch_module=torch,
                    )
                    if phase == "complete":
                        break
                    continue
                if final_mode:
                    if current_epoch < max_epochs:
                        phase = "post_eval"
                        _save_dev_generation(
                            output_dir,
                            epoch=current_epoch,
                            phase=phase,
                            payload=state_payload(current_epoch, phase),
                            torch_module=torch,
                        )
                        continue
                    fixed_path = output_dir / "fixed_final_model.pth"
                    _atomic_torch_save(fixed_path, model.state_dict(), torch)
                    best_checkpoint_sha256 = _sha256(fixed_path)
                    best_epoch = current_epoch
                    last_metrics = evaluate()
                    best_metrics = last_metrics
                    best_map = float(best_metrics["fused"]["mAP"])
                    _atomic_json(
                        output_dir / "official_test_metrics.json",
                        {
                            "schema_version": "trifusion-official-fixed-v1",
                            "fixed_epoch": current_epoch,
                            "metrics_percent": best_metrics,
                            "query_records": data.num_query,
                            "gallery_records": (
                                len(data.eval_loader.dataset) - data.num_query
                            ),
                            "official_test_evaluation_count": 1,
                            "official_test_access_count": 1,
                            "further_model_selection": False,
                            "checkpoint_sha256": best_checkpoint_sha256,
                        },
                    )
                    phase = "complete"
                    _save_dev_generation(
                        output_dir,
                        epoch=current_epoch,
                        phase=phase,
                        payload=state_payload(current_epoch, phase),
                        torch_module=torch,
                    )
                    break
                last_metrics = evaluate()
                dev_evaluation_count += 1
                _atomic_json(
                    output_dir / "metrics" / f"epoch-{current_epoch:04d}.json",
                    {
                        "epoch": current_epoch,
                        "metrics_percent": last_metrics,
                        "query_records": data.num_query,
                        "gallery_records": len(data.eval_loader.dataset) - data.num_query,
                        "official_test_access_count": 0,
                    },
                )
                selection_output = (
                    "fused"
                    if "fused" in contract["evaluation_outputs"]
                    else contract["active_experts"][0]
                )
                selection_map = float(last_metrics[selection_output]["mAP"])
                if selection_map > best_map:
                    best_map = selection_map
                    best_epoch = current_epoch
                    best_metrics = last_metrics
                    best_path = output_dir / "best_dev_model.pth"
                    _atomic_torch_save(best_path, model.state_dict(), torch)
                    best_checkpoint_sha256 = _sha256(best_path)
                    _atomic_json(
                        output_dir / "best_dev_receipt.json",
                        {
                            "schema_version": "trifusion-dev-selection-v1",
                            "variant": variant,
                            "epoch": best_epoch,
                            "selection_output": selection_output,
                            "dev_selection_mAP": best_map,
                            "metrics_percent": best_metrics,
                            "checkpoint": str(best_path),
                            "checkpoint_sha256": best_checkpoint_sha256,
                            "config_sha256": _sha256(config_path),
                            "circ_protocol_sha256": registered_circ_protocol_sha256,
                            "circ_training_evidence": circ_training_evidence,
                            "variant_contract_sha256": variant_sha256(contract),
                            "selection_split": "train_171 held-out dev identities",
                            "official_test_access_count": 0,
                            "contract_testing": os.environ.get(
                                "TRIFUSION_CONTRACT_TESTING"
                            )
                            == "1",
                            "scientific_evidence_eligible": os.environ.get(
                                "TRIFUSION_CONTRACT_TESTING"
                            )
                            != "1",
                        },
                    )
                phase = "complete" if current_epoch == max_epochs else "post_eval"
                _save_dev_generation(
                    output_dir,
                    epoch=current_epoch,
                    phase=phase,
                    payload=state_payload(current_epoch, phase),
                    torch_module=torch,
                )
                if phase == "complete":
                    break

        if circ_target_cache is not None:
            calibration_checkpoint = output_dir / (
                "fixed_final_model.pth" if final_mode else "best_dev_model.pth"
            )
            if not calibration_checkpoint.is_file():
                raise RuntimeError("CIRC endpoint lacks a calibration checkpoint")
            router_calibration_evidence = audit_router_calibration(
                calibration_checkpoint
            )

        if final_mode:
            fixed_receipt_path = output_dir / "fixed_final_receipt.json"
            if best_metrics is None or best_checkpoint_sha256 is None:
                raise RuntimeError("final run completed without fixed metrics/checkpoint")
            fixed_receipt = {
                "schema_version": "trifusion-postfreeze-final-v1",
                "mode": "postfreeze-final",
                "variant": variant,
                "epoch": max_epochs,
                "phase": "complete",
                "metrics_percent": best_metrics,
                "checkpoint": str(output_dir / "fixed_final_model.pth"),
                "checkpoint_sha256": best_checkpoint_sha256,
                "config_sha256": _sha256(config_path),
                "circ_protocol_sha256": registered_circ_protocol_sha256,
                "circ_training_evidence": circ_training_evidence,
                "router_calibration_receipt": str(
                    output_dir / "router_calibration_receipt.json"
                ),
                "router_calibration_receipt_sha256": _sha256(
                    output_dir / "router_calibration_receipt.json"
                ),
                "router_calibration": router_calibration_evidence,
                "variant_contract_sha256": variant_sha256(contract),
                "training_split": "RGBNT201/train_171 all identities",
                "evaluation_split": "RGBNT201 official test",
                "further_model_selection": False,
                "official_test_evaluation_count": 1,
                "official_test_access_count": 1,
                "run_identity": str(output_dir / "run_identity.json"),
                "run_identity_sha256": identity_hash,
                "recovery_manifest": str(latest_path),
                "recovery_manifest_sha256": _sha256(latest_path),
                "model_constructed": True,
                "training_started": True,
                "fatal_or_nonfinite_detected": False,
                "contract_testing": os.environ.get("TRIFUSION_CONTRACT_TESTING")
                == "1",
                "scientific_evidence_eligible": os.environ.get(
                    "TRIFUSION_CONTRACT_TESTING"
                )
                != "1",
            }
            _atomic_json(fixed_receipt_path, fixed_receipt)
        elif not oof_mode:
            best_receipt_path = output_dir / "best_dev_receipt.json"
            if not best_receipt_path.is_file():
                raise RuntimeError("dev run completed without a best receipt")
            completed_selection = json.loads(
                best_receipt_path.read_text(encoding="utf-8")
            )
            completed_selection.update(
                {
                    "phase": "complete",
                    "schedule_horizon_epochs": schedule_horizon_epochs,
                    "dev_evaluation_count": dev_evaluation_count,
                    "run_identity": str(output_dir / "run_identity.json"),
                    "run_identity_sha256": identity_hash,
                    "recovery_manifest": str(latest_path),
                    "recovery_manifest_sha256": _sha256(latest_path),
                    "model_constructed": True,
                    "training_started": True,
                    "fatal_or_nonfinite_detected": False,
                    "circ_training_evidence": circ_training_evidence,
                    "router_calibration_receipt": (
                        str(output_dir / "router_calibration_receipt.json")
                        if router_calibration_evidence
                        else None
                    ),
                    "router_calibration_receipt_sha256": (
                        _sha256(output_dir / "router_calibration_receipt.json")
                        if router_calibration_evidence
                        else None
                    ),
                    "router_calibration": router_calibration_evidence,
                }
            )
            _atomic_json(best_receipt_path, completed_selection)

        if oof_mode:
            checkpoint_path = output_dir / "generator.pth"
            if (
                not checkpoint_path.is_file()
                or generator_checkpoint_sha256 is None
                or generator_model_state_sha256 is None
                or _sha256(checkpoint_path) != generator_checkpoint_sha256
            ):
                raise RuntimeError("OOF generator checkpoint is not recovery-bound")
            result = {
                "status": "COMPLETE",
                "mode": data_mode,
                "epoch": max_epochs,
                "phase": "complete",
                "target_fold": int(oof_target_fold),
                "fixed_endpoint": max_epochs,
                "schedule_horizon_epochs": schedule_horizon_epochs,
                "generator_target_identity_overlap": int(
                    data.provenance["generator_target_identity_overlap"]
                ),
                "dev_evaluation_count": 0,
                "target_loader_iteration_count": eval_loader_iteration_count,
                "official_test_access_count": 0,
                "query_records": data.num_query,
                "gallery_records": len(data.eval_loader.dataset) - data.num_query,
                "train_records": len(data.train_loader.dataset),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": generator_checkpoint_sha256,
                "model_state_sha256": generator_model_state_sha256,
                "data_provenance": dict(data.provenance),
                "train_history": train_history,
                "resume_history": resume_history,
                "parameter_budget_pass": bool(
                    built.provenance["parameter_budget_pass"]
                ),
                "total_parameters": int(built.provenance["total_parameters"]),
                "variant_contract": contract,
                "variant_contract_sha256": variant_sha256(contract),
                "variant": variant,
                "source_sha256": trifusion_source_hashes(),
                "circ_protocol_sha256": registered_circ_protocol_sha256,
                "config_sha256": _sha256(config_path),
                "run_identity_sha256": identity_hash,
                "recovery_manifest": str(latest_path),
                "recovery_manifest_sha256": _sha256(latest_path),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "fatal_or_nonfinite_detected": False,
                "model_constructed": True,
                "training_started": True,
                "second_gpu_gate": second_gpu_gate,
                "contract_testing": os.environ.get("TRIFUSION_CONTRACT_TESTING")
                == "1",
                "scientific_evidence_eligible": os.environ.get(
                    "TRIFUSION_CONTRACT_TESTING"
                )
                != "1",
            }
        elif final_mode:
            if best_metrics is None or last_metrics is None:
                raise RuntimeError("final run completed without official metrics")
            result = {
                "status": "COMPLETE",
                "mode": "postfreeze-final",
                "epoch": max_epochs,
                "phase": "complete",
                "metrics_percent": best_metrics,
                "last_metrics_percent": last_metrics,
                "dev_evaluation_count": 0,
                "official_test_evaluation_count": 1,
                "official_test_access_count": 1,
                "further_model_selection": False,
                "query_records": data.num_query,
                "gallery_records": len(data.eval_loader.dataset) - data.num_query,
                "train_records": len(data.train_loader.dataset),
                "fixed_checkpoint": str(output_dir / "fixed_final_model.pth"),
                "fixed_checkpoint_sha256": _sha256(
                    output_dir / "fixed_final_model.pth"
                ),
                "train_history": train_history,
                "resume_history": resume_history,
                "parameter_budget_pass": bool(
                    built.provenance["parameter_budget_pass"]
                ),
                "total_parameters": int(built.provenance["total_parameters"]),
                "variant_contract": contract,
                "variant_contract_sha256": variant_sha256(contract),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "fatal_or_nonfinite_detected": False,
                "model_constructed": True,
                "training_started": True,
                "second_gpu_gate": second_gpu_gate,
                "contract_testing": os.environ.get("TRIFUSION_CONTRACT_TESTING")
                == "1",
                "scientific_evidence_eligible": os.environ.get(
                    "TRIFUSION_CONTRACT_TESTING"
                )
                != "1",
                "circ_training_evidence": circ_training_evidence,
                "router_calibration_receipt": str(
                    output_dir / "router_calibration_receipt.json"
                ),
                "router_calibration_receipt_sha256": _sha256(
                    output_dir / "router_calibration_receipt.json"
                ),
                "router_calibration": router_calibration_evidence,
            }
        else:
            if best_metrics is None or last_metrics is None:
                raise RuntimeError("dev run completed without metrics")
            result = {
            "status": "COMPLETE",
            "epoch": max_epochs,
            "phase": "complete",
            "best_epoch": best_epoch,
            "selection_output": (
                "fused"
                if "fused" in contract["evaluation_outputs"]
                else contract["active_experts"][0]
            ),
            "metrics_percent": best_metrics,
            "last_metrics_percent": last_metrics,
            "dev_evaluation_count": dev_evaluation_count,
            "official_test_access_count": 0,
            "query_records": data.num_query,
            "gallery_records": len(data.eval_loader.dataset) - data.num_query,
            "train_records": len(data.train_loader.dataset),
            "best_checkpoint": str(output_dir / "best_dev_model.pth"),
            "best_checkpoint_sha256": _sha256(output_dir / "best_dev_model.pth"),
            "train_history": train_history,
            "resume_history": resume_history,
            "parameter_budget_pass": bool(built.provenance["parameter_budget_pass"]),
            "total_parameters": int(built.provenance["total_parameters"]),
            "variant_contract": contract,
            "variant_contract_sha256": variant_sha256(contract),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "fatal_or_nonfinite_detected": False,
            "model_constructed": True,
            "training_started": True,
            "second_gpu_gate": second_gpu_gate,
            "contract_testing": os.environ.get("TRIFUSION_CONTRACT_TESTING")
            == "1",
            "scientific_evidence_eligible": os.environ.get(
                "TRIFUSION_CONTRACT_TESTING"
            )
            != "1",
            "circ_training_evidence": circ_training_evidence,
            }
        _atomic_json(result_path, result)
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        _atomic_json(
            result_path,
            {
                "status": "OOM" if "out of memory" in message.lower() else "FAILED",
                "official_test_access_count": (
                    1 if final_mode and eval_loader_iteration_count else 0
                ),
                "error": message,
                "traceback": traceback.format_exc(),
                "second_gpu_gate": second_gpu_gate,
            },
        )
        return 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "capacity", "overfit", "dev", "final")
    )
    parser.add_argument("--variant", required=True, choices=variant_names())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--_worker",
        choices=("capacity", "overfit", "dev", "final"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args._worker:
        if args._worker == "capacity":
            return _worker_capacity(args.config.resolve(), args.variant, args.output_dir)
        if args._worker in ("dev", "final"):
            return _worker_dev(
                args.config.resolve(),
                args.variant,
                args.output_dir,
                data_mode=(
                    "development"
                    if args._worker == "dev"
                    else "postfreeze-final"
                ),
            )
        return _worker_capacity(
            args.config.resolve(), args.variant, args.output_dir, overfit=True
        )
    if args.mode is None:
        print("--mode is required", file=sys.stderr)
        return 2
    receipt = _preflight(args.config.resolve(), args.variant)
    if args.mode == "capacity":
        receipt, returncode = _capacity(
            args.config.resolve(), args.variant, args.output_dir
        )
        receipt_path = args.output_dir / "capacity.json"
    elif args.mode == "overfit":
        receipt, returncode = _overfit(
            args.config.resolve(), args.variant, args.output_dir
        )
        receipt_path = args.output_dir / "overfit.json"
    elif args.mode == "dev":
        receipt, returncode = _dev(args.config.resolve(), args.variant, args.output_dir)
        receipt_path = args.output_dir / "run_summary.json"
    elif args.mode == "final":
        receipt, returncode = _dev(
            args.config.resolve(),
            args.variant,
            args.output_dir,
            data_mode="postfreeze-final",
        )
        receipt_path = args.output_dir / "run_summary.json"
    elif args.mode != "preflight":
        receipt["mode"] = args.mode
        receipt["status"] = "FAILED"
        receipt["launch_allowed"] = False
        receipt["blockers"].append(f"{args.mode}_vertical_slice_not_implemented")
        receipt_path = args.output_dir / "run_summary.json"
        returncode = 2
    else:
        receipt_path = args.output_dir / "preflight.json"
        returncode = 0
    _atomic_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
