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
DEFAULT_CONFIG = PROJECT / "configs/RGBNT201/TriFusion.yml"
DEV_PROTOCOL = PROJECT / "protocols/rgbnt201_dev_v1.json"
DATASET_RECEIPT = PROJECT / "evidence/rgbnt201_audit_20260831.json"
EXPECTED = {
    "clip_sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "dev_protocol_sha256": "d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946",
    "dataset_receipt_sha256": "ec36309921a3dd7c12d46bb60a83406440ba316f171e419a67ad2cc83bf24318",
}


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


def _preflight(config_path: Path, variant: str) -> dict[str, Any]:
    blockers: list[str] = []
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("SCHEMA_VERSION") != 1:
        blockers.append("config_schema_drift")
    if variant != config.get("EXPERIMENT", {}).get("VARIANT"):
        blockers.append("variant_config_mismatch")
    clip = Path(config["MODEL"]["CLIP_CHECKPOINT"]).expanduser().resolve()
    immutable = {
        "clip": (clip, EXPECTED["clip_sha256"]),
        "dev_protocol": (DEV_PROTOCOL, EXPECTED["dev_protocol_sha256"]),
        "dataset_receipt": (DATASET_RECEIPT, EXPECTED["dataset_receipt_sha256"]),
    }
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
    protocol = json.loads(DEV_PROTOCOL.read_text(encoding="utf-8"))
    data_protocol = {
        "fit_identities": len(protocol["train_ids"]),
        "fit_records": int(protocol["counts"]["train_triplets"]),
        "dev_identities": len(protocol["dev_ids"]),
        "query_records": int(protocol["evaluation"]["query_triplets"]),
        "gallery_records": int(protocol["evaluation"]["gallery_triplets"]),
        "identity_overlap": len(set(protocol["train_ids"]) & set(protocol["dev_ids"])),
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
    if data_protocol != expected_data_protocol:
        blockers.append("development_protocol_drift")
    optimization = {
        "train_batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "eval_batch_size": int(config["DATA"]["EVAL_BATCH_SIZE"]),
        "num_workers": int(config["DATA"]["NUM_WORKERS"]),
        "gradient_accumulation": int(config["OPTIMIZATION"]["GRADIENT_ACCUMULATION"]),
        "amp": bool(config["OPTIMIZATION"]["AMP"]),
        "max_epochs": int(config["OPTIMIZATION"]["MAX_EPOCHS"]),
    }
    if optimization["train_batch_size"] != 16 or optimization["num_instances"] != 4:
        blockers.append("pk_batch_drift")
    if optimization["gradient_accumulation"] != 1:
        blockers.append("batch_hard_accumulation_forbidden")
    gpu = _gpu_state()
    if not gpu.get("query_passed"):
        blockers.append("gpu_query_failed")
    elif int(gpu["memory_used_mib"]) >= 500:
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
        "variant": variant,
        "status": "READY" if not blockers else "BLOCKED",
        "launch_allowed": not blockers,
        "required_memory_used_strictly_below_mib": 500,
        "blockers": blockers,
        "gpu": gpu,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "repository_head": head.stdout.strip() if head.returncode == 0 else None,
        "runner_sha256": _sha256(Path(__file__)),
        "file_checks": file_checks,
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
        "batch_size": 16,
        "num_instances": 4,
        "finite_losses": True,
        "finite_gradients": True,
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
        "batch_size": 16,
        "num_instances": 4,
        "finite_losses": True,
        "finite_gradients": True,
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
    return {
        "schema_version": "1.0",
        "run_type": "TriFusion/RGBNT201/train-only-dev",
        "variant": variant,
        "repository_head": preflight["repository_head"],
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": preflight["config_sha256"],
        "clip_sha256": EXPECTED["clip_sha256"],
        "dev_protocol_sha256": EXPECTED["dev_protocol_sha256"],
        "dataset_receipt_sha256": EXPECTED["dataset_receipt_sha256"],
        "data_protocol": preflight["data_protocol"],
        "optimization": preflight["optimization"],
        "official_test_access_during_development": False,
    }


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
    config_path: Path, variant: str, output_dir: Path
) -> tuple[dict[str, Any], int]:
    recovery = _validate_recovery(output_dir)
    preflight = _preflight(config_path, variant)
    preflight["mode"] = "dev"
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
            preflight["claim_boundary"] = "dev run blocked before model construction"
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
            preflight["claim_boundary"] = "dev resume blocked before model construction"
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
            "dev",
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
            "dev",
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
    worker_log = output_dir / "dev_worker.log"
    worker_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    result_path = output_dir / "dev_worker_result.json"
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
    required = {
        "status": "COMPLETE",
        "epoch": 60,
        "phase": "complete",
        "official_test_access_count": 0,
        "dev_evaluation_count": 60,
        "query_records": 825,
        "gallery_records": 825,
        "train_records": 3126,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in required.items()
        if result.get(key) != value
    }
    metrics = result.get("metrics_percent", {})
    if set(metrics) != {"fused", "cnn", "transformer", "mamba"}:
        mismatches["metrics_percent"] = {"expected": "four named outputs", "actual": sorted(metrics)}
    if not recovery_after["valid"] or recovery_after.get("phase") != "complete":
        mismatches["recovery"] = {"expected": "valid complete", "actual": recovery_after}
    preflight.update(result)
    preflight["worker_result_sha256"] = _sha256(result_path)
    preflight["recovery"] = recovery_after
    preflight["claim_scope"] = "train-only development result"
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
    second_gpu = _gpu_state()
    if not second_gpu.get("query_passed") or int(
        second_gpu.get("memory_used_mib", 500)
    ) >= 500:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "steps": 0,
                "official_test_access_count": 0,
                "dev_loader_iterations": 0,
                "second_gpu_gate": second_gpu,
            },
        )
        return 3
    try:
        import random

        import numpy as np
        import torch

        from modeling.trifusion.builder import build_trifusion_from_clip
        from modeling.trifusion.criterion import TriFusionCriterion
        from modeling.trifusion.data import build_rgbnt201_dev_loaders

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if variant != "core_pre_circ":
            raise ValueError(f"unsupported capacity variant: {variant}")
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
        built = build_trifusion_from_clip(
            config["MODEL"]["CLIP_CHECKPOINT"],
            num_classes=data.num_classes,
            image_size=tuple(config["MODEL"]["IMAGE_SIZE"]),
            patch_size=int(config["MODEL"]["PATCH_SIZE"]),
            cnn_width=int(config["MODEL"]["CNN_WIDTH"]),
            mamba_width=int(config["MODEL"]["MAMBA_WIDTH"]),
            relay_rank=int(config["MODEL"]["RELAY_RANK"]),
            embedding_width=int(config["MODEL"]["EMBEDDING_WIDTH"]),
            private_width=int(config["MODEL"]["PRIVATE_WIDTH"]),
        )
        model = built.model.cuda()
        for name, parameter in model.named_parameters():
            if "private_projection" in name:
                parameter.requires_grad_(False)
        pretrained_tokens = (
            "encoder.tokenizer.patch_projection",
            "encoder.tokenizer.positional_embedding",
            "encoder.experts.transformer.blocks",
            "encoder.experts.transformer.class_embedding",
            "encoder.experts.transformer.class_position",
            "encoder.experts.transformer.pre_norm",
            "encoder.experts.transformer.post_norm",
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
        criterion = TriFusionCriterion(
            target_cache=None,
            triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
        ).cuda()
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config["OPTIMIZATION"]["AMP"]))
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
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                finite = bool(torch.isfinite(parameter.grad).all().item())
                all_gradients_finite = all_gradients_finite and finite
                if finite:
                    finite_gradient_names.add(name)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        coverage = len(finite_gradient_names) / len(trainable_names)
        initial_loss = losses_by_step[0]["total"]
        final_loss = losses_by_step[-1]["total"]
        loss_ratio = final_loss / max(abs(initial_loss), 1e-12)
        gate_pass = all_losses_finite and all_gradients_finite and coverage == 1.0
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
                "gradient_parameter_coverage": coverage,
                "trainable_parameter_tensors": len(trainable_names),
                "finite_gradient_parameter_tensors": len(finite_gradient_names),
                "missing_gradient_parameters": sorted(trainable_names - finite_gradient_names),
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
                "second_gpu_gate": second_gpu,
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
                "second_gpu_gate": second_gpu,
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


def _worker_dev(config_path: Path, variant: str, output_dir: Path) -> int:
    result_path = output_dir / "dev_worker_result.json"
    second_gpu = _gpu_state()
    if not second_gpu.get("query_passed") or int(
        second_gpu.get("memory_used_mib", 500)
    ) >= 500:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "epoch": 0,
                "phase": "worker_gpu_recheck",
                "official_test_access_count": 0,
                "dev_evaluation_count": 0,
                "second_gpu_gate": second_gpu,
            },
        )
        return 3
    try:
        import math
        import random

        import numpy as np
        import torch
        import torch.nn.functional as functional

        from modeling.trifusion.builder import build_trifusion_from_clip
        from modeling.trifusion.criterion import TriFusionCriterion
        from modeling.trifusion.data import build_rgbnt201_dev_loaders
        from utils.reid_evaluation import evaluate_reid

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if variant != "core_pre_circ":
            raise ValueError(f"unsupported dev variant: {variant}")
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
        built = build_trifusion_from_clip(
            config["MODEL"]["CLIP_CHECKPOINT"],
            num_classes=data.num_classes,
            image_size=tuple(config["MODEL"]["IMAGE_SIZE"]),
            patch_size=int(config["MODEL"]["PATCH_SIZE"]),
            cnn_width=int(config["MODEL"]["CNN_WIDTH"]),
            mamba_width=int(config["MODEL"]["MAMBA_WIDTH"]),
            relay_rank=int(config["MODEL"]["RELAY_RANK"]),
            embedding_width=int(config["MODEL"]["EMBEDDING_WIDTH"]),
            private_width=int(config["MODEL"]["PRIVATE_WIDTH"]),
        )
        model = built.model.cuda()
        for name, parameter in model.named_parameters():
            if "private_projection" in name:
                parameter.requires_grad_(False)
        pretrained_tokens = (
            "encoder.tokenizer.patch_projection",
            "encoder.tokenizer.positional_embedding",
            "encoder.experts.transformer.blocks",
            "encoder.experts.transformer.class_embedding",
            "encoder.experts.transformer.class_position",
            "encoder.experts.transformer.pre_norm",
            "encoder.experts.transformer.post_norm",
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
        max_epochs = int(config["OPTIMIZATION"]["MAX_EPOCHS"])
        warmup_epochs = int(config["OPTIMIZATION"]["WARMUP_EPOCHS"])
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(1, warmup_epochs),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs - warmup_epochs),
            eta_min=float(config["OPTIMIZATION"]["PRETRAINED_LR"]) * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        criterion = TriFusionCriterion(
            target_cache=None,
            triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
        ).cuda()
        amp_enabled = bool(config["OPTIMIZATION"]["AMP"])
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        loss_weights = {
            "id_fused": float(config["LOSS"]["ID_FUSED"]),
            "triplet_fused": float(config["LOSS"]["TRIPLET_FUSED"]),
            "id_cnn": float(config["LOSS"]["ID_BRANCH"]),
            "id_transformer": float(config["LOSS"]["ID_BRANCH"]),
            "id_mamba": float(config["LOSS"]["ID_BRANCH"]),
            "triplet_cnn": float(config["LOSS"]["TRIPLET_BRANCH"]),
            "triplet_transformer": float(config["LOSS"]["TRIPLET_BRANCH"]),
            "triplet_mamba": float(config["LOSS"]["TRIPLET_BRANCH"]),
            "reliability": 0.0,
            "peer_logits": 0.0,
            "peer_role": 0.0,
            "private_diversity": 0.0,
        }
        identity_hash = _sha256(output_dir / "run_identity.json")
        latest_path = output_dir / ".resume/latest.json"
        current_epoch = 0
        phase = "epoch_boundary"
        best_epoch = 0
        best_map = float("-inf")
        best_metrics: dict[str, Any] | None = None
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
            return {
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
                "dev_evaluation_count": dev_evaluation_count,
                "run_identity_sha256": identity_hash,
                "train_history": train_history,
                "resume_history": resume_history,
            }

        def evaluate() -> dict[str, dict[str, float]]:
            model.eval()
            features: dict[str, list[Any]] = {
                "fused": [],
                "cnn": [],
                "transformer": [],
                "mamba": [],
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
                features["fused"].append(output.fused_embedding.detach().float().cpu())
                for expert in ("cnn", "transformer", "mamba"):
                    features[expert].append(output.branch_embeddings[expert].detach().float().cpu())
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

        torch.cuda.reset_peak_memory_stats()
        last_metrics = best_metrics
        while current_epoch < max_epochs or phase == "post_train":
            if phase in {"epoch_boundary", "post_eval"}:
                epoch = current_epoch + 1
                model.train()
                total_sum = 0.0
                sample_count = 0
                named_sums: dict[str, float] = {name: 0.0 for name in loss_weights}
                for images, labels, _camera_ids, _view_ids, _sample_keys in data.train_loader:
                    images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
                    labels = labels.cuda(non_blocking=False)
                    mask = torch.ones(labels.shape[0], 3, dtype=torch.bool, device="cuda")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        output = model(
                            {"images": images, "modality_mask": mask},
                            targets=labels,
                            return_aux=True,
                        )
                        named_losses = criterion(output, labels)
                        total_loss = sum(
                            named_losses[name] * weight
                            for name, weight in loss_weights.items()
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
                train_history[str(epoch)] = {
                    "total": total_sum / sample_count,
                    "named": {name: value / sample_count for name, value in named_sums.items()},
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
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
                fused_map = float(last_metrics["fused"]["mAP"])
                if fused_map > best_map:
                    best_map = fused_map
                    best_epoch = current_epoch
                    best_metrics = last_metrics
                    best_path = output_dir / "best_dev_model.pth"
                    _atomic_torch_save(best_path, model.state_dict(), torch)
                    _atomic_json(
                        output_dir / "best_dev_receipt.json",
                        {
                            "epoch": best_epoch,
                            "dev_fused_mAP": best_map,
                            "metrics_percent": best_metrics,
                            "checkpoint": str(best_path),
                            "checkpoint_sha256": _sha256(best_path),
                            "selection_split": "train_171 held-out dev identities",
                            "official_test_access_count": 0,
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

        if best_metrics is None or last_metrics is None:
            raise RuntimeError("dev run completed without metrics")
        result = {
            "status": "COMPLETE",
            "epoch": max_epochs,
            "phase": "complete",
            "best_epoch": best_epoch,
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
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "fatal_or_nonfinite_detected": False,
            "second_gpu_gate": second_gpu,
        }
        _atomic_json(result_path, result)
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        _atomic_json(
            result_path,
            {
                "status": "OOM" if "out of memory" in message.lower() else "FAILED",
                "official_test_access_count": 0,
                "error": message,
                "traceback": traceback.format_exc(),
                "second_gpu_gate": second_gpu,
            },
        )
        return 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "capacity", "overfit", "dev"))
    parser.add_argument("--variant", required=True, choices=("core_pre_circ",))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--_worker", choices=("capacity", "overfit", "dev"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args._worker:
        if args._worker == "capacity":
            return _worker_capacity(args.config.resolve(), args.variant, args.output_dir)
        if args._worker == "dev":
            return _worker_dev(args.config.resolve(), args.variant, args.output_dir)
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
