#!/usr/bin/env python3
"""Run the Signal-preserving TriFusion V5 development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import yaml


GATE_OUTPUTS = ("baseline_only", "cnn", "transformer", "mamba")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_V5 = "signal_preserving_collaborative_v5"
ARCHITECTURE_V6 = "signal_preserving_collaborative_v6"
ARCHITECTURE_V7 = "signal_preserving_collaborative_v7"
ARCHITECTURE_V8 = "signal_preserving_collaborative_v8_expert_formation"


def receipt_schema(architecture: str, stage: str) -> str:
    version = {
        ARCHITECTURE_V5: "v5",
        ARCHITECTURE_V6: "v6",
        ARCHITECTURE_V7: "v7",
        ARCHITECTURE_V8: "v8",
    }[architecture]
    return f"trifusion-signal-preserving-{version}-{stage}-v1"


def architecture_source_paths(architecture: str) -> dict[str, Path]:
    version = {
        ARCHITECTURE_V5: "v5",
        ARCHITECTURE_V6: "v6",
        ARCHITECTURE_V7: "v7",
        ARCHITECTURE_V8: "v8",
    }[architecture]
    return {
        "model": PROJECT_ROOT
        / "modeling"
        / "trifusion"
        / f"signal_preserving_{version}.py",
        "builder": PROJECT_ROOT
        / "modeling"
        / "trifusion"
        / f"signal_preserving_{version}_builder.py",
    }


def load_contract(path: Path) -> dict[str, Any]:
    """Load the frozen public experiment contract used by every runner mode."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "architecture": str(config["MODEL"]["ARCHITECTURE"]),
        "train_batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "max_epochs": int(config["OPTIMIZATION"]["MAX_EPOCHS"]),
        "signal_checkpoint_sha256": str(config["SIGNAL"]["CHECKPOINT_SHA256"]),
        "baseline_width": int(config["SIGNAL"]["BASELINE_WIDTH"]),
        "retrieval_outputs": tuple(config["MODEL"]["RETRIEVAL_OUTPUTS"]),
        "dev_min_map": float(config["GATES"]["DEV_MIN_MAP"]),
        "official_test_during_development": bool(
            config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"]
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    config = load_raw_config(path)
    contract = load_contract(path)
    if contract["seed"] != 42:
        raise ValueError("V5 main experiment is frozen to seed 42")
    if contract["official_test_during_development"] is not False:
        raise ValueError("V5 development must not access the official test")
    return config


def weighted_training_loss(
    losses: dict[str, Any], config: dict[str, Any], *, phase: str = "joint"
) -> Any:
    weights = config["LOSS"]
    architecture = config["MODEL"]["ARCHITECTURE"]
    if architecture == ARCHITECTURE_V7 and phase == "router_warmup":
        return (
            losses["peer_logits"] * float(weights["PEER_LOGITS"])
            + losses["alpha"] * float(weights["ALPHA"])
            + losses["reliability"] * float(weights["RELIABILITY"])
        )
    branch_id = sum(losses[f"id_{expert}"] for expert in ("cnn", "transformer", "mamba"))
    branch_triplet = sum(
        losses[f"triplet_{expert}"] for expert in ("cnn", "transformer", "mamba")
    )
    total = (
        losses["id_fused"] * float(weights["ID_FUSED"])
        + losses["triplet_fused"] * float(weights["TRIPLET_FUSED"])
        + branch_id * float(weights["ID_BRANCH"])
        + branch_triplet * float(weights["TRIPLET_BRANCH"])
    )
    if architecture != ARCHITECTURE_V8:
        total = total + losses["peer_logits"] * float(weights["PEER_LOGITS"])
    if architecture in (ARCHITECTURE_V6, ARCHITECTURE_V7, ARCHITECTURE_V8):
        residual_id = sum(
            losses[f"id_residual_{expert}"]
            for expert in ("cnn", "transformer", "mamba")
        )
        residual_triplet = sum(
            losses[f"triplet_residual_{expert}"]
            for expert in ("cnn", "transformer", "mamba")
        )
        total = (
            total
            + residual_id * float(weights["ID_RESIDUAL"])
            + residual_triplet * float(weights["TRIPLET_RESIDUAL"])
        )
    if architecture == ARCHITECTURE_V7:
        total = (
            total
            + losses["alpha"] * float(weights["ALPHA"])
            + losses["reliability"] * float(weights["RELIABILITY"])
        )
    return total


def apply_controlled_modality_degradation(
    images: dict[str, Any],
    *,
    selected_modalities: Any,
    selected_samples: Any,
    degraded_quality: float,
) -> tuple[dict[str, Any], Any]:
    """Blur one selected modality and return its explicit quality target."""

    import torch
    import torch.nn.functional as F

    modalities = ("RGB", "NI", "TI")
    batch_size = next(iter(images.values())).shape[0]
    quality = torch.ones(
        batch_size,
        len(modalities),
        dtype=next(iter(images.values())).dtype,
        device=selected_modalities.device,
    )
    degraded = {name: tensor.clone() for name, tensor in images.items()}
    for modality_index, modality in enumerate(modalities):
        rows = selected_samples & (selected_modalities == modality_index)
        if bool(rows.any()):
            degraded[modality][rows] = F.avg_pool2d(
                degraded[modality][rows], kernel_size=5, stride=1, padding=2
            )
            quality[rows, modality_index] = float(degraded_quality)
    return degraded, quality


def quality_response_gate(
    clean_probabilities: Any,
    corrupted_probabilities: dict[str, Any],
) -> dict[str, Any]:
    """Require each modality's mean mass to fall under its own corruption."""

    modalities = ("RGB", "NI", "TI")
    results = {}
    for index, modality in enumerate(modalities):
        clean_mass = float(clean_probabilities[:, index].mean())
        corrupted_mass = float(corrupted_probabilities[modality][:, index].mean())
        results[modality] = {
            "clean_mean_mass": clean_mass,
            "corrupted_mean_mass": corrupted_mass,
            "decrease": clean_mass - corrupted_mass,
            "decreased": corrupted_mass < clean_mass,
        }
    return {
        "passed": all(result["decreased"] for result in results.values()),
        "modalities": results,
    }


def load_v7_initialization(
    model: Any,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    """Load V6 expert state while leaving only V7's new alpha gate initialized."""

    import torch

    checkpoint = Path(initialization["V6_CHECKPOINT"]).resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != str(initialization["V6_CHECKPOINT_SHA256"]):
        raise ValueError("V6 initialization checkpoint SHA-256 differs from contract")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    expected_missing = [
        "fusion.alpha_predictor.0.bias",
        "fusion.alpha_predictor.0.weight",
        "fusion.alpha_predictor.2.bias",
        "fusion.alpha_predictor.2.weight",
    ]
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            f"V6-to-V7 initialization mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def set_v7_training_phase(
    model: Any,
    *,
    phase: str,
    joint_trainable_names: set[str],
) -> set[str]:
    """Select the frozen-expert router warmup or restore the joint V7 phase."""

    if phase == "router_warmup":
        selected = {
            name
            for name in joint_trainable_names
            if name.startswith("encoder.reliability_gate.")
            or name.startswith("fusion.alpha_predictor.")
        }
    elif phase == "joint":
        selected = set(joint_trainable_names)
    else:
        raise ValueError(f"unsupported V7 training phase: {phase}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    return selected


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _build_runtime(config: dict[str, Any]) -> dict[str, Any]:
    from run_signal_baseline_dev import _build_loaders, _configure_signal_source

    signal_config = config["SIGNAL"]
    data_config = config["DATA"]
    model_config = config["MODEL"]
    architecture = str(model_config["ARCHITECTURE"])
    seed = int(config["EXPERIMENT"]["SEED"])
    source = Path(signal_config["SOURCE"]).resolve()
    checkpoint = Path(signal_config["CHECKPOINT"]).resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != str(signal_config["CHECKPOINT_SHA256"]):
        raise ValueError("Signal checkpoint SHA-256 differs from the frozen contract")

    source_commit = _configure_signal_source(source)
    from config import cfg as signal_cfg
    from modeling import make_frame

    _set_seed(seed)
    signal_cfg.merge_from_file(str(Path(signal_config["CONFIG"]).resolve()))
    signal_cfg.defrost()
    signal_cfg.MODEL.PRETRAIN_PATH_T = str(Path(signal_config["CLIP_WEIGHT"]).resolve())
    signal_cfg.SOLVER.SEED = seed
    signal_cfg.SOLVER.IMS_PER_BATCH = int(data_config["TRAIN_BATCH_SIZE"])
    signal_cfg.DATALOADER.NUM_INSTANCE = int(data_config["NUM_INSTANCES"])
    signal_cfg.DATALOADER.NUM_WORKERS = int(data_config["NUM_WORKERS"])
    signal_cfg.TEST.IMS_PER_BATCH = int(data_config["EVAL_BATCH_SIZE"])
    signal_cfg.freeze()

    train_loader, eval_loader, train_records, dev_records = _build_loaders(
        dataset_root=Path(data_config["DATASET_ROOT"]).resolve(),
        protocol_path=(PROJECT_ROOT / data_config["DEV_PROTOCOL"]).resolve(),
        batch_size=int(data_config["TRAIN_BATCH_SIZE"]),
        num_instances=int(data_config["NUM_INSTANCES"]),
        eval_batch_size=int(data_config["EVAL_BATCH_SIZE"]),
        num_workers=int(data_config["NUM_WORKERS"]),
        seed=seed,
    )
    signal_model = make_frame(
        signal_cfg,
        num_class=len({record[1] for record in train_records}),
        camera_num=len({record[2] for record in train_records}),
        view_num=len({record[3] for record in train_records}),
    )

    import torch

    signal_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    signal_model.load_state_dict(signal_state, strict=True)
    signal_model.cuda()

    project_modeling = str(PROJECT_ROOT / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    if architecture in (ARCHITECTURE_V7, ARCHITECTURE_V8):
        from trifusion.aligned_data import build_aligned_train_loader

        train_loader = build_aligned_train_loader(
            train_records,
            batch_size=int(data_config["TRAIN_BATCH_SIZE"]),
            num_instances=int(data_config["NUM_INSTANCES"]),
            num_workers=int(data_config["NUM_WORKERS"]),
            seed=seed,
        )
    initialization = None
    common_build_arguments = {
        "signal_checkpoint_sha256": checkpoint_sha256,
        "num_classes": len({record[1] for record in train_records}),
        "feature_width": int(model_config["FEATURE_WIDTH"]),
        "grid_size": tuple(model_config["GRID_SIZE"]),
        "adapter_width": int(model_config["ADAPTER_WIDTH"]),
    }
    if architecture == ARCHITECTURE_V5:
        from trifusion.signal_preserving_v5_builder import (
            build_signal_preserving_trifusion_v5,
        )

        build = build_signal_preserving_trifusion_v5(
            signal_model,
            residual_scale_init=float(model_config["RESIDUAL_SCALE_INIT"]),
            residual_width=int(model_config["RESIDUAL_WIDTH"]),
            relay_rank=int(model_config["RELAY_RANK"]),
            private_width=int(model_config["PRIVATE_WIDTH"]),
            reliability_hidden_width=int(model_config["RELIABILITY_HIDDEN_WIDTH"]),
            **common_build_arguments,
        )
    elif architecture == ARCHITECTURE_V6:
        from trifusion.signal_preserving_v6_builder import (
            build_signal_preserving_trifusion_v6,
        )

        build = build_signal_preserving_trifusion_v6(
            signal_model,
            residual_width=int(model_config["RESIDUAL_WIDTH"]),
            relay_rank=int(model_config["RELAY_RANK"]),
            private_width=int(model_config["PRIVATE_WIDTH"]),
            reliability_hidden_width=int(model_config["RELIABILITY_HIDDEN_WIDTH"]),
            **common_build_arguments,
        )
    elif architecture == ARCHITECTURE_V7:
        from trifusion.signal_preserving_v7_builder import (
            build_signal_preserving_trifusion_v7,
        )

        build = build_signal_preserving_trifusion_v7(
            signal_model,
            alpha_max=float(model_config["ALPHA_MAX"]),
            alpha_init=float(model_config["ALPHA_INIT"]),
            residual_width=int(model_config["RESIDUAL_WIDTH"]),
            relay_rank=int(model_config["RELAY_RANK"]),
            private_width=int(model_config["PRIVATE_WIDTH"]),
            reliability_hidden_width=int(model_config["RELIABILITY_HIDDEN_WIDTH"]),
            **common_build_arguments,
        )
        initialization = load_v7_initialization(build.model, config["INITIALIZATION"])
    elif architecture == ARCHITECTURE_V8:
        from trifusion.signal_preserving_v8_builder import (
            build_signal_preserving_trifusion_v8_expert_formation,
        )

        build = build_signal_preserving_trifusion_v8_expert_formation(
            signal_model,
            semantic_width=int(model_config["SEMANTIC_WIDTH"]),
            branch_after_block=int(model_config["BRANCH_AFTER_BLOCK"]),
            expert_modal_width=int(model_config["EXPERT_MODAL_WIDTH"]),
            scale_init=float(model_config["SCALE_INIT"]),
            gradient_checkpointing=bool(model_config["GRADIENT_CHECKPOINTING"]),
            **common_build_arguments,
        )
    else:
        raise ValueError(f"unsupported Signal-preserving architecture: {architecture}")
    build.model.cuda()
    build_provenance = dict(build.provenance)
    if initialization is not None:
        build_provenance["initialization"] = initialization
    return {
        "model": build.model,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_records": train_records,
        "dev_records": dev_records,
        "signal_stage": signal_cfg.MODEL.stageName,
        "signal_source_commit": source_commit,
        "signal_source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
        ).hexdigest(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "build_provenance": build_provenance,
    }


def _evaluate_baseline_parity(
    model: Any,
    loader: Any,
    *,
    num_query: int,
    signal_stage: str,
) -> tuple[dict[str, float], int, bool]:
    import numpy as np
    import torch
    from utils.metrics import R1_mAP_eval

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")
    feature_width = 0
    exact_reference_match = True
    model.eval()
    for images, identities, camera_ids, camera_ids_batch, view_ids, paths in loader:
        del view_ids
        images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
        camera_ids_cuda = camera_ids_batch.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_ids_cuda.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_ids_cuda,
        }
        with torch.no_grad():
            reference = model.baseline.signal(
                images,
                cam_label=camera_ids_cuda,
                view_label=None,
                training=False,
                sge=signal_stage,
            )
            baseline = model(batch, retrieval_output="baseline_only")
        exact_reference_match = exact_reference_match and torch.equal(
            reference, baseline
        )
        feature_width = int(baseline.shape[1])
        evaluator.update((baseline, identities, camera_ids, paths))
    cmc, mean_ap, *_ = evaluator.compute()
    metrics = {
        "mAP": float(mean_ap * 100.0),
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[4] * 100.0),
        "Rank-10": float(cmc[9] * 100.0),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite V5 baseline parity metric")
    return metrics, feature_width, exact_reference_match


def _run_preflight(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.time()
    runtime = _build_runtime(config)
    metrics, feature_width, exact_reference_match = _evaluate_baseline_parity(
        runtime["model"],
        runtime["eval_loader"],
        num_query=len(runtime["dev_records"]),
        signal_stage=runtime["signal_stage"],
    )
    expected_metrics = {
        key: float(value)
        for key, value in config["SIGNAL"]["EXPECTED_DEV_METRICS"].items()
    }
    metric_parity = {
        key: metrics[key] == expected_metrics[key] for key in expected_metrics
    }
    if feature_width != int(config["SIGNAL"]["BASELINE_WIDTH"]):
        raise ValueError("V5 baseline width differs from the Signal contract")
    if not exact_reference_match or not all(metric_parity.values()):
        raise ValueError("V5 baseline-only output failed exact Signal parity")
    result = {
        "schema_version": receipt_schema(config["MODEL"]["ARCHITECTURE"], "preflight"),
        "status": "PASS",
        "mode": "preflight",
        "signal_source_commit": runtime["signal_source_commit"],
        "signal_source_diff_sha256": runtime["signal_source_diff_sha256"],
        "signal_checkpoint": str(runtime["checkpoint"]),
        "signal_checkpoint_sha256": runtime["checkpoint_sha256"],
        "baseline_metrics_percent": metrics,
        "expected_baseline_metrics_percent": expected_metrics,
        "metric_parity": metric_parity,
        "exact_reference_feature_match": exact_reference_match,
        "baseline_feature_width": feature_width,
        "fit_triplets": len(runtime["train_records"]),
        "dev_query_triplets": len(runtime["dev_records"]),
        "dev_gallery_triplets": len(runtime["dev_records"]),
        "build_provenance": runtime["build_provenance"],
        "official_test_access_count": 0,
        "training_started": False,
        "optimizer_steps": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _criterion_class(config: dict[str, Any]) -> Any:
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    if architecture == ARCHITECTURE_V5:
        from trifusion.signal_preserving_v5 import SignalPreservingV5Criterion

        return SignalPreservingV5Criterion
    if architecture == ARCHITECTURE_V6:
        from trifusion.signal_preserving_v6 import ComplementarityActivatedV6Criterion

        return ComplementarityActivatedV6Criterion
    raise ValueError(f"unsupported Signal-preserving architecture: {architecture}")


def _build_criterion(config: dict[str, Any]) -> Any:
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    if architecture == ARCHITECTURE_V8:
        from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion

        return ExpertFormationV8Criterion(
            triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
            label_smoothing=float(config["LOSS"]["LABEL_SMOOTHING"]),
        ).cuda()
    if architecture == ARCHITECTURE_V7:
        from trifusion.signal_preserving_v7 import MarginalGainV7Criterion

        return MarginalGainV7Criterion(
            triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
            label_smoothing=float(config["LOSS"]["LABEL_SMOOTHING"]),
            utility_temperature=float(config["LOSS"]["UTILITY_TEMPERATURE"]),
            alpha_gain_scale=float(config["LOSS"]["ALPHA_GAIN_SCALE"]),
        ).cuda()
    return _criterion_class(config)(
        target_cache=None,
        triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
    ).cuda()


def _training_batch(raw_batch: Any) -> tuple[dict[str, Any], Any]:
    import torch

    images, labels, camera_ids, _view_ids, _paths = raw_batch
    images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
    labels = labels.cuda(non_blocking=True)
    camera_ids = camera_ids.cuda(non_blocking=True)
    return (
        {
            "images": images,
            "modality_mask": torch.ones(
                labels.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_ids,
        },
        labels,
    )


def _training_views(
    raw_batch: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, Any]:
    import torch

    clean_batch, labels = _training_batch(raw_batch)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V7:
        return clean_batch, None, labels
    batch_size = labels.shape[0]
    selected_samples = torch.rand(batch_size, device="cuda") < float(
        config["QUALITY"]["DEGRADATION_PROBABILITY"]
    )
    selected_modalities = torch.randint(3, (batch_size,), device="cuda")
    degraded_images, quality = apply_controlled_modality_degradation(
        clean_batch["images"],
        selected_modalities=selected_modalities,
        selected_samples=selected_samples,
        degraded_quality=float(config["QUALITY"]["DEGRADED_QUALITY"]),
    )
    quality_batch = {
        "images": degraded_images,
        "modality_mask": clean_batch["modality_mask"],
        "camera_ids": clean_batch["camera_ids"],
        "modality_quality": quality,
    }
    return clean_batch, quality_batch, labels


def _criterion_losses(
    criterion: Any,
    output: Any,
    labels: Any,
    *,
    architecture: str,
    quality_output: Any | None,
) -> dict[str, Any]:
    if architecture == ARCHITECTURE_V7:
        return criterion(output, labels, quality_output=quality_output)
    return criterion(output, labels)


def _evaluate_v7_quality_response(
    model: Any,
    raw_batch: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    clean_batch, _labels = _training_batch(raw_batch)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["OPTIMIZATION"]["AMP"]),
    ):
        clean = model(clean_batch, return_aux=True).modal_probabilities.detach()
        corrupted = {}
        selected_samples = torch.ones(clean.shape[0], dtype=torch.bool, device="cuda")
        for index, modality in enumerate(("RGB", "NI", "TI")):
            selected_modalities = torch.full(
                (clean.shape[0],), index, dtype=torch.long, device="cuda"
            )
            degraded_images, quality = apply_controlled_modality_degradation(
                clean_batch["images"],
                selected_modalities=selected_modalities,
                selected_samples=selected_samples,
                degraded_quality=float(config["QUALITY"]["DEGRADED_QUALITY"]),
            )
            degraded_batch = {
                "images": degraded_images,
                "modality_mask": clean_batch["modality_mask"],
                "camera_ids": clean_batch["camera_ids"],
                "modality_quality": quality,
            }
            corrupted[modality] = model(
                degraded_batch, return_aux=True
            ).modal_probabilities.detach()
    return quality_response_gate(clean, corrupted)


def _run_capacity(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    started = time.time()
    runtime = _build_runtime(config)
    model = runtime["model"]
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    criterion = _build_criterion(config)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"])
    )
    baseline_before = _module_state_sha256(model.baseline.signal)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    gradient_names: set[str] = set()
    losses_by_step: list[float] = []
    overflow_events = 0
    steps = int(config["GATES"]["CAPACITY_STEPS"])
    iterator = iter(runtime["train_loader"])
    for _step in range(steps):
        batch, quality_batch, labels = _training_views(next(iterator), config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config["OPTIMIZATION"]["AMP"]),
        ):
            output = model(batch, return_aux=True)
            quality_output = (
                model(quality_batch, return_aux=True)
                if architecture == ARCHITECTURE_V7
                else None
            )
            if not output.diagnostics["all_finite"]:
                raise FloatingPointError("V5 forward emitted a nonfinite tensor")
            loss_parts = _criterion_losses(
                criterion,
                output,
                labels,
                architecture=architecture,
                quality_output=quality_output,
            )
            total_loss = weighted_training_loss(loss_parts, config)
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("V5 capacity loss is nonfinite")
        scale_before = scaler.get_scale()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in trainable.items():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"nonfinite gradient: {name}")
                gradient_names.add(name)
        scaler.step(optimizer)
        scaler.update()
        overflow_events += int(scaler.get_scale() < scale_before)
        losses_by_step.append(float(total_loss.detach()))

    baseline_after = _module_state_sha256(model.baseline.signal)
    missing_gradients = sorted(set(trainable) - gradient_names)
    passed = (
        not missing_gradients
        and overflow_events == 0
        and baseline_before == baseline_after
    )
    result = {
        "schema_version": receipt_schema(config["MODEL"]["ARCHITECTURE"], "capacity"),
        "status": "PASS" if passed else "FAIL",
        "mode": "capacity",
        "steps": steps,
        "batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "losses": losses_by_step,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "trainable_gradient_tensors": len(gradient_names),
        "trainable_tensors": len(trainable),
        "missing_gradient_tensors": missing_gradients,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "signal_state_sha256_before": baseline_before,
        "signal_state_sha256_after": baseline_after,
        "signal_state_unchanged": baseline_before == baseline_after,
        "signal_checkpoint_sha256": runtime["checkpoint_sha256"],
        "build_provenance": runtime["build_provenance"],
        "official_test_access_count": 0,
        "optimizer_steps": steps,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V5 capacity gate failed; inspect capacity.json")
    return result


def _run_overfit(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    started = time.time()
    runtime = _build_runtime(config)
    model = runtime["model"]
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    criterion = _build_criterion(config)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"])
    )
    fixed_batch, fixed_quality_batch, fixed_labels = _training_views(
        next(iter(runtime["train_loader"])), config
    )
    baseline_before = _module_state_sha256(model.baseline.signal)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    gradient_names: set[str] = set()
    losses_by_step: list[float] = []
    overflow_events = 0
    steps = int(config["GATES"]["OVERFIT_STEPS"])
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config["OPTIMIZATION"]["AMP"]),
        ):
            output = model(fixed_batch, return_aux=True)
            quality_output = (
                model(fixed_quality_batch, return_aux=True)
                if architecture == ARCHITECTURE_V7
                else None
            )
            if not output.diagnostics["all_finite"]:
                raise FloatingPointError("V5 forward emitted a nonfinite tensor")
            loss_parts = _criterion_losses(
                criterion,
                output,
                fixed_labels,
                architecture=architecture,
                quality_output=quality_output,
            )
            total_loss = weighted_training_loss(loss_parts, config)
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("V5 overfit loss is nonfinite")
        scale_before = scaler.get_scale()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in trainable.items():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"nonfinite gradient: {name}")
                gradient_names.add(name)
        scaler.step(optimizer)
        scaler.update()
        overflow_events += int(scaler.get_scale() < scale_before)
        losses_by_step.append(float(total_loss.detach()))

    baseline_after = _module_state_sha256(model.baseline.signal)
    missing_gradients = sorted(set(trainable) - gradient_names)
    gate = evaluate_overfit_gate(
        losses_by_step,
        max_ratio=float(config["GATES"]["OVERFIT_MAX_LOSS_RATIO"]),
        minimum_loss=overfit_loss_floor(config, num_classes=model.num_classes),
    )
    passed = (
        gate["passed"]
        and not missing_gradients
        and overflow_events == 0
        and baseline_before == baseline_after
    )
    result = {
        "schema_version": receipt_schema(config["MODEL"]["ARCHITECTURE"], "overfit"),
        "status": "PASS" if passed else "FAIL",
        "mode": "overfit",
        "steps": steps,
        "batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "losses": losses_by_step,
        "overfit_gate": gate,
        "trainable_gradient_tensors": len(gradient_names),
        "trainable_tensors": len(trainable),
        "missing_gradient_tensors": missing_gradients,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "signal_state_sha256_before": baseline_before,
        "signal_state_sha256_after": baseline_after,
        "signal_state_unchanged": baseline_before == baseline_after,
        "signal_checkpoint_sha256": runtime["checkpoint_sha256"],
        "build_provenance": runtime["build_provenance"],
        "official_test_access_count": 0,
        "optimizer_steps": steps,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "overfit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V5 overfit gate failed; inspect overfit.json")
    return result


def _evaluate_outputs(
    model: Any,
    loader: Any,
    *,
    num_query: int,
    retrieval_outputs: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    import numpy as np
    import torch
    from utils.metrics import R1_mAP_eval

    evaluators = {
        name: R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")
        for name in retrieval_outputs
    }
    feature_widths = {name: 0 for name in retrieval_outputs}
    model.eval()
    for images, identities, camera_ids, camera_ids_batch, _view_ids, paths in loader:
        images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
        camera_ids_cuda = camera_ids_batch.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_ids_cuda.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_ids_cuda,
        }
        with torch.no_grad():
            output = model(batch, return_aux=True)
        if not output.diagnostics["all_finite"]:
            raise FloatingPointError("V5 evaluation emitted a nonfinite tensor")
        if not output.diagnostics["baseline_exact_prefix"]:
            raise RuntimeError("V5 fused output lost the exact Signal prefix")
        features = {
            "baseline_only": output.baseline_embedding,
            "fused": output.fused_embedding,
            **dict(output.branch_embeddings),
        }
        for name in retrieval_outputs:
            feature_widths[name] = int(features[name].shape[1])
            evaluators[name].update((features[name], identities, camera_ids, paths))

    metrics: dict[str, dict[str, float]] = {}
    for name in retrieval_outputs:
        cmc, mean_ap, *_ = evaluators[name].compute()
        metrics[name] = {
            "mAP": float(mean_ap * 100.0),
            "Rank-1": float(cmc[0] * 100.0),
            "Rank-5": float(cmc[4] * 100.0),
            "Rank-10": float(cmc[9] * 100.0),
        }
        if not all(np.isfinite(value) for value in metrics[name].values()):
            raise FloatingPointError(f"nonfinite V5 retrieval metric: {name}")
    return metrics, feature_widths


def _metric_parity(
    actual: dict[str, dict[str, float]],
    expected: dict[str, dict[str, float]],
) -> dict[str, dict[str, bool]]:
    return {
        output: {
            metric: actual[output][metric] == expected[output][metric]
            for metric in expected[output]
        }
        for output in expected
    }


def _run_dev(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    started = time.time()
    runtime = _build_runtime(config)
    model = runtime["model"]
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    criterion = _build_criterion(config)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    base_lr = float(config["OPTIMIZATION"]["NEW_MODULE_LR"])
    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=base_lr,
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"])
    )
    max_epochs = int(config["OPTIMIZATION"]["MAX_EPOCHS"])
    warmup_epochs = int(config["OPTIMIZATION"]["WARMUP_EPOCHS"])
    retrieval_outputs = tuple(config["MODEL"]["RETRIEVAL_OUTPUTS"])
    expected_baseline = {
        key: float(value)
        for key, value in config["SIGNAL"]["EXPECTED_DEV_METRICS"].items()
    }
    source_paths = architecture_source_paths(architecture)
    identity = {
        "schema_version": receipt_schema(
            config["MODEL"]["ARCHITECTURE"], "dev-identity"
        ),
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "model_sha256": _sha256(source_paths["model"]),
        "builder_sha256": _sha256(source_paths["builder"]),
        "signal_source_commit": runtime["signal_source_commit"],
        "signal_source_diff_sha256": runtime["signal_source_diff_sha256"],
        "signal_checkpoint_sha256": runtime["checkpoint_sha256"],
        "fit_triplets": len(runtime["train_records"]),
        "dev_query_triplets": len(runtime["dev_records"]),
        "dev_gallery_triplets": len(runtime["dev_records"]),
        "retrieval_outputs": list(retrieval_outputs),
        "official_test_access_count": 0,
    }
    (output_dir / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_before = _module_state_sha256(model.baseline.signal)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_fused_map = float("-inf")
    best_metrics: dict[str, dict[str, float]] | None = None
    best_checkpoint = output_dir / "best_model.pth"
    optimizer_steps = 0
    overflow_events = 0
    torch.cuda.reset_peak_memory_stats()
    joint_trainable_names = set(trainable)

    for epoch in range(1, max_epochs + 1):
        multiplier = learning_rate_multiplier(
            epoch,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
        )
        learning_rate = base_lr * multiplier
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        phase = "joint"
        if architecture == ARCHITECTURE_V7:
            phase = (
                "router_warmup"
                if epoch <= int(config["OPTIMIZATION"]["ROUTER_WARMUP_EPOCHS"])
                else "joint"
            )
            set_v7_training_phase(
                model,
                phase=phase,
                joint_trainable_names=joint_trainable_names,
            )
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        quality_gate_raw_batch = None
        for raw_batch in runtime["train_loader"]:
            if (
                architecture == ARCHITECTURE_V7
                and epoch == int(config["OPTIMIZATION"]["ROUTER_WARMUP_EPOCHS"])
                and quality_gate_raw_batch is None
            ):
                quality_gate_raw_batch = raw_batch
            batch, quality_batch, labels = _training_views(raw_batch, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(config["OPTIMIZATION"]["AMP"]),
            ):
                output = model(batch, return_aux=True)
                quality_output = (
                    model(quality_batch, return_aux=True)
                    if architecture == ARCHITECTURE_V7
                    else None
                )
                if not output.diagnostics["all_finite"]:
                    raise FloatingPointError("V5 training emitted a nonfinite tensor")
                loss_parts = _criterion_losses(
                    criterion,
                    output,
                    labels,
                    architecture=architecture,
                    quality_output=quality_output,
                )
                total_loss = weighted_training_loss(loss_parts, config, phase=phase)
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("V5 dev loss is nonfinite")
            scale_before = scaler.get_scale()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in trainable.items():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(f"nonfinite gradient: {name}")
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            overflow_events += int(overflow)
            optimizer_steps += int(not overflow)
            epoch_loss += float(total_loss.detach())
            epoch_steps += 1

        quality_gate_result = None
        if (
            architecture == ARCHITECTURE_V7
            and epoch == int(config["OPTIMIZATION"]["ROUTER_WARMUP_EPOCHS"])
        ):
            quality_gate_result = _evaluate_v7_quality_response(
                model, quality_gate_raw_batch, config
            )
        metrics, feature_widths = _evaluate_outputs(
            model,
            runtime["eval_loader"],
            num_query=len(runtime["dev_records"]),
            retrieval_outputs=retrieval_outputs,
        )
        if metrics["baseline_only"] != expected_baseline:
            raise RuntimeError("frozen Signal baseline metrics changed during V5 training")
        epoch_result = {
            "epoch": epoch,
            "phase": phase,
            "learning_rate": learning_rate,
            "mean_training_loss": epoch_loss / epoch_steps,
            "metrics_percent": metrics,
            "feature_widths": feature_widths,
        }
        if quality_gate_result is not None:
            epoch_result["quality_response_gate"] = quality_gate_result
        history.append(epoch_result)
        if metrics["fused"]["mAP"] > best_fused_map:
            best_epoch = epoch
            best_fused_map = metrics["fused"]["mAP"]
            best_metrics = metrics
            torch.save(
                {
                    "schema_version": receipt_schema(
                        config["MODEL"]["ARCHITECTURE"], "checkpoint"
                    ),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics_percent": metrics,
                    "run_identity_sha256": _sha256(output_dir / "run_identity.json"),
                },
                best_checkpoint,
            )
        (output_dir / "history.json").write_text(
            json.dumps(
                {
                    "schema_version": receipt_schema(
                        config["MODEL"]["ARCHITECTURE"], "history"
                    ),
                    "epochs": history,
                    "best_epoch": best_epoch,
                    "best_fused_mAP": best_fused_map,
                    "official_test_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(epoch_result, sort_keys=True), flush=True)
        if quality_gate_result is not None and not quality_gate_result["passed"]:
            raise RuntimeError(
                "V7 quality-response gate failed after router warmup; joint phase blocked"
            )

    if best_metrics is None:
        raise RuntimeError("V5 dev training did not select a checkpoint")
    baseline_after_training = _module_state_sha256(model.baseline.signal)
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda()
    final_metrics, final_feature_widths = _evaluate_outputs(
        model,
        runtime["eval_loader"],
        num_query=len(runtime["dev_records"]),
        retrieval_outputs=retrieval_outputs,
    )
    reload_parity = _metric_parity(final_metrics, best_metrics)
    baseline_after_reload = _module_state_sha256(model.baseline.signal)
    dev_gate = evaluate_dev_gate(
        final_metrics,
        min_map=float(config["GATES"]["DEV_MIN_MAP"]),
    )
    completed = (
        all(all(values.values()) for values in reload_parity.values())
        and final_metrics["baseline_only"] == expected_baseline
        and baseline_before == baseline_after_training == baseline_after_reload
        and overflow_events == 0
    )
    result = {
        "schema_version": receipt_schema(
            config["MODEL"]["ARCHITECTURE"], "dev-result"
        ),
        "status": "PASS" if completed else "FAIL",
        "mode": "dev",
        "epochs_completed": max_epochs,
        "best_epoch": best_epoch,
        "selection_metric": "fused_dev_mAP",
        "selected_metrics_percent": best_metrics,
        "final_reloaded_metrics_percent": final_metrics,
        "reload_metric_parity": reload_parity,
        "feature_widths": final_feature_widths,
        "dev_gate": dev_gate,
        "claim_supported": bool(dev_gate["passed"]),
        "checkpoint": str(best_checkpoint),
        "checkpoint_sha256": _sha256(best_checkpoint),
        "signal_checkpoint_sha256": runtime["checkpoint_sha256"],
        "signal_state_sha256_before": baseline_before,
        "signal_state_sha256_after_training": baseline_after_training,
        "signal_state_sha256_after_reload": baseline_after_reload,
        "signal_state_unchanged": (
            baseline_before == baseline_after_training == baseline_after_reload
        ),
        "build_provenance": runtime["build_provenance"],
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "fit_triplets": len(runtime["train_records"]),
        "dev_query_triplets": len(runtime["dev_records"]),
        "dev_gallery_triplets": len(runtime["dev_records"]),
        "official_test_access_count": 0,
        "run_identity_sha256": _sha256(output_dir / "run_identity.json"),
        "history_sha256": _sha256(output_dir / "history.json"),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not completed:
        raise RuntimeError("V5 dev completion gate failed; inspect run_summary.json")
    return result


def evaluate_dev_gate(
    metrics: dict[str, dict[str, float]], *, min_map: float
) -> dict[str, Any]:
    """Judge the frozen same-checkpoint fused promotion contract."""

    fused_map = float(metrics["fused"]["mAP"])
    strictly_beaten = {
        output: fused_map > float(metrics[output]["mAP"])
        for output in GATE_OUTPUTS
    }
    return {
        "passed": fused_map >= min_map and all(strictly_beaten.values()),
        "fused_mAP": fused_map,
        "minimum_mAP": float(min_map),
        "strictly_beaten": strictly_beaten,
    }


def overfit_loss_floor(config: dict[str, Any], *, num_classes: int) -> float:
    """Return the analytic weighted CE floor introduced by label smoothing."""

    if str(config["MODEL"]["ARCHITECTURE"]) not in (
        ARCHITECTURE_V7,
        ARCHITECTURE_V8,
    ):
        return 0.0
    smoothing = float(config["LOSS"]["LABEL_SMOOTHING"])
    correct_probability = 1.0 - smoothing + smoothing / num_classes
    other_probability = smoothing / num_classes
    entropy = -correct_probability * math.log(correct_probability)
    entropy -= (num_classes - 1) * other_probability * math.log(other_probability)
    identity_weight = (
        float(config["LOSS"]["ID_FUSED"])
        + 3.0 * float(config["LOSS"]["ID_BRANCH"])
        + 3.0 * float(config["LOSS"]["ID_RESIDUAL"])
    )
    return identity_weight * entropy


def evaluate_overfit_gate(
    losses: list[float], *, max_ratio: float, minimum_loss: float = 0.0
) -> dict[str, Any]:
    initial_loss = float(losses[0])
    final_loss = float(losses[-1])
    initial_excess = initial_loss - minimum_loss
    final_excess = final_loss - minimum_loss
    loss_ratio = final_excess / initial_excess
    return {
        "passed": loss_ratio <= max_ratio,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "minimum_loss": float(minimum_loss),
        "initial_excess_loss": initial_excess,
        "final_excess_loss": final_excess,
        "loss_ratio": loss_ratio,
        "maximum_loss_ratio": float(max_ratio),
    }


def learning_rate_multiplier(
    epoch: int, *, max_epochs: int, warmup_epochs: int
) -> float:
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    progress = (epoch - warmup_epochs - 1) / (max_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "capacity", "overfit", "dev"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config = _load_config(config_path)
    output_dir.mkdir(parents=True)
    if args.mode == "preflight":
        return _run_preflight(config, output_dir)
    if args.mode == "capacity":
        return _run_capacity(config, output_dir)
    if args.mode == "overfit":
        return _run_overfit(config, output_dir)
    if args.mode == "dev":
        return _run_dev(config, config_path, output_dir)
    raise NotImplementedError(f"V5 mode is not implemented yet: {args.mode}")


__all__ = [
    "architecture_source_paths",
    "evaluate_dev_gate",
    "evaluate_overfit_gate",
    "learning_rate_multiplier",
    "load_contract",
    "load_raw_config",
    "load_v7_initialization",
    "overfit_loss_floor",
    "quality_response_gate",
    "receipt_schema",
    "parse_args",
    "run",
    "set_v7_training_phase",
    "weighted_training_loss",
]


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
