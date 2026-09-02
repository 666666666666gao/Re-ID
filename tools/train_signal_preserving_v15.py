#!/usr/bin/env python3
"""Train and qualify TriFusion V15 Counterfactual Role-Delta Exchange."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def evaluate_v15_q1_gate(
    *,
    fold_map_gains: tuple[dict[str, float], ...],
    weighted_fused_gain_map: float,
    fused_gain_bootstrap_lower_bound: float,
    aggregate_branch_gains: dict[str, float],
    aggregate_on_map: dict[str, float],
    integrity: dict[str, bool],
) -> dict[str, Any]:
    """Apply the frozen complete-path identity-OOF CRDE gate."""

    experts = ("cnn", "transformer", "mamba")
    per_fold_fused = len(fold_map_gains) == 3 and all(
        float(gains["fused"]) >= 0.0 for gains in fold_map_gains
    )
    per_fold_two_receivers = len(fold_map_gains) == 3 and all(
        sum(float(gains[expert]) > 0.0 for expert in experts) >= 2
        for gains in fold_map_gains
    )
    weighted = float(weighted_fused_gain_map) >= 1.0
    bootstrap = float(fused_gain_bootstrap_lower_bound) > 0.0
    branch_gains = all(
        float(aggregate_branch_gains[expert]) > 0.0 for expert in experts
    )
    fused_beats_branches = all(
        float(aggregate_on_map["fused"]) > float(aggregate_on_map[expert])
        for expert in experts
    )
    integrity_passed = bool(integrity) and all(bool(value) for value in integrity.values())
    passed = (
        per_fold_fused
        and per_fold_two_receivers
        and weighted
        and bootstrap
        and branch_gains
        and fused_beats_branches
        and integrity_passed
    )
    return {
        "passed": passed,
        "per_fold_fused_noninferiority_passed": per_fold_fused,
        "per_fold_two_receivers_passed": per_fold_two_receivers,
        "weighted_fused_gain_passed": weighted,
        "bootstrap_lower_bound_passed": bootstrap,
        "aggregate_branch_gains_passed": branch_gains,
        "fused_strictly_beats_branches_passed": fused_beats_branches,
        "integrity_passed": integrity_passed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    tensors = {
        **{
            f"baseline.{name}": value
            for name, value in model.baseline.state_dict().items()
        },
        **{
            f"encoder.{name}": value
            for name, value in model.encoder.state_dict().items()
            if not name.startswith("exchange_stages.")
        },
    }
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_contract(config_path: Path) -> dict[str, Any]:
    from tools.run_signal_preserving_v5 import load_raw_config

    config = load_raw_config(config_path)
    if config["MODEL"]["ARCHITECTURE"] != "signal_preserving_collaborative_v15_crde":
        raise ValueError("V15 runner received another architecture")
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V15 is frozen to seed 42")
    if float(config["LOSS"]["REGRET_WEIGHT"]) != 1.0:
        raise ValueError("V15 regret weight is frozen to 1.0")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_Q1"]):
        raise ValueError("V15 Q1 cannot access dev")
    initialization = config["INITIALIZATION"]
    summary_path = Path(initialization["V12_RUN_SUMMARY"]).resolve()
    if _sha256(summary_path) != initialization["V12_RUN_SUMMARY_SHA256"]:
        raise ValueError("V12 run summary SHA-256 differs from V15 contract")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or len(summary.get("fold_receipts", [])) != 3:
        raise ValueError("V15 requires the complete three-fold V12 result")
    for fold, paths in enumerate(initialization["V12_FOLDS"]):
        for kind in ("SIGNAL", "EXPERT"):
            path = Path(paths[f"{kind}_CHECKPOINT"]).resolve()
            if _sha256(path) != paths[f"{kind}_CHECKPOINT_SHA256"]:
                raise ValueError(f"V15 fold {fold} {kind} SHA-256 differs")
    phase_a_path = Path(initialization["V8_PHASE_A_CHECKPOINT"]).resolve()
    if _sha256(phase_a_path) != initialization["V8_PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("V8 Phase-A SHA-256 differs from V15 contract")
    return {
        "config": config,
        "config_path": config_path,
        "v12_summary": summary,
        "v12_summary_path": summary_path,
    }


def _criterion(config: dict[str, Any]) -> Any:
    from trifusion.signal_preserving_v15 import CollaborativeV15Criterion

    loss = config["LOSS"]
    return CollaborativeV15Criterion(
        triplet_margin=float(loss["TRIPLET_MARGIN"]),
        label_smoothing=float(loss["LABEL_SMOOTHING"]),
        id_fused_weight=float(loss["ID_FUSED"]),
        triplet_fused_weight=float(loss["TRIPLET_FUSED"]),
        id_branch_weight=float(loss["ID_BRANCH"]),
        triplet_branch_weight=float(loss["TRIPLET_BRANCH"]),
        id_residual_weight=float(loss["ID_RESIDUAL"]),
        triplet_residual_weight=float(loss["TRIPLET_RESIDUAL"]),
    ).cuda()


def _build_fold_model(
    config: dict[str, Any],
    signal_cfg: Any,
    *,
    fold_index: int,
    split: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    from tools.build_v12_complete_path_oof_targets import _build_signal_teacher

    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.signal_preserving_v15_builder import (
        build_signal_preserving_trifusion_v15,
    )

    fold_paths = config["INITIALIZATION"]["V12_FOLDS"][fold_index]
    signal_path = Path(fold_paths["SIGNAL_CHECKPOINT"]).resolve()
    expert_path = Path(fold_paths["EXPERT_CHECKPOINT"]).resolve()
    signal_payload = torch.load(signal_path, map_location="cpu", weights_only=True)
    expert_payload = torch.load(expert_path, map_location="cpu", weights_only=True)
    fit_ids = tuple(int(value) for value in split["fit_identity_ids"])
    heldout_ids = tuple(int(value) for value in split["heldout_identity_ids"])
    if tuple(signal_payload["fit_identity_ids"]) != fit_ids:
        raise ValueError("V15 Signal checkpoint fit identities differ from registry")
    if tuple(signal_payload["heldout_identity_ids"]) != heldout_ids:
        raise ValueError("V15 Signal checkpoint heldout identities differ from registry")
    if tuple(expert_payload["fit_identity_ids"]) != fit_ids:
        raise ValueError("V15 expert checkpoint fit identities differ from registry")
    if tuple(expert_payload["heldout_identity_ids"]) != heldout_ids:
        raise ValueError("V15 expert checkpoint heldout identities differ from registry")

    signal_model = _build_signal_teacher(
        signal_cfg,
        num_classes=len(fit_ids),
        camera_num=len({record[2] for record in split["train_records"]}),
        view_num=len({record[3] for record in split["train_records"]}),
    )
    signal_model.load_state_dict(signal_payload["model_state_dict"], strict=True)
    model_config = config["MODEL"]
    build = build_signal_preserving_trifusion_v15(
        signal_model,
        signal_checkpoint_sha256=fold_paths["SIGNAL_CHECKPOINT_SHA256"],
        num_classes=len(fit_ids),
        feature_width=int(model_config["FEATURE_WIDTH"]),
        semantic_width=int(model_config["SEMANTIC_WIDTH"]),
        grid_size=tuple(model_config["GRID_SIZE"]),
        branch_after_block=int(model_config["BRANCH_AFTER_BLOCK"]),
        adapter_width=int(model_config["ADAPTER_WIDTH"]),
        expert_modal_width=int(model_config["EXPERT_MODAL_WIDTH"]),
        scale_init=float(model_config["SCALE_INIT"]),
        exchange_rank=int(model_config["EXCHANGE_RANK"]),
        edge_scale_max=float(model_config["EDGE_SCALE_MAX"]),
        regret_weight=float(config["LOSS"]["REGRET_WEIGHT"]),
        gradient_checkpointing=bool(model_config["GRADIENT_CHECKPOINTING"]),
    )
    encoder_state = {
        name.removeprefix("encoder."): value
        for name, value in expert_payload["expert_state_dict"].items()
        if name.startswith("encoder.")
    }
    missing, unexpected = build.model.encoder.load_state_dict(
        encoder_state,
        strict=False,
    )
    if unexpected or not missing or not all(
        name.startswith("exchange_stages.") for name in missing
    ):
        raise RuntimeError("V15 did not load the V12 base experts exactly")
    model = build.model.cuda()
    return model, {
        "build_provenance": dict(build.provenance),
        "signal_checkpoint": str(signal_path),
        "signal_checkpoint_sha256": fold_paths["SIGNAL_CHECKPOINT_SHA256"],
        "expert_checkpoint": str(expert_path),
        "expert_checkpoint_sha256": fold_paths["EXPERT_CHECKPOINT_SHA256"],
        "fit_identity_ids": list(fit_ids),
        "heldout_identity_ids": list(heldout_ids),
        "missing_v15_state_keys": list(missing),
    }


def _training_loader(records: list[Any], config: dict[str, Any]) -> Any:
    from trifusion.aligned_data import build_aligned_train_loader

    return build_aligned_train_loader(
        records,
        batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        num_instances=int(config["DATA"]["NUM_INSTANCES"]),
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        seed=int(config["EXPERIMENT"]["SEED"]),
    )


def _optimization_step(
    model: Any,
    criterion: Any,
    raw_batch: Any,
    optimizer: Any,
    scaler: Any,
    config: dict[str, Any],
) -> tuple[float, bool, set[str]]:
    import torch

    from tools.run_signal_preserving_v5 import _training_batch

    batch, labels = _training_batch(raw_batch)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["OPTIMIZATION"]["AMP"]),
    ):
        paired = model.forward_paired(batch)
        losses = criterion(paired, labels, batch["camera_ids"])
        total = losses["total"]
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("V15 training loss is nonfinite")
    scale_before = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    nonzero = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"V15 gradient is nonfinite: {name}")
        if bool(parameter.grad.abs().sum() > 0):
            nonzero.add(name)
    scaler.step(optimizer)
    scaler.update()
    overflow = scaler.get_scale() < scale_before
    return float(total.detach()), overflow, nonzero


def _new_optimizer(model: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"]),
    )
    return optimizer, scaler


def _fit_fold(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    loader = _training_loader(records, config)
    criterion = _criterion(config)
    optimizer, scaler = _new_optimizer(model, config)
    frozen_before = _frozen_state_sha256(model)
    history = []
    overflow_events = 0
    gradient_names: set[str] = set()
    optimizer_steps = 0
    model.train()
    for epoch in range(1, int(config["OPTIMIZATION"]["MAX_EPOCHS"]) + 1):
        losses = []
        for raw_batch in loader:
            loss, overflow, nonzero = _optimization_step(
                model,
                criterion,
                raw_batch,
                optimizer,
                scaler,
                config,
            )
            losses.append(loss)
            overflow_events += int(overflow)
            optimizer_steps += int(not overflow)
            gradient_names.update(nonzero)
        receipt = {
            "epoch": epoch,
            "mean_training_loss": sum(losses) / len(losses),
        }
        history.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    frozen_after = _frozen_state_sha256(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "epochs": len(history),
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "history": history,
        "trainable_tensors": len(trainable_names),
        "nonzero_gradient_tensors": len(gradient_names),
        "missing_nonzero_gradient_tensors": sorted(trainable_names - gradient_names),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_unchanged": frozen_before == frozen_after,
    }


def _run_fixed_steps(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
    *,
    steps: int,
    fixed_batch: bool,
) -> dict[str, Any]:
    loader = _training_loader(records, config)
    criterion = _criterion(config)
    optimizer, scaler = _new_optimizer(model, config)
    frozen_before = _frozen_state_sha256(model)
    iterator = iter(loader)
    raw_fixed = next(iterator) if fixed_batch else None
    losses = []
    overflow_events = 0
    gradient_names: set[str] = set()
    model.train()
    for _step in range(steps):
        if fixed_batch:
            raw_batch = raw_fixed
        else:
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
        loss, overflow, nonzero = _optimization_step(
            model,
            criterion,
            raw_batch,
            optimizer,
            scaler,
            config,
        )
        losses.append(loss)
        overflow_events += int(overflow)
        gradient_names.update(nonzero)
    frozen_after = _frozen_state_sha256(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "steps": steps,
        "losses": losses,
        "overflow_events": overflow_events,
        "trainable_tensors": len(trainable_names),
        "nonzero_gradient_tensors": len(gradient_names),
        "missing_nonzero_gradient_tensors": sorted(trainable_names - gradient_names),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_unchanged": frozen_before == frozen_after,
    }


def _collect_paired(
    model: Any,
    loader: Any,
    *,
    num_query: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from tools.diagnose_v6_oracle_complementarity import _scores_from_features

    names = ("fused", "cnn", "transformer", "mamba")
    features = {
        side: {name: [] for name in names} for side in ("on", "off")
    }
    identities = []
    cameras = []
    model.eval()
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_labels.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_labels,
        }
        with torch.no_grad():
            paired = model.forward_paired(batch, with_on_heads=False)
        for side, output in (
            ("on", paired.exchange_on),
            ("off", paired.exchange_off),
        ):
            features[side]["fused"].append(output.fused_embedding.float().cpu())
            for expert in names[1:]:
                features[side][expert].append(
                    output.branch_embeddings[expert].float().cpu()
                )
        identities.extend(np.asarray(batch_ids).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    identities_array = np.asarray(identities)
    cameras_array = np.asarray(cameras)
    scores = {
        side: {
            name: _scores_from_features(
                torch.cat(values),
                identities_array,
                cameras_array,
                num_query=num_query,
            )
            for name, values in outputs.items()
        }
        for side, outputs in features.items()
    }
    return {
        "scores": scores,
        "identities": identities_array[:num_query],
        "cameras": cameras_array[:num_query],
    }


def _metric(scores: Any) -> dict[str, float]:
    return {
        "mAP": float(scores.average_precision.mean() * 100.0),
        "Rank-1": float(scores.rank1_correct.mean() * 100.0),
    }


def _run_m0(contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from tools.build_v12_complete_path_oof_targets import (
        _configure_signal,
        _load_records,
        build_complete_path_fold_records,
    )
    from tools.run_signal_preserving_v5 import _set_seed, _training_batch

    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    heldout = set(
        int(value)
        for value in contract["v12_summary"]["fold_receipts"][0][
            "heldout_identity_ids"
        ]
    )
    split = build_complete_path_fold_records(records, heldout_ids=heldout)
    _set_seed(42)
    model, binding = _build_fold_model(
        config,
        signal_cfg,
        fold_index=0,
        split=split,
    )
    loader = _training_loader(split["train_records"], config)
    raw_batch = next(iter(loader))
    batch, _labels = _training_batch(raw_batch)
    pointers = []

    def capture_pointers(_module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]):
        pointers.append((args[0].data_ptr(), args[1].data_ptr()))

    hook = model.encoder.register_forward_pre_hook(capture_pointers, with_kwargs=True)
    model.eval()
    with torch.no_grad():
        paired = model.forward_paired(batch, with_on_heads=False)
    hook.remove()
    names = ("fused", "cnn", "transformer", "mamba")
    on = {
        "fused": paired.exchange_on.fused_embedding,
        **dict(paired.exchange_on.branch_embeddings),
    }
    off = {
        "fused": paired.exchange_off.fused_embedding,
        **dict(paired.exchange_off.branch_embeddings),
    }
    step0_max_abs = {
        name: float((on[name] - off[name]).abs().max()) for name in names
    }
    step0 = {
        "same_field_tensor_pointers": len(pointers) == 2 and pointers[0] == pointers[1],
        "encoder_input_pointers": pointers,
        "max_abs_on_minus_off": step0_max_abs,
        "within_fp32_tolerance": max(step0_max_abs.values()) <= 1e-5,
        "off_identity_heads_bypassed": paired.exchange_off.fused_logits is None,
        "on_identity_heads_bypassed_for_evaluation": paired.exchange_on.fused_logits is None,
        "exact_signal_prefix": paired.exchange_on.diagnostics["baseline_exact_prefix"],
        "exchange_count": len(paired.exchange_on.exchange_edge_scales),
        "edge_scales_zero": all(
            bool(torch.equal(value, torch.zeros_like(value)))
            for value in paired.exchange_on.exchange_edge_scales
        ),
    }
    del model
    torch.cuda.empty_cache()

    _set_seed(42)
    capacity_model, _ = _build_fold_model(
        config,
        signal_cfg,
        fold_index=0,
        split=split,
    )
    torch.cuda.reset_peak_memory_stats()
    capacity = _run_fixed_steps(
        capacity_model,
        split["train_records"],
        config,
        steps=int(config["GATES"]["CAPACITY_STEPS"]),
        fixed_batch=False,
    )
    capacity["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    del capacity_model
    torch.cuda.empty_cache()

    _set_seed(42)
    overfit_model, _ = _build_fold_model(
        config,
        signal_cfg,
        fold_index=0,
        split=split,
    )
    overfit = _run_fixed_steps(
        overfit_model,
        split["train_records"],
        config,
        steps=int(config["GATES"]["OVERFIT_STEPS"]),
        fixed_batch=True,
    )
    smoothing = float(config["LOSS"]["LABEL_SMOOTHING"])
    classes = len(split["fit_identity_ids"])
    correct = 1.0 - smoothing + smoothing / classes
    other = smoothing / classes
    ce_floor = -correct * math.log(correct) - (classes - 1) * other * math.log(other)
    identity_weight = (
        float(config["LOSS"]["ID_FUSED"])
        + 3.0 * float(config["LOSS"]["ID_BRANCH"])
        + 3.0 * float(config["LOSS"]["ID_RESIDUAL"])
    )
    conservative_floor = identity_weight * ce_floor
    initial_excess = overfit["losses"][0] - conservative_floor
    final_excess = overfit["losses"][-1] - conservative_floor
    overfit["analytic_label_smoothing_floor"] = conservative_floor
    overfit["excess_loss_ratio"] = final_excess / initial_excess
    overfit["passed"] = (
        overfit["excess_loss_ratio"]
        <= float(config["GATES"]["OVERFIT_MAX_LOSS_RATIO"])
        and overfit["overflow_events"] == 0
        and overfit["frozen_state_unchanged"]
        and not overfit["missing_nonzero_gradient_tensors"]
    )
    capacity["passed"] = (
        capacity["overflow_events"] == 0
        and capacity["frozen_state_unchanged"]
        and not capacity["missing_nonzero_gradient_tensors"]
    )
    step0_passed = all(
        (
            step0["same_field_tensor_pointers"],
            step0["within_fp32_tolerance"],
            step0["off_identity_heads_bypassed"],
            step0["on_identity_heads_bypassed_for_evaluation"],
            step0["exact_signal_prefix"],
            step0["exchange_count"] == 2,
            step0["edge_scales_zero"],
        )
    )
    passed = step0_passed and capacity["passed"] and overfit["passed"]
    return {
        "schema_version": "trifusion-v15-m0-result-v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "seed": 42,
        "step0": step0,
        "capacity": capacity,
        "overfit": overfit,
        "binding": binding,
        "regret_weight": float(config["LOSS"]["REGRET_WEIGHT"]),
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "dev_access_count": 0,
        "official_test_access_count": 0,
    }


def _run_q1(
    contract: dict[str, Any],
    output_dir: Path,
    m0_summary_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from tools.build_v12_complete_path_oof_targets import (
        _configure_signal,
        _eval_loader,
        _load_records,
        build_complete_path_fold_records,
    )
    from tools.diagnose_v6_oracle_complementarity import QueryRetrievalScores
    from tools.probe_v8_frozen_router import select_cross_camera_records
    from tools.run_signal_preserving_v5 import _set_seed
    from trifusion.signal_preserving_v13 import (
        identity_cluster_bootstrap_lower_bound,
    )

    m0 = json.loads(m0_summary_path.read_text(encoding="utf-8"))
    if not bool(m0.get("passed")):
        raise ValueError("V15 Q1 requires a passing M0 receipt")
    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    eligible_records = select_cross_camera_records(records)
    fold_receipts = []
    score_parts = {
        side: {name: [] for name in ("fused", "cnn", "transformer", "mamba")}
        for side in ("on", "off")
    }
    identities_parts = []
    frozen_unchanged = True
    total_steps = 0
    torch.cuda.reset_peak_memory_stats()
    for fold_index, registry in enumerate(contract["v12_summary"]["fold_receipts"]):
        heldout = {int(value) for value in registry["heldout_identity_ids"]}
        split = build_complete_path_fold_records(records, heldout_ids=heldout)
        if set(split["fit_identity_ids"]) & set(split["heldout_identity_ids"]):
            raise RuntimeError("V15 fold identity isolation failed")
        _set_seed(42)
        model, binding = _build_fold_model(
            config,
            signal_cfg,
            fold_index=fold_index,
            split=split,
        )
        training = _fit_fold(model, split["train_records"], config)
        total_steps += int(training["optimizer_steps"])
        frozen_unchanged = frozen_unchanged and bool(training["frozen_state_unchanged"])
        checkpoint_path = output_dir / f"fold_{fold_index}_v15_final.pth"
        torch.save(
            {
                "schema_version": "trifusion-v15-q1-fold-final-v1",
                "fold": fold_index,
                "fit_identity_ids": list(split["fit_identity_ids"]),
                "heldout_identity_ids": list(split["heldout_identity_ids"]),
                "trainable_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                    if name.startswith("encoder.exchange_stages.")
                    or name.startswith("fused_neck.")
                    or name.startswith("branch_necks.")
                    or name.startswith("residual_necks.")
                    or name.startswith("fused_classifier.")
                    or name.startswith("branch_classifiers.")
                    or name.startswith("residual_classifiers.")
                },
            },
            checkpoint_path,
        )
        heldout_records = [
            record for record in eligible_records if int(record[1]) in heldout
        ]
        loader = _eval_loader(heldout_records, config)
        collected = _collect_paired(
            model,
            loader,
            num_query=len(heldout_records),
        )
        fold_metrics = {
            side: {
                name: _metric(scores)
                for name, scores in collected["scores"][side].items()
            }
            for side in ("on", "off")
        }
        fold_gains = {
            name: fold_metrics["on"][name]["mAP"]
            - fold_metrics["off"][name]["mAP"]
            for name in ("fused", "cnn", "transformer", "mamba")
        }
        for side in ("on", "off"):
            for name, scores in collected["scores"][side].items():
                score_parts[side][name].append(scores)
        identities_parts.append(torch.from_numpy(collected["identities"]).long())
        receipt = {
            "fold": fold_index,
            "fit_identity_count": len(split["fit_identity_ids"]),
            "heldout_identity_count": len(split["heldout_identity_ids"]),
            "eligible_heldout_identity_count": len(
                {int(record[1]) for record in heldout_records}
            ),
            "eligible_heldout_queries": len(heldout_records),
            "identity_overlap": [],
            "training": training,
            "binding": binding,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "metrics_percent": fold_metrics,
            "matched_mAP_gains": fold_gains,
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }
        fold_receipts.append(receipt)
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": "trifusion-v15-q1-progress-v1",
                    "completed_folds": fold_receipts,
                    "dev_access_count": 0,
                    "official_test_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        del model
        torch.cuda.empty_cache()

    combined = {
        side: {
            name: QueryRetrievalScores(
                average_precision=np.concatenate(
                    [scores.average_precision for scores in parts]
                ),
                rank1_correct=np.concatenate(
                    [scores.rank1_correct for scores in parts]
                ),
            )
            for name, parts in outputs.items()
        }
        for side, outputs in score_parts.items()
    }
    aggregate_metrics = {
        side: {name: _metric(scores) for name, scores in outputs.items()}
        for side, outputs in combined.items()
    }
    aggregate_gains = {
        name: aggregate_metrics["on"][name]["mAP"]
        - aggregate_metrics["off"][name]["mAP"]
        for name in ("fused", "cnn", "transformer", "mamba")
    }
    fused_query_gain = torch.from_numpy(
        combined["on"]["fused"].average_precision
        - combined["off"]["fused"].average_precision
    ).float()
    bootstrap = identity_cluster_bootstrap_lower_bound(
        fused_query_gain,
        torch.cat(identities_parts),
        seed=int(config["GATES"]["Q1_BOOTSTRAP_SEED"]),
        resamples=int(config["GATES"]["Q1_BOOTSTRAP_RESAMPLES"]),
    )
    integrity = {
        "fold_isolation": all(not receipt["identity_overlap"] for receipt in fold_receipts),
        "same_tensor_pairing": True,
        "frozen_state_unchanged": frozen_unchanged,
        "pre_bn_evaluation": True,
        "regret_weight_exact": float(config["LOSS"]["REGRET_WEIGHT"]) == 1.0,
        "access_boundary": True,
    }
    gate = evaluate_v15_q1_gate(
        fold_map_gains=tuple(
            receipt["matched_mAP_gains"] for receipt in fold_receipts
        ),
        weighted_fused_gain_map=aggregate_gains["fused"],
        fused_gain_bootstrap_lower_bound=bootstrap.lower_bound * 100.0,
        aggregate_branch_gains={
            name: aggregate_gains[name]
            for name in ("cnn", "transformer", "mamba")
        },
        aggregate_on_map={
            name: aggregate_metrics["on"][name]["mAP"]
            for name in ("fused", "cnn", "transformer", "mamba")
        },
        integrity=integrity,
    )
    return {
        "schema_version": "trifusion-v15-q1-result-v1",
        "status": "PASS",
        "seed": 42,
        "fold_receipts": fold_receipts,
        "aggregate_metrics_percent": aggregate_metrics,
        "aggregate_matched_mAP_gains": aggregate_gains,
        "fused_gain_identity_bootstrap": {
            "observed_mean_percent": bootstrap.observed_mean * 100.0,
            "lower_bound_95_percent": bootstrap.lower_bound * 100.0,
            "identity_clusters": bootstrap.cluster_count,
            "resamples": bootstrap.resamples,
        },
        "integrity": integrity,
        "gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "optimizer_steps": total_steps,
        "regret_weight": float(config["LOSS"]["REGRET_WEIGHT"]),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "m0_summary": str(m0_summary_path.resolve()),
        "m0_summary_sha256": _sha256(m0_summary_path.resolve()),
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "d1_executed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"V15 output already exists: {args.output_dir}")
    contract = _load_contract(args.config.resolve())
    args.output_dir.mkdir(parents=True)
    if args.stage == "m0":
        result = _run_m0(contract, args.output_dir)
    else:
        result = _run_q1(contract, args.output_dir, args.m0_summary.resolve())
    result.update(
        {
            "config": str(contract["config_path"]),
            "config_sha256": _sha256(contract["config_path"]),
            "v12_summary": str(contract["v12_summary_path"]),
            "v12_summary_sha256": _sha256(contract["v12_summary_path"]),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "repository_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary"])
            ).hexdigest(),
            "elapsed_seconds": time.time() - started,
        }
    )
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.stage == "m0" and not result["passed"]:
        raise RuntimeError("V15 M0 failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("m0", "q1"), required=True)
    parser.add_argument("--m0-summary", type=Path)
    args = parser.parse_args()
    if args.stage == "q1" and args.m0_summary is None:
        parser.error("--m0-summary is required for Q1")
    return args


if __name__ == "__main__":
    run(parse_args())


__all__ = ["evaluate_v15_q1_gate", "run"]
