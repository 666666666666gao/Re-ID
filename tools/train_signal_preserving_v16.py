#!/usr/bin/env python3
"""Train, qualify, and conditionally evaluate TriFusion V16 SATR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


V16_INITIAL_COVERAGE_MIN = 0.005
V16_INITIAL_COVERAGE_MAX = 0.25


def evaluate_v16_m0_gate(
    *,
    exact_prefix: bool,
    paired_receipts_equal: bool,
    initial_activity: bool,
    capacity_overflow_events: int,
    capacity_all_trainable_tensors_reached: bool,
    capacity_frozen_state_unchanged: bool,
    capacity_peak_reserved_mib: float,
    overfit_overflow_events: int,
    overfit_frozen_state_unchanged: bool,
    overfit_excess_loss_ratio: float,
    overfit_max_loss_ratio: float,
) -> dict[str, bool]:
    preflight_passed = exact_prefix and paired_receipts_equal and initial_activity
    capacity_passed = all(
        (
            capacity_overflow_events == 0,
            capacity_all_trainable_tensors_reached,
            capacity_frozen_state_unchanged,
            capacity_peak_reserved_mib < 24.0 * 1024.0,
        )
    )
    overfit_passed = all(
        (
            overfit_overflow_events == 0,
            overfit_frozen_state_unchanged,
            overfit_excess_loss_ratio <= overfit_max_loss_ratio,
        )
    )
    return {
        "passed": preflight_passed and capacity_passed and overfit_passed,
        "preflight_passed": preflight_passed,
        "capacity_passed": capacity_passed,
        "overfit_passed": overfit_passed,
    }


def evaluate_v16_d1_gate(
    *,
    metrics_percent: Mapping[str, Mapping[str, float]],
    minimum_fused_map: float,
    v8_phase_b_map: float,
    strict_reload: bool,
    frozen_state_unchanged: bool,
) -> dict[str, bool]:
    branches = ("cnn", "transformer", "mamba")
    fused = metrics_percent["fused"]
    minimum_map = float(fused["mAP"]) >= float(minimum_fused_map)
    strict_map_wins = all(
        float(fused["mAP"]) > comparator
        for comparator in (
            float(metrics_percent["baseline_only"]["mAP"]),
            float(v8_phase_b_map),
            *(float(metrics_percent[expert]["mAP"]) for expert in branches),
        )
    )
    strict_rank1_wins = all(
        float(fused["Rank-1"]) > float(metrics_percent[expert]["Rank-1"])
        for expert in branches
    )
    passed = all(
        (
            minimum_map,
            strict_map_wins,
            strict_rank1_wins,
            strict_reload,
            frozen_state_unchanged,
        )
    )
    return {
        "passed": passed,
        "minimum_fused_mAP_passed": minimum_map,
        "strict_mAP_wins_passed": strict_map_wins,
        "strict_branch_Rank1_wins_passed": strict_rank1_wins,
        "strict_reload_passed": strict_reload,
        "frozen_state_unchanged_passed": frozen_state_unchanged,
    }


def evaluate_v16_q1_gate(
    *,
    fold_map_gains: Sequence[Mapping[str, float]],
    weighted_fused_gain_map: float,
    fused_gain_bootstrap_lower_bound: float,
    aggregate_branch_gains: Mapping[str, float],
    aggregate_satr_map: Mapping[str, float],
    initial_coverages: Sequence[Mapping[str, float]],
    integrity: Mapping[str, bool],
) -> dict[str, Any]:
    experts = ("cnn", "transformer", "mamba")
    fold_count = len(fold_map_gains) == 3 and len(initial_coverages) == 3
    per_fold_fused = fold_count and all(
        float(fold["fused"]) >= 0.0 for fold in fold_map_gains
    )
    per_fold_receivers = fold_count and all(
        sum(float(fold[expert]) > 0.0 for expert in experts) >= 2
        for fold in fold_map_gains
    )
    weighted = float(weighted_fused_gain_map) >= 1.0
    bootstrap = float(fused_gain_bootstrap_lower_bound) > 0.0
    aggregate_branches = all(
        float(aggregate_branch_gains[expert]) > 0.0 for expert in experts
    )
    fused_beats_branches = all(
        float(aggregate_satr_map["fused"]) > float(aggregate_satr_map[expert])
        for expert in experts
    )
    activity = fold_count and all(
        V16_INITIAL_COVERAGE_MIN
        <= float(fold[expert])
        <= V16_INITIAL_COVERAGE_MAX
        for fold in initial_coverages
        for expert in experts
    )
    integrity_passed = bool(integrity) and all(bool(value) for value in integrity.values())
    passed = all(
        (
            per_fold_fused,
            per_fold_receivers,
            weighted,
            bootstrap,
            aggregate_branches,
            fused_beats_branches,
            activity,
            integrity_passed,
        )
    )
    return {
        "passed": passed,
        "per_fold_fused_nonnegative_passed": per_fold_fused,
        "per_fold_two_receivers_passed": per_fold_receivers,
        "weighted_fused_gain_passed": weighted,
        "fused_bootstrap_lower_bound_passed": bootstrap,
        "aggregate_branch_gains_passed": aggregate_branches,
        "fused_strictly_beats_branches_passed": fused_beats_branches,
        "fixed_initial_activity_passed": activity,
        "integrity_passed": integrity_passed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_mapping_sha256(tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: Any) -> str:
    return _tensor_mapping_sha256(model.state_dict())


def _frozen_state_sha256(model: Any) -> str:
    return _tensor_mapping_sha256(
        {f"baseline.{name}": value for name, value in model.baseline.state_dict().items()}
    )


def _trainable_names(model: Any) -> tuple[str, ...]:
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _load_contract(config_path: Path) -> dict[str, Any]:
    from tools.run_signal_preserving_v5 import load_raw_config

    config = load_raw_config(config_path)
    if config["MODEL"]["ARCHITECTURE"] != "signal_anchored_triadic_repair_v16":
        raise ValueError("V16 runner received another architecture")
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V16 is frozen to seed 42")
    if (
        int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        int(config["DATA"]["NUM_INSTANCES"]),
    ) != (64, 8):
        raise ValueError("V16 requires a physical B64/K8 batch")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_Q1"]):
        raise ValueError("V16 Q1 cannot access dev")
    if bool(config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"]):
        raise ValueError("V16 development cannot access official test")
    expected_loss = {
        "ID_FUSED": 0.25,
        "TRIPLET_FUSED": 1.0,
        "ID_BRANCH": 1.0 / 12.0,
        "TRIPLET_BRANCH": 0.25,
        "ID_RESIDUAL": 1.0 / 12.0,
        "TRIPLET_RESIDUAL": 0.25,
        "TRIPLET_MARGIN": 0.3,
        "LABEL_SMOOTHING": 0.1,
    }
    if any(float(config["LOSS"][name]) != value for name, value in expected_loss.items()):
        raise ValueError("V16 ReID loss weights differ from the frozen contract")
    expected_satr = {
        "RELATION_GAP": 0.05,
        "PROTECTION_THRESHOLD": 0.30,
        "PROTECTION_TOLERANCE": 0.02,
        "REPAIR_WEIGHT": 1.0,
        "PROTECTION_WEIGHT": 0.25,
    }
    if any(float(config["SATR"][name]) != value for name, value in expected_satr.items()):
        raise ValueError("V16 SATR constants differ from the frozen contract")
    if list(config["PROTOCOL"]["Q1_ENDPOINTS"]) != ["no_satr", "satr"]:
        raise ValueError("V16 Q1 endpoint order differs from the frozen contract")

    initialization = config["INITIALIZATION"]
    summary_path = Path(initialization["V12_RUN_SUMMARY"]).resolve()
    if _sha256(summary_path) != initialization["V12_RUN_SUMMARY_SHA256"]:
        raise ValueError("V12 run summary SHA-256 differs from V16 contract")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or len(summary.get("fold_receipts", [])) != 3:
        raise ValueError("V16 requires the complete three-fold V12 result")
    for fold, paths in enumerate(initialization["V12_FOLDS"]):
        for kind in ("SIGNAL", "EXPERT"):
            path = Path(paths[f"{kind}_CHECKPOINT"]).resolve()
            if _sha256(path) != paths[f"{kind}_CHECKPOINT_SHA256"]:
                raise ValueError(f"V16 fold {fold} {kind} SHA-256 differs")
    phase_a_path = Path(initialization["V8_PHASE_A_CHECKPOINT"]).resolve()
    if _sha256(phase_a_path) != initialization["V8_PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("V8 Phase-A SHA-256 differs from V16 contract")
    signal_path = Path(config["SIGNAL"]["CHECKPOINT"]).resolve()
    if _sha256(signal_path) != config["SIGNAL"]["CHECKPOINT_SHA256"]:
        raise ValueError("all-fit Signal SHA-256 differs from V16 contract")
    threshold_path = _project_path(config["EVIDENCE"]["THRESHOLD_FREEZE"])
    if _sha256(threshold_path) != config["EVIDENCE"]["THRESHOLD_FREEZE_SHA256"]:
        raise ValueError("V16 threshold-freeze evidence SHA-256 differs")
    return {
        "config": config,
        "config_path": config_path,
        "v12_summary": summary,
        "v12_summary_path": summary_path,
        "threshold_path": threshold_path,
    }


def _criterion(config: dict[str, Any], *, satr_enabled: bool) -> Any:
    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.signal_preserving_v16 import SATRV16Criterion

    return SATRV16Criterion(
        triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
        label_smoothing=float(config["LOSS"]["LABEL_SMOOTHING"]),
        satr_enabled=satr_enabled,
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
    from trifusion.signal_preserving_v16_builder import (
        build_signal_preserving_trifusion_v16,
    )

    fold_paths = config["INITIALIZATION"]["V12_FOLDS"][fold_index]
    signal_path = Path(fold_paths["SIGNAL_CHECKPOINT"]).resolve()
    expert_path = Path(fold_paths["EXPERT_CHECKPOINT"]).resolve()
    signal_payload = torch.load(signal_path, map_location="cpu", weights_only=True)
    expert_payload = torch.load(expert_path, map_location="cpu", weights_only=True)
    fit_ids = tuple(int(value) for value in split["fit_identity_ids"])
    heldout_ids = tuple(int(value) for value in split["heldout_identity_ids"])
    for payload, name in ((signal_payload, "Signal"), (expert_payload, "expert")):
        if tuple(payload["fit_identity_ids"]) != fit_ids:
            raise ValueError(f"V16 {name} fit identities differ from registry")
        if tuple(payload["heldout_identity_ids"]) != heldout_ids:
            raise ValueError(f"V16 {name} heldout identities differ from registry")

    signal_model = _build_signal_teacher(
        signal_cfg,
        num_classes=len(fit_ids),
        camera_num=len({record[2] for record in split["train_records"]}),
        view_num=len({record[3] for record in split["train_records"]}),
    )
    signal_model.load_state_dict(signal_payload["model_state_dict"], strict=True)
    model_config = config["MODEL"]
    build = build_signal_preserving_trifusion_v16(
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
        gradient_checkpointing=bool(model_config["GRADIENT_CHECKPOINTING"]),
    )
    missing, unexpected = build.model.load_state_dict(
        expert_payload["expert_state_dict"],
        strict=False,
    )
    if unexpected or not missing or not all(name.startswith("baseline.") for name in missing):
        raise RuntimeError("V16 did not load the complete V12 expert endpoint exactly")
    model = build.model.cuda()
    return model, {
        "build_provenance": dict(build.provenance),
        "signal_checkpoint": str(signal_path),
        "signal_checkpoint_sha256": fold_paths["SIGNAL_CHECKPOINT_SHA256"],
        "expert_checkpoint": str(expert_path),
        "expert_checkpoint_sha256": fold_paths["EXPERT_CHECKPOINT_SHA256"],
        "fit_identity_ids": list(fit_ids),
        "heldout_identity_ids": list(heldout_ids),
        "missing_baseline_state_keys": list(missing),
    }


def _training_loader(records: list[Any], config: dict[str, Any]) -> Any:
    from tools.train_signal_preserving_v15 import _training_loader as build_loader

    return build_loader(records, config)


def _new_optimizer(model: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    from tools.train_signal_preserving_v15 import _new_optimizer as build_optimizer

    return build_optimizer(model, config)


def _optimization_step(
    model: Any,
    criterion: Any,
    raw_batch: Any,
    optimizer: Any,
    scaler: Any,
    config: dict[str, Any],
) -> tuple[dict[str, float], bool, set[str]]:
    import torch

    from tools.run_signal_preserving_v5 import _training_batch

    batch, labels = _training_batch(raw_batch)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["OPTIMIZATION"]["AMP"]),
    ):
        output = model(batch, return_aux=True)
        losses = criterion(output, labels, batch["camera_ids"])
        total = losses["total"]
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("V16 training loss is nonfinite")
    scale_before = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    nonzero = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"V16 gradient is nonfinite: {name}")
        if bool(parameter.grad.abs().sum() > 0):
            nonzero.add(name)
    scaler.step(optimizer)
    scaler.update()
    overflow = scaler.get_scale() < scale_before
    scalar_losses = {
        name: float(value.detach())
        for name, value in losses.items()
        if name == "total" or name == "satr_total" or name.startswith("coverage_")
    }
    return scalar_losses, overflow, nonzero


def _record_index_by_path(records: Sequence[Any]) -> dict[str, int]:
    mapping = {
        Path(record[0][0]).name: index for index, record in enumerate(records)
    }
    if len(mapping) != len(records):
        raise ValueError("V16 sampler receipt requires unique RGB filenames")
    return mapping


def _raw_batch_receipt(
    raw_batch: Any,
    *,
    record_index_by_path: Mapping[str, int],
) -> dict[str, Any]:
    images, labels, cameras, views, paths = raw_batch
    tensor_sha256 = {
        modality: _tensor_mapping_sha256({modality: tensor})
        for modality, tensor in images.items()
    }
    sampler_indices = [record_index_by_path[str(path)] for path in paths]
    metadata = {
        "labels": labels,
        "physical_cameras": cameras,
        "model_camera_labels": cameras,
        "views": views,
    }
    return {
        "paths": [str(path) for path in paths],
        "sampler_indices": sampler_indices,
        "tensor_sha256": tensor_sha256,
        "metadata_sha256": _tensor_mapping_sha256(metadata),
    }


def _initial_endpoint_receipt(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from tools.run_signal_preserving_v5 import _set_seed, _training_batch
    from trifusion.signal_preserving_v16 import satr_relation_objective_v16

    _set_seed(42)
    loader = _training_loader(records, config)
    record_index = _record_index_by_path(records)
    batch_receipts = []
    eligible = {expert: 0 for expert in ("cnn", "transformer", "mamba")}
    valid_queries = 0
    protected_queries = 0
    exact_prefix = True
    frozen_before = _frozen_state_sha256(model)
    model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= int(config["GATES"]["INITIAL_ACTIVITY_BATCHES"]):
            break
        batch_receipts.append(
            _raw_batch_receipt(
                raw_batch,
                record_index_by_path=record_index,
            )
        )
        batch, labels = _training_batch(raw_batch)
        with torch.no_grad():
            output = model(batch, return_aux=True)
            relation = satr_relation_objective_v16(
                output.baseline_embedding,
                output.fused_embedding,
                output.branch_embeddings,
                labels,
                batch["camera_ids"],
            )
        exact_prefix = exact_prefix and bool(output.diagnostics["baseline_exact_prefix"])
        valid_queries += int(relation.pairs.valid_query_mask.sum())
        protected_queries += int(relation.protection_mask.sum())
        for expert in eligible:
            eligible[expert] += int(relation.eligible_masks[expert].sum())
    frozen_after = _frozen_state_sha256(model)
    coverages = {
        expert: count / valid_queries for expert, count in eligible.items()
    }
    return {
        "initial_state_sha256": _model_state_sha256(model),
        "trainable_names": list(_trainable_names(model)),
        "trainable_names_sha256": _names_sha256(_trainable_names(model)),
        "batch_receipts": batch_receipts,
        "valid_queries": valid_queries,
        "eligible_queries": eligible,
        "coverages": coverages,
        "protection_coverage": protected_queries / valid_queries,
        "exact_signal_prefix": exact_prefix,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_unchanged": frozen_before == frozen_after,
        "seed_contract": {
            "global_seed": 42,
            "sampler_seed": 42,
            "batch_size": 64,
            "instances_per_identity": 8,
        },
    }


def _fit_endpoint(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
    *,
    satr_enabled: bool,
) -> dict[str, Any]:
    from tools.run_signal_preserving_v5 import _set_seed

    _set_seed(42)
    loader = _training_loader(records, config)
    criterion = _criterion(config, satr_enabled=satr_enabled)
    optimizer, scaler = _new_optimizer(model, config)
    record_index = _record_index_by_path(records)
    frozen_before = _frozen_state_sha256(model)
    initial_state = _model_state_sha256(model)
    trainable_names = _trainable_names(model)
    history = []
    first_batch_receipts = []
    sample_order_digest = hashlib.sha256()
    overflow_events = 0
    optimizer_steps = 0
    gradient_names: set[str] = set()
    model.train()
    for epoch in range(1, int(config["OPTIMIZATION"]["MAX_EPOCHS"]) + 1):
        totals = []
        satr_totals = []
        coverage = {expert: [] for expert in ("cnn", "transformer", "mamba")}
        for raw_batch in loader:
            receipt = _raw_batch_receipt(
                raw_batch,
                record_index_by_path=record_index,
            )
            sample_order_digest.update(
                json.dumps(receipt["sampler_indices"], separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            if len(first_batch_receipts) < 8:
                first_batch_receipts.append(receipt)
            losses, overflow, nonzero = _optimization_step(
                model,
                criterion,
                raw_batch,
                optimizer,
                scaler,
                config,
            )
            totals.append(losses["total"])
            satr_totals.append(losses["satr_total"])
            for expert in coverage:
                coverage[expert].append(losses[f"coverage_{expert}"])
            overflow_events += int(overflow)
            optimizer_steps += int(not overflow)
            gradient_names.update(nonzero)
        epoch_receipt = {
            "epoch": epoch,
            "mean_training_loss": sum(totals) / len(totals),
            "mean_satr_loss": sum(satr_totals) / len(satr_totals),
            "mean_relation_coverage": {
                expert: sum(values) / len(values) for expert, values in coverage.items()
            },
        }
        history.append(epoch_receipt)
        print(json.dumps(epoch_receipt, sort_keys=True), flush=True)
    frozen_after = _frozen_state_sha256(model)
    return {
        "endpoint": "satr" if satr_enabled else "no_satr",
        "epochs": len(history),
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "history": history,
        "initial_state_sha256": initial_state,
        "final_state_sha256": _model_state_sha256(model),
        "trainable_names": list(trainable_names),
        "trainable_names_sha256": _names_sha256(trainable_names),
        "trainable_tensors": len(trainable_names),
        "nonzero_gradient_tensors": len(gradient_names),
        "missing_nonzero_gradient_tensors": sorted(set(trainable_names) - gradient_names),
        "sample_order_sha256": sample_order_digest.hexdigest(),
        "first_eight_batch_receipts": first_batch_receipts,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_unchanged": frozen_before == frozen_after,
        "seed_contract": {
            "global_seed": 42,
            "sampler_seed": 42,
            "batch_size": 64,
            "instances_per_identity": 8,
        },
    }


def _run_fixed_steps(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
    *,
    steps: int,
    fixed_batch: bool,
) -> dict[str, Any]:
    from tools.run_signal_preserving_v5 import _set_seed

    _set_seed(42)
    loader = _training_loader(records, config)
    criterion = _criterion(config, satr_enabled=True)
    optimizer, scaler = _new_optimizer(model, config)
    frozen_before = _frozen_state_sha256(model)
    iterator = iter(loader)
    fixed = next(iterator) if fixed_batch else None
    losses = []
    overflow_events = 0
    gradient_names: set[str] = set()
    model.train()
    for _step in range(steps):
        if fixed_batch:
            raw_batch = fixed
        else:
            raw_batch = next(iterator)
        scalar_losses, overflow, nonzero = _optimization_step(
            model,
            criterion,
            raw_batch,
            optimizer,
            scaler,
            config,
        )
        losses.append(scalar_losses["total"])
        overflow_events += int(overflow)
        gradient_names.update(nonzero)
    frozen_after = _frozen_state_sha256(model)
    trainable_names = set(_trainable_names(model))
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


def _run_m0(contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from tools.build_v12_complete_path_oof_targets import (
        _configure_signal,
        _load_records,
        build_complete_path_fold_records,
    )
    from tools.run_signal_preserving_v5 import _set_seed

    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    fold_preflights = []
    bindings = []
    for fold_index, registry in enumerate(contract["v12_summary"]["fold_receipts"]):
        heldout = {int(value) for value in registry["heldout_identity_ids"]}
        split = build_complete_path_fold_records(records, heldout_ids=heldout)
        _set_seed(42)
        model_a, binding = _build_fold_model(
            config,
            signal_cfg,
            fold_index=fold_index,
            split=split,
        )
        receipt_a = _initial_endpoint_receipt(
            model_a,
            split["train_records"],
            config,
        )
        del model_a
        torch.cuda.empty_cache()

        _set_seed(42)
        model_b, _ = _build_fold_model(
            config,
            signal_cfg,
            fold_index=fold_index,
            split=split,
        )
        receipt_b = _initial_endpoint_receipt(
            model_b,
            split["train_records"],
            config,
        )
        del model_b
        torch.cuda.empty_cache()
        paired = {
            "initial_state": (
                receipt_a["initial_state_sha256"]
                == receipt_b["initial_state_sha256"]
            ),
            "trainable_names": (
                receipt_a["trainable_names_sha256"]
                == receipt_b["trainable_names_sha256"]
            ),
            "sample_order_and_transformed_batches": (
                receipt_a["batch_receipts"] == receipt_b["batch_receipts"]
            ),
            "seed_contract": receipt_a["seed_contract"] == receipt_b["seed_contract"],
        }
        fold_preflights.append(
            {
                "fold": fold_index,
                "fit_identity_count": len(split["fit_identity_ids"]),
                "heldout_identity_count": len(split["heldout_identity_ids"]),
                "identity_overlap": sorted(
                    set(split["fit_identity_ids"]) & set(split["heldout_identity_ids"])
                ),
                "endpoint_a": receipt_a,
                "endpoint_b": receipt_b,
                "paired": paired,
            }
        )
        bindings.append(binding)

    fold0_heldout = {
        int(value)
        for value in contract["v12_summary"]["fold_receipts"][0][
            "heldout_identity_ids"
        ]
    }
    split0 = build_complete_path_fold_records(records, heldout_ids=fold0_heldout)
    _set_seed(42)
    capacity_model, _ = _build_fold_model(
        config,
        signal_cfg,
        fold_index=0,
        split=split0,
    )
    torch.cuda.reset_peak_memory_stats()
    capacity = _run_fixed_steps(
        capacity_model,
        split0["train_records"],
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
        split=split0,
    )
    overfit = _run_fixed_steps(
        overfit_model,
        split0["train_records"],
        config,
        steps=int(config["GATES"]["OVERFIT_STEPS"]),
        fixed_batch=True,
    )
    smoothing = float(config["LOSS"]["LABEL_SMOOTHING"])
    classes = len(split0["fit_identity_ids"])
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
    del overfit_model
    torch.cuda.empty_cache()

    exact_prefix = all(
        receipt["endpoint_a"]["exact_signal_prefix"]
        and receipt["endpoint_b"]["exact_signal_prefix"]
        for receipt in fold_preflights
    )
    paired_receipts_equal = all(
        all(receipt["paired"].values()) for receipt in fold_preflights
    )
    initial_coverages = tuple(
        receipt["endpoint_a"]["coverages"] for receipt in fold_preflights
    )
    initial_activity = all(
        float(config["GATES"]["INITIAL_COVERAGE_MIN"])
        <= float(coverage[expert])
        <= float(config["GATES"]["INITIAL_COVERAGE_MAX"])
        for coverage in initial_coverages
        for expert in ("cnn", "transformer", "mamba")
    )
    gate = evaluate_v16_m0_gate(
        exact_prefix=exact_prefix,
        paired_receipts_equal=paired_receipts_equal,
        initial_activity=initial_activity,
        capacity_overflow_events=capacity["overflow_events"],
        capacity_all_trainable_tensors_reached=(
            not capacity["missing_nonzero_gradient_tensors"]
        ),
        capacity_frozen_state_unchanged=capacity["frozen_state_unchanged"],
        capacity_peak_reserved_mib=capacity["peak_reserved_mib"],
        overfit_overflow_events=overfit["overflow_events"],
        overfit_frozen_state_unchanged=overfit["frozen_state_unchanged"],
        overfit_excess_loss_ratio=overfit["excess_loss_ratio"],
        overfit_max_loss_ratio=float(config["GATES"]["OVERFIT_MAX_LOSS_RATIO"]),
    )
    return {
        "schema_version": "trifusion-v16-m0-result-v1",
        "status": "PASS" if gate["passed"] else "FAIL",
        "passed": gate["passed"],
        "seed": 42,
        "fold_preflights": fold_preflights,
        "initial_coverages": initial_coverages,
        "capacity": capacity,
        "overfit": overfit,
        "m0_gate": gate,
        "bindings": bindings,
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "threshold_freeze": str(contract["threshold_path"]),
        "threshold_freeze_sha256": _sha256(contract["threshold_path"]),
        "dev_access_count": 0,
        "official_test_access_count": 0,
    }


def _collect_endpoint(
    model: Any,
    loader: Any,
    *,
    num_query: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from tools.diagnose_v6_oracle_complementarity import _scores_from_features

    names = ("baseline_only", "fused", "cnn", "transformer", "mamba")
    features = {name: [] for name in names}
    identities = []
    cameras = []
    exact_prefix = True
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
            output = model(batch, return_aux=True)
        features["baseline_only"].append(output.baseline_embedding.float().cpu())
        features["fused"].append(output.fused_embedding.float().cpu())
        for expert in ("cnn", "transformer", "mamba"):
            features[expert].append(output.branch_embeddings[expert].float().cpu())
        exact_prefix = exact_prefix and bool(output.diagnostics["baseline_exact_prefix"])
        identities.extend(np.asarray(batch_ids).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    identities_array = np.asarray(identities)
    cameras_array = np.asarray(cameras)
    concatenated = {name: torch.cat(values) for name, values in features.items()}
    scores = {
        name: _scores_from_features(
            values,
            identities_array,
            cameras_array,
            num_query=num_query,
        )
        for name, values in concatenated.items()
    }
    return {
        "scores": scores,
        "feature_widths": {
            name: int(values.shape[1]) for name, values in concatenated.items()
        },
        "query_identities": identities_array[:num_query],
        "query_cameras": cameras_array[:num_query],
        "exact_signal_prefix": exact_prefix,
    }


def _metric(scores: Any) -> dict[str, float]:
    from tools.train_signal_preserving_v15 import _metric as metric

    return metric(scores)


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
    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.signal_preserving_v13 import (
        identity_cluster_bootstrap_lower_bound,
    )

    m0 = json.loads(m0_summary_path.read_text(encoding="utf-8"))
    if not bool(m0.get("passed")):
        raise ValueError("V16 Q1 requires a passing M0 receipt")
    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    eligible_records = select_cross_camera_records(records)
    endpoint_names = ("no_satr", "satr")
    output_names = ("fused", "cnn", "transformer", "mamba")
    fold_receipts = []
    score_parts = {
        endpoint: {name: [] for name in output_names}
        for endpoint in endpoint_names
    }
    identities_parts = []
    total_steps = 0
    torch.cuda.reset_peak_memory_stats()
    for fold_index, registry in enumerate(contract["v12_summary"]["fold_receipts"]):
        heldout = {int(value) for value in registry["heldout_identity_ids"]}
        split = build_complete_path_fold_records(records, heldout_ids=heldout)
        identity_overlap = sorted(
            set(split["fit_identity_ids"]) & set(split["heldout_identity_ids"])
        )
        if identity_overlap:
            raise RuntimeError("V16 fold identity isolation failed")
        heldout_records = [
            record for record in eligible_records if int(record[1]) in heldout
        ]
        endpoint_receipts: dict[str, Any] = {}
        endpoint_metrics: dict[str, Any] = {}
        endpoint_scores: dict[str, Any] = {}
        query_identities = None
        for endpoint in endpoint_names:
            satr_enabled = endpoint == "satr"
            _set_seed(42)
            model, binding = _build_fold_model(
                config,
                signal_cfg,
                fold_index=fold_index,
                split=split,
            )
            training = _fit_endpoint(
                model,
                split["train_records"],
                config,
                satr_enabled=satr_enabled,
            )
            total_steps += int(training["optimizer_steps"])
            checkpoint_path = output_dir / f"fold_{fold_index}_{endpoint}_final.pth"
            torch.save(
                {
                    "schema_version": "trifusion-v16-q1-fold-endpoint-final-v1",
                    "fold": fold_index,
                    "endpoint": endpoint,
                    "fit_identity_ids": list(split["fit_identity_ids"]),
                    "heldout_identity_ids": list(split["heldout_identity_ids"]),
                    "nonbaseline_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                        if not name.startswith("baseline.")
                    },
                },
                checkpoint_path,
            )
            loader = _eval_loader(heldout_records, config)
            collected = _collect_endpoint(
                model,
                loader,
                num_query=len(heldout_records),
            )
            endpoint_metrics[endpoint] = {
                name: _metric(collected["scores"][name])
                for name in ("baseline_only", *output_names)
            }
            endpoint_scores[endpoint] = collected["scores"]
            for name in output_names:
                score_parts[endpoint][name].append(collected["scores"][name])
            if query_identities is None:
                query_identities = collected["query_identities"]
            elif not np.array_equal(query_identities, collected["query_identities"]):
                raise RuntimeError("V16 endpoint query identities differ")
            endpoint_receipts[endpoint] = {
                "training": training,
                "binding": binding,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "metrics_percent": endpoint_metrics[endpoint],
                "feature_widths": collected["feature_widths"],
                "exact_signal_prefix": collected["exact_signal_prefix"],
            }
            del model
            torch.cuda.empty_cache()

        no_satr_training = endpoint_receipts["no_satr"]["training"]
        satr_training = endpoint_receipts["satr"]["training"]
        paired = {
            "initial_state": (
                no_satr_training["initial_state_sha256"]
                == satr_training["initial_state_sha256"]
            ),
            "trainable_names": (
                no_satr_training["trainable_names_sha256"]
                == satr_training["trainable_names_sha256"]
            ),
            "sample_order": (
                no_satr_training["sample_order_sha256"]
                == satr_training["sample_order_sha256"]
            ),
            "transformed_batches": (
                no_satr_training["first_eight_batch_receipts"]
                == satr_training["first_eight_batch_receipts"]
            ),
            "seed_contract": (
                no_satr_training["seed_contract"] == satr_training["seed_contract"]
            ),
            "optimizer_steps": (
                no_satr_training["optimizer_steps"]
                == satr_training["optimizer_steps"]
            ),
        }
        fold_gains = {
            name: endpoint_metrics["satr"][name]["mAP"]
            - endpoint_metrics["no_satr"][name]["mAP"]
            for name in output_names
        }
        fold_receipt = {
            "fold": fold_index,
            "fit_identity_count": len(split["fit_identity_ids"]),
            "heldout_identity_count": len(split["heldout_identity_ids"]),
            "eligible_heldout_identity_count": len(
                {int(record[1]) for record in heldout_records}
            ),
            "eligible_heldout_queries": len(heldout_records),
            "identity_overlap": identity_overlap,
            "endpoints": endpoint_receipts,
            "paired_integrity": paired,
            "matched_mAP_gains": fold_gains,
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }
        fold_receipts.append(fold_receipt)
        identities_parts.append(torch.from_numpy(query_identities).long())
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": "trifusion-v16-q1-progress-v1",
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

    combined = {
        endpoint: {
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
        for endpoint, outputs in score_parts.items()
    }
    aggregate_metrics = {
        endpoint: {name: _metric(scores) for name, scores in outputs.items()}
        for endpoint, outputs in combined.items()
    }
    aggregate_gains = {
        name: aggregate_metrics["satr"][name]["mAP"]
        - aggregate_metrics["no_satr"][name]["mAP"]
        for name in output_names
    }
    fused_query_gain = torch.from_numpy(
        combined["satr"]["fused"].average_precision
        - combined["no_satr"]["fused"].average_precision
    ).float()
    bootstrap = identity_cluster_bootstrap_lower_bound(
        fused_query_gain,
        torch.cat(identities_parts),
        seed=int(config["GATES"]["Q1_BOOTSTRAP_SEED"]),
        resamples=int(config["GATES"]["Q1_BOOTSTRAP_RESAMPLES"]),
    )
    integrity = {
        "fold_isolation": all(not receipt["identity_overlap"] for receipt in fold_receipts),
        "paired_initial_state": all(
            receipt["paired_integrity"]["initial_state"] for receipt in fold_receipts
        ),
        "paired_trainable_names": all(
            receipt["paired_integrity"]["trainable_names"] for receipt in fold_receipts
        ),
        "paired_sample_order": all(
            receipt["paired_integrity"]["sample_order"] for receipt in fold_receipts
        ),
        "paired_transformed_batches": all(
            receipt["paired_integrity"]["transformed_batches"]
            for receipt in fold_receipts
        ),
        "paired_seed_contract": all(
            receipt["paired_integrity"]["seed_contract"] for receipt in fold_receipts
        ),
        "paired_optimizer_steps": all(
            receipt["paired_integrity"]["optimizer_steps"] for receipt in fold_receipts
        ),
        "frozen_state_unchanged": all(
            endpoint["training"]["frozen_state_unchanged"]
            for receipt in fold_receipts
            for endpoint in receipt["endpoints"].values()
        ),
        "zero_overflow": all(
            endpoint["training"]["overflow_events"] == 0
            for receipt in fold_receipts
            for endpoint in receipt["endpoints"].values()
        ),
        "exact_signal_prefix": all(
            endpoint["exact_signal_prefix"]
            for receipt in fold_receipts
            for endpoint in receipt["endpoints"].values()
        ),
        "pre_bn_evaluation": True,
        "final_epoch_only": True,
        "access_boundary": True,
    }
    gate = evaluate_v16_q1_gate(
        fold_map_gains=tuple(
            receipt["matched_mAP_gains"] for receipt in fold_receipts
        ),
        weighted_fused_gain_map=aggregate_gains["fused"],
        fused_gain_bootstrap_lower_bound=bootstrap.lower_bound * 100.0,
        aggregate_branch_gains={
            name: aggregate_gains[name]
            for name in ("cnn", "transformer", "mamba")
        },
        aggregate_satr_map={
            name: aggregate_metrics["satr"][name]["mAP"]
            for name in output_names
        },
        initial_coverages=tuple(m0["initial_coverages"]),
        integrity=integrity,
    )
    return {
        "schema_version": "trifusion-v16-q1-result-v1",
        "status": "PASS" if gate["passed"] else "FAIL",
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
        "initial_coverages": m0["initial_coverages"],
        "integrity": integrity,
        "gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "optimizer_steps": total_steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "m0_summary": str(m0_summary_path),
        "m0_summary_sha256": _sha256(m0_summary_path),
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "d1_executed": False,
    }


def _d1_records(config: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    from tools.run_signal_baseline_dev import _records_for_ids

    protocol_path = _project_path(config["DATA"]["DEV_PROTOCOL"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    train_ids = {int(value) for value in protocol["train_ids"]}
    dev_ids = {int(value) for value in protocol["dev_ids"]}
    if len(train_ids) != 141 or len(dev_ids) != 30 or train_ids & dev_ids:
        raise ValueError("V16 D1 requires the frozen 141-fit/30-dev registry")
    root = Path(config["DATA"]["DATASET_ROOT"]).resolve() / "train_171"
    train_records = list(_records_for_ids(root, train_ids, relabel=True))
    dev_records = list(_records_for_ids(root, dev_ids, relabel=False))
    if len(train_records) != int(protocol["counts"]["train_triplets"]):
        raise ValueError("V16 D1 fit records differ from protocol")
    if len(dev_records) != int(protocol["counts"]["dev_triplets"]):
        raise ValueError("V16 D1 dev records differ from protocol")
    return train_records, dev_records


def _build_d1_model(
    config: dict[str, Any],
    signal_cfg: Any,
    train_records: list[Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    from tools.build_v12_complete_path_oof_targets import _build_signal_teacher

    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.signal_preserving_v16_builder import (
        build_signal_preserving_trifusion_v16,
    )

    signal_model = _build_signal_teacher(
        signal_cfg,
        num_classes=len({record[1] for record in train_records}),
        camera_num=len({record[2] for record in train_records}),
        view_num=len({record[3] for record in train_records}),
    )
    signal_path = Path(config["SIGNAL"]["CHECKPOINT"]).resolve()
    signal_model.load_state_dict(
        torch.load(signal_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    model_config = config["MODEL"]
    build = build_signal_preserving_trifusion_v16(
        signal_model,
        signal_checkpoint_sha256=config["SIGNAL"]["CHECKPOINT_SHA256"],
        num_classes=len({record[1] for record in train_records}),
        feature_width=int(model_config["FEATURE_WIDTH"]),
        semantic_width=int(model_config["SEMANTIC_WIDTH"]),
        grid_size=tuple(model_config["GRID_SIZE"]),
        branch_after_block=int(model_config["BRANCH_AFTER_BLOCK"]),
        adapter_width=int(model_config["ADAPTER_WIDTH"]),
        expert_modal_width=int(model_config["EXPERT_MODAL_WIDTH"]),
        scale_init=float(model_config["SCALE_INIT"]),
        gradient_checkpointing=bool(model_config["GRADIENT_CHECKPOINTING"]),
    )
    phase_a_path = Path(config["INITIALIZATION"]["V8_PHASE_A_CHECKPOINT"]).resolve()
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=True)
    build.model.load_state_dict(phase_a["model_state_dict"], strict=True)
    return build.model.cuda(), {
        "build_provenance": dict(build.provenance),
        "signal_checkpoint": str(signal_path),
        "signal_checkpoint_sha256": config["SIGNAL"]["CHECKPOINT_SHA256"],
        "v8_phase_a_checkpoint": str(phase_a_path),
        "v8_phase_a_checkpoint_sha256": config["INITIALIZATION"][
            "V8_PHASE_A_CHECKPOINT_SHA256"
        ],
    }


def _run_d1(
    contract: dict[str, Any],
    output_dir: Path,
    q1_summary_path: Path,
) -> dict[str, Any]:
    import torch

    from tools.build_v12_complete_path_oof_targets import _configure_signal, _eval_loader
    from tools.run_signal_preserving_v5 import _set_seed

    q1 = json.loads(q1_summary_path.read_text(encoding="utf-8"))
    if not bool(q1.get("next_phase_authorized")):
        raise ValueError("V16 D1 requires a Q1 receipt that authorizes the next phase")
    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    train_records, dev_records = _d1_records(config)
    _set_seed(42)
    model, binding = _build_d1_model(config, signal_cfg, train_records)
    frozen_before = _frozen_state_sha256(model)
    training = _fit_endpoint(
        model,
        train_records,
        config,
        satr_enabled=True,
    )
    frozen_after_training = _frozen_state_sha256(model)
    final_state_before_save = _model_state_sha256(model)
    checkpoint_path = output_dir / "final_model.pth"
    torch.save(
        {
            "schema_version": "trifusion-v16-d1-final-v1",
            "epoch": int(config["OPTIMIZATION"]["MAX_EPOCHS"]),
            "model_state_dict": model.state_dict(),
            "q1_summary_sha256": _sha256(q1_summary_path),
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final_state_after_reload = _model_state_sha256(model)
    strict_reload = (
        not load_result.missing_keys
        and not load_result.unexpected_keys
        and final_state_before_save == final_state_after_reload
    )
    loader = _eval_loader(dev_records, config)
    collected = _collect_endpoint(model, loader, num_query=len(dev_records))
    metrics = {
        name: _metric(scores) for name, scores in collected["scores"].items()
    }
    frozen_after_evaluation = _frozen_state_sha256(model)
    frozen_unchanged = (
        frozen_before == frozen_after_training == frozen_after_evaluation
    )
    gate = evaluate_v16_d1_gate(
        metrics_percent=metrics,
        minimum_fused_map=float(config["GATES"]["DEV_MIN_MAP"]),
        v8_phase_b_map=float(config["COMPARATORS"]["V8_PHASE_B_MAP"]),
        strict_reload=strict_reload,
        frozen_state_unchanged=frozen_unchanged,
    )
    return {
        "schema_version": "trifusion-v16-d1-result-v1",
        "status": "PASS" if gate["passed"] else "FAIL",
        "claim_supported": bool(gate["passed"]),
        "seed": 42,
        "epochs_completed": training["epochs"],
        "model_selection": "none_final_epoch_only",
        "training": training,
        "metrics_percent": metrics,
        "feature_widths": collected["feature_widths"],
        "exact_signal_prefix": collected["exact_signal_prefix"],
        "dev_gate": gate,
        "binding": binding,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "strict_reload": strict_reload,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after_training": frozen_after_training,
        "frozen_state_sha256_after_evaluation": frozen_after_evaluation,
        "frozen_state_unchanged": frozen_unchanged,
        "v8_phase_b_comparator_percent": {
            "mAP": float(config["COMPARATORS"]["V8_PHASE_B_MAP"]),
            "Rank-1": float(config["COMPARATORS"]["V8_PHASE_B_RANK1"]),
        },
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "q1_summary": str(q1_summary_path),
        "q1_summary_sha256": _sha256(q1_summary_path),
        "dev_access_count": 1,
        "official_test_access_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"V16 output already exists: {args.output_dir}")
    contract = _load_contract(args.config.resolve())
    args.output_dir.mkdir(parents=True)
    if args.stage == "m0":
        result = _run_m0(contract, args.output_dir)
    elif args.stage == "q1":
        result = _run_q1(contract, args.output_dir, args.m0_summary.resolve())
    else:
        result = _run_d1(contract, args.output_dir, args.q1_summary.resolve())
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
        raise RuntimeError("V16 M0 failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("m0", "q1", "d1"), required=True)
    parser.add_argument("--m0-summary", type=Path)
    parser.add_argument("--q1-summary", type=Path)
    args = parser.parse_args()
    if args.stage == "q1" and args.m0_summary is None:
        parser.error("--m0-summary is required for Q1")
    if args.stage == "d1" and args.q1_summary is None:
        parser.error("--q1-summary is required for D1")
    return args


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "V16_INITIAL_COVERAGE_MAX",
    "V16_INITIAL_COVERAGE_MIN",
    "evaluate_v16_d1_gate",
    "evaluate_v16_m0_gate",
    "evaluate_v16_q1_gate",
    "run",
]
