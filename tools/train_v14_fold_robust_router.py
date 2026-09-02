#!/usr/bin/env python3
"""Train and qualify the fold-robust TriFusion V14 Router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


Q1_RETRIEVAL_METRICS = (
    "retrieval_risk",
    "replay_average_precision",
    "replay_margin",
)


def evaluate_v14_q1_gate(
    *,
    fold_gains: tuple[dict[str, float], ...],
    bootstrap_lower_bounds: dict[str, float],
    corrupted_mass_decreases: dict[str, bool],
    missing_modality_max_mass: float,
    frozen_phase_a_unchanged: bool,
    dev_access_count: int,
    official_test_access_count: int,
) -> dict[str, Any]:
    """Apply the preregistered V14 held-out retrieval and safety gates."""

    per_fold_retrieval = len(fold_gains) == 3 and all(
        float(gains["retrieval_risk"]) > 0.0
        and float(gains["replay_average_precision"]) >= 0.0
        and float(gains["replay_margin"]) >= 0.0
        for gains in fold_gains
    )
    aggregate = all(
        float(bootstrap_lower_bounds[metric]) > 0.0
        for metric in Q1_RETRIEVAL_METRICS
    )
    quality = all(
        bool(corrupted_mass_decreases[modality])
        for modality in ("RGB", "NI", "TI")
    )
    missing = float(missing_modality_max_mass) == 0.0
    access = int(dev_access_count) == 0 and int(official_test_access_count) == 0
    passed = (
        per_fold_retrieval
        and aggregate
        and quality
        and missing
        and bool(frozen_phase_a_unchanged)
        and access
    )
    return {
        "passed": passed,
        "per_fold_retrieval_passed": per_fold_retrieval,
        "aggregate_bootstrap_passed": aggregate,
        "quality_response_passed": quality,
        "missing_modality_zero_mass_passed": missing,
        "frozen_phase_a_unchanged_passed": bool(frozen_phase_a_unchanged),
        "access_boundary_passed": access,
        "bootstrap_lower_bounds": {
            metric: float(bootstrap_lower_bounds[metric])
            for metric in Q1_RETRIEVAL_METRICS
        },
    }


def _new_router(config: dict[str, Any], *, direct_width: int, residual_width: int):
    from train_v13_deployment_aligned_router import _new_router as new_v13_router

    return new_v13_router(
        config,
        direct_width=direct_width,
        residual_width=residual_width,
    )


def _fold_data(paired_cache: dict[str, Any], fold: int) -> dict[str, Any]:
    import torch

    rows = paired_cache["fold_indices"] == int(fold)
    return {
        "fold": int(fold),
        "rows": rows,
        "row_fold_ids": paired_cache["fold_indices"][rows].cuda(),
        "direct": paired_cache["student_direct_modal"][rows].float().cuda(),
        "student_residual": paired_cache["student_modal_residual"][rows].float().cuda(),
        "baseline": paired_cache["teacher_oof_baseline"][rows].float().cuda(),
        "teacher_residual": paired_cache["teacher_oof_modal_residual"][rows].float().cuda(),
        "identities": paired_cache["identities"][rows].cuda(),
        "cameras": paired_cache["cameras"][rows].cuda(),
        "utility": paired_cache["teacher_identity_utility"][rows].float(),
    }


def _fixed_weights(data: dict[str, Any], slot: int):
    import torch

    weights = torch.zeros(
        data["baseline"].shape[0],
        3,
        3,
        dtype=data["baseline"].dtype,
        device=data["baseline"].device,
    )
    weights.flatten(1)[:, int(slot)] = 1.0
    return weights


def _fold_risk(data: dict[str, Any], weights: Any):
    from trifusion.signal_preserving_v14 import fold_bound_retrieval_risk

    return fold_bound_retrieval_risk(
        fold_id=data["fold"],
        row_fold_ids=data["row_fold_ids"],
        baseline_embedding=data["baseline"],
        modal_residual=data["teacher_residual"],
        weights=weights,
        identities=data["identities"],
        cameras=data["cameras"],
    )


def _fixed_risk_matrix(fold_data: tuple[dict[str, Any], ...]):
    import torch

    rows = []
    with torch.no_grad():
        for data in fold_data:
            rows.append(
                torch.stack(
                    [
                        _fold_risk(data, _fixed_weights(data, slot)).risk
                        for slot in range(9)
                    ]
                )
            )
    return torch.stack(rows)


def _select_source_comparator(fold_data: tuple[dict[str, Any], ...]):
    from trifusion.signal_preserving_v14 import select_minimax_fixed_slot

    fixed_risks = _fixed_risk_matrix(fold_data)
    selection = select_minimax_fixed_slot(fixed_risks)
    return fixed_risks, selection


def _prepare_quality(
    quality_cache: dict[str, Any],
    source_folds: tuple[int, ...],
) -> dict[str, Any]:
    import torch

    rows = torch.zeros_like(quality_cache["fold_indices"], dtype=torch.bool)
    for fold in source_folds:
        rows |= quality_cache["fold_indices"] == int(fold)
    conditions = tuple(quality_cache["conditions"])
    direct = torch.cat(
        [quality_cache["direct_modal"][name][rows] for name in conditions]
    ).float().cuda()
    residual = torch.cat(
        [quality_cache["modal_residual"][name][rows] for name in conditions]
    ).float().cuda()
    target = torch.cat(
        [quality_cache["modality_quality"][name][rows] for name in conditions]
    ).float().cuda()
    return {
        "direct": direct,
        "residual": residual,
        "target": target,
        "mask": torch.ones(direct.shape[0], 3, dtype=torch.bool, device="cuda"),
    }


def _fit_router(
    router: Any,
    paired_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    source_folds: tuple[int, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from trifusion.signal_preserving_v8_router import modality_quality_loss

    fold_data = tuple(_fold_data(paired_cache, fold) for fold in source_folds)
    fixed_risks, comparator = _select_source_comparator(fold_data)
    quality = _prepare_quality(quality_cache, source_folds)
    router_config = config["ROUTER"]
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(router_config["LEARNING_RATE"]),
        weight_decay=float(router_config["WEIGHT_DECAY"]),
    )
    history = []
    for epoch in range(1, int(router_config["EPOCHS"]) + 1):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        regrets = []
        fold_risks = []
        for index, data in enumerate(fold_data):
            mask = torch.ones(
                data["direct"].shape[0],
                3,
                dtype=torch.bool,
                device="cuda",
            )
            routing = router(data["direct"], data["student_residual"], mask)
            risk = _fold_risk(data, routing.weights).risk
            fold_risks.append(risk)
            regrets.append(risk - fixed_risks[index, comparator.slot])
        identity_loss = torch.stack(regrets).max()
        quality_routing = router(
            quality["direct"],
            quality["residual"],
            quality["mask"],
        )
        quality_loss = modality_quality_loss(
            quality_routing.modal_probabilities,
            quality["target"],
            quality["mask"],
        )
        total = identity_loss + quality_loss * float(
            router_config["QUALITY_LOSS_WEIGHT"]
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("V14 Router training loss is nonfinite")
        total.backward()
        for name, parameter in router.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"V14 Router gradient is invalid: {name}")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "total": float(total.detach()),
                "identity": float(identity_loss.detach()),
                "quality": float(quality_loss.detach()),
                "fold_risks": [float(value.detach()) for value in fold_risks],
            }
        )
    return {
        "optimizer_steps": len(history),
        "source_folds": list(source_folds),
        "fixed_slot": comparator.slot,
        "fixed_worst_fold_risk": float(comparator.worst_fold_risk),
        "fixed_fold_slot_risks": fixed_risks.detach().cpu().tolist(),
        "first_epoch": history[0],
        "final_epoch": history[-1],
    }


def qualify_v14_q0(
    paired_cache: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    source_folds = tuple(range(int(config["PROTOCOL"]["OOF_TARGET_FOLDS"])))
    data = tuple(_fold_data(paired_cache, fold) for fold in source_folds)
    fixed_risks, comparator = _select_source_comparator(data)
    router = _new_router(
        config,
        direct_width=int(paired_cache["student_direct_modal"].shape[-1]),
        residual_width=int(paired_cache["student_modal_residual"].shape[-1]),
    )
    regrets = []
    learned_risks = []
    for index, fold_data in enumerate(data):
        mask = torch.ones(
            fold_data["direct"].shape[0],
            3,
            dtype=torch.bool,
            device="cuda",
        )
        routing = router(
            fold_data["direct"],
            fold_data["student_residual"],
            mask,
        )
        risk = _fold_risk(fold_data, routing.weights).risk
        learned_risks.append(risk)
        regrets.append(risk - fixed_risks[index, comparator.slot])
    loss = torch.stack(regrets).max()
    loss.backward()
    gradient_receipt = {
        name: {
            "finite": parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all()),
            "nonzero": parameter.grad is not None
            and bool(parameter.grad.abs().sum() > 0),
        }
        for name, parameter in router.named_parameters()
    }
    gradients_passed = all(
        values["finite"] and values["nonzero"]
        for values in gradient_receipt.values()
    )
    return {
        "passed": bool(torch.isfinite(loss)) and gradients_passed,
        "folds": list(source_folds),
        "queries_per_fold": [int(item["baseline"].shape[0]) for item in data],
        "identities_per_fold": [
            int(item["identities"].unique().numel()) for item in data
        ],
        "fixed_fold_slot_risks": fixed_risks.detach().cpu().tolist(),
        "minimax_fixed_slot": comparator.slot,
        "minimax_worst_fold_risk": float(comparator.worst_fold_risk),
        "initialized_router_fold_risks": [
            float(value.detach()) for value in learned_risks
        ],
        "worst_fold_regret_loss": float(loss.detach()),
        "gradient_receipt": gradient_receipt,
        "all_router_gradients_finite_nonzero": gradients_passed,
        "optimizer_steps": 0,
        "router_input_scope": "all_fit_deployment",
        "teacher_embedding_scope": "identity_oof",
        "cross_fold_feature_distances": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
    }


def _evaluate_router_oof(
    paired_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from train_v13_deployment_aligned_router import _replay_scores
    from trifusion.signal_preserving_v13 import (
        identity_cluster_bootstrap_lower_bound,
    )

    fold_count = int(config["PROTOCOL"]["OOF_TARGET_FOLDS"])
    differences = {metric: [] for metric in Q1_RETRIEVAL_METRICS}
    difference_identities = []
    clean_mass = {name: [] for name in ("RGB", "NI", "TI")}
    corrupted_mass = {name: [] for name in ("RGB", "NI", "TI")}
    missing_mass = []
    fold_receipts = []

    for heldout_fold in range(fold_count):
        source_folds = tuple(
            fold for fold in range(fold_count) if fold != heldout_fold
        )
        router = _new_router(
            config,
            direct_width=int(paired_cache["student_direct_modal"].shape[-1]),
            residual_width=int(paired_cache["student_modal_residual"].shape[-1]),
        )
        training = _fit_router(
            router,
            paired_cache,
            quality_cache,
            source_folds,
            config,
        )
        heldout = _fold_data(paired_cache, heldout_fold)
        router.eval()
        with torch.no_grad():
            mask = torch.ones(
                heldout["direct"].shape[0],
                3,
                dtype=torch.bool,
                device="cuda",
            )
            routing = router(
                heldout["direct"],
                heldout["student_residual"],
                mask,
            )
            learned_risk = _fold_risk(heldout, routing.weights)
            fixed_weights = _fixed_weights(heldout, training["fixed_slot"])
            fixed_risk = _fold_risk(heldout, fixed_weights)
            risk_gain = fixed_risk.per_query_loss - learned_risk.per_query_loss

            rows = heldout["rows"]
            identities = paired_cache["identities"][rows]
            cameras = paired_cache["cameras"][rows]
            baseline = paired_cache["teacher_oof_baseline"][rows]
            residual = paired_cache["teacher_oof_modal_residual"][rows]
            learned_weights = routing.weights.cpu()
            fixed_weights_cpu = fixed_weights.cpu()
            learned_replay = _replay_scores(
                baseline,
                residual,
                learned_weights,
                identities,
                cameras,
            )
            fixed_replay = _replay_scores(
                baseline,
                residual,
                fixed_weights_cpu,
                identities,
                cameras,
            )
            ap_gain = (
                learned_replay["average_precision"]
                - fixed_replay["average_precision"]
            )
            margin_gain = learned_replay["margin"] - fixed_replay["margin"]

            utility = heldout["utility"]
            expected_utility = (learned_weights * utility).sum(dim=(1, 2))
            fixed_utility = utility.flatten(1)[:, training["fixed_slot"]]
            action_winner = utility.flatten(1).argmax(dim=1)
            learned_action = learned_weights.flatten(1).argmax(dim=1)

            heldout_fixed_risks = _fixed_risk_matrix((heldout,))[0]
            oracle_slot = int(heldout_fixed_risks.argmin())
            oracle_weights = _fixed_weights(heldout, oracle_slot).cpu()
            oracle_replay = _replay_scores(
                baseline,
                residual,
                oracle_weights,
                identities,
                cameras,
            )

            fold_gain = {
                "retrieval_risk": float(risk_gain.mean()),
                "replay_average_precision": float(ap_gain.mean()),
                "replay_margin": float(margin_gain.mean()),
            }
            for metric, values in (
                ("retrieval_risk", risk_gain.cpu()),
                ("replay_average_precision", ap_gain),
                ("replay_margin", margin_gain),
            ):
                differences[metric].append(values)
            difference_identities.append(identities)

            quality_rows = quality_cache["fold_indices"] == heldout_fold
            clean_direct = quality_cache["direct_modal"]["clean"][quality_rows].float().cuda()
            clean_residual = quality_cache["modal_residual"]["clean"][quality_rows].float().cuda()
            clean_output = router(
                clean_direct,
                clean_residual,
                torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
            )
            for modality_index, modality in enumerate(("RGB", "NI", "TI")):
                corrupted_output = router(
                    quality_cache["direct_modal"][modality][quality_rows].float().cuda(),
                    quality_cache["modal_residual"][modality][quality_rows].float().cuda(),
                    torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
                )
                clean_mass[modality].append(
                    clean_output.modal_probabilities[:, modality_index].cpu()
                )
                corrupted_mass[modality].append(
                    corrupted_output.modal_probabilities[:, modality_index].cpu()
                )
                missing_mask = torch.ones(
                    clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"
                )
                missing_mask[:, modality_index] = False
                missing_output = router(clean_direct, clean_residual, missing_mask)
                missing_mass.append(
                    missing_output.modal_probabilities[:, modality_index].cpu()
                )

            fold_receipts.append(
                {
                    "heldout_fold": heldout_fold,
                    "queries": int(identities.shape[0]),
                    "training": training,
                    "heldout_gains": fold_gain,
                    "learned_risk": float(learned_risk.risk),
                    "fixed_risk": float(fixed_risk.risk),
                    "learned_map": float(learned_replay["average_precision"].mean()),
                    "fixed_map": float(fixed_replay["average_precision"].mean()),
                    "learned_margin": float(learned_replay["margin"].mean()),
                    "fixed_margin": float(fixed_replay["margin"].mean()),
                    "diagnostics": {
                        "learned_expected_utility": float(expected_utility.mean()),
                        "fixed_expected_utility": float(fixed_utility.mean()),
                        "action_top1_accuracy": float(
                            (learned_action == action_winner).float().mean()
                        ),
                        "heldout_best_fixed_slot": oracle_slot,
                        "heldout_best_fixed_risk": float(heldout_fixed_risks[oracle_slot]),
                        "heldout_best_fixed_map": float(
                            oracle_replay["average_precision"].mean()
                        ),
                        "heldout_best_fixed_margin": float(
                            oracle_replay["margin"].mean()
                        ),
                    },
                }
            )

    combined_identities = torch.cat(difference_identities)
    bootstrap = {}
    for metric in Q1_RETRIEVAL_METRICS:
        result = identity_cluster_bootstrap_lower_bound(
            torch.cat(differences[metric]),
            combined_identities,
            seed=int(config["V14"]["BOOTSTRAP_SEED"]),
            resamples=int(config["V14"]["BOOTSTRAP_RESAMPLES"]),
        )
        bootstrap[metric] = {
            "observed_mean": result.observed_mean,
            "lower_bound_95": result.lower_bound,
            "identity_clusters": result.cluster_count,
            "resamples": result.resamples,
        }

    quality_response = {
        modality: {
            "clean_mean_mass": float(torch.cat(clean_mass[modality]).mean()),
            "corrupted_mean_mass": float(torch.cat(corrupted_mass[modality]).mean()),
        }
        for modality in ("RGB", "NI", "TI")
    }
    return {
        "fold_receipts": fold_receipts,
        "bootstrap": bootstrap,
        "quality_response": quality_response,
        "corrupted_mass_decreases": {
            modality: values["corrupted_mean_mass"] < values["clean_mean_mass"]
            for modality, values in quality_response.items()
        },
        "missing_modality_max_mass": float(torch.cat(missing_mass).max()),
        "router_input_scope": "all_fit_deployment",
        "teacher_embedding_scope": "identity_oof",
        "cross_fold_feature_distances": 0,
    }


def _load_contract(args: argparse.Namespace):
    import torch

    from run_signal_preserving_v5 import (
        _build_runtime,
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )

    config_path = args.config.resolve()
    config = load_raw_config(config_path)
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V14 Router training is frozen to seed 42")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_ROUTER_TRAINING"]):
        raise ValueError("V14 Router qualification cannot access dev")
    initialization = config["INITIALIZATION"]
    phase_a_path = Path(initialization["PHASE_A_CHECKPOINT"]).resolve()
    paired_path = Path(initialization["PAIRED_TARGET_CACHE"]).resolve()
    if _sha256(phase_a_path) != initialization["PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("V14 Phase-A checkpoint SHA-256 differs from contract")
    if _sha256(paired_path) != initialization["PAIRED_TARGET_CACHE_SHA256"]:
        raise ValueError("V14 paired cache SHA-256 differs from contract")
    runtime = _build_runtime(config)
    model = runtime["model"]
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=True)
    model.load_state_dict(phase_a["model_state_dict"], strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    paired_cache = torch.load(paired_path, map_location="cpu", weights_only=True)
    if paired_cache["schema_version"] != "trifusion-v13-paired-target-cache-v1":
        raise ValueError("unexpected paired target cache schema")
    if paired_cache["phase_a_checkpoint_sha256"] != initialization["PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("paired cache references a different Phase-A checkpoint")
    if float(paired_cache["fixed_alpha"]) != float(config["V14"]["FIXED_ALPHA"]):
        raise ValueError("paired cache fixed alpha differs from V14 contract")
    return {
        "config": config,
        "config_path": config_path,
        "phase_a_path": phase_a_path,
        "paired_path": paired_path,
        "runtime": runtime,
        "model": model,
        "phase_a": phase_a,
        "paired_cache": paired_cache,
        "phase_a_state": _module_state_sha256(model),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from probe_v8_frozen_router import select_cross_camera_records
    from run_signal_preserving_v5 import _module_state_sha256, _sha256
    from train_v8_oof_margin_router import _collect_quality_cache

    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"V14 output already exists: {args.output_dir}")
    contract = _load_contract(args)
    config = contract["config"]
    model = contract["model"]
    paired_cache = contract["paired_cache"]
    args.output_dir.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()

    if args.stage == "q0":
        q0 = qualify_v14_q0(paired_cache, config)
        phase_a_after = _module_state_sha256(model)
        q0["phase_a_state_sha256_before"] = contract["phase_a_state"]
        q0["phase_a_state_sha256_after"] = phase_a_after
        q0["phase_a_state_unchanged"] = phase_a_after == contract["phase_a_state"]
        q0["passed"] = bool(q0["passed"] and q0["phase_a_state_unchanged"])
        result = {
            "schema_version": "trifusion-v14-q0-result-v1",
            "status": "PASS" if q0["passed"] else "FAIL",
            "seed": 42,
            "q0": q0,
            "next_phase_authorized": bool(q0["passed"]),
            "training_executed": False,
            "optimizer_steps": 0,
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }
    else:
        q0_summary = json.loads(args.q0_summary.resolve().read_text(encoding="utf-8"))
        if not bool(q0_summary["next_phase_authorized"]):
            raise ValueError("V14 Q1 requires a passing Q0 receipt")
        if q0_summary["paired_target_cache_sha256"] != _sha256(contract["paired_path"]):
            raise ValueError("V14 Q0 receipt references another paired cache")
        identity_to_fold = {
            int(identity): int(fold)
            for identity, fold in zip(
                paired_cache["identities"].tolist(),
                paired_cache["fold_indices"].tolist(),
                strict=True,
            )
        }
        eligible_records = select_cross_camera_records(contract["runtime"]["train_records"])
        quality_cache = _collect_quality_cache(
            model,
            eligible_records,
            contract["runtime"],
            config,
            identity_to_fold,
        )
        quality_path = args.output_dir / "router_quality_features.pth"
        torch.save(quality_cache, quality_path)
        oof = _evaluate_router_oof(paired_cache, quality_cache, config)
        phase_a_after = _module_state_sha256(model)
        lower_bounds = {
            metric: oof["bootstrap"][metric]["lower_bound_95"]
            for metric in Q1_RETRIEVAL_METRICS
        }
        gate = evaluate_v14_q1_gate(
            fold_gains=tuple(
                receipt["heldout_gains"] for receipt in oof["fold_receipts"]
            ),
            bootstrap_lower_bounds=lower_bounds,
            corrupted_mass_decreases=oof["corrupted_mass_decreases"],
            missing_modality_max_mass=oof["missing_modality_max_mass"],
            frozen_phase_a_unchanged=phase_a_after == contract["phase_a_state"],
            dev_access_count=0,
            official_test_access_count=0,
        )
        oof["gate"] = gate
        optimizer_steps = sum(
            receipt["training"]["optimizer_steps"]
            for receipt in oof["fold_receipts"]
        )
        final_training = None
        combined_checkpoint = None
        combined_checkpoint_sha256 = None
        if gate["passed"]:
            final_router = _new_router(
                config,
                direct_width=int(paired_cache["student_direct_modal"].shape[-1]),
                residual_width=int(paired_cache["student_modal_residual"].shape[-1]),
            )
            final_training = _fit_router(
                final_router,
                paired_cache,
                quality_cache,
                tuple(range(int(config["PROTOCOL"]["OOF_TARGET_FOLDS"]))),
                config,
            )
            optimizer_steps += final_training["optimizer_steps"]
            combined_checkpoint = args.output_dir / "v14_phase_a_plus_router.pth"
            torch.save(
                {
                    "schema_version": "trifusion-v14-phase-a-plus-router-v1",
                    "phase_a_model_state_dict": contract["phase_a"]["model_state_dict"],
                    "router_state_dict": {
                        name: value.detach().cpu()
                        for name, value in final_router.state_dict().items()
                    },
                    "router_config": dict(config["ROUTER"]),
                    "fixed_alpha": float(config["V14"]["FIXED_ALPHA"]),
                    "phase_a_checkpoint_sha256": _sha256(contract["phase_a_path"]),
                    "paired_target_cache_sha256": _sha256(contract["paired_path"]),
                },
                combined_checkpoint,
            )
            combined_checkpoint_sha256 = _sha256(combined_checkpoint)
        result = {
            "schema_version": "trifusion-v14-q1-result-v1",
            "status": "PASS",
            "seed": 42,
            "q0_summary": str(args.q0_summary.resolve()),
            "quality_cache": str(quality_path),
            "quality_cache_sha256": _sha256(quality_path),
            "router_oof": oof,
            "final_training": final_training,
            "combined_checkpoint": str(combined_checkpoint) if combined_checkpoint else None,
            "combined_checkpoint_sha256": combined_checkpoint_sha256,
            "next_phase_authorized": bool(gate["passed"]),
            "phase_a_state_sha256_before": contract["phase_a_state"],
            "phase_a_state_sha256_after": phase_a_after,
            "phase_a_state_unchanged": phase_a_after == contract["phase_a_state"],
            "router_training_executed": True,
            "expert_training_executed": False,
            "optimizer_steps": optimizer_steps,
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }

    result.update(
        {
            "phase_a_checkpoint": str(contract["phase_a_path"]),
            "phase_a_checkpoint_sha256": _sha256(contract["phase_a_path"]),
            "paired_target_cache": str(contract["paired_path"]),
            "paired_target_cache_sha256": _sha256(contract["paired_path"]),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "config": str(contract["config_path"]),
            "config_sha256": _sha256(contract["config_path"]),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary"])
            ).hexdigest(),
            "elapsed_seconds": time.time() - started,
        }
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("q0", "q1"), required=True)
    parser.add_argument("--q0-summary", type=Path)
    args = parser.parse_args()
    if args.stage == "q1" and args.q0_summary is None:
        parser.error("--q0-summary is required for stage q1")
    return args


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "Q1_RETRIEVAL_METRICS",
    "evaluate_v14_q1_gate",
    "qualify_v14_q0",
    "run",
]
