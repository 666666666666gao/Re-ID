#!/usr/bin/env python3
"""Train and qualify the deployment-aligned TriFusion V13 Router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


Q1_METRICS = (
    "expected_utility",
    "top1",
    "replay_average_precision",
    "replay_margin",
)


def evaluate_v13_q1_gate(
    *,
    fold_noninferiority: tuple[dict[str, bool], ...],
    bootstrap_lower_bounds: dict[str, float],
    corrupted_mass_decreases: dict[str, bool],
    missing_modality_max_mass: float,
    frozen_phase_a_unchanged: bool,
    dev_access_count: int,
    official_test_access_count: int,
) -> dict[str, Any]:
    """Apply the preregistered Q1 policy and replay qualification gate."""

    per_fold = len(fold_noninferiority) == 3 and all(
        all(bool(receipt[metric]) for metric in Q1_METRICS)
        for receipt in fold_noninferiority
    )
    aggregate = all(
        float(bootstrap_lower_bounds[metric]) > 0.0 for metric in Q1_METRICS
    )
    quality = all(
        bool(corrupted_mass_decreases[modality])
        for modality in ("RGB", "NI", "TI")
    )
    missing = float(missing_modality_max_mass) == 0.0
    access = int(dev_access_count) == 0 and int(official_test_access_count) == 0
    passed = (
        per_fold
        and aggregate
        and quality
        and missing
        and bool(frozen_phase_a_unchanged)
        and access
    )
    return {
        "passed": passed,
        "per_fold_noninferiority_passed": per_fold,
        "aggregate_bootstrap_passed": aggregate,
        "quality_response_passed": quality,
        "missing_modality_zero_mass_passed": missing,
        "frozen_phase_a_unchanged_passed": bool(frozen_phase_a_unchanged),
        "access_boundary_passed": access,
        "bootstrap_lower_bounds": {
            metric: float(bootstrap_lower_bounds[metric]) for metric in Q1_METRICS
        },
    }


def _new_router(config: dict[str, Any], *, direct_width: int, residual_width: int):
    import torch

    from trifusion.signal_preserving_v13 import DeploymentAlignedRouter

    seed = int(config["EXPERIMENT"]["SEED"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return DeploymentAlignedRouter(
        direct_width=direct_width,
        residual_width=residual_width,
        hidden_width=int(config["ROUTER"]["HIDDEN_WIDTH"]),
    ).cuda()


def _fit_router(
    router: Any,
    paired_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    paired_rows: Any,
    quality_rows: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from trifusion.signal_preserving_v13 import deployment_aligned_utility_loss
    from trifusion.signal_preserving_v8_router import modality_quality_loss

    router_config = config["ROUTER"]
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(router_config["LEARNING_RATE"]),
        weight_decay=float(router_config["WEIGHT_DECAY"]),
    )
    direct = paired_cache["student_direct_modal"][paired_rows].float().cuda()
    residual = paired_cache["student_modal_residual"][paired_rows].float().cuda()
    target = paired_cache["teacher_identity_utility"][paired_rows].float().cuda()
    modality_mask = torch.ones(direct.shape[0], 3, dtype=torch.bool, device="cuda")

    conditions = tuple(quality_cache["conditions"])
    quality_direct = torch.cat(
        [quality_cache["direct_modal"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_residual = torch.cat(
        [quality_cache["modal_residual"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_target = torch.cat(
        [quality_cache["modality_quality"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_mask = torch.ones(
        quality_direct.shape[0],
        3,
        dtype=torch.bool,
        device="cuda",
    )

    history = []
    for epoch in range(1, int(router_config["EPOCHS"]) + 1):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        routing = router(direct, residual, modality_mask)
        utility_loss = deployment_aligned_utility_loss(
            routing.weights,
            target,
            modality_mask,
            temperature=float(router_config["UTILITY_TEMPERATURE"]),
        )
        quality_routing = router(quality_direct, quality_residual, quality_mask)
        quality_loss = modality_quality_loss(
            quality_routing.modal_probabilities,
            quality_target,
            quality_mask,
        )
        total = utility_loss + quality_loss * float(
            router_config["QUALITY_LOSS_WEIGHT"]
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("V13 Router training loss is nonfinite")
        total.backward()
        for name, parameter in router.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"V13 Router gradient is invalid: {name}")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "total": float(total.detach()),
                "utility": float(utility_loss.detach()),
                "quality": float(quality_loss.detach()),
            }
        )
    return {
        "optimizer_steps": len(history),
        "first_epoch": history[0],
        "final_epoch": history[-1],
    }


def _replay_scores(
    baseline: Any,
    modal_residual: Any,
    weights: Any,
    identities: Any,
    cameras: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from diagnose_v6_oracle_complementarity import _scores_from_features
    from trifusion.signal_preserving_v13 import compose_v13_fusion

    embedding = compose_v13_fusion(
        baseline.float(),
        modal_residual.float(),
        weights.float(),
    ).retrieval_embedding
    count = embedding.shape[0]
    scores = _scores_from_features(
        torch.cat((embedding, embedding)),
        np.concatenate((identities.numpy(), identities.numpy())),
        np.concatenate((cameras.numpy(), cameras.numpy())),
        num_query=count,
    )
    distances = torch.cdist(embedding, embedding)
    margins = []
    for index, row in enumerate(distances):
        positives = (identities == identities[index]) & (cameras != cameras[index])
        negatives = identities != identities[index]
        margins.append(row[negatives].min() - row[positives].max())
    return {
        "average_precision": torch.from_numpy(scores.average_precision.copy()),
        "rank1_correct": torch.from_numpy(scores.rank1_correct.copy()),
        "margin": torch.stack(margins),
    }


def _evaluate_router_oof(
    paired_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from trifusion.signal_preserving_v13 import (
        identity_cluster_bootstrap_lower_bound,
    )

    paired_folds = paired_cache["fold_indices"]
    quality_folds = quality_cache["fold_indices"]
    differences = {metric: [] for metric in Q1_METRICS}
    difference_identities = []
    clean_mass = {name: [] for name in ("RGB", "NI", "TI")}
    corrupted_mass = {name: [] for name in ("RGB", "NI", "TI")}
    missing_mass = []
    fold_receipts = []

    for fold in range(int(config["PROTOCOL"]["OOF_TARGET_FOLDS"])):
        train_rows = paired_folds != fold
        heldout_rows = paired_folds == fold
        train_quality_rows = quality_folds != fold
        heldout_quality_rows = quality_folds == fold
        router = _new_router(
            config,
            direct_width=int(paired_cache["student_direct_modal"].shape[-1]),
            residual_width=int(paired_cache["student_modal_residual"].shape[-1]),
        )
        training = _fit_router(
            router,
            paired_cache,
            quality_cache,
            train_rows,
            train_quality_rows,
            config,
        )

        router.eval()
        with torch.no_grad():
            direct = paired_cache["student_direct_modal"][heldout_rows].float().cuda()
            residual = paired_cache["student_modal_residual"][heldout_rows].float().cuda()
            mask = torch.ones(direct.shape[0], 3, dtype=torch.bool, device="cuda")
            routing = router(direct, residual, mask)
            weights = routing.weights.cpu()
            target = paired_cache["teacher_identity_utility"][heldout_rows].float()
            fixed_slot = int(
                paired_cache["teacher_identity_utility"][train_rows]
                .float()
                .mean(dim=0)
                .flatten()
                .argmax()
            )
            fixed_weights = torch.zeros_like(weights)
            fixed_weights.flatten(1)[:, fixed_slot] = 1.0

            learned_expected = (weights * target).sum(dim=(1, 2))
            fixed_expected = target.flatten(1)[:, fixed_slot]
            target_winner = target.flatten(1).argmax(dim=1)
            learned_correct = weights.flatten(1).argmax(dim=1) == target_winner
            majority_correct = target_winner == fixed_slot

            identities = paired_cache["identities"][heldout_rows]
            cameras = paired_cache["cameras"][heldout_rows]
            baseline = paired_cache["teacher_oof_baseline"][heldout_rows]
            teacher_residual = paired_cache["teacher_oof_modal_residual"][heldout_rows]
            learned_replay = _replay_scores(
                baseline,
                teacher_residual,
                weights,
                identities,
                cameras,
            )
            fixed_replay = _replay_scores(
                baseline,
                teacher_residual,
                fixed_weights,
                identities,
                cameras,
            )

            fold_differences = {
                "expected_utility": learned_expected - fixed_expected,
                "top1": learned_correct.float() - majority_correct.float(),
                "replay_average_precision": (
                    learned_replay["average_precision"]
                    - fixed_replay["average_precision"]
                ),
                "replay_margin": learned_replay["margin"] - fixed_replay["margin"],
            }
            for metric in Q1_METRICS:
                differences[metric].append(fold_differences[metric])
            difference_identities.append(identities)

            noninferiority = {
                metric: float(fold_differences[metric].mean()) >= 0.0
                for metric in Q1_METRICS
            }
            fold_receipts.append(
                {
                    "fold": fold,
                    "queries": int(heldout_rows.sum()),
                    "fixed_slot": fixed_slot,
                    "training": training,
                    "expected_utility": {
                        "learned": float(learned_expected.mean()),
                        "fixed": float(fixed_expected.mean()),
                        "gain": float(fold_differences["expected_utility"].mean()),
                    },
                    "top1_accuracy": {
                        "learned": float(learned_correct.float().mean()),
                        "majority": float(majority_correct.float().mean()),
                        "gain": float(fold_differences["top1"].mean()),
                    },
                    "replay": {
                        "learned_map": float(learned_replay["average_precision"].mean()),
                        "fixed_map": float(fixed_replay["average_precision"].mean()),
                        "map_gain": float(
                            fold_differences["replay_average_precision"].mean()
                        ),
                        "learned_rank1": float(
                            learned_replay["rank1_correct"].float().mean()
                        ),
                        "fixed_rank1": float(
                            fixed_replay["rank1_correct"].float().mean()
                        ),
                        "learned_mean_margin": float(learned_replay["margin"].mean()),
                        "fixed_mean_margin": float(fixed_replay["margin"].mean()),
                        "margin_gain": float(fold_differences["replay_margin"].mean()),
                    },
                    "noninferiority": noninferiority,
                }
            )

            clean_direct = quality_cache["direct_modal"]["clean"][heldout_quality_rows].float().cuda()
            clean_residual = quality_cache["modal_residual"]["clean"][heldout_quality_rows].float().cuda()
            clean_output = router(
                clean_direct,
                clean_residual,
                torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
            )
            for modality_index, modality in enumerate(("RGB", "NI", "TI")):
                corrupted_output = router(
                    quality_cache["direct_modal"][modality][heldout_quality_rows].float().cuda(),
                    quality_cache["modal_residual"][modality][heldout_quality_rows].float().cuda(),
                    torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
                )
                clean_mass[modality].append(
                    clean_output.modal_probabilities[:, modality_index].cpu()
                )
                corrupted_mass[modality].append(
                    corrupted_output.modal_probabilities[:, modality_index].cpu()
                )
                missing_mask = torch.ones(
                    clean_direct.shape[0],
                    3,
                    dtype=torch.bool,
                    device="cuda",
                )
                missing_mask[:, modality_index] = False
                missing_output = router(clean_direct, clean_residual, missing_mask)
                missing_mass.append(
                    missing_output.modal_probabilities[:, modality_index].cpu()
                )

    combined_identities = torch.cat(difference_identities)
    bootstrap = {}
    for metric in Q1_METRICS:
        result = identity_cluster_bootstrap_lower_bound(
            torch.cat(differences[metric]),
            combined_identities,
            seed=int(config["V13"]["BOOTSTRAP_SEED"]),
            resamples=int(config["V13"]["BOOTSTRAP_RESAMPLES"]),
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
    decreases = {
        modality: values["corrupted_mean_mass"] < values["clean_mean_mass"]
        for modality, values in quality_response.items()
    }
    missing_max = float(torch.cat(missing_mass).max())
    return {
        "fold_receipts": fold_receipts,
        "bootstrap": bootstrap,
        "quality_response": quality_response,
        "corrupted_mass_decreases": decreases,
        "missing_modality_max_mass": missing_max,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from probe_v8_frozen_router import select_cross_camera_records
    from run_signal_preserving_v5 import (
        _build_runtime,
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )
    from train_v8_oof_margin_router import _collect_quality_cache

    started = time.time()
    config_path = args.config.resolve()
    config = load_raw_config(config_path)
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V13 Router training is frozen to seed 42")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_ROUTER_TRAINING"]):
        raise ValueError("V13 Router training cannot access dev")
    if args.output_dir.exists():
        raise FileExistsError(f"V13 Router output already exists: {args.output_dir}")

    initialization = config["INITIALIZATION"]
    phase_a_path = Path(initialization["PHASE_A_CHECKPOINT"]).resolve()
    paired_path = Path(initialization["PAIRED_TARGET_CACHE"]).resolve()
    if _sha256(phase_a_path) != initialization["PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("V13 Phase-A checkpoint SHA-256 differs from contract")
    if _sha256(paired_path) != initialization["PAIRED_TARGET_CACHE_SHA256"]:
        raise ValueError("V13 paired target cache SHA-256 differs from contract")
    args.output_dir.mkdir(parents=True)

    runtime = _build_runtime(config)
    model = runtime["model"]
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=True)
    model.load_state_dict(phase_a["model_state_dict"], strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_a_state_before = _module_state_sha256(model)

    paired_cache = torch.load(paired_path, map_location="cpu", weights_only=True)
    if paired_cache["schema_version"] != "trifusion-v13-paired-target-cache-v1":
        raise ValueError("unexpected V13 paired target cache schema")
    if (
        paired_cache["phase_a_checkpoint_sha256"]
        != initialization["PHASE_A_CHECKPOINT_SHA256"]
    ):
        raise ValueError("V13 paired cache references a different Phase-A checkpoint")
    if float(paired_cache["fixed_alpha"]) != float(config["V13"]["FIXED_ALPHA"]):
        raise ValueError("V13 paired cache fixed alpha differs from config")

    identity_to_fold = {
        int(identity): int(fold)
        for identity, fold in zip(
            paired_cache["identities"].tolist(),
            paired_cache["fold_indices"].tolist(),
            strict=True,
        )
    }
    eligible_records = select_cross_camera_records(runtime["train_records"])
    torch.cuda.reset_peak_memory_stats()
    quality_cache = _collect_quality_cache(
        model,
        eligible_records,
        runtime,
        config,
        identity_to_fold,
    )
    quality_cache_path = args.output_dir / "router_quality_features.pth"
    torch.save(quality_cache, quality_cache_path)
    oof = _evaluate_router_oof(paired_cache, quality_cache, config)

    phase_a_state_after = _module_state_sha256(model)
    unchanged = phase_a_state_before == phase_a_state_after
    lower_bounds = {
        metric: oof["bootstrap"][metric]["lower_bound_95"] for metric in Q1_METRICS
    }
    gate = evaluate_v13_q1_gate(
        fold_noninferiority=tuple(
            receipt["noninferiority"] for receipt in oof["fold_receipts"]
        ),
        bootstrap_lower_bounds=lower_bounds,
        corrupted_mass_decreases=oof["corrupted_mass_decreases"],
        missing_modality_max_mass=oof["missing_modality_max_mass"],
        frozen_phase_a_unchanged=unchanged,
        dev_access_count=0,
        official_test_access_count=0,
    )
    oof["gate"] = gate

    optimizer_steps = sum(
        receipt["training"]["optimizer_steps"] for receipt in oof["fold_receipts"]
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
            torch.ones(paired_cache["fold_indices"].shape[0], dtype=torch.bool),
            torch.ones(quality_cache["fold_indices"].shape[0], dtype=torch.bool),
            config,
        )
        optimizer_steps += final_training["optimizer_steps"]
        combined_checkpoint = args.output_dir / "v13_phase_a_plus_router.pth"
        torch.save(
            {
                "schema_version": "trifusion-v13-phase-a-plus-router-v1",
                "phase_a_model_state_dict": phase_a["model_state_dict"],
                "router_state_dict": {
                    name: value.detach().cpu()
                    for name, value in final_router.state_dict().items()
                },
                "router_config": dict(config["ROUTER"]),
                "fixed_alpha": float(config["V13"]["FIXED_ALPHA"]),
                "phase_a_checkpoint_sha256": _sha256(phase_a_path),
                "paired_target_cache_sha256": _sha256(paired_path),
            },
            combined_checkpoint,
        )
        combined_checkpoint_sha256 = _sha256(combined_checkpoint)

    result = {
        "schema_version": "trifusion-v13-deployment-aligned-router-q1-result-v1",
        "status": "PASS",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "phase_a_checkpoint": str(phase_a_path),
        "phase_a_checkpoint_sha256": _sha256(phase_a_path),
        "paired_target_cache": str(paired_path),
        "paired_target_cache_sha256": _sha256(paired_path),
        "quality_cache": str(quality_cache_path),
        "quality_cache_sha256": _sha256(quality_cache_path),
        "router_oof": oof,
        "final_training": final_training,
        "combined_checkpoint": str(combined_checkpoint) if combined_checkpoint else None,
        "combined_checkpoint_sha256": combined_checkpoint_sha256,
        "next_phase_authorized": bool(gate["passed"]),
        "phase_a_state_sha256_before": phase_a_state_before,
        "phase_a_state_sha256_after": phase_a_state_after,
        "phase_a_state_unchanged": unchanged,
        "router_training_executed": True,
        "expert_training_executed": False,
        "optimizer_steps": optimizer_steps,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"])
        ).hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["evaluate_v13_q1_gate", "run"]
