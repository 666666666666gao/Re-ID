#!/usr/bin/env python3
"""Read-only terminal diagnostics for Signal-preserving V7."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


def summarize_v7_routing(
    baseline: torch.Tensor,
    fused: torch.Tensor,
    branches: Mapping[str, torch.Tensor],
    router_weights: torch.Tensor,
    modal_probabilities: torch.Tensor,
    expert_probabilities: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, Any]:
    """Summarize bounded residual energy and the joint hierarchical router."""

    baseline_width = baseline.shape[1]
    baseline_norm = baseline.norm(dim=1).clamp_min(1e-12)
    fused_suffix = fused[:, baseline_width:]
    branch_suffixes = {
        expert: branches[expert][:, baseline_width:] for expert in EXPERTS
    }
    pairwise = {}
    for left_index, left in enumerate(EXPERTS):
        for right in EXPERTS[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = float(
                F.cosine_similarity(
                    branch_suffixes[left], branch_suffixes[right], dim=1
                ).mean()
            )
    joint_entropy = -(
        router_weights.clamp_min(1e-12)
        * router_weights.clamp_min(1e-12).log()
    ).sum(dim=(1, 2)) / math.log(len(EXPERTS) * len(MODALITIES))
    modal_entropy = -(
        modal_probabilities.clamp_min(1e-12)
        * modal_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(len(MODALITIES))
    expert_entropy = -(
        expert_probabilities.clamp_min(1e-12)
        * expert_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(len(EXPERTS))
    return {
        "samples": int(baseline.shape[0]),
        "baseline_width": int(baseline_width),
        "fused_width": int(fused.shape[1]),
        "fused_suffix_to_baseline_norm_mean": float(
            (fused_suffix.norm(dim=1) / baseline_norm).mean()
        ),
        "branch_suffix_to_baseline_norm_mean": {
            expert: float(
                (branch_suffixes[expert].norm(dim=1) / baseline_norm).mean()
            )
            for expert in EXPERTS
        },
        "branch_suffix_pairwise_cosine_mean": pairwise,
        "joint_router_normalized_entropy_mean": float(joint_entropy.mean()),
        "joint_router_normalized_entropy_std": float(
            joint_entropy.std(unbiased=False)
        ),
        "modal_normalized_entropy_mean": float(modal_entropy.mean()),
        "modal_normalized_entropy_std": float(modal_entropy.std(unbiased=False)),
        "conditional_expert_entropy_mean": float(expert_entropy.mean()),
        "conditional_expert_entropy_std": float(
            expert_entropy.std(unbiased=False)
        ),
        "modal_probability_mean": dict(
            zip(
                MODALITIES,
                [float(value) for value in modal_probabilities.mean(dim=0)],
                strict=True,
            )
        ),
        "modal_probability_std": dict(
            zip(
                MODALITIES,
                [
                    float(value)
                    for value in modal_probabilities.std(dim=0, unbiased=False)
                ],
                strict=True,
            )
        ),
        "router_weight_mean": {
            expert: [float(value) for value in router_weights[:, index].mean(dim=0)]
            for index, expert in enumerate(EXPERTS)
        },
        "router_weight_std": {
            expert: [
                float(value)
                for value in router_weights[:, index].std(dim=0, unbiased=False)
            ]
            for index, expert in enumerate(EXPERTS)
        },
        "alpha": {
            "mean": float(alpha.mean()),
            "std": float(alpha.std(unbiased=False)),
            "minimum": float(alpha.min()),
            "maximum": float(alpha.max()),
        },
    }


def summarize_pairwise_query_outcomes(fused: Any, mamba: Any) -> dict[str, Any]:
    """Count query-wise AP and Rank-1 wins for fused versus Mamba."""

    ap_delta = fused.average_precision - mamba.average_precision
    return {
        "queries": int(ap_delta.size),
        "mean_ap_delta_percent": float(ap_delta.mean() * 100.0),
        "fused_ap_wins": int((ap_delta > 1e-12).sum()),
        "mamba_ap_wins": int((ap_delta < -1e-12).sum()),
        "ap_ties": int((np.abs(ap_delta) <= 1e-12).sum()),
        "fused_rank1_repairs": int(
            (fused.rank1_correct & ~mamba.rank1_correct).sum()
        ),
        "fused_rank1_breaks": int(
            (~fused.rank1_correct & mamba.rank1_correct).sum()
        ),
    }


def _slot_alignment(routing: Any) -> dict[str, Any]:
    target = routing.target_weights
    predicted = routing.predicted_weights
    target_top = target.flatten(1).argmax(dim=1)
    predicted_top = predicted.flatten(1).argmax(dim=1)
    utilities = routing.utilities
    return {
        "samples": int(target.shape[0]),
        "top_slot_agreement": float((target_top == predicted_top).float().mean()),
        "target_expert_mass_mean": {
            expert: float(target[:, index].sum(dim=1).mean())
            for index, expert in enumerate(EXPERTS)
        },
        "predicted_expert_mass_mean": {
            expert: float(predicted[:, index].sum(dim=1).mean())
            for index, expert in enumerate(EXPERTS)
        },
        "utility_by_expert": {
            expert: {
                "mean": float(utilities[:, index].mean()),
                "maximum": float(utilities[:, index].max()),
                "positive_fraction": float((utilities[:, index] > 0).float().mean()),
            }
            for index, expert in enumerate(EXPERTS)
        },
        "alpha_target_mean": float(routing.alpha_target.mean()),
    }


def _phase_summary(history: dict[str, Any]) -> dict[str, Any]:
    phases = {}
    for phase in ("router_warmup", "joint"):
        rows = [row for row in history["epochs"] if row["phase"] == phase]
        best = max(rows, key=lambda row: row["metrics_percent"]["fused"]["mAP"])
        phases[phase] = {
            "best_epoch": int(best["epoch"]),
            "best_metrics_percent": best["metrics_percent"],
        }
    phases["epoch60"] = history["epochs"][-1]
    return phases


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")

    from diagnose_signal_preserving_v5 import (
        _distance_change,
        _parameter_updates,
        _sha256,
    )
    from diagnose_v6_oracle_complementarity import (
        _percent,
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V7,
        _build_runtime,
        _evaluate_v7_quality_response,
        _load_config,
        _training_batch,
    )
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = _load_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V7:
        raise ValueError("V7 terminal diagnostic requires the V7 architecture")
    runtime = _build_runtime(config)
    from trifusion.signal_preserving_v7 import marginal_gain_router_loss

    model = runtime["model"]
    initial = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    feature_names = (
        "baseline_only",
        "fused",
        "cnn",
        "transformer",
        "mamba",
        "residual_cnn",
        "residual_transformer",
        "residual_mamba",
    )
    collected = {name: [] for name in feature_names}
    query_router = {name: [] for name in ("weights", "modal", "expert", "alpha")}
    identities_all = []
    camera_ids_all = []
    remaining_queries = len(runtime["dev_records"])
    for images, identities, camera_ids, camera_labels, _view_ids, _paths in runtime[
        "eval_loader"
    ]:
        images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(
                {
                    "images": images,
                    "modality_mask": torch.ones(
                        camera_labels.shape[0], 3, dtype=torch.bool, device="cuda"
                    ),
                    "camera_ids": camera_labels,
                },
                return_aux=True,
            )
        batch_features = {
            "baseline_only": output.baseline_embedding,
            "fused": output.fused_embedding,
            **dict(output.branch_embeddings),
            **{
                f"residual_{expert}": output.residual_embeddings[expert]
                for expert in EXPERTS
            },
        }
        for name in feature_names:
            collected[name].append(batch_features[name].float().cpu())
        if remaining_queries:
            take = min(remaining_queries, output.baseline_embedding.shape[0])
            query_router["weights"].append(output.router_weights[:take].float().cpu())
            query_router["modal"].append(
                output.modal_probabilities[:take].float().cpu()
            )
            query_router["expert"].append(
                output.expert_probabilities[:take].float().cpu()
            )
            query_router["alpha"].append(output.alpha[:take].float().cpu())
            remaining_queries -= take
        identities_all.extend(np.asarray(identities).tolist())
        camera_ids_all.extend(np.asarray(camera_ids).tolist())

    features = {name: torch.cat(values) for name, values in collected.items()}
    routing_tensors = {
        name: torch.cat(values) for name, values in query_router.items()
    }
    identities_array = np.asarray(identities_all)
    cameras_array = np.asarray(camera_ids_all)
    num_query = len(runtime["dev_records"])
    scores = {
        name: _scores_from_features(
            tensor,
            identities_array,
            cameras_array,
            num_query=num_query,
        )
        for name, tensor in features.items()
    }
    query_features = {name: tensor[:num_query] for name, tensor in features.items()}
    branches = {expert: query_features[expert] for expert in EXPERTS}

    raw_train_batch = next(iter(runtime["train_loader"]))
    quality_response = _evaluate_v7_quality_response(model, raw_train_batch, config)
    clean_batch, labels = _training_batch(raw_train_batch)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["OPTIMIZATION"]["AMP"]),
    ):
        train_output = model(clean_batch, return_aux=True)
        routing = marginal_gain_router_loss(
            train_output.baseline_embedding,
            train_output.contribution_embeddings,
            train_output.router_weights,
            train_output.alpha,
            labels,
            modality_mask=train_output.modality_mask,
            alpha_max=train_output.alpha_max,
            utility_temperature=float(config["LOSS"]["UTILITY_TEMPERATURE"]),
            alpha_gain_scale=float(config["LOSS"]["ALPHA_GAIN_SCALE"]),
        )

    history_path = checkpoint_path.parent / "history.json"
    run_summary_path = checkpoint_path.parent / "run_summary.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    branch_names = ("baseline_only", "cnn", "transformer", "mamba")
    result = {
        "schema_version": "trifusion-signal-preserving-v7-terminal-diagnostic-v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config_sha256": _sha256(config_path),
        "diagnostic_sha256": _sha256(Path(__file__).resolve()),
        "source_run_summary_sha256": _sha256(run_summary_path),
        "source_history_sha256": _sha256(history_path),
        "samples": num_query,
        "routing": summarize_v7_routing(
            query_features["baseline_only"],
            query_features["fused"],
            branches,
            routing_tensors["weights"],
            routing_tensors["modal"],
            routing_tensors["expert"],
            routing_tensors["alpha"],
        ),
        "distance_change_from_baseline": {
            output: _distance_change(
                query_features["baseline_only"], query_features[output]
            )
            for output in ("fused", *EXPERTS)
        },
        "branch_oracle": summarize_oracle_complementarity(
            {name: scores[name] for name in branch_names}
        ),
        "residual_only_oracle": summarize_oracle_complementarity(
            {expert: scores[f"residual_{expert}"] for expert in EXPERTS}
        ),
        "fused_vs_mamba": summarize_pairwise_query_outcomes(
            scores["fused"], scores["mamba"]
        ),
        "fused_metrics_percent": {
            "mAP": _percent(scores["fused"].average_precision.mean()),
            "Rank-1": _percent(scores["fused"].rank1_correct.mean()),
        },
        "selected_checkpoint_quality_response_train_batch": {
            "scope": "one deterministic 64-sample fit batch",
            **quality_response,
        },
        "selected_checkpoint_slot_alignment_train_batch": {
            "scope": "one deterministic 64-sample fit batch",
            **_slot_alignment(routing),
        },
        "phase_summary": _phase_summary(history),
        "trainable_parameter_updates_from_v6_initialization": _parameter_updates(
            initial, model
        ),
        "oracle_uses_ground_truth": True,
        "oracle_is_deployment_result": False,
        "training_executed": False,
        "optimizer_steps": 0,
        "official_test_access_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["summarize_pairwise_query_outcomes", "summarize_v7_routing"]
