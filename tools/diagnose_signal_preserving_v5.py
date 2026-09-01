#!/usr/bin/env python3
"""Read-only diagnostics for completed Signal-preserving checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


EXPERTS = ("cnn", "transformer", "mamba")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_collaboration(
    baseline: torch.Tensor,
    fused: torch.Tensor,
    branches: Mapping[str, torch.Tensor],
    router_weights: torch.Tensor,
    *,
    reliability_r: torch.Tensor,
    reliability_u: torch.Tensor,
) -> dict[str, Any]:
    """Summarize the observable scale and diversity of appended V5 features."""

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
    entropy = -(
        router_weights.clamp_min(1e-12) * router_weights.clamp_min(1e-12).log()
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
        "router_normalized_entropy_mean": float(entropy.mean()),
        "router_normalized_entropy_std": float(entropy.std(unbiased=False)),
        "router_weight_mean": {
            expert: [
                float(value)
                for value in router_weights[:, index].mean(dim=0)
            ]
            for index, expert in enumerate(EXPERTS)
        },
        "router_weight_std": {
            expert: [
                float(value)
                for value in router_weights[:, index].std(dim=0, unbiased=False)
            ]
            for index, expert in enumerate(EXPERTS)
        },
        "reliability_r_mean": float(reliability_r.mean()),
        "reliability_r_std": float(reliability_r.std(unbiased=False)),
        "reliability_u_mean": float(reliability_u.mean()),
        "reliability_u_std": float(reliability_u.std(unbiased=False)),
    }


def _distance_change(
    baseline: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    baseline_distances = torch.cdist(
        F.normalize(baseline.float(), dim=1),
        F.normalize(baseline.float(), dim=1),
    )
    candidate_distances = torch.cdist(
        F.normalize(candidate.float(), dim=1),
        F.normalize(candidate.float(), dim=1),
    )
    delta = (candidate_distances - baseline_distances).abs()
    correlation = torch.corrcoef(
        torch.stack((baseline_distances.flatten(), candidate_distances.flatten()))
    )[0, 1]
    diagonal = torch.eye(
        baseline.shape[0], dtype=torch.bool, device=baseline_distances.device
    )
    baseline_neighbors = baseline_distances.masked_fill(diagonal, float("inf")).topk(
        10, largest=False
    ).indices
    candidate_neighbors = candidate_distances.masked_fill(diagonal, float("inf")).topk(
        10, largest=False
    ).indices
    overlaps = []
    for baseline_row, candidate_row in zip(
        baseline_neighbors, candidate_neighbors, strict=True
    ):
        overlaps.append(
            torch.isin(baseline_row, candidate_row).float().mean()
        )
    return {
        "distance_absolute_delta_mean": float(delta.mean()),
        "distance_absolute_delta_max": float(delta.max()),
        "distance_pearson_correlation": float(correlation),
        "top10_neighbor_overlap_mean": float(torch.stack(overlaps).mean()),
    }


def _parameter_group(name: str) -> str:
    for expert in EXPERTS:
        if name.startswith(f"encoder.experts.{expert}."):
            return expert
    if name.startswith("encoder.collaborator."):
        return "relay"
    if name.startswith("encoder.reliability_gate."):
        return "reliability"
    if name.startswith("fusion."):
        return "fusion"
    return "retrieval_heads"


def _parameter_updates(
    initial: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, dict[str, float]]:
    accumulators: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if name not in initial:
            continue
        group = _parameter_group(name)
        values = accumulators.setdefault(
            group,
            {"initial_squared_norm": 0.0, "delta_squared_norm": 0.0},
        )
        initial_tensor = initial[name]
        trained_tensor = parameter.detach().cpu().float()
        values["initial_squared_norm"] += float(initial_tensor.square().sum())
        values["delta_squared_norm"] += float(
            (trained_tensor - initial_tensor).square().sum()
        )
    return {
        group: {
            "initial_norm": math.sqrt(values["initial_squared_norm"]),
            "delta_norm": math.sqrt(values["delta_squared_norm"]),
            "delta_to_initial_norm": math.sqrt(values["delta_squared_norm"])
            / max(math.sqrt(values["initial_squared_norm"]), 1e-12),
        }
        for group, values in accumulators.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V5,
        ARCHITECTURE_V6,
        _build_runtime,
        _load_config,
        receipt_schema,
    )

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = _load_config(config_path)
    architecture = str(config["MODEL"]["ARCHITECTURE"])
    runtime = _build_runtime(config)
    model = runtime["model"]
    initial = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    collected: dict[str, list[torch.Tensor]] = {
        "baseline_only": [],
        "fused": [],
        "cnn": [],
        "transformer": [],
        "mamba": [],
        "router": [],
        "r": [],
        "u": [],
    }
    if architecture == ARCHITECTURE_V6:
        for expert in EXPERTS:
            collected[f"residual_{expert}"] = []
        from utils.metrics import R1_mAP_eval

        residual_evaluators = {
            expert: R1_mAP_eval(
                len(runtime["dev_records"]), max_rank=50, feat_norm="yes"
            )
            for expert in EXPERTS
        }
    else:
        residual_evaluators = {}
    remaining = len(runtime["dev_records"])
    for images, identities, camera_ids, camera_ids_batch, _view_ids, paths in runtime[
        "eval_loader"
    ]:
        images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
        camera_labels = camera_ids_batch.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_labels.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_labels,
        }
        with torch.no_grad():
            output = model(batch, return_aux=True)
        if architecture == ARCHITECTURE_V6:
            for expert in EXPERTS:
                residual_evaluators[expert].update(
                    (
                        output.residual_embeddings[expert],
                        identities,
                        camera_ids,
                        paths,
                    )
                )
        if remaining == 0:
            continue
        take = min(remaining, output.baseline_embedding.shape[0])
        collected["baseline_only"].append(output.baseline_embedding[:take].float().cpu())
        collected["fused"].append(output.fused_embedding[:take].float().cpu())
        for expert in EXPERTS:
            collected[expert].append(
                output.branch_embeddings[expert][:take].float().cpu()
            )
            if architecture == ARCHITECTURE_V6:
                collected[f"residual_{expert}"].append(
                    output.residual_embeddings[expert][:take].float().cpu()
                )
        collected["router"].append(output.router_weights[:take].float().cpu())
        collected["r"].append(output.reliability.r[:take].float().cpu())
        collected["u"].append(output.reliability.u[:take].float().cpu())
        remaining -= take
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    branches = {expert: tensors[expert] for expert in EXPERTS}
    collaboration = summarize_collaboration(
        tensors["baseline_only"],
        tensors["fused"],
        branches,
        tensors["router"],
        reliability_r=tensors["r"],
        reliability_u=tensors["u"],
    )
    distance_change = {
        output: _distance_change(tensors["baseline_only"], tensors[output])
        for output in ("fused", *EXPERTS)
    }
    run_summary = checkpoint_path.parent / "run_summary.json"
    result = {
        "schema_version": receipt_schema(architecture, "diagnostic"),
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(
            Path(__file__).resolve().parent / "run_signal_preserving_v5.py"
        ),
        "diagnostic_sha256": _sha256(Path(__file__).resolve()),
        "samples": int(tensors["baseline_only"].shape[0]),
        "collaboration": collaboration,
        "distance_change_from_baseline": distance_change,
        "trainable_parameter_updates": _parameter_updates(initial, model),
        "source_run_summary_sha256": _sha256(run_summary),
        "official_test_access_count": 0,
        "training_executed": False,
        "optimizer_steps": 0,
    }
    if architecture == ARCHITECTURE_V5:
        scale_values = [
            float(value)
            for value in torch.sigmoid(
                model.fusion.residual_scale_logits.detach().cpu()
            )
        ]
        result["trained_residual_scales"] = dict(
            zip(EXPERTS, scale_values, strict=True)
        )
    elif architecture == ARCHITECTURE_V6:
        result["residual_energy_activation"] = {
            "has_free_scale": False,
            "target_suffix_to_baseline_norm": 1.0,
        }
        residual_metrics = {}
        for expert, evaluator in residual_evaluators.items():
            cmc, mean_ap, *_ = evaluator.compute()
            residual_metrics[expert] = {
                "mAP": float(mean_ap * 100.0),
                "Rank-1": float(cmc[0] * 100.0),
                "Rank-5": float(cmc[4] * 100.0),
                "Rank-10": float(cmc[9] * 100.0),
            }
        result["residual_only_metrics_percent"] = residual_metrics
    else:
        raise ValueError(f"unsupported Signal-preserving architecture: {architecture}")
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


__all__ = ["summarize_collaboration"]
