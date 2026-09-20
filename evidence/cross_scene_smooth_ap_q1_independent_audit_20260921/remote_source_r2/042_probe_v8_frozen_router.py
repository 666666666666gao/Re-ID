#!/usr/bin/env python3
"""Frozen-feature utility-routing probe for a possible TriFusion V8."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LeastSquaresUtilityTeacher:
    coefficients: torch.Tensor


def fit_least_squares_utility_teacher(
    features: torch.Tensor,
    utilities: torch.Tensor,
) -> LeastSquaresUtilityTeacher:
    design = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    coefficients = torch.linalg.lstsq(design, utilities).solution
    return LeastSquaresUtilityTeacher(coefficients=coefficients)


def predict_utility_probabilities(
    teacher: LeastSquaresUtilityTeacher,
    features: torch.Tensor,
) -> torch.Tensor:
    design = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    return torch.softmax(design @ teacher.coefficients, dim=1)


def compose_equal_energy_fused(
    baseline: torch.Tensor,
    contributions: torch.Tensor,
    expert_probabilities: torch.Tensor,
    modal_probabilities: torch.Tensor,
) -> torch.Tensor:
    joint = expert_probabilities[:, :, None] * modal_probabilities[:, None, :]
    routed = (contributions * joint[..., None]).flatten(1)
    activated = F.normalize(routed, dim=1) * baseline.norm(dim=1, keepdim=True)
    return torch.cat((baseline, activated), dim=1)


def summarize_winner_alignment(
    utilities: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    majority_expert_index: int,
) -> dict[str, Any]:
    targets = utilities.argmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    accuracy = float((predicted == targets).float().mean())
    majority_accuracy = float((targets == majority_expert_index).float().mean())
    return {
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "beats_majority": accuracy > majority_accuracy,
        "target_distribution": [
            int((targets == index).sum()) / targets.numel() for index in range(3)
        ],
        "predicted_distribution": [
            int((predicted == index).sum()) / predicted.numel() for index in range(3)
        ],
    }


def select_cross_camera_records(records: list[tuple[Any, int, int, int]]):
    cameras_by_identity: dict[int, set[int]] = {}
    for _paths, identity, camera, _view in records:
        cameras_by_identity.setdefault(identity, set()).add(camera)
    eligible = {
        identity
        for identity, cameras in cameras_by_identity.items()
        if len(cameras) >= 2
    }
    return [record for record in records if record[1] in eligible]


def _fit_eval_loader(runtime: dict[str, Any], config: dict[str, Any]):
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    from torch.utils.data import DataLoader

    records = select_cross_camera_records(runtime["train_records"])
    transform = runtime["eval_loader"].dataset.transform
    return DataLoader(
        ImageDataset(records + records, transform),
        batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        collate_fn=val_collate_fn,
    )


def _collect_split(model: Any, loader: Any) -> dict[str, Any]:
    tensors: dict[str, list[torch.Tensor]] = {
        "baseline": [],
        "fused": [],
        "contributions": [],
        "modal_probabilities": [],
        "current_expert_probabilities": [],
        "quality_features": [],
        "residual_cnn": [],
        "residual_transformer": [],
        "residual_mamba": [],
    }
    identities: list[int] = []
    cameras: list[int] = []
    model.eval()
    for images, batch_identities, batch_cameras, camera_labels, _views, _paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
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
        tensors["baseline"].append(output.baseline_embedding.float().cpu())
        tensors["fused"].append(output.fused_embedding.float().cpu())
        tensors["contributions"].append(
            output.contribution_embeddings.float().cpu()
        )
        tensors["modal_probabilities"].append(
            output.modal_probabilities.float().cpu()
        )
        tensors["current_expert_probabilities"].append(
            output.router_weights.sum(dim=2).float().cpu()
        )
        tensors["quality_features"].append(
            torch.cat((output.reliability.r, output.reliability.u), dim=1)
            .flatten(1)
            .float()
            .cpu()
        )
        for expert in ("cnn", "transformer", "mamba"):
            tensors[f"residual_{expert}"].append(
                output.residual_embeddings[expert].float().cpu()
            )
        identities.extend(np.asarray(batch_identities).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    return {
        **{name: torch.cat(values) for name, values in tensors.items()},
        "identities": np.asarray(identities),
        "cameras": np.asarray(cameras),
    }


def _retrieval_scores(split: dict[str, Any], *, num_query: int) -> dict[str, Any]:
    from diagnose_v6_oracle_complementarity import _scores_from_features

    names = ("baseline", "fused", "residual_cnn", "residual_transformer", "residual_mamba")
    return {
        name: _scores_from_features(
            split[name],
            split["identities"],
            split["cameras"],
            num_query=num_query,
        )
        for name in names
    }


def _utilities(scores: dict[str, Any]) -> torch.Tensor:
    baseline = scores["baseline"].average_precision
    return torch.from_numpy(
        np.stack(
            [
                scores[f"residual_{expert}"].average_precision - baseline
                for expert in ("cnn", "transformer", "mamba")
            ],
            axis=1,
        )
    ).float()


def _score_summary(scores: Any) -> dict[str, float]:
    return {
        "mAP": float(scores.average_precision.mean() * 100.0),
        "Rank-1": float(scores.rank1_correct.mean() * 100.0),
    }


def _score_embedding(
    embedding: torch.Tensor,
    split: dict[str, Any],
    *,
    num_query: int,
) -> dict[str, float]:
    from diagnose_v6_oracle_complementarity import _scores_from_features

    return _score_summary(
        _scores_from_features(
            embedding,
            split["identities"],
            split["cameras"],
            num_query=num_query,
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"probe output already exists: {args.output}")

    from run_signal_preserving_v5 import _build_runtime, _load_config, _sha256

    config = _load_config(args.config.resolve())
    runtime = _build_runtime(config)
    torch.backends.cudnn.benchmark = False
    model = runtime["model"]
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    fit_loader = _fit_eval_loader(runtime, config)
    fit = _collect_split(model, fit_loader)
    dev = _collect_split(model, runtime["eval_loader"])
    fit_queries = len(fit_loader.dataset) // 2
    dev_queries = len(runtime["dev_records"])
    fit_scores = _retrieval_scores(fit, num_query=fit_queries)
    dev_scores = _retrieval_scores(dev, num_query=dev_queries)
    fit_utilities = _utilities(fit_scores)
    dev_utilities = _utilities(dev_scores)

    teacher = fit_least_squares_utility_teacher(
        fit["quality_features"][:fit_queries], fit_utilities
    )
    fit_probabilities = predict_utility_probabilities(
        teacher, fit["quality_features"]
    )
    dev_probabilities = predict_utility_probabilities(
        teacher, dev["quality_features"]
    )
    majority_expert_index = int(
        torch.bincount(fit_utilities.argmax(dim=1), minlength=3).argmax()
    )

    uniform_probabilities = torch.full_like(dev_probabilities, 1.0 / 3.0)
    equal_energy_embeddings = {
        "uniform": compose_equal_energy_fused(
            dev["baseline"],
            dev["contributions"],
            uniform_probabilities,
            dev["modal_probabilities"],
        ),
        "current_router": compose_equal_energy_fused(
            dev["baseline"],
            dev["contributions"],
            dev["current_expert_probabilities"],
            dev["modal_probabilities"],
        ),
        "fit_utility_teacher": compose_equal_energy_fused(
            dev["baseline"],
            dev["contributions"],
            dev_probabilities,
            dev["modal_probabilities"],
        ),
    }
    equal_energy_metrics = {
        name: _score_embedding(embedding, dev, num_query=dev_queries)
        for name, embedding in equal_energy_embeddings.items()
    }
    fixed_metrics = {name: _score_summary(value) for name, value in dev_scores.items()}
    best_fixed_map = max(value["mAP"] for value in fixed_metrics.values())
    teacher_map = equal_energy_metrics["fit_utility_teacher"]["mAP"]
    dev_alignment = summarize_winner_alignment(
        dev_utilities,
        dev_probabilities[:dev_queries],
        majority_expert_index=majority_expert_index,
    )
    result = {
        "schema_version": "trifusion-v8-frozen-router-probe-v1",
        "status": "PASS",
        "scope": "fit-identity utility teacher evaluated on disjoint dev identities",
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "probe_sha256": _sha256(Path(__file__).resolve()),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "fit_queries": fit_queries,
        "fit_identities": int(np.unique(fit["identities"][:fit_queries]).size),
        "dev_queries": dev_queries,
        "dev_identities": int(np.unique(dev["identities"][:dev_queries]).size),
        "quality_feature_width": int(fit["quality_features"].shape[1]),
        "majority_expert": ("cnn", "transformer", "mamba")[majority_expert_index],
        "fit_alignment": summarize_winner_alignment(
            fit_utilities,
            fit_probabilities[:fit_queries],
            majority_expert_index=majority_expert_index,
        ),
        "dev_alignment": dev_alignment,
        "current_router_dev_alignment": summarize_winner_alignment(
            dev_utilities,
            dev["current_expert_probabilities"][:dev_queries],
            majority_expert_index=majority_expert_index,
        ),
        "fixed_metrics_percent": fixed_metrics,
        "equal_energy_metrics_percent": equal_energy_metrics,
        "v8_frozen_router_probe_gate": {
            "passed": bool(
                dev_alignment["beats_majority"]
                and teacher_map > best_fixed_map
                and teacher_map >= 65.0
            ),
            "teacher_beats_fit_majority_on_dev": dev_alignment["beats_majority"],
            "teacher_fused_mAP": teacher_map,
            "best_fixed_mAP": best_fixed_map,
            "minimum_mAP": 65.0,
        },
        "model_training_executed": False,
        "optimizer_steps": 0,
        "official_test_access_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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


__all__ = [
    "LeastSquaresUtilityTeacher",
    "compose_equal_energy_fused",
    "fit_least_squares_utility_teacher",
    "predict_utility_probabilities",
    "run",
    "select_cross_camera_records",
    "summarize_winner_alignment",
]
