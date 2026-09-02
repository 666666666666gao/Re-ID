#!/usr/bin/env python3
"""Diagnose V13 target scale, stability, and sample-local observability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = left.norm() * right.norm()
    return float((left * right).sum() / denominator)


def analyze_v13_target_learnability(
    cache: dict[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    utility = cache["teacher_identity_utility"].float()
    identities = cache["identities"]
    folds = cache["fold_indices"]
    direct = cache["student_direct_modal"].float()
    residual = cache["student_modal_residual"].float()
    flat = utility.flatten(1)

    probability = torch.softmax(flat / float(temperature), dim=1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
    normalized_entropy = entropy / np.log(flat.shape[1])
    sorted_utility = flat.sort(dim=1, descending=True).values
    top_gap = sorted_utility[:, 0] - sorted_utility[:, 1]
    winner = flat.argmax(dim=1)

    fold_means = []
    fixed_slots = []
    for fold in range(3):
        mean = flat[folds == fold].mean(dim=0)
        fold_means.append(mean)
        fixed_slots.append(int(mean.argmax()))
    fold_rank_correlations = {}
    for left in range(3):
        for right in range(left + 1, 3):
            fold_rank_correlations[f"fold_{left}_vs_{right}"] = float(
                np.corrcoef(
                    _rank(fold_means[left].numpy()),
                    _rank(fold_means[right].numpy()),
                )[0, 1]
            )

    identity_majority_correct = []
    identity_majority_count = 0
    for identity in identities.unique(sorted=True):
        rows = identities == identity
        counts = torch.bincount(winner[rows], minlength=flat.shape[1])
        identity_majority_correct.append(counts.max().float() / rows.sum())
        identity_majority_count += int(counts.max())

    centered = flat - flat.mean(dim=1, keepdim=True)
    centered = F.normalize(centered, dim=1)
    similarity = centered @ centered.T
    off_diagonal = ~torch.eye(flat.shape[0], dtype=torch.bool)
    same_identity = identities[:, None] == identities[None, :]
    within_identity = similarity[off_diagonal & same_identity]
    between_identity = similarity[off_diagonal & ~same_identity]

    residual_norm = residual.norm(dim=-1)
    direct_by_slot = direct[:, None].expand_as(residual)
    direct_residual_cosine = F.cosine_similarity(
        residual,
        direct_by_slot,
        dim=-1,
    )
    norm_correlations = torch.empty(3, 3)
    cosine_correlations = torch.empty(3, 3)
    for expert in range(3):
        for modality in range(3):
            target = utility[:, expert, modality]
            norm_correlations[expert, modality] = _correlation(
                residual_norm[:, expert, modality],
                target,
            )
            cosine_correlations[expert, modality] = _correlation(
                direct_residual_cosine[:, expert, modality],
                target,
            )

    quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "query_count": int(flat.shape[0]),
        "identity_count": int(identities.unique().numel()),
        "utility": {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "top1_minus_top2_quantiles": torch.quantile(top_gap, quantiles).tolist(),
        },
        "distillation_target": {
            "temperature": float(temperature),
            "normalized_entropy_mean": float(normalized_entropy.mean()),
            "normalized_entropy_min": float(normalized_entropy.min()),
            "normalized_entropy_max": float(normalized_entropy.max()),
            "mean_max_probability": float(probability.max(dim=1).values.mean()),
            "uniform_probability": 1.0 / flat.shape[1],
        },
        "cross_fold_slot_semantics": {
            "fixed_slots": fixed_slots,
            "slot_means": [mean.tolist() for mean in fold_means],
            "rank_correlations": fold_rank_correlations,
        },
        "identity_stability": {
            "identity_majority_winner_macro_accuracy": float(
                torch.stack(identity_majority_correct).mean()
            ),
            "identity_majority_winner_micro_accuracy": (
                identity_majority_count / flat.shape[0]
            ),
            "within_identity_centered_utility_cosine": float(within_identity.mean()),
            "between_identity_centered_utility_cosine": float(between_identity.mean()),
        },
        "sample_local_observability": {
            "residual_norm_utility_correlation": norm_correlations.tolist(),
            "direct_residual_cosine_utility_correlation": cosine_correlations.tolist(),
            "max_abs_residual_norm_correlation": float(norm_correlations.abs().max()),
            "max_abs_direct_residual_cosine_correlation": float(
                cosine_correlations.abs().max()
            ),
        },
        "training_executed": False,
        "optimizer_steps": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_path = Path(config["INITIALIZATION"]["PAIRED_TARGET_CACHE"]).resolve()
    expected_sha = config["INITIALIZATION"]["PAIRED_TARGET_CACHE_SHA256"]
    if _sha256(cache_path) != expected_sha:
        raise ValueError("V13 paired cache SHA-256 differs from config")
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    result = analyze_v13_target_learnability(
        cache,
        temperature=float(config["ROUTER"]["UTILITY_TEMPERATURE"]),
    )
    result.update(
        {
            "schema_version": "trifusion-v13-target-learnability-diagnostic-v1",
            "seed": int(config["EXPERIMENT"]["SEED"]),
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config.resolve()),
            "cache": str(cache_path),
            "cache_sha256": expected_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary"])
            ).hexdigest(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["analyze_v13_target_learnability", "run"]
