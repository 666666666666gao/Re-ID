#!/usr/bin/env python3
"""Replace saturated OOF AP labels with continuous identity-margin targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


def per_query_identity_margin(
    distances: np.ndarray,
    *,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray,
    gallery_cameras: np.ndarray,
) -> np.ndarray:
    """Return nearest-negative minus farthest-positive distance per query."""

    values = []
    for query_index, row in enumerate(np.asarray(distances)):
        same_identity = gallery_ids == query_ids[query_index]
        same_camera = gallery_cameras == query_cameras[query_index]
        positives = same_identity & ~same_camera
        negatives = ~same_identity
        if not np.any(positives) or not np.any(negatives):
            raise RuntimeError("identity margin requires positives and negatives")
        values.append(float(row[negatives].min() - row[positives].max()))
    return np.asarray(values, dtype=np.float64)


def evaluate_margin_target_gate(
    *,
    expert_winner_counts: dict[str, int],
    modality_winner_counts: dict[str, int],
    oracle_margin_gain: float,
) -> dict[str, Any]:
    expert_diversity = all(int(expert_winner_counts[name]) > 0 for name in EXPERTS)
    modality_diversity = all(
        int(modality_winner_counts[name]) > 0 for name in MODALITIES
    )
    positive_gain = float(oracle_margin_gain) > 0.0
    return {
        "passed": expert_diversity and modality_diversity and positive_gain,
        "expert_diversity_passed": expert_diversity,
        "modality_diversity_passed": modality_diversity,
        "oracle_margin_gain_passed": positive_gain,
        "oracle_margin_gain": float(oracle_margin_gain),
        "expert_unique_winner_counts": {
            name: int(expert_winner_counts[name]) for name in EXPERTS
        },
        "modality_unique_winner_counts": {
            name: int(modality_winner_counts[name]) for name in MODALITIES
        },
    }


def _margins_from_features(
    features: Any,
    identities: np.ndarray,
    cameras: np.ndarray,
    *,
    num_query: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    normalized = F.normalize(features.float(), dim=1)
    distances = torch.cdist(
        normalized[:num_query],
        normalized[num_query:],
    ).cpu().numpy()
    return per_query_identity_margin(
        distances,
        query_ids=identities[:num_query],
        gallery_ids=identities[num_query:],
        query_cameras=cameras[:num_query],
        gallery_cameras=cameras[num_query:],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from build_v8_oof_router_targets import (
        _collect_fold,
        _eval_loader,
        build_identity_folds,
    )
    from probe_v8_frozen_router import select_cross_camera_records
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_runtime,
        _load_config,
        _sha256,
    )

    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"OOF margin output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source_dir = args.source_dir.resolve()
    source_summary = source_dir / "run_summary.json"
    source_cache = source_dir / "oof_router_targets.pth"
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V8:
        raise ValueError("OOF margin targets require V8 expert formation")
    runtime = _build_runtime(config)
    model = runtime["model"]
    folds = build_identity_folds(
        runtime["train_records"],
        num_folds=int(config["PROTOCOL"]["OOF_TARGET_FOLDS"]),
    )
    eligible_records = select_cross_camera_records(runtime["train_records"])
    slot_margins = []
    expert_margins = {expert: [] for expert in EXPERTS}
    query_direct = []
    query_modal_residual = []
    query_identities = []
    query_cameras = []
    query_folds = []
    fold_receipts = []

    for fold_index, heldout_ids in enumerate(folds):
        checkpoint_path = source_dir / f"fold_{fold_index}_experts.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = model.state_dict()
        state.update(checkpoint["expert_state_dict"])
        model.load_state_dict(state, strict=True)
        model.cuda().eval()
        heldout_records = [
            record
            for record in eligible_records
            if int(record[1]) in heldout_ids
        ]
        loader = _eval_loader(heldout_records, runtime, config)
        split = _collect_fold(model, loader)
        num_query = len(heldout_records)
        identities = split["identities"]
        cameras = split["cameras"]
        fold_slot_margins = np.empty(
            (num_query, len(EXPERTS), len(MODALITIES)),
            dtype=np.float64,
        )
        for expert_index, expert in enumerate(EXPERTS):
            expert_margins[expert].append(
                _margins_from_features(
                    split[f"residual_{expert}"],
                    identities,
                    cameras,
                    num_query=num_query,
                )
            )
            for modality_index, modality in enumerate(MODALITIES):
                fold_slot_margins[:, expert_index, modality_index] = (
                    _margins_from_features(
                        split[f"slot_{expert}_{modality}"],
                        identities,
                        cameras,
                        num_query=num_query,
                    )
                )
        slot_margins.append(torch.from_numpy(fold_slot_margins).float())
        query_direct.append(split["direct_modal"][:num_query].half())
        query_modal_residual.append(split["modal_residual"][:num_query].half())
        query_identities.extend(identities[:num_query].tolist())
        query_cameras.extend(cameras[:num_query].tolist())
        query_folds.extend([fold_index] * num_query)
        fold_receipts.append(
            {
                "fold": fold_index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "heldout_identity_count": len(heldout_ids),
                "eligible_heldout_queries": num_query,
            }
        )

    target_margin = torch.cat(slot_margins)
    maximum = target_margin.amax(dim=(1, 2), keepdim=True)
    ties = torch.isclose(target_margin, maximum, rtol=0.0, atol=1e-8)
    unique = ties.flatten(1).sum(dim=1) == 1
    winners = target_margin.flatten(1).argmax(dim=1)
    expert_counts = {
        expert: int((unique & (winners // len(MODALITIES) == index)).sum())
        for index, expert in enumerate(EXPERTS)
    }
    modality_counts = {
        modality: int((unique & (winners % len(MODALITIES) == index)).sum())
        for index, modality in enumerate(MODALITIES)
    }
    fixed_margin = target_margin.mean(dim=0)
    oracle_margin = float(target_margin.amax(dim=(1, 2)).mean())
    best_fixed_margin = float(fixed_margin.max())
    oracle_margin_gain = oracle_margin - best_fixed_margin
    gate = evaluate_margin_target_gate(
        expert_winner_counts=expert_counts,
        modality_winner_counts=modality_counts,
        oracle_margin_gain=oracle_margin_gain,
    )
    expert_fixed_margin = {
        expert: float(np.concatenate(expert_margins[expert]).mean())
        for expert in EXPERTS
    }
    expert_oracle_margin = float(
        np.stack(
            [np.concatenate(expert_margins[expert]) for expert in EXPERTS],
            axis=1,
        ).max(axis=1).mean()
    )
    cache_path = args.output_dir / "oof_router_margin_targets.pth"
    torch.save(
        {
            "schema_version": "trifusion-v8-oof-router-margin-cache-v1",
            "direct_modal": torch.cat(query_direct),
            "modal_residual": torch.cat(query_modal_residual),
            "target_identity_margin": target_margin,
            "identities": torch.tensor(query_identities, dtype=torch.long),
            "cameras": torch.tensor(query_cameras, dtype=torch.long),
            "fold_indices": torch.tensor(query_folds, dtype=torch.long),
            "experts": EXPERTS,
            "modalities": MODALITIES,
        },
        cache_path,
    )
    result = {
        "schema_version": "trifusion-v8-oof-router-margin-result-v1",
        "status": "PASS",
        "source_oof_summary": str(source_summary),
        "source_oof_summary_sha256": _sha256(source_summary),
        "source_ap_cache": str(source_cache),
        "source_ap_cache_sha256": _sha256(source_cache),
        "fold_receipts": fold_receipts,
        "oof_queries": int(target_margin.shape[0]),
        "slot_fixed_mean_identity_margin": {
            expert: {
                modality: float(fixed_margin[expert_index, modality_index])
                for modality_index, modality in enumerate(MODALITIES)
            }
            for expert_index, expert in enumerate(EXPERTS)
        },
        "slot_oracle_mean_identity_margin": oracle_margin,
        "slot_oracle_minus_best_fixed_margin": oracle_margin_gain,
        "expert_fixed_mean_identity_margin": expert_fixed_margin,
        "expert_oracle_mean_identity_margin": expert_oracle_margin,
        "margin_target_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "target_cache": str(cache_path),
        "target_cache_sha256": _sha256(cache_path),
        "training_executed": False,
        "optimizer_steps": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
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
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "evaluate_margin_target_gate",
    "per_query_identity_margin",
    "run",
]
