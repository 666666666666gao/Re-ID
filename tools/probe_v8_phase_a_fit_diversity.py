#!/usr/bin/env python3
"""Read-only fit-domain expert-diversity gate for V8 Phase A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPERTS = ("cnn", "transformer", "mamba")


def evaluate_fit_diversity_gate(
    oracle: dict[str, Any],
    *,
    min_oracle_gain_map: float,
) -> dict[str, Any]:
    """Require non-collapsed fit winners before fitting a Router."""

    gain = float(oracle["oracle_minus_best_fixed_percent"]["mAP"])
    experts = {}
    for expert in EXPERTS:
        unique_wins = int(oracle["unique_ap_wins"][expert])
        marginal_map = float(
            oracle["leave_one_expert_out"][expert]["marginal_mAP"]
        )
        experts[expert] = {
            "unique_ap_wins": unique_wins,
            "marginal_mAP": marginal_map,
            "passed": unique_wins > 0 and marginal_map > 0.0,
        }
    return {
        "passed": gain >= float(min_oracle_gain_map)
        and all(result["passed"] for result in experts.values()),
        "oracle_gain_mAP": gain,
        "minimum_oracle_gain_mAP": float(min_oracle_gain_map),
        "experts": experts,
    }


def _collect_features(model: Any, loader: Any) -> dict[str, Any]:
    import torch

    names = (
        "baseline_only",
        "cnn",
        "transformer",
        "mamba",
        "residual_cnn",
        "residual_transformer",
        "residual_mamba",
    )
    collected = {name: [] for name in names}
    identities = []
    cameras = []
    model.eval()
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
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
        features = {
            "baseline_only": output.baseline_embedding,
            **dict(output.branch_embeddings),
            **{
                f"residual_{expert}": output.residual_embeddings[expert]
                for expert in EXPERTS
            },
        }
        for name in names:
            collected[name].append(features[name].float().cpu())
        identities.extend(np.asarray(batch_ids).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    return {
        **{name: torch.cat(values) for name, values in collected.items()},
        "identities": np.asarray(identities),
        "cameras": np.asarray(cameras),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from diagnose_v6_oracle_complementarity import (
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from probe_v8_frozen_router import _fit_eval_loader
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_runtime,
        _load_config,
        _sha256,
    )

    if args.output.exists():
        raise FileExistsError(f"fit-diversity output already exists: {args.output}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = _load_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V8:
        raise ValueError("fit-diversity gate is frozen to V8 expert formation")
    runtime = _build_runtime(config)
    model = runtime["model"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    loader = _fit_eval_loader(runtime, config)
    split = _collect_features(model, loader)
    num_query = len(loader.dataset) // 2
    if split["identities"].size != num_query * 2:
        raise ValueError("fit-diversity gate requires one query/gallery pair")
    scores = {
        name: _scores_from_features(
            split[name],
            split["identities"],
            split["cameras"],
            num_query=num_query,
        )
        for name in (
            "baseline_only",
            "cnn",
            "transformer",
            "mamba",
            "residual_cnn",
            "residual_transformer",
            "residual_mamba",
        )
    }
    branch_oracle = summarize_oracle_complementarity(
        {
            name: scores[name]
            for name in ("baseline_only", "cnn", "transformer", "mamba")
        }
    )
    residual_oracle = summarize_oracle_complementarity(
        {expert: scores[f"residual_{expert}"] for expert in EXPERTS}
    )
    min_gain = float(config["GATES"]["FORMATION_MIN_ORACLE_GAIN_MAP"])
    branch_gate = evaluate_fit_diversity_gate(
        branch_oracle,
        min_oracle_gain_map=min_gain,
    )
    residual_gate = evaluate_fit_diversity_gate(
        residual_oracle,
        min_oracle_gain_map=min_gain,
    )
    result = {
        "schema_version": "trifusion-v8-phase-a-fit-diversity-v1",
        "status": "PASS",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "probe_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "fit_queries": num_query,
        "fit_identities": int(np.unique(split["identities"][:num_query]).size),
        "branch_oracle": branch_oracle,
        "residual_only_oracle": residual_oracle,
        "branch_gate": branch_gate,
        "residual_only_gate": residual_gate,
        "next_phase_authorized": bool(
            branch_gate["passed"] and residual_gate["passed"]
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["evaluate_fit_diversity_gate", "run"]
