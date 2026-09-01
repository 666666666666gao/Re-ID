#!/usr/bin/env python3
"""Qualify DINOv2 against identity-OOF TriFusion residual representations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EXPERTS = ("cnn", "transformer", "mamba")
EXPECTED_FOLDS = 3
EXPECTED_QUERIES = 571


def compose_residual_bank(residuals: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the three fixed role-specific residual embeddings."""

    return torch.cat(
        [F.normalize(residuals[expert], dim=1) for expert in EXPERTS],
        dim=1,
    )


def aggregate_fold_scores(
    fold_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-query fold results without comparing cross-fold embeddings."""

    from tools.diagnose_v6_oracle_complementarity import QueryRetrievalScores

    names = tuple(fold_scores[0])
    return {
        name: QueryRetrievalScores(
            average_precision=np.concatenate(
                [scores[name].average_precision for scores in fold_scores]
            ),
            rank1_correct=np.concatenate(
                [scores[name].rank1_correct for scores in fold_scores]
            ),
        )
        for name in names
    }


def evaluate_qualification_gate(
    *,
    fixed_map: Mapping[str, float],
    oracle_map: float,
    unique_ap_wins: Mapping[str, int],
    fold_count: int,
    query_count: int,
    min_concat_gain_map: float = 1.0,
    min_oracle_gain_map: float = 2.0,
    saturation_map: float = 99.0,
) -> dict[str, Any]:
    """Apply the one preregistered V11 fail-closed qualification decision."""

    best_source_map = max(
        float(fixed_map["residual_bank"]),
        float(fixed_map["dinov2"]),
    )
    concat_gain = float(fixed_map["concat"]) - best_source_map
    oracle_gain = float(oracle_map) - best_source_map
    protocol_passed = (
        int(fold_count) == EXPECTED_FOLDS and int(query_count) == EXPECTED_QUERIES
    )
    non_saturation = max(float(value) for value in fixed_map.values()) < float(
        saturation_map
    )
    concat_passed = concat_gain >= float(min_concat_gain_map)
    oracle_passed = oracle_gain >= float(min_oracle_gain_map)
    two_source_wins = all(
        int(unique_ap_wins[name]) > 0 for name in ("residual_bank", "dinov2")
    )
    return {
        "passed": (
            protocol_passed
            and non_saturation
            and concat_passed
            and oracle_passed
            and two_source_wins
        ),
        "protocol_passed": protocol_passed,
        "non_saturation_passed": non_saturation,
        "concat_gain_passed": concat_passed,
        "oracle_gain_passed": oracle_passed,
        "two_source_unique_wins_passed": two_source_wins,
        "fold_count": int(fold_count),
        "query_count": int(query_count),
        "best_fixed_source_mAP": best_source_map,
        "concat_gain_mAP": concat_gain,
        "minimum_concat_gain_mAP": float(min_concat_gain_map),
        "oracle_gain_mAP": oracle_gain,
        "minimum_oracle_gain_mAP": float(min_oracle_gain_map),
        "saturation_mAP": float(saturation_map),
        "unique_ap_wins": {
            name: int(unique_ap_wins[name])
            for name in ("residual_bank", "dinov2")
        },
    }


def _collect_fold_features(
    *,
    model: torch.nn.Module,
    dinov2: torch.nn.Module,
    loader: Any,
) -> dict[str, Any]:
    from tools.probe_v10_dinov2_fit_utility import (
        DINO_TOKEN_SHAPE,
        MODALITIES,
        compose_equal_block_embedding,
        dinov2_global_embedding,
        prepare_dinov2_input,
    )

    collected = {
        name: []
        for name in (*EXPERTS, "residual_bank", "dinov2", "concat")
    }
    identities: list[int] = []
    cameras: list[int] = []
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
        images = {
            name: value.cuda(non_blocking=True) for name, value in images.items()
        }
        camera_labels = camera_labels.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_labels.shape[0],
                3,
                dtype=torch.bool,
                device="cuda",
            ),
            "camera_ids": camera_labels,
        }
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            phase = model(batch, return_aux=True)
            residuals = {
                expert: phase.residual_embeddings[expert] for expert in EXPERTS
            }
            residual_bank = compose_residual_bank(residuals)
            modality_tokens = []
            for modality in MODALITIES:
                tokens = dinov2.forward_features(
                    prepare_dinov2_input(images[modality])
                )
                if tuple(tokens.shape[1:]) != DINO_TOKEN_SHAPE:
                    raise RuntimeError(
                        "DINOv2 token shape differs from the preregistered contract"
                    )
                modality_tokens.append(tokens)
            dino = dinov2_global_embedding(torch.stack(modality_tokens, dim=1))
            concat = compose_equal_block_embedding(residual_bank, dino)
        for expert in EXPERTS:
            collected[expert].append(residuals[expert].float().cpu())
        collected["residual_bank"].append(residual_bank.float().cpu())
        collected["dinov2"].append(dino.float().cpu())
        collected["concat"].append(concat.float().cpu())
        identities.extend(torch.as_tensor(batch_ids).tolist())
        cameras.extend(torch.as_tensor(batch_cameras).tolist())
    return {
        **{name: torch.cat(values) for name, values in collected.items()},
        "identities": np.asarray(identities),
        "cameras": np.asarray(cameras),
    }


def _metric_summary(scores: Any) -> dict[str, float]:
    return {
        "mAP": float(scores.average_precision.mean() * 100.0),
        "Rank-1": float(scores.rank1_correct.mean() * 100.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from tools.build_v8_oof_router_targets import (
        _eval_loader,
        build_identity_folds,
    )
    from tools.diagnose_v6_oracle_complementarity import (
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from tools.probe_v10_dinov2_fit_utility import (
        DINO_IMAGE_SIZE,
        DINO_TOKEN_SHAPE,
        _build_dinov2,
    )
    from tools.probe_v8_frozen_router import select_cross_camera_records
    from tools.run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_runtime,
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )

    started = time.time()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"V11 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    config_path = args.config.resolve()
    source_dir = args.source_dir.resolve()
    source_summary_path = source_dir / "run_summary.json"
    dino_weight_path = args.dino_weight.resolve()
    if _sha256(dino_weight_path) != args.dino_weight_sha256:
        raise ValueError("DINOv2 weight SHA-256 differs from the contract")

    config = load_raw_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V8:
        raise ValueError("V11 qualification requires the V8 expert formation model")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if (
        source_summary["schema_version"]
        != "trifusion-v8-oof-router-target-result-v1"
    ):
        raise ValueError("unexpected V8 OOF source schema")
    runtime = _build_runtime(config)
    model = runtime["model"].cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dinov2 = _build_dinov2(dino_weight_path)
    dino_state_before = _module_state_sha256(dinov2)

    folds = build_identity_folds(
        runtime["train_records"],
        num_folds=int(config["PROTOCOL"]["OOF_TARGET_FOLDS"]),
    )
    eligible_records = select_cross_camera_records(runtime["train_records"])
    receipts_by_fold = {
        int(receipt["fold"]): receipt for receipt in source_summary["fold_receipts"]
    }
    fold_scores = []
    fold_results = []
    model_states_unchanged = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    for fold_index, heldout_ids in enumerate(folds):
        checkpoint_path = source_dir / f"fold_{fold_index}_experts.pth"
        receipt = receipts_by_fold[fold_index]
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != receipt["checkpoint_sha256"]:
            raise ValueError("V8 OOF expert checkpoint SHA-256 differs from its receipt")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if int(checkpoint["fold"]) != fold_index or checkpoint[
            "heldout_identities"
        ] != sorted(int(value) for value in heldout_ids):
            raise ValueError("V8 OOF expert checkpoint fold contract failed")
        state = model.state_dict()
        state.update(checkpoint["expert_state_dict"])
        model.load_state_dict(state, strict=True)
        model.eval()
        model_state_before = _module_state_sha256(model)
        heldout_records = [
            record
            for record in eligible_records
            if int(record[1]) in heldout_ids
        ]
        loader = _eval_loader(heldout_records, runtime, config)
        features = _collect_fold_features(
            model=model,
            dinov2=dinov2,
            loader=loader,
        )
        model_state_after = _module_state_sha256(model)
        model_states_unchanged &= model_state_before == model_state_after
        num_query = len(heldout_records)
        scores = {
            name: _scores_from_features(
                features[name],
                features["identities"],
                features["cameras"],
                num_query=num_query,
            )
            for name in (*EXPERTS, "residual_bank", "dinov2", "concat")
        }
        fold_scores.append(scores)
        fold_results.append(
            {
                "fold": fold_index,
                "fit_identity_count": int(receipt["fit_identity_count"]),
                "heldout_identity_count": len(heldout_ids),
                "eligible_heldout_identity_count": len(
                    {int(record[1]) for record in heldout_records}
                ),
                "queries": num_query,
                "gallery": len(loader.dataset) - num_query,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "model_state_sha256_before": model_state_before,
                "model_state_sha256_after": model_state_after,
                "metrics_percent": {
                    name: _metric_summary(value) for name, value in scores.items()
                },
            }
        )

    combined = aggregate_fold_scores(fold_scores)
    metrics = {name: _metric_summary(value) for name, value in combined.items()}
    oracle = summarize_oracle_complementarity(
        {
            name: combined[name] for name in ("residual_bank", "dinov2")
        }
    )
    fixed_map = {
        name: metrics[name]["mAP"]
        for name in ("residual_bank", "dinov2", "concat")
    }
    gate = evaluate_qualification_gate(
        fixed_map=fixed_map,
        oracle_map=float(oracle["oracle_metrics_percent"]["mAP"]),
        unique_ap_wins=oracle["unique_ap_wins"],
        fold_count=len(fold_results),
        query_count=int(combined["residual_bank"].average_precision.size),
        min_concat_gain_map=float(args.min_concat_gain_map),
        min_oracle_gain_map=float(args.min_oracle_gain_map),
    )
    dino_state_after = _module_state_sha256(dinov2)
    states_unchanged = model_states_unchanged and dino_state_before == dino_state_after
    if not states_unchanged:
        raise RuntimeError("V11 qualification changed a frozen model state")

    result = {
        "schema_version": "trifusion-v11-dinov2-oof-residual-complement-v1",
        "status": "COMPLETE",
        "qualification_status": "PASS" if gate["passed"] else "FAIL",
        "scope": "three-fold fit-identity OOF residual-only retrieval",
        "evaluation_type": "real_gt_fit_identity_oof_residual_only",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_summary": str(source_summary_path),
        "source_summary_sha256": _sha256(source_summary_path),
        "dinov2_weight": str(dino_weight_path),
        "dinov2_weight_sha256": _sha256(dino_weight_path),
        "dinov2_model": "vit_base_patch14_dinov2",
        "dinov2_source": "https://github.com/facebookresearch/dinov2",
        "dinov2_input_size": list(DINO_IMAGE_SIZE),
        "dinov2_token_shape": list(DINO_TOKEN_SHAPE),
        "strict_load": True,
        "removed_pretraining_only_key": "mask_token",
        "fold_distance_policy": "within_fold_only_query_weighted_aggregation",
        "signal_or_phase_b_in_qualification_metrics": False,
        "frozen_signal_field_seen_all_fit_identities": True,
        "fold_results": fold_results,
        "metrics_percent": metrics,
        "residual_bank_dinov2_oracle": oracle,
        "qualification_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "dinov2_state_sha256_before": dino_state_before,
        "dinov2_state_sha256_after": dino_state_after,
        "checkpoint_states_unchanged": states_unchanged,
        "oracle_uses_ground_truth": True,
        "oracle_is_deployment_result": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--dino-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-concat-gain-map", type=float, default=1.0)
    parser.add_argument("--min-oracle-gain-map", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "aggregate_fold_scores",
    "compose_residual_bank",
    "evaluate_qualification_gate",
    "run",
]
