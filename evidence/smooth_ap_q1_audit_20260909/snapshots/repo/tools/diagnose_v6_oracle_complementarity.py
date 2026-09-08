#!/usr/bin/env python3
"""Read-only query-wise Oracle complementarity diagnostic for V6."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueryRetrievalScores:
    average_precision: np.ndarray
    rank1_correct: np.ndarray


def _percent(value: float) -> float:
    return round(float(value) * 100.0, 10)


def summarize_oracle_complementarity(
    scores: Mapping[str, QueryRetrievalScores],
) -> dict[str, object]:
    """Summarize ground-truth query-wise Oracle headroom and unique wins."""

    names = tuple(scores)
    average_precision = np.stack(
        [scores[name].average_precision for name in names], axis=0
    )
    rank1 = np.stack([scores[name].rank1_correct for name in names], axis=0)
    fixed = {
        name: {
            "mAP": _percent(scores[name].average_precision.mean()),
            "Rank-1": _percent(scores[name].rank1_correct.mean()),
        }
        for name in names
    }
    best_fixed = max(names, key=lambda name: fixed[name]["mAP"])
    oracle_map = _percent(average_precision.max(axis=0).mean())
    oracle_rank1 = _percent(rank1.any(axis=0).mean())
    ap_max = average_precision.max(axis=0)
    ap_ties = np.isclose(average_precision, ap_max[None], rtol=0.0, atol=1e-12)
    rank1_winners = rank1.sum(axis=0)
    leave_one_out = {}
    for index, name in enumerate(names):
        if name == "baseline_only":
            continue
        remaining = np.delete(average_precision, index, axis=0)
        without_map = _percent(remaining.max(axis=0).mean())
        leave_one_out[name] = {
            "oracle_mAP": without_map,
            "marginal_mAP": round(oracle_map - without_map, 10),
        }
    return {
        "queries": int(average_precision.shape[1]),
        "fixed_metrics_percent": fixed,
        "best_fixed_output": best_fixed,
        "oracle_metrics_percent": {"mAP": oracle_map, "Rank-1": oracle_rank1},
        "oracle_minus_best_fixed_percent": {
            "mAP": round(oracle_map - fixed[best_fixed]["mAP"], 10),
            "Rank-1": round(oracle_rank1 - fixed[best_fixed]["Rank-1"], 10),
        },
        "unique_ap_wins": {
            name: int((ap_ties[index] & (ap_ties.sum(axis=0) == 1)).sum())
            for index, name in enumerate(names)
        },
        "unique_rank1_wins": {
            name: int((rank1[index] & (rank1_winners == 1)).sum())
            for index, name in enumerate(names)
        },
        "all_rank1_failures": int((rank1_winners == 0).sum()),
        "leave_one_expert_out": leave_one_out,
    }


def per_query_reid_scores(
    distances: np.ndarray,
    *,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray,
    gallery_cameras: np.ndarray,
) -> QueryRetrievalScores:
    """Return per-query AP and Rank-1 under the project ReID protocol."""

    ranked_gallery = np.argsort(np.asarray(distances), axis=1, kind="stable")
    average_precision = []
    rank1_correct = []
    for query_index, order in enumerate(ranked_gallery):
        junk = (gallery_ids[order] == query_ids[query_index]) & (
            gallery_cameras[order] == query_cameras[query_index]
        )
        matches = gallery_ids[order][~junk] == query_ids[query_index]
        if not np.any(matches):
            raise RuntimeError("every diagnostic query must appear in the gallery")
        cumulative = np.cumsum(matches)
        precision = cumulative / np.arange(1, matches.size + 1)
        average_precision.append(float(precision[matches].mean()))
        rank1_correct.append(bool(matches[0]))
    return QueryRetrievalScores(
        average_precision=np.asarray(average_precision, dtype=np.float64),
        rank1_correct=np.asarray(rank1_correct, dtype=bool),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scores_from_features(
    features: object,
    identities: np.ndarray,
    camera_ids: np.ndarray,
    *,
    num_query: int,
) -> QueryRetrievalScores:
    import torch
    import torch.nn.functional as F

    normalized = F.normalize(features.float(), dim=1)
    distances = torch.cdist(
        normalized[:num_query], normalized[num_query:]
    ).cpu().numpy()
    return per_query_reid_scores(
        distances,
        query_ids=identities[:num_query],
        gallery_ids=identities[num_query:],
        query_cameras=camera_ids[:num_query],
        gallery_cameras=camera_ids[num_query:],
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V6,
        _build_runtime,
        _load_config,
    )

    if args.output.exists():
        raise FileExistsError(f"oracle output already exists: {args.output}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = _load_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V6:
        raise ValueError("Oracle complementarity is frozen to V6")
    runtime = _build_runtime(config)
    model = runtime["model"]
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
    identities_all = []
    camera_ids_all = []
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
                for expert in ("cnn", "transformer", "mamba")
            },
        }
        for name in feature_names:
            collected[name].append(batch_features[name].float().cpu())
        identities_all.extend(np.asarray(identities).tolist())
        camera_ids_all.extend(np.asarray(camera_ids).tolist())

    features = {name: torch.cat(values) for name, values in collected.items()}
    identities_array = np.asarray(identities_all)
    camera_ids_array = np.asarray(camera_ids_all)
    num_query = len(runtime["dev_records"])
    if identities_array.size != num_query * 2:
        raise ValueError("V6 Oracle requires one complete query/gallery dev pair")
    scores = {
        name: _scores_from_features(
            tensor,
            identities_array,
            camera_ids_array,
            num_query=num_query,
        )
        for name, tensor in features.items()
    }
    branch_oracle = summarize_oracle_complementarity(
        {
            name: scores[name]
            for name in ("baseline_only", "cnn", "transformer", "mamba")
        }
    )
    residual_oracle = summarize_oracle_complementarity(
        {
            expert: scores[f"residual_{expert}"]
            for expert in ("cnn", "transformer", "mamba")
        }
    )
    fused_metrics = {
        "mAP": _percent(scores["fused"].average_precision.mean()),
        "Rank-1": _percent(scores["fused"].rank1_correct.mean()),
    }
    run_summary_path = checkpoint_path.parent / "run_summary.json"
    result = {
        "schema_version": "trifusion-signal-preserving-v6-oracle-complementarity-v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config_sha256": _sha256(config_path),
        "diagnostic_sha256": _sha256(Path(__file__).resolve()),
        "source_run_summary_sha256": _sha256(run_summary_path),
        "queries": num_query,
        "gallery": identities_array.size - num_query,
        "branch_oracle": branch_oracle,
        "residual_only_oracle": residual_oracle,
        "fused_metrics_percent": fused_metrics,
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
