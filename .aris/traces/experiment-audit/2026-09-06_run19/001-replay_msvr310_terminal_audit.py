#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

NUMPY_PATH = Path("D:/Program Files/UserCache/gb/uv/archive-v0/92ICCcZmeDTDf2G1EDSF0/Lib/site-packages")
sys.path.insert(0, str(NUMPY_PATH))
import numpy as np


OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")
EXPERTS = ("cnn", "transformer", "mamba")
OUTPUT_WIDTHS = {
    "baseline_only": 3072,
    "fused": 7680,
    "cnn": 4608,
    "transformer": 4608,
    "mamba": 4608,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).isoformat()


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("/", "\\")
    except ValueError:
        return str(path)


def hash_manifest_files(root: Path, manifest: dict[str, object], phase: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for index, entry in enumerate(manifest["files"]):
        path = Path(entry["path"])
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        row = {
            "phase": phase,
            "index": index,
            "path": rel(root, path),
            "expected_bytes": entry["bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_sha,
            "matches": actual_bytes == entry["bytes"] and actual_sha == entry["sha256"],
        }
        rows.append(row)
        if not row["matches"]:
            mismatches.append(row)
    summary = {
        "phase": phase,
        "file_count": len(rows),
        "manifest_bytes_total": int(sum(entry["bytes"] for entry in manifest["files"])),
        "mismatches": mismatches,
        "all_match": not mismatches,
    }
    return rows, summary


def learning_rate(epoch: int, base_lr: float, max_epochs: int, warmup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        multiplier = epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs - 1) / (max_epochs - warmup_epochs)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * multiplier


def weighted_loss(components: dict[str, float], loss_config: dict[str, float]) -> float:
    branch_id = sum(components[f"id_{expert}"] for expert in EXPERTS)
    branch_triplet = sum(components[f"triplet_{expert}"] for expert in EXPERTS)
    residual_id = sum(components[f"id_residual_{expert}"] for expert in EXPERTS)
    residual_triplet = sum(components[f"triplet_residual_{expert}"] for expert in EXPERTS)
    return (
        components["id_fused"] * loss_config["ID_FUSED"]
        + components["triplet_fused"] * loss_config["TRIPLET_FUSED"]
        + branch_id * loss_config["ID_BRANCH"]
        + branch_triplet * loss_config["TRIPLET_BRANCH"]
        + residual_id * loss_config["ID_RESIDUAL"]
        + residual_triplet * loss_config["TRIPLET_RESIDUAL"]
    )


def ap_and_rank(order: list[int], ids: np.ndarray, scenes: np.ndarray, query_id: int, query_scene: int) -> tuple[float, int, int]:
    kept: list[int] = []
    matches: list[bool] = []
    for gallery_index in order:
        if ids[gallery_index] == query_id and scenes[gallery_index] == query_scene:
            continue
        kept.append(gallery_index)
        matches.append(bool(ids[gallery_index] == query_id))
    match_array = np.asarray(matches, dtype=np.bool_)
    positions = np.flatnonzero(match_array)
    if positions.size == 0:
        raise AssertionError("query has no positive after scene filtering")
    precision = np.cumsum(match_array)[positions] / (positions + 1)
    return float(np.mean(precision)), int(positions[0] + 1), int(kept[0])


def metrics_from(ap_values: list[float], first_ranks: list[int]) -> dict[str, float]:
    first = np.asarray(first_ranks)
    return {
        "mAP": float(np.mean(ap_values) * 100),
        "Rank-1": float(np.mean(first <= 1) * 100),
        "Rank-5": float(np.mean(first <= 5) * 100),
        "Rank-10": float(np.mean(first <= 10) * 100),
    }


def max_metric_diff(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> float:
    max_diff = 0.0
    for output in a:
        for key in ("mAP", "Rank-1", "Rank-5", "Rank-10"):
            max_diff = max(max_diff, abs(float(a[output][key]) - float(b[output][key])))
    return max_diff


def bootstrap_lower_bound(differences: np.ndarray, identities: np.ndarray, *, seed: int, resamples: int) -> dict[str, object]:
    clusters = np.unique(identities)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    rows_by_cluster = {identity: np.flatnonzero(identities == identity) for identity in clusters}
    for index in range(resamples):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([rows_by_cluster[identity] for identity in sampled])
        means[index] = differences[rows].mean()
    return {
        "observed_mean_pp": float(differences.mean()),
        "lower_bound_pp": float(np.percentile(means, 2.5)),
        "cluster_count": int(len(clusters)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def compare_float_lists(computed: list[float], stored: list[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(computed, stored, strict=True)), default=0.0)


def replay_query_outputs(root: Path, protocol: dict[str, object], comparison: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    query_rows_out: list[dict[str, object]] = []
    ap_by_output = {output: [] for output in OUTPUTS}
    first_by_output = {output: [] for output in OUTPUTS}
    identities_all: list[int] = []
    max_ap_diff = 0.0
    max_rank_mismatch = 0
    max_fold_metric_diff = 0.0
    fold_summaries: list[dict[str, object]] = []

    for fold in protocol["folds"]:
        fold_index = int(fold["fold"])
        receipt = load_json(root / f"evidence/msvr310_trifusion_v1_complete/fold_{fold_index}_receipt.json")
        rankings = load_json(root / f"evidence/msvr310_trifusion_v1_complete/fold_{fold_index}_rankings.json")
        rows = receipt["retrieval"]["gallery_manifest"]
        query_rows = receipt["retrieval"]["query_rows"]
        ids = np.asarray([row["identity"] for row in rows])
        cameras = np.asarray([row["camera"] for row in rows])
        scenes = np.asarray([row["scene"] for row in rows])
        positions = [query["gallery_position"] for query in query_rows]

        if receipt != comparison["folds"][fold_index]:
            raise AssertionError(f"fold {fold_index} receipt differs from complete summary")
        if query_rows != fold["query_rows"]:
            raise AssertionError(f"fold {fold_index} query rows differ from protocol")
        if [row["index"] for row in rows] != fold["gallery_record_indices"]:
            raise AssertionError(f"fold {fold_index} gallery manifest differs from protocol")

        fold_metrics: dict[str, dict[str, float]] = {}
        for output in OUTPUTS:
            computed_ap: list[float] = []
            computed_first: list[int] = []
            stored_output = receipt["retrieval"]["outputs"][output]
            if len(rankings[output]) != len(query_rows):
                raise AssertionError(f"fold {fold_index} {output} ranking query count mismatch")
            for query_index, (query, order) in enumerate(zip(query_rows, rankings[output], strict=True)):
                if sorted(order) != list(range(len(rows))):
                    raise AssertionError(f"fold {fold_index} {output} query {query_index} ranking is not a permutation")
                pos = int(query["gallery_position"])
                if rows[pos]["index"] != query["record_index"]:
                    raise AssertionError(f"fold {fold_index} query position mismatch")
                if rows[pos]["identity"] != query["identity"] or rows[pos]["scene"] != query["scene"]:
                    raise AssertionError(f"fold {fold_index} query metadata mismatch")
                ap, first_rank, first_retained = ap_and_rank(
                    order,
                    ids,
                    scenes,
                    int(query["identity"]),
                    int(query["scene"]),
                )
                retained_gallery = int(np.sum(~((ids == query["identity"]) & (scenes == query["scene"]))))
                valid_positives = int(np.sum((ids == query["identity"]) & (scenes != query["scene"])))
                if retained_gallery != query["retained_gallery"]:
                    raise AssertionError(f"fold {fold_index} query {query_index} retained count mismatch")
                if valid_positives != query["valid_positives"]:
                    raise AssertionError(f"fold {fold_index} query {query_index} positive count mismatch")
                computed_ap.append(ap)
                computed_first.append(first_rank)
                stored_ap = float(stored_output["average_precision"][query_index])
                stored_rank = int(stored_output["first_match_rank"][query_index])
                ap_diff = abs(ap - stored_ap)
                rank_mismatch = int(first_rank != stored_rank)
                max_ap_diff = max(max_ap_diff, ap_diff)
                max_rank_mismatch = max(max_rank_mismatch, rank_mismatch)
                top = rows[first_retained]
                query_rows_out.append(
                    {
                        "fold": fold_index,
                        "output": output,
                        "query_index_in_fold": query_index,
                        "query_record_index": int(query["record_index"]),
                        "identity": int(query["identity"]),
                        "query_camera": int(cameras[pos]),
                        "query_scene": int(query["scene"]),
                        "computed_average_precision": ap,
                        "stored_average_precision": stored_ap,
                        "average_precision_abs_diff": ap_diff,
                        "computed_first_match_rank": first_rank,
                        "stored_first_match_rank": stored_rank,
                        "first_rank_mismatch": rank_mismatch,
                        "top_retained_gallery_position": int(first_retained),
                        "top_retained_record_index": int(top["index"]),
                        "top_retained_identity": int(top["identity"]),
                        "top_retained_camera": int(top["camera"]),
                        "top_retained_scene": int(top["scene"]),
                        "top_retained_same_camera": bool(int(top["camera"]) == int(cameras[pos])),
                        "top_retained_same_scene": bool(int(top["scene"]) == int(query["scene"])),
                        "top_retained_correct_identity": bool(int(top["identity"]) == int(query["identity"])),
                    }
                )
            fold_metrics[output] = metrics_from(computed_ap, computed_first)
            max_fold_metric_diff = max(
                max_fold_metric_diff,
                max_metric_diff({output: fold_metrics[output]}, {output: stored_output["metrics"]}),
            )
            ap_by_output[output].extend(computed_ap)
            first_by_output[output].extend(computed_first)
        identities_all.extend(int(query["identity"]) for query in query_rows)
        fold_summaries.append(
            {
                "fold": fold_index,
                "queries": len(query_rows),
                "gallery_records": len(rows),
                "query_identities": len(set(query["identity"] for query in query_rows)),
                "metrics": fold_metrics,
            }
        )

    ap_arrays = {output: np.asarray(values, dtype=np.float64) for output, values in ap_by_output.items()}
    first_arrays = {output: np.asarray(values, dtype=np.int64) for output, values in first_by_output.items()}
    identities = np.asarray(identities_all, dtype=np.int64)
    aggregate_metrics = {
        output: metrics_from(ap_arrays[output].tolist(), first_arrays[output].tolist())
        for output in OUTPUTS
    }
    gains = {
        output: aggregate_metrics[output]["mAP"] - aggregate_metrics["baseline_only"]["mAP"]
        for output in OUTPUTS
    }
    fold_fused_gains = [
        row["metrics"]["fused"]["mAP"] - row["metrics"]["baseline_only"]["mAP"]
        for row in fold_summaries
    ]
    differences = {
        output: (ap_arrays[output] - ap_arrays["baseline_only"]) * 100.0
        for output in OUTPUTS
    }
    bootstrap = bootstrap_lower_bound(
        differences["fused"],
        identities,
        seed=42,
        resamples=10000,
    )
    scientific_checks = {
        "fused_gain_at_least_1pp": bool(gains["fused"] >= 1.0),
        "all_fold_fused_gains_nonnegative": bool(all(value >= 0 for value in fold_fused_gains)),
        "all_full_branches_not_below_signal": bool(all(gains[output] >= 0 for output in EXPERTS)),
        "identity_bootstrap_lower_positive": bool(bootstrap["lower_bound_pp"] > 0),
        "fused_strictly_best": bool(
            all(aggregate_metrics["fused"]["mAP"] > aggregate_metrics[output]["mAP"] for output in ("baseline_only", *EXPERTS))
        ),
    }
    per_identity = []
    for identity in np.unique(identities):
        mask = identities == identity
        per_identity.append(
            {
                "identity": int(identity),
                "query_count": int(mask.sum()),
                "map_by_output": {
                    output: float(ap_arrays[output][mask].mean() * 100.0)
                    for output in OUTPUTS
                },
            }
        )
    query_changes = {}
    for output in EXPERTS + ("fused",):
        values = differences[output]
        query_changes[output] = {
            "ap_improved": int(np.sum(values > 0)),
            "ap_declined": int(np.sum(values < 0)),
            "ap_unchanged": int(np.sum(values == 0)),
            "rank1_repaired": int(np.sum((first_arrays["baseline_only"] > 1) & (first_arrays[output] == 1))),
            "rank1_new_errors": int(np.sum((first_arrays["baseline_only"] == 1) & (first_arrays[output] > 1))),
        }

    comparison_replay = {
        "metrics": aggregate_metrics,
        "gains_over_signal_pp": gains,
        "fold_fused_gains_pp": fold_fused_gains,
        "identity_bootstrap": {
            "lower_bound_pp": bootstrap["lower_bound_pp"],
            "seed": 42,
            "resamples": 10000,
            "cluster_count": bootstrap["cluster_count"],
            "weighting": "resample whole identities with replacement; retain query weights",
            "percentile": 2.5,
            "quantile_method": "linear",
        },
        "scientific_checks": scientific_checks,
        "scientific_passed": bool(all(scientific_checks.values())),
        "per_identity": per_identity,
        "query_changes": query_changes,
    }

    summary = {
        "query_output_evaluations": len(query_rows_out),
        "queries": int(len(identities)),
        "query_identities": int(len(np.unique(identities))),
        "max_average_precision_abs_diff_to_receipts": max_ap_diff,
        "rank_mismatch_count_upper_bound": max_rank_mismatch,
        "max_fold_metric_abs_diff_to_receipts": max_fold_metric_diff,
        "max_aggregate_metric_abs_diff_to_summary": max_metric_diff(aggregate_metrics, comparison["comparison"]["metrics"]),
        "bootstrap_lower_abs_diff_to_summary": abs(
            float(comparison_replay["identity_bootstrap"]["lower_bound_pp"])
            - float(comparison["comparison"]["identity_bootstrap"]["lower_bound_pp"])
        ),
        "scientific_checks_match_summary": scientific_checks == comparison["comparison"]["scientific_checks"],
        "scientific_passed_matches_summary": comparison_replay["scientific_passed"] == comparison["comparison"]["scientific_passed"],
        "comparison_replay": comparison_replay,
        "fold_summaries": fold_summaries,
    }
    return summary, query_rows_out, ap_arrays, first_arrays, identities


def replay_training(root: Path, protocol: dict[str, object], config: dict[str, object], comparison: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    step_rows_out: list[dict[str, object]] = []
    epoch_rows_out: list[dict[str, object]] = []
    max_weighted_loss_diff = 0.0
    max_epoch_mean_diff = 0.0
    max_lr_diff = 0.0
    total_source_exposures = 0
    total_same_pairs = 0
    total_cross_scene_pairs = 0
    fold_summaries: list[dict[str, object]] = []

    record_by_index = {int(row["index"]): row for row in protocol["records"]}

    for fold in protocol["folds"]:
        fold_index = int(fold["fold"])
        training_path = root / f"evidence/msvr310_trifusion_v1_complete/fold_{fold_index}_training.json"
        receipt_path = root / f"evidence/msvr310_trifusion_v1_complete/fold_{fold_index}_receipt.json"
        training = load_json(training_path)
        receipt = load_json(receipt_path)
        if training != receipt["training"]:
            raise AssertionError(f"fold {fold_index} training file differs from receipt")
        if training != comparison["folds"][fold_index]["training"]:
            raise AssertionError(f"fold {fold_index} training differs from complete summary")
        if fold_index == 0:
            failed_training = load_json(root / "evidence/msvr310_trifusion_v1_failed_comparison/fold_0_training.json")
            if failed_training != training:
                raise AssertionError("complete fold0 training is not byte-equivalent JSON content to failed original fold0 training")

        source_set = set(int(index) for index in fold["source_record_indices"])
        source_identities = set(int(identity) for identity in fold["source_ids"])
        unique_records_seen: set[int] = set()
        identities_seen: set[int] = set()
        fold_same_pairs = 0
        fold_cross_scene_pairs = 0

        for row in training["steps"]:
            components = {key: float(value) for key, value in row["components"].items()}
            computed_loss = weighted_loss(components, config["LOSS"])
            weighted_diff = abs(computed_loss - float(row["loss"]))
            max_weighted_loss_diff = max(max_weighted_loss_diff, weighted_diff)
            sampled = [int(index) for index in row["sampled_record_indices"]]
            if len(sampled) != 64:
                raise AssertionError(f"fold {fold_index} step {row['step']} does not have 64 samples")
            if any(index not in source_set for index in sampled):
                raise AssertionError(f"fold {fold_index} step {row['step']} sampled outside source records")
            identities = [int(record_by_index[index]["identity"]) for index in sampled]
            scenes = [int(record_by_index[index]["scene"]) for index in sampled]
            if any(identity not in source_identities for identity in identities):
                raise AssertionError(f"fold {fold_index} step {row['step']} sampled identity outside source ids")
            identity_counts = sorted(identities.count(identity) for identity in set(identities))
            if identity_counts != [8] * 8:
                raise AssertionError(f"fold {fold_index} step {row['step']} is not B64/K8")
            same_pairs = 0
            cross_scene_pairs = 0
            for left in range(len(sampled)):
                for right in range(left + 1, len(sampled)):
                    if identities[left] == identities[right]:
                        same_pairs += 1
                        if scenes[left] != scenes[right]:
                            cross_scene_pairs += 1
            unique_records_seen.update(sampled)
            identities_seen.update(identities)
            fold_same_pairs += same_pairs
            fold_cross_scene_pairs += cross_scene_pairs
            total_source_exposures += len(sampled)
            step_rows_out.append(
                {
                    "fold": fold_index,
                    "step": int(row["step"]),
                    "epoch": int(row["epoch"]),
                    "stored_loss": float(row["loss"]),
                    "computed_weighted_loss": computed_loss,
                    "weighted_loss_abs_diff": weighted_diff,
                    "sample_count": len(sampled),
                    "unique_identity_count": len(set(identities)),
                    "identity_counts": identity_counts,
                    "same_identity_positive_pairs": same_pairs,
                    "cross_scene_positive_pairs": cross_scene_pairs,
                    "all_samples_in_source": True,
                    "amp_scale_before": float(row["amp_scale_before"]),
                    "amp_scale_after": float(row["amp_scale_after"]),
                }
            )

        by_epoch: dict[int, list[float]] = {}
        for row in training["steps"]:
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["loss"]))
        for history in training["history"]:
            epoch = int(history["epoch"])
            mean_loss = float(np.mean(by_epoch[epoch]))
            expected_lr = learning_rate(epoch, config["OPTIMIZATION"]["NEW_MODULE_LR"], 20, 5)
            epoch_mean_diff = abs(mean_loss - float(history["mean_loss"]))
            lr_diff = abs(expected_lr - float(history["learning_rate"]))
            max_epoch_mean_diff = max(max_epoch_mean_diff, epoch_mean_diff)
            max_lr_diff = max(max_lr_diff, lr_diff)
            if int(history["optimizer_steps"]) != len(by_epoch[epoch]):
                raise AssertionError(f"fold {fold_index} epoch {epoch} optimizer step count mismatch")
            epoch_rows_out.append(
                {
                    "fold": fold_index,
                    "epoch": epoch,
                    "stored_mean_loss": float(history["mean_loss"]),
                    "computed_mean_loss": mean_loss,
                    "mean_loss_abs_diff": epoch_mean_diff,
                    "stored_learning_rate": float(history["learning_rate"]),
                    "computed_learning_rate": expected_lr,
                    "learning_rate_abs_diff": lr_diff,
                    "optimizer_steps": int(history["optimizer_steps"]),
                }
            )

        total_same_pairs += fold_same_pairs
        total_cross_scene_pairs += fold_cross_scene_pairs
        fold_summaries.append(
            {
                "fold": fold_index,
                "optimizer_steps": int(training["optimizer_steps"]),
                "epochs": int(training["epochs"]),
                "source_record_exposures": int(len(training["steps"]) * 64),
                "unique_source_records_exposed": int(len(unique_records_seen)),
                "source_identities_exposed": int(len(identities_seen)),
                "same_identity_positive_pairs": int(fold_same_pairs),
                "cross_scene_positive_pairs": int(fold_cross_scene_pairs),
            }
        )

    summary = {
        "optimizer_steps": len(step_rows_out),
        "epochs": sum(int(load_json(root / f"evidence/msvr310_trifusion_v1_complete/fold_{i}_training.json")["epochs"]) for i in range(3)),
        "source_record_exposures": int(total_source_exposures),
        "same_identity_positive_pairs": int(total_same_pairs),
        "cross_scene_positive_pairs": int(total_cross_scene_pairs),
        "max_weighted_loss_abs_diff": max_weighted_loss_diff,
        "max_epoch_mean_abs_diff": max_epoch_mean_diff,
        "max_learning_rate_abs_diff": max_lr_diff,
        "fold_summaries": fold_summaries,
        "matches_summary_optimizer_steps": int(comparison["optimizer_steps"]) == len(step_rows_out),
        "matches_summary_new_optimizer_steps": int(comparison["new_optimizer_steps"]) == 520,
    }
    return summary, step_rows_out, epoch_rows_out


def replay_protocol(protocol: dict[str, object]) -> dict[str, object]:
    all_gallery_records: list[int] = []
    all_query_identities: list[int] = []
    fold_summaries: list[dict[str, object]] = []
    records = {int(row["index"]): row for row in protocol["records"]}
    all_paths = []

    for fold in protocol["folds"]:
        fold_index = int(fold["fold"])
        source_ids = set(int(identity) for identity in fold["source_ids"])
        heldout_ids = set(int(identity) for identity in fold["heldout_ids"])
        if source_ids & heldout_ids:
            raise AssertionError(f"fold {fold_index} source/heldout overlap")
        source_records = [records[int(index)] for index in fold["source_record_indices"]]
        gallery_records = [records[int(index)] for index in fold["gallery_record_indices"]]
        if any(int(row["identity"]) not in source_ids for row in source_records):
            raise AssertionError(f"fold {fold_index} source record identity outside source ids")
        if any(int(row["identity"]) not in heldout_ids for row in gallery_records):
            raise AssertionError(f"fold {fold_index} gallery record identity outside heldout ids")
        if any("bounding_box_train/" not in path for row in gallery_records for path in row["paths"]):
            raise AssertionError(f"fold {fold_index} gallery path outside training split")
        for row in source_records + gallery_records:
            all_paths.extend(row["paths"])
        for query in fold["query_rows"]:
            gallery_row = gallery_records[int(query["gallery_position"])]
            if int(gallery_row["index"]) != int(query["record_index"]):
                raise AssertionError(f"fold {fold_index} query record position mismatch")
            same_scene = [
                row for row in gallery_records
                if int(row["identity"]) == int(query["identity"]) and int(row["scene"]) == int(query["scene"])
            ]
            cross_scene = [
                row for row in gallery_records
                if int(row["identity"]) == int(query["identity"]) and int(row["scene"]) != int(query["scene"])
            ]
            if len(same_scene) != int(query["removed_same_identity_same_scene"]):
                raise AssertionError(f"fold {fold_index} removed same-scene count mismatch")
            if len(cross_scene) != int(query["valid_positives"]):
                raise AssertionError(f"fold {fold_index} cross-scene positive count mismatch")
        for index in fold["excluded_query_record_indices"]:
            row = records[int(index)]
            gallery_for_identity = [records[int(g)] for g in fold["gallery_record_indices"] if int(records[int(g)]["identity"]) == int(row["identity"])]
            if len({int(item["scene"]) for item in gallery_for_identity}) != 1:
                raise AssertionError(f"fold {fold_index} excluded query has cross-scene positive")

        query_identities = sorted({int(query["identity"]) for query in fold["query_rows"]})
        all_gallery_records.extend(int(index) for index in fold["gallery_record_indices"])
        all_query_identities.extend(query_identities)
        fold_summaries.append(
            {
                "fold": fold_index,
                "source_identities": len(source_ids),
                "source_records": len(source_records),
                "heldout_identities": len(heldout_ids),
                "gallery_records": len(gallery_records),
                "query_rows": len(fold["query_rows"]),
                "query_identities": len(query_identities),
                "excluded_query_records": len(fold["excluded_query_record_indices"]),
                "source_heldout_disjoint": True,
            }
        )

    return {
        "fold_summaries": fold_summaries,
        "unique_gallery_records": len(set(all_gallery_records)),
        "valid_queries": sum(row["query_rows"] for row in fold_summaries),
        "unique_query_identities": len(set(all_query_identities)),
        "gallery_only_distractor_identities": int(protocol["aggregate_counts"]["gallery_only_distractor_identities"]),
        "excluded_query_records_retained_in_gallery": int(protocol["aggregate_counts"]["excluded_query_records_retained_in_gallery"]),
        "all_protocol_paths_from_training_split": all(path.startswith("bounding_box_train/") for path in all_paths),
        "official_test_paths_seen": sum(1 for path in all_paths if path.startswith("bounding_box_test/")),
        "aggregate_counts": protocol["aggregate_counts"],
    }


def replay_error_census(query_rows: list[dict[str, object]], comparison: dict[str, object], error_census: dict[str, object], ap_arrays: dict[str, np.ndarray], first_arrays: dict[str, np.ndarray], identities: np.ndarray) -> dict[str, object]:
    rows_by_output = {output: [row for row in query_rows if row["output"] == output] for output in OUTPUTS}
    baseline_rows = rows_by_output["baseline_only"]
    baseline_summary = {
        "total": 0,
        "same_camera": 0,
        "same_scene": 0,
    }
    for row in baseline_rows:
        if row["computed_first_match_rank"] > 1:
            baseline_summary["total"] += 1
            baseline_summary["same_camera"] += int(bool(row["top_retained_same_camera"]))
            baseline_summary["same_scene"] += int(bool(row["top_retained_same_scene"]))

    summary = {}
    max_query_row_diff = 0.0
    for output in EXPERTS + ("fused",):
        rows = rows_by_output[output]
        output_summary = {
            "all_queries": len(rows),
            "rank1_repaired": 0,
            "rank1_new_errors": 0,
            "new_errors_same_camera": 0,
            "new_errors_same_scene": 0,
            "all_rank1_errors": 0,
            "all_rank1_errors_same_camera": 0,
            "all_rank1_errors_same_scene": 0,
        }
        stored_rows = error_census["query_rows"][output]
        for index, row in enumerate(rows):
            baseline_rank = int(first_arrays["baseline_only"][index])
            output_rank = int(row["computed_first_match_rank"])
            rank1_repaired = baseline_rank > 1 and output_rank == 1
            rank1_new_error = baseline_rank == 1 and output_rank > 1
            if rank1_repaired:
                output_summary["rank1_repaired"] += 1
            if rank1_new_error:
                output_summary["rank1_new_errors"] += 1
                output_summary["new_errors_same_camera"] += int(bool(row["top_retained_same_camera"]))
                output_summary["new_errors_same_scene"] += int(bool(row["top_retained_same_scene"]))
            if output_rank > 1:
                output_summary["all_rank1_errors"] += 1
                output_summary["all_rank1_errors_same_camera"] += int(bool(row["top_retained_same_camera"]))
                output_summary["all_rank1_errors_same_scene"] += int(bool(row["top_retained_same_scene"]))
            max_query_row_diff = max(
                max_query_row_diff,
                abs(float(row["computed_average_precision"] - rows_by_output["baseline_only"][index]["computed_average_precision"]) * 100.0 - float(stored_rows[index]["ap_delta_pp"])),
            )
            if int(stored_rows[index]["candidate_first_identity"]) != int(row["top_retained_identity"]):
                raise AssertionError(f"{output} census first identity mismatch at global query {index}")
            if int(stored_rows[index]["candidate_first_camera"]) != int(row["top_retained_camera"]):
                raise AssertionError(f"{output} census first camera mismatch at global query {index}")
            if int(stored_rows[index]["candidate_first_scene"]) != int(row["top_retained_scene"]):
                raise AssertionError(f"{output} census first scene mismatch at global query {index}")
        summary[output] = output_summary

    identity_deltas = []
    for identity in np.unique(identities):
        mask = identities == identity
        delta = float((ap_arrays["fused"][mask].mean() - ap_arrays["baseline_only"][mask].mean()) * 100.0)
        identity_deltas.append(delta)
    identity_directions = {
        "improved": int(np.sum(np.asarray(identity_deltas) > 0)),
        "declined": int(np.sum(np.asarray(identity_deltas) < 0)),
        "unchanged": int(np.sum(np.asarray(identity_deltas) == 0)),
    }
    return {
        "summary": summary,
        "baseline_rank1_errors": baseline_summary,
        "identity_directions": identity_directions,
        "summary_matches_error_census": summary == error_census["summary"],
        "baseline_rank1_errors_match_error_census": baseline_summary == error_census["baseline_rank1_errors"],
        "identity_directions_match_error_census": identity_directions == error_census["identity_directions"],
        "max_ap_delta_abs_diff_to_error_census_rows": max_query_row_diff,
        "causal_limit_preserved": "does not identify" in error_census.get("causal_limit", ""),
    }


def check_receipts(root: Path, comparison: dict[str, object], terminal_files: dict[str, object], exact_verification: dict[str, object], sim_diagnosis: dict[str, object], baseline_parity: dict[str, object], m0: dict[str, object], signal_b0: dict[str, object]) -> dict[str, object]:
    receipts = [load_json(root / f"evidence/msvr310_trifusion_v1_complete/fold_{fold}_receipt.json") for fold in range(3)]
    training_reuse = {
        "fold0_training_execution_commit": receipts[0].get("training_execution_commit"),
        "fold0_training_reused_from_original_run": bool(receipts[0].get("training_reused_from_original_run")),
        "fold0_new_heldout_record_forwards": int(receipts[0].get("new_heldout_record_forwards", -1)),
        "fold1_training_execution_commit": receipts[1].get("training_execution_commit"),
        "fold2_training_execution_commit": receipts[2].get("training_execution_commit"),
        "fold1_training_reused_from_original_run": bool(receipts[1].get("training_reused_from_original_run")),
        "fold2_training_reused_from_original_run": bool(receipts[2].get("training_reused_from_original_run")),
        "summary_new_optimizer_steps": int(comparison["new_optimizer_steps"]),
        "summary_optimizer_steps": int(comparison["optimizer_steps"]),
        "summary_original_training_updates_reused": int(comparison["original_training_updates_reused"]),
        "summary_fold0_retrained": bool(comparison["fold0_retrained"]),
    }

    stage_rows = {}
    for stage in sim_diagnosis["stages"]:
        counts = {row["name"]: row["count"] for row in stage["operator_counts"]}
        stage_rows[stage["name"]] = {
            "bmm": int(counts.get("aten::bmm", 0)),
            "mm": int(counts.get("aten::mm", 0)),
            "matmul": int(counts.get("aten::matmul", 0)),
            "sim_requires_grad_parameters": int(sum(1 for meta in stage["sim_parameters"].values() if meta.get("requires_grad"))),
            "signal_requires_grad_parameters": int(len(stage["signal_requires_grad_names"])),
            "bitwise_equal_to_initial_sim": bool(sim_diagnosis["stage_sim_vs_initial"][stage["name"]]["bitwise_equal"]),
        }
    array_rows = terminal_files["array_checks"]
    array_checks = {
        "array_check_count": len(array_rows),
        "all_flags_true": all(
            row["distance_recomputation_bitwise_equal"]
            and row["saved_rankings_equal_full_saved_distance_sort"]
            and row["maximum_absolute_distance_difference"] == 0.0
            for row in array_rows
        ),
        "shapes_match_receipts": True,
        "retrieval_array_hashes_match_receipts": True,
        "checkpoint_hashes_match_receipts": True,
        "local_binary_arrays_available": False,
    }
    terminal_file_table = terminal_files["files"]
    for receipt in receipts:
        fold = int(receipt["fold"])
        expected_gallery = int(receipt["counts"]["gallery_records"])
        expected_queries = int(receipt["counts"]["valid_queries"])
        for output in OUTPUTS:
            row = next(item for item in array_rows if int(item["fold"]) == fold and item["output"] == output)
            array_checks["shapes_match_receipts"] = array_checks["shapes_match_receipts"] and row["feature_shape"] == [expected_gallery, OUTPUT_WIDTHS[output]]
            array_checks["shapes_match_receipts"] = array_checks["shapes_match_receipts"] and row["distance_shape"] == [expected_queries, expected_gallery]
        retrieval_remote_root = f"comparison_resume_r3/fold_{fold}"
        checkpoint_remote_root = "comparison/fold_0" if fold == 0 else f"comparison_resume_r3/fold_{fold}"
        retrieval_entry = terminal_file_table[f"{retrieval_remote_root}/retrieval_arrays.pt"]
        array_checks["retrieval_array_hashes_match_receipts"] = (
            array_checks["retrieval_array_hashes_match_receipts"]
            and retrieval_entry["sha256"] == receipt["retrieval"]["retrieval_arrays_sha256"]
        )
        checkpoint_entry = terminal_file_table[f"{checkpoint_remote_root}/roles_epoch20.pth"]
        array_checks["checkpoint_hashes_match_receipts"] = (
            array_checks["checkpoint_hashes_match_receipts"]
            and checkpoint_entry["sha256"] == receipt["checkpoint_sha256"]
        )

    return {
        "training_reuse": training_reuse,
        "training_reuse_supported_by_text_receipts": (
            training_reuse["fold0_training_execution_commit"] == "1c444cdf72e13fd041afd0c641dc8f522faa5844"
            and training_reuse["fold0_training_reused_from_original_run"]
            and training_reuse["fold0_new_heldout_record_forwards"] == 0
            and training_reuse["fold1_training_execution_commit"] == comparison["project_commit"]
            and training_reuse["fold2_training_execution_commit"] == comparison["project_commit"]
            and not training_reuse["fold1_training_reused_from_original_run"]
            and not training_reuse["fold2_training_reused_from_original_run"]
            and training_reuse["summary_new_optimizer_steps"] == 520
            and training_reuse["summary_optimizer_steps"] == 780
            and training_reuse["summary_original_training_updates_reused"] == 260
            and not training_reuse["summary_fold0_retrained"]
        ),
        "exact_inference_repair": {
            "status": exact_verification["status"],
            "all_checks_true": all(exact_verification["checks"].values()),
            "batch_sizes": [row["batch_size"] for row in exact_verification["batch_checks"]],
            "optimizer_updates": exact_verification["optimizer_updates"],
            "backward_calls": exact_verification["backward_calls"],
            "checkpoint_writes": exact_verification["checkpoint_writes"],
            "ranking_AP_Rank_evaluations": exact_verification["ranking_AP_Rank_evaluations"],
        },
        "sim_operation_stage_counts": stage_rows,
        "sim_freeze_changes_dispatch_supported": (
            stage_rows["01_repeat_same_sim"]["mm"] == 2
            and stage_rows["02_freeze_sim_only"]["mm"] == 0
            and stage_rows["01_repeat_same_sim"]["bmm"] == 6
            and stage_rows["02_freeze_sim_only"]["bmm"] == 8
            and not stage_rows["02_freeze_sim_only"]["bitwise_equal_to_initial_sim"]
            and stage_rows["03_restore_requires_grad"]["bitwise_equal_to_initial_sim"]
        ),
        "array_receipts": array_checks,
        "three_baseline_parity_receipts": {
            "all_receipt_flags_true": all(row["retrieval"]["baseline_features_and_distances_bitwise_equal_to_b0"] for row in receipts),
            "terminal_all_three_flag": terminal_files["all_three_baselines_bitwise_equal_b0_features_and_distances"],
            "baseline_parity_all_full_model_or_paths_exact": all(
                baseline_parity["baseline_comparisons"][key]["bitwise_equal"]
                for key in ("standalone_before_wrapping", "standalone_after_wrapping", "hierarchical_baseline", "full_model_baseline")
            ),
            "fold0_exact_verification_features_and_distances": (
                exact_verification["checks"]["repaired_all360_baseline_features_bitwise_equal"]
                and exact_verification["checks"]["repaired_all210x360_baseline_distances_bitwise_equal"]
            ),
        },
        "m0_binding": {
            "status": m0["status"],
            "project_commit": m0["project_commit"],
            "config_sha256": m0["config_sha256"],
            "protocol_sha256": m0["protocol_sha256"],
            "optimizer_steps": m0["optimizer_steps"],
            "heldout_record_forwards": m0["heldout_record_forwards"],
            "fold_initial_state_sha256": [fold["initialization"]["initial_state_sha256"] for fold in m0["folds"]],
        },
        "signal_b0_binding": {
            "status": signal_b0["status"],
            "project_commit": signal_b0["project_commit"],
            "config_sha256": signal_b0["config_sha256"],
            "protocol_sha256": signal_b0["protocol_sha256"],
            "optimizer_steps": signal_b0["optimizer_steps"],
            "aggregate": signal_b0["aggregate"],
            "fold_checkpoint_sha256": [fold["checkpoint_sha256"] for fold in signal_b0["folds"]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--epoch-output", type=Path, required=True)
    parser.add_argument("--hash-pre-output", type=Path, required=True)
    parser.add_argument("--hash-post-output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(__file__).resolve().parents[4]
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)

    pre_rows, pre_summary = hash_manifest_files(root, manifest, "pre")
    write_jsonl(args.hash_pre_output, pre_rows)

    config = load_json(root / "configs/MSVR310/TriFusion-source-oof-v1.json")
    protocol = load_json(root / "protocols/msvr310_train_oof_v1.json")
    comparison = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json")
    error_census = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json")
    terminal_files = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json")
    exact_verification = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_exact_signal_inference_verification_20260906.json")
    sim_diagnosis = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json")
    baseline_parity = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json")
    m0 = load_json(root / "evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json")
    signal_b0 = load_json(root / "evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json")

    protocol_summary = replay_protocol(protocol)
    query_summary, query_rows, ap_arrays, first_arrays, identities = replay_query_outputs(root, protocol, comparison)
    write_jsonl(args.query_output, query_rows)
    training_summary, training_rows, epoch_rows = replay_training(root, protocol, config, comparison)
    write_jsonl(args.training_output, training_rows)
    write_jsonl(args.epoch_output, epoch_rows)
    error_summary = replay_error_census(query_rows, comparison, error_census, ap_arrays, first_arrays, identities)
    receipt_summary = check_receipts(root, comparison, terminal_files, exact_verification, sim_diagnosis, baseline_parity, m0, signal_b0)

    post_rows, post_summary = hash_manifest_files(root, manifest, "post")
    write_jsonl(args.hash_post_output, post_rows)

    output_artifacts = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in {
            "query_output_jsonl": args.query_output,
            "training_output_jsonl": args.training_output,
            "epoch_output_jsonl": args.epoch_output,
            "hash_pre_jsonl": args.hash_pre_output,
            "hash_post_jsonl": args.hash_post_output,
        }.items()
    }
    result = {
        "status": "PASS",
        "script": rel(root, Path(__file__)),
        "started_at": now_iso(),
        "completed_at": None,
        "elapsed_seconds": None,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "numpy_path": str(NUMPY_PATH),
        "manifest": {
            "path": rel(root, manifest_path),
            "sha256": manifest_sha,
            "expected_sha256_from_task": "af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059",
            "matches_expected_sha256_from_task": manifest_sha == "af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059",
            "pre_hashcheck": pre_summary,
            "post_hashcheck": post_summary,
            "raw_tensor_image_files_in_package": bool(manifest["raw_tensor_image_files_in_package"]),
            "actual_remote_source_snapshots": manifest["actual_remote_source_snapshots"],
            "signal_source_snapshots": manifest["signal_source_snapshots"],
        },
        "protocol_replay": protocol_summary,
        "query_metric_replay": query_summary,
        "training_scalar_replay": training_summary,
        "error_census_replay": error_summary,
        "receipt_checks": receipt_summary,
        "output_artifacts": output_artifacts,
    }
    result["completed_at"] = now_iso()
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output, result)

    for path in (args.output, args.query_output, args.training_output, args.epoch_output, args.hash_pre_output, args.hash_post_output):
        print(json.dumps({"artifact": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}, sort_keys=True))
    print(json.dumps({
        "status": "PASS",
        "elapsed_seconds": result["elapsed_seconds"],
        "manifest_sha256": manifest_sha,
        "query_output_evaluations": query_summary["query_output_evaluations"],
        "optimizer_steps": training_summary["optimizer_steps"],
        "scientific_passed": query_summary["comparison_replay"]["scientific_passed"],
        "scientific_checks": query_summary["comparison_replay"]["scientific_checks"],
        "max_metric_abs_diff": query_summary["max_aggregate_metric_abs_diff_to_summary"],
        "max_weighted_loss_abs_diff": training_summary["max_weighted_loss_abs_diff"],
        "error_census_matches": error_summary["summary_matches_error_census"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
