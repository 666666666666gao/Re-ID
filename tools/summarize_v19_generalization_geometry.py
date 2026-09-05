#!/usr/bin/env python3
"""Aggregate complete V19 diagnostic arrays without model or data access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENDPOINTS = ("frozen_private_tail", "trained_private_tail")
EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)


def summarize(diagnostic):
    assert diagnostic["status"] == "ALL_SIX_FINAL_MODELS_REPLAY_AND_SOURCE_DIAGNOSIS_COMPLETE"
    assert diagnostic["optimizer_steps"] == diagnostic["checkpoint_writes"] == 0
    assert diagnostic["dev_access_count"] == diagnostic["official_test_access_count"] == 0
    aggregates = {}
    for endpoint in ENDPOINTS:
        aggregates[endpoint] = {}
        for scope in ("source", "heldout"):
            rows = [fold["endpoints"][endpoint][scope] for fold in diagnostic["folds"]]
            count = sum(row["gallery"] for row in rows)
            queries = sum(row["eligible_queries"] for row in rows)
            assert (count, queries) == ((6252, 1142) if scope == "source" else (3126, 571))
            metrics = {}
            for output in OUTPUTS:
                ap = [x for row in rows for x in row["outputs"][output]["scores"]["average_precision"]]
                rank = np.array([x for row in rows for x in row["outputs"][output]["scores"]["first_match_rank"]])
                assert len(ap) == len(rank) == queries
                metrics[output] = {"mAP": float(np.mean(ap) * 100),
                                   **{f"Rank-{k}": float(np.mean(rank <= k) * 100) for k in (1, 5, 10)}}
            pair_rows, grouped = {}, {}
            for expert in EXPERTS:
                pair_rows[expert], grouped[expert] = {}, {}
                groups = {"same_modality": [], "different_modality": []}
                for pair in rows[0]["modal_geometry"][expert]:
                    arrays = [row["modal_geometry"][expert][pair]["arrays"] for row in rows]
                    keys = ("same_instance_cosine", "mean_cross_camera_positive_cosine",
                            "nearest_positive_cosine", "nearest_negative_cosine", "nearest_negative_same_camera")
                    data = np.array([[x for arr in arrays for x in arr[key]] for key in keys]).T
                    assert data.shape == (queries, 5) and np.isfinite(data).all()
                    pair_rows[expert][pair] = {
                        **{key: float(data[:, i].mean()) for i, key in enumerate(keys)},
                        "nearest_cosine_margin": float(np.mean(data[:, 2] - data[:, 3])),
                        "negative_at_least_as_close_percent": float(np.mean(data[:, 3] >= data[:, 2]) * 100),
                    }
                    left, right = pair.split("_to_")
                    groups["same_modality" if left == right else "different_modality"].append(data)
                for group, matrices in groups.items():
                    data = np.concatenate(matrices)
                    grouped[expert][group] = {
                        "query_modality_pairs": len(data),
                        **{key: float(data[:, i].mean()) for i, key in enumerate(keys)},
                        "nearest_cosine_margin": float(np.mean(data[:, 2] - data[:, 3])),
                        "negative_at_least_as_close_percent": float(np.mean(data[:, 3] >= data[:, 2]) * 100),
                    }
            aggregates[endpoint][scope] = {"gallery_memberships": count, "eligible_query_memberships": queries,
                                           "metrics_percent": metrics, "all_nine_modality_pairs": pair_rows,
                                           "grouped_modality_geometry": grouped}
            if scope == "source":
                classification = {}
                for head in rows[0]["clean_source_classification"]:
                    labels = np.array([x["identity"] for row in rows for x in row["gallery_manifest"]])
                    pred = np.array([x for row in rows for x in row["clean_source_classification"][head]["predicted_identity"]])
                    ce = [x for row in rows for x in row["clean_source_classification"][head]["smoothed_cross_entropy"]]
                    assert len(labels) == len(pred) == len(ce) == count
                    classification[head] = {"correct": int((labels == pred).sum()), "total": count,
                                            "accuracy_percent": float(np.mean(labels == pred) * 100),
                                            "mean_smoothed_cross_entropy": float(np.mean(ce))}
                aggregates[endpoint][scope]["clean_source_classification"] = classification
    changes = {}
    for output in OUTPUTS:
        deltas = []
        for fold in diagnostic["folds"]:
            a, b = [fold["endpoints"][end]["heldout"]["outputs"][output]["queries"] for end in ENDPOINTS]
            for left, right in zip(a, b, strict=True):
                assert left["query_index"] == right["query_index"]
                deltas.append({key: right[key] - left[key] for key in ("positive_distance", "negative_distance", "nearest_margin")})
        assert len(deltas) == 571
        changes[output] = {key: float(np.mean([row[key] for row in deltas])) for key in deltas[0]}
    return {"scope": "arithmetic_summary_of_all_terminal_diagnostic_arrays_no_model_replay",
            "numpy_version": np.__version__, "aggregates": aggregates, "heldout_paired_distance_changes": changes,
            "source_and_heldout_triplet_forwards": 18756, "scientific_status_unchanged": "Q1_FAIL"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.diagnostic.read_bytes()
    report = summarize(json.loads(raw))
    report["diagnostic_sha256"] = hashlib.sha256(raw).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({"written": str(args.output), "scientific_status_unchanged": report["scientific_status_unchanged"]}))
