#!/usr/bin/env python3
"""Verify saved V24 source-diagnostic arrays without loading models or features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def audit(diagnostic, q1):
    assert diagnostic["status"] == "ALL_NINE_SOURCE_MODELS_DIAGNOSED_READ_ONLY"
    assert (diagnostic["models_extracted"], diagnostic["source_record_forwards"],
            diagnostic["source_batch_forward_calls"]) == (9, 18756, 153)
    for key in ("optimizer_steps", "backward_calls", "checkpoint_writes", "heldout_image_forward_calls",
                "dev_access_count", "official_test_access_count"):
        assert diagnostic[key] == 0
    registry = sorted([row for fold in q1["folds"] for row in fold["gallery_manifest"]], key=lambda row: row["file"])
    numerical_errors, probability_errors, rows = [], [], []

    def check_distribution(values, reported):
        values = np.asarray(values, dtype=np.float64)
        assert len(values) == reported["count"] and np.isfinite(values).all()
        numerical_errors.append(abs(float(values.mean()) - reported["mean"]))
        for level, expected in reported["quantiles"].items():
            numerical_errors.append(abs(float(np.quantile(values, float(level))) - expected))

    def check_columns(block, size):
        assert set(block["per_record"]) == set(block["distributions"])
        for name, values in block["per_record"].items():
            assert len(values) == size
            check_distribution(values, block["distributions"][name])

    for index, fold in enumerate(diagnostic["folds"]):
        assert fold["fold"] == index
        fit = fold["source_identity_mapping"]
        mapping = {identity: label for label, identity in enumerate(fit)}
        expected = [{"file": row["file"], "source_label": mapping[row["identity"]], "camera": row["camera"],
                     "fit_registry_identity": row["identity"]} for row in registry if row["identity"] in mapping]
        assert fold["source_manifest"] == expected and len(fit) == 94
        labels = np.array([row["source_label"] for row in expected])
        cameras = np.array([row["camera"] for row in expected])
        size = len(labels)
        assert size == (2126, 2075, 2051)[index]
        cross = np.array([len(set(cameras[labels == label])) > 1 for label in labels])
        cross_indices = np.flatnonzero(cross)
        assert len(np.unique(labels[cross])) == 14
        assert set(fold["models"]) == {"initial", "ordinary_two_view", "environment_identity_prototype"}
        for name, model in fold["models"].items():
            endpoint = "ordinary_two_view" if name == "initial" else name
            saved = q1["folds"][index]["endpoints"][endpoint]
            assert model["binding"] == saved["binding"]
            stage = "initial" if name == "initial" else "final"
            extraction = model["extraction"]
            assert extraction["state_unchanged"] and extraction["record_forwards"] == size
            assert extraction["batch_forward_calls"] == 17
            assert extraction["model_state_sha256"] == saved["training"][stage + "_state_sha256"]
            receipt = saved["training"]["memory_" + stage]
            assert model["memory_file"] == receipt["file"] and model["memory_file_sha256"] == receipt["file_sha256"]
            assert model["cached_memory_state_sha256"] == receipt["state_sha256"]
            if name == "initial":
                assert model["fresh_memory_state_sha256"] == receipt["state_sha256"]
            drift = model["memory_drift"]
            for key in ("pair_labels", "pair_cameras", "last_update", "update_count"):
                assert drift[key] == receipt[key]
            pair_labels, pair_cameras = np.array(drift["pair_labels"]), np.array(drift["pair_cameras"])
            assert len(pair_labels) == len(drift["pair_cosine"]) == 108 and len(drift["global_cosine"]) == 94
            for kind in ("pair", "global"):
                check_distribution(drift[kind + "_cosine"], drift[kind + "_cosine_distribution"])
            for kind in ("cached", "fresh"):
                block = model[kind + "_prototype_scores"]
                check_columns(block, size)
                data = block["per_record"]
                assert np.array_equal(data["global_correct"], np.array(block["global_predicted_source_label"]) == labels)
                true_pair, predicted_pair = np.array(block["environment_true_pair"]), np.array(block["environment_predicted_pair"])
                assert np.array_equal(pair_labels[true_pair], labels) and np.array_equal(pair_cameras[true_pair], cameras)
                assert np.array_equal(pair_cameras[predicted_pair], cameras)
                assert np.array_equal(data["environment_correct"], true_pair == predicted_pair)
                assert np.all(np.array(block["global_nearest_negative_source_label"]) != labels)
                assert np.array_equal(data["environment_candidate_pairs"], (cameras[:, None] == pair_cameras).sum(axis=1))
                for term in ("global", "environment"):
                    ce = np.asarray(data[term + "_cross_entropy"])
                    probability = np.asarray(data[term + "_true_probability"])
                    assert np.all(ce >= 0) and np.all((probability >= 0) & (probability <= 1))
                    probability_errors.append(float(np.max(np.abs(np.exp(-ce) - probability))))
                combined = (np.asarray(data["global_cross_entropy"]) + data["environment_cross_entropy"]) / 2
                probability_errors.append(float(np.max(np.abs(combined - data["combined_cross_entropy"]))))
                margin = np.asarray(data["global_positive_cosine"]) - data["global_nearest_negative_cosine"]
                probability_errors.append(float(np.max(np.abs(margin - data["global_positive_minus_nearest_negative_cosine"]))))
                assert block["original_loss_mean_absolute_error"] < 1e-6
                means = block["all_source_identity_means"]
                assert [row["source_label"] for row in means] == list(range(94))
                for row in means:
                    selected = labels == row["source_label"]
                    assert int(selected.sum()) == row["records"]
                    for column, values in data.items():
                        numerical_errors.append(abs(float(np.asarray(values, dtype=float)[selected].mean()) - row[column]))
            geometry = model["sample_geometry"]
            check_columns(geometry, size)
            values = geometry["per_record"]
            negative = np.array(geometry["nearest_negative_record_index"])
            assert np.all((negative >= 0) & (negative < size)) and np.all(labels[negative] != labels)
            assert np.array_equal(values["nearest_negative_same_camera"], cameras[negative] == cameras)
            for kind in ("nearest", "hardest"):
                positive = np.array(geometry[kind + "_positive_record_index"])
                assert np.all((positive >= 0) & (positive < size))
                assert np.array_equal(labels[positive], labels) and np.all(positive != np.arange(size))
                margin = np.asarray(values[kind + "_positive_cosine"]) - values["nearest_negative_cosine"]
                probability_errors.append(float(np.max(np.abs(margin - values[kind + "_positive_minus_nearest_negative_cosine"]))))
            assert geometry["hardest_positive_margin_negative_records"] == int(np.sum(np.asarray(values["hardest_positive_cosine"]) < values["nearest_negative_cosine"]))
            fresh_negative = model["fresh_prototype_scores"]["per_record"]["global_nearest_negative_cosine"]
            gap = np.asarray(values["nearest_negative_cosine"]) - fresh_negative
            probability_errors.append(float(np.max(np.abs(gap - values["sample_negative_minus_fresh_negative_prototype_cosine"]))))
            subset = geometry["real_cross_camera_positive_subset"]
            assert subset["query_indices"] == cross_indices.tolist() and subset["identity_count"] == 14
            check_columns(subset, len(cross_indices))
            for kind in ("nearest", "hardest"):
                positive = np.array(subset[kind + "_cross_camera_positive_index"])
                assert np.array_equal(labels[positive], labels[cross]) and np.all(cameras[positive] != cameras[cross])
                margin = np.asarray(subset["per_record"][kind + "_cross_camera_positive_cosine"]) - np.asarray(values["nearest_negative_cosine"])[cross]
                probability_errors.append(float(np.max(np.abs(margin - subset["per_record"][kind + "_cross_positive_minus_nearest_negative_cosine"]))))
            cached = model["cached_prototype_scores"]["distributions"]
            fresh = model["fresh_prototype_scores"]["distributions"]
            rows.append({"fold": index, "model": name, "source_records": size, "cross_camera_source_queries": int(cross.sum()),
                         "cached_combined_ce": cached["combined_cross_entropy"]["mean"],
                         "fresh_combined_ce": fresh["combined_cross_entropy"]["mean"],
                         "cached_global_accuracy_percent": cached["global_correct"]["mean"] * 100,
                         "fresh_global_accuracy_percent": fresh["global_correct"]["mean"] * 100,
                         "sample_hardest_positive_margin_negative_records": geometry["hardest_positive_margin_negative_records"],
                         "sample_negative_minus_prototype_negative_cosine_mean": float(gap.mean()),
                         "cached_fresh_pair_cosine_mean": drift["pair_cosine_distribution"]["mean"]})
    assert len(rows) == 9 and max(numerical_errors) < 1e-9 and max(probability_errors) < 1e-6
    return {"verified": True, "scope": "Saved JSON distributions, source labels, index masks and algebra only; no feature/distance or tensor replay",
            "all_source_record_arrays_checked": 18756, "prototype_identity_mean_rows_checked": 1692,
            "max_distribution_or_mean_error": max(numerical_errors),
            "max_probability_or_float32_algebra_error": max(probability_errors), "all_model_rows": rows,
            "numpy_version": np.__version__, "local_model_tensor_image_loads": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    raw, qraw = args.diagnostic.read_bytes(), args.q1_summary.read_bytes()
    diagnostic, q1 = json.loads(raw), json.loads(qraw)
    assert diagnostic["q1_summary_sha256"] == hashlib.sha256(qraw).hexdigest()
    result = audit(diagnostic, q1)
    result.update({"input_diagnostic_sha256": hashlib.sha256(raw).hexdigest(),
                   "input_q1_sha256": hashlib.sha256(qraw).hexdigest(),
                   "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "elapsed_seconds": time.perf_counter() - started})
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
