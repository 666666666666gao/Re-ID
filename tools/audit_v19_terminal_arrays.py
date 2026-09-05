#!/usr/bin/env python3
"""Recompute V19 terminal metrics and label masks without loading any model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ENDPOINTS = ("frozen_private_tail", "trained_private_tail")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def metrics(ap, rank):
    return {"mAP": float(np.mean(ap) * 100),
            **{f"Rank-{k}": float(np.mean(np.asarray(rank) <= k) * 100)
               for k in (1, 5, 10)}}


def audit(summary):
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert summary["repository_commit"] == "4b749cd92735c228a4bdb1cfacb0b2c6cb80cfe9"
    assert summary["seed"] == 42 and summary["epochs_per_endpoint"] == 20
    assert summary["dev_access_count"] == summary["official_test_access_count"] == 0
    assert not summary["d1_executed"] and summary["source_checkpoint_files_unchanged"]
    folds = summary["folds"]
    assert [fold["fold"] for fold in folds] == [0, 1, 2]
    aps = {end: {name: [] for name in OUTPUTS} for end in ENDPOINTS}
    ranks = {end: {name: [] for name in OUTPUTS} for end in ENDPOINTS}
    identities, fold_gains, scopes, errors, heldout_sets = [], [], [], [], []
    steps = 0
    for fold in folds:
        gallery = fold["gallery_manifest"]
        counts = Counter(row["identity"] for row in gallery)
        camera_counts = Counter((row["identity"], row["camera"]) for row in gallery)
        assert len(counts) == 47 and len({row["file"] for row in gallery}) == len(gallery)
        query = [i for i, row in enumerate(gallery)
                 if counts[row["identity"]] > camera_counts[(row["identity"], row["camera"])]]
        excluded = sorted(set(range(len(gallery))) - set(query))
        identities.extend(gallery[i]["identity"] for i in query)
        heldout_sets.append(set(counts))
        scopes.append({"fold": fold["fold"], "gallery": len(gallery), "queries": len(query),
                       "excluded_only_from_query": len(excluded), "gallery_identities": len(counts)})
        ends = fold["endpoints"]
        assert set(ends) == set(ENDPOINTS)
        a, b = [ends[end] for end in ENDPOINTS]
        for key in ("sample_order_sha256", "first_eight_batch_receipts", "initial_state_sha256"):
            assert a["training"][key] == b["training"][key]
        assert a["outputs"]["baseline_only"] == b["outputs"]["baseline_only"]
        assert a["binding"]["source"] == b["binding"]["source"]
        for end, receipt in ends.items():
            trained = end == ENDPOINTS[1]
            training, binding = receipt["training"], receipt["binding"]
            assert receipt["strict_reload"] and receipt["read_only_evaluation"]
            assert training["epochs"] == len(training["history"]) == 20
            assert [row["epoch"] for row in training["history"]] == list(range(1, 21))
            assert training["optimizer_steps"] == sum(row["batches"] for row in training["history"])
            assert training["overflow_events"] == 0 and training["frozen_state_unchanged"]
            assert training["private_tail_changed"] == trained
            assert not training["missing_nonzero_gradients"]
            assert training["nonzero_gradient_tensors"] == training["trainable_tensors"]
            assert len(set(training["trainable_names"])) == training["trainable_tensors"]
            assert set(binding["heldout_identity_ids"]) == set(counts)
            assert len(binding["fit_identity_ids"]) == 94
            assert set(binding["fit_identity_ids"]).isdisjoint(counts)
            assert binding["train_private_tail"] == trained and binding["baseline_frozen"]
            assert binding["private_storage_disjoint"] and binding["source_state_strictly_loaded"]
            steps += training["optimizer_steps"]
            assert set(receipt["outputs"]) == set(OUTPUTS)
            for name, output in receipt["outputs"].items():
                assert output["query_indices"] == query
                assert output["excluded_no_cross_camera_positive"] == excluded
                ap = np.asarray(output["average_precision"], dtype=np.float64)
                rank = np.asarray(output["first_match_rank"])
                assert len(ap) == len(rank) == len(query)
                assert np.all(np.isfinite(ap)) and np.all((ap >= 0) & (ap <= 1))
                assert np.all((rank >= 1) & (rank <= len(gallery)))
                recalculated = metrics(ap, rank)
                errors.extend(abs(value - output["metrics_percent"][key])
                              for key, value in recalculated.items())
                aps[end][name].extend(ap.tolist())
                ranks[end][name].extend(rank.tolist())
        fold_gains.append(metrics(b["outputs"]["fused"]["average_precision"],
                                  b["outputs"]["fused"]["first_match_rank"])["mAP"]
                          - metrics(a["outputs"]["fused"]["average_precision"],
                                    a["outputs"]["fused"]["first_match_rank"])["mAP"])
    assert sum(len(ids) for ids in heldout_sets) == len(set.union(*heldout_sets)) == 141
    assert steps == 3360 and len(identities) == summary["total_eligible_queries"] == 571
    assert sum(row["gallery"] for row in scopes) == summary["total_gallery_records"] == 3126
    aggregate = {end: {name: metrics(aps[end][name], ranks[end][name]) for name in OUTPUTS}
                 for end in ENDPOINTS}
    for end in ENDPOINTS:
        for name in OUTPUTS:
            errors.extend(abs(value - summary["aggregate"][end][name][key])
                          for key, value in aggregate[end][name].items())
    a, b = ENDPOINTS
    gains = {name: aggregate[b][name]["mAP"] - aggregate[a][name]["mAP"] for name in OUTPUTS}
    differences = np.asarray(aps[b]["fused"]) - np.asarray(aps[a]["fused"])
    ids = np.asarray(identities)
    clusters = np.unique(ids)
    assert len(clusters) == summary["bootstrap"]["clusters"] == 21
    cluster_sums = np.array([differences[ids == identity].sum() for identity in clusters])
    cluster_sizes = np.array([np.count_nonzero(ids == identity) for identity in clusters])
    # Sum/count resampling independently avoids the runner's concatenated query arrays.
    samples = np.random.default_rng(42).integers(0, len(clusters), size=(10000, len(clusters)))
    bootstrap_means = cluster_sums[samples].sum(axis=1) / cluster_sizes[samples].sum(axis=1)
    lower = float(np.quantile(bootstrap_means, 0.025) * 100)
    assert summary["bootstrap"]["resamples"] == 10000
    errors.append(abs(lower - summary["bootstrap"]["lower_bound_95_mAP"]))
    errors.extend(abs(gains[name] - summary["matched_gains_mAP"][name]) for name in OUTPUTS)
    errors.extend(abs(x - y) for x, y in zip(fold_gains, summary["fold_fused_gains_mAP"], strict=True))
    assert max(errors) < 1e-9, max(errors)
    checks = {
        "aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.0,
        "all_fold_fused_nonnegative": all(value >= 0 for value in fold_gains),
        "all_expert_aggregate_nonnegative": all(gains[name] >= 0 for name in OUTPUTS[2:]),
        "fused_bootstrap_lower_positive": lower > 0,
        "fused_beats_baseline_and_experts": all(aggregate[b]["fused"]["mAP"] > aggregate[b][name]["mAP"]
                                                for name in ("baseline_only", *OUTPUTS[2:])),
    }
    assert checks == summary["scientific_checks"]
    assert all(checks.values()) == summary["next_phase_qualified"] == (summary["status"] == "Q1_PASS")
    identity_rows = [{"identity": int(identity), "queries": int(size),
                      "mean_gain_mAP": float(total / size * 100),
                      "weighted_contribution_mAP": float(total / len(ids) * 100)}
                     for identity, size, total in zip(clusters, cluster_sizes, cluster_sums, strict=True)]
    paired_rows = {}
    for name in OUTPUTS:
        delta = np.asarray(aps[b][name]) - np.asarray(aps[a][name])
        ra, rb = np.asarray(ranks[a][name]), np.asarray(ranks[b][name])
        paired_rows[name] = {"ap_improved": int(np.count_nonzero(delta > 0)),
                             "ap_declined": int(np.count_nonzero(delta < 0)),
                             "ap_equal": int(np.count_nonzero(delta == 0)),
                             "rank1_repaired": int(np.count_nonzero((ra > 1) & (rb == 1))),
                             "rank1_broken": int(np.count_nonzero((ra == 1) & (rb > 1)))}
    return {"verification_passed": True, "scientific_status": summary["status"],
            "scope": "All six endpoint receipts, gallery label masks, AP/rank aggregation and bootstrap; no model, feature or distance replay",
            "numpy_version": np.__version__, "optimizer_steps_recounted": steps,
            "max_absolute_numeric_difference_percent": max(errors), "fold_scopes": scopes,
            "aggregate": aggregate, "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains,
            "bootstrap_lower_bound_95_mAP": lower, "scientific_checks": checks,
            "all_identity_fused_gains": identity_rows, "all_query_paired_changes": paired_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    report = audit(json.loads(raw))
    report["input_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("verification_passed", "scientific_status",
                                                 "max_absolute_numeric_difference_percent",
                                                 "bootstrap_lower_bound_95_mAP")}))
