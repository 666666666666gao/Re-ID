#!/usr/bin/env python3
"""Recompute V23 terminal metrics and label masks without loading any model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ENDPOINTS = ("frozen_zero_spectral_adapter", "trained_spectral_adapter")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def metrics(ap, rank):
    return {"mAP": float(np.mean(ap) * 100),
            **{f"Rank-{k}": float(np.mean(np.asarray(rank) <= k) * 100)
               for k in (1, 5, 10)}}


def audit_training(training, binding, fold, trained):
    assert training["spectral_adapter_enabled"] is trained
    assert binding["spectral_adapters_trainable"] is trained
    assert training["optimizer_steps"] == (580, 560, 540)[fold]
    expected_parameters, expected_tensors = (9618828, 239) if trained else (7841292, 203)
    assert (binding["total_parameters"], binding["trainable_parameters"],
            binding["trainable_tensors"]) == (100577677, expected_parameters, expected_tensors)
    assert training["trainable_tensors"] == expected_tensors
    assert binding["spectral_adapter_modules"] == 9 and binding["spectral_adapter_tensors"] == 36
    assert binding["adapter_parameters_shared_across_expert_roles"]
    loss_errors = []
    for row in training["history"]:
        assert row["batches"] == (29, 28, 27)[fold]
        ordinary = .25 * sum(row[f"mean_triplet_residual_{e}"] for e in OUTPUTS[2:])
        common = .25 * row["mean_id_fused"] + row["mean_triplet_fused"]
        for e in OUTPUTS[2:]:
            common += row[f"mean_id_{e}"] / 12 + .25 * row[f"mean_triplet_{e}"] + row[f"mean_id_residual_{e}"] / 12
        loss_errors.extend([abs(ordinary - row["mean_ordinary_residual_triplet"]),
                            abs(common - row["mean_common_identity_and_branch_triplet"]),
                            abs(common + ordinary - row["mean_total"])])
    assert max(loss_errors) < 1e-6
    return {"fold": fold, "spectral_adapter_enabled": trained,
            "optimizer_steps": training["optimizer_steps"],
            "trainable_parameters": expected_parameters, "trainable_tensors": expected_tensors,
            "max_loss_component_roundoff": max(loss_errors)}


def audit(summary):
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert summary["repository_commit"] == "9f4a10b6162b9658ba103cd92466411ebb6ccd8f"
    assert summary["seed"] == 42 and summary["epochs_per_endpoint"] == 20
    assert summary["m0"]["passed"] and all(summary["m0"]["checks"].values())
    assert summary["new_inference_parameters"] == 1777536
    assert summary["dev_access_count"] == summary["official_test_access_count"] == 0
    assert not summary["d1_executed"] and summary["source_checkpoint_files_unchanged"]
    assert summary["evaluation_type"] == "real_gt_train_internal_complete_path_oof_reused_development_qualification"
    assert summary["oof_is_reused_development_qualification"]
    training_audits = []
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
        for key in ("source", "source_model_state_sha256", "fit_identity_ids", "heldout_identity_ids",
                    "new_inference_parameters", "total_parameters"):
            assert a["binding"][key] == b["binding"][key], key
        for end, receipt in ends.items():
            trained = end == ENDPOINTS[1]
            training, binding = receipt["training"], receipt["binding"]
            preflight = summary["preflight"][fold["fold"]]["endpoints"][int(trained)]
            assert preflight["endpoint"] == end and preflight["binding"] == binding
            assert preflight["initial_state_sha256"] == training["initial_state_sha256"]
            assert preflight["legacy_zero_adapter_first_batch_exact"] and preflight["state_unchanged"]
            assert receipt["strict_reload"] and receipt["read_only_evaluation"]
            assert training["epochs"] == len(training["history"]) == 20
            assert [row["epoch"] for row in training["history"]] == list(range(1, 21))
            assert training["optimizer_steps"] == sum(row["batches"] for row in training["history"])
            training_audits.append(audit_training(training, binding, fold["fold"], trained))
            assert training["overflow_events"] == 0 and training["frozen_state_unchanged"]
            assert not training["missing_nonzero_gradients"]
            assert training["nonzero_gradient_tensors"] == training["trainable_tensors"]
            assert len(set(training["trainable_names"])) == training["trainable_tensors"]
            assert set(binding["heldout_identity_ids"]) == set(counts)
            assert len(binding["fit_identity_ids"]) == 94
            assert set(binding["fit_identity_ids"]).isdisjoint(counts)
            assert binding["baseline_frozen"] and binding["shared_tail_frozen"]
            assert binding["source_state_strictly_loaded"] and binding["new_inference_parameters"] == 1777536
            assert binding["architecture"] == "signal_preserving_v23_spectral_adapter"
            assert not binding["router_enabled"] and not binding["hfer_enabled"]
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
            "training_loss_and_parameter_audits": training_audits,
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
    summary = json.loads(raw)
    report = audit(summary)
    report["input_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: report[key] for key in ("verification_passed", "scientific_status",
                                                 "max_absolute_numeric_difference_percent",
                                                 "bootstrap_lower_bound_95_mAP")}))
