#!/usr/bin/env python3
"""Independently recompute all3000 query outputs and780 logged training updates."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def metrics(aps, ranks):
    return {"mAP": statistics.fmean(aps) * 100,
            **{f"Rank-{k}": statistics.fmean(r <= k for r in ranks) * 100 for k in (1, 5, 10)}}


def main(args):
    import numpy as np

    started = time.perf_counter()
    assert not args.output.exists()
    summary, config, protocol, m0 = [json.loads(p.read_text(encoding="utf-8"))
                                     for p in (args.summary, args.config, args.protocol, args.m0)]
    assert summary["mode"] == "comparison_resume_r3" and len(summary["folds"]) == 3
    assert summary["optimizer_steps"] == 780 and summary["new_optimizer_steps"] == 520
    assert summary["original_training_updates_reused"] == 260 and not summary["fold0_retrained"]
    assert summary["new_heldout_record_forwards"] == 672 and summary["evaluated_gallery_records"] == 1032
    assert summary["official_test_image_access"] == summary["rgbnt201_dev_image_access"] == 0
    assert summary["seed"] == 42 and summary["checkpoint_selection"] == "fixed_epoch20"
    assert summary["source_only_training"] and not summary["m0_weights_reused"] and not summary["rgbnt201_role_weights_reused"]
    assert summary["original_failed_run_files_unchanged"]
    records, weights = protocol["records"], config["LOSS"]
    assert len(records) == 1032 and len({r["identity"] for r in records}) == 155
    all_aps, all_ranks = {k: [] for k in OUTPUTS}, {k: [] for k in OUTPUTS}
    all_ids, gallery_union, fold_rows, history_rows, per_identity = [], [], [], [], []
    loss_errors, metric_errors, mean_errors, epoch_log_rows = [], [], [], []
    for actual, fold, preflight in zip(summary["folds"], protocol["folds"], m0["folds"], strict=True):
        index = fold["fold"]
        assert actual["fold"] == index == preflight["fold"]
        assert actual["counts"] == fold["counts"]
        assert actual["training_reused_from_original_run"] == (index == 0)
        assert actual["new_heldout_record_forwards"] == (0 if index == 0 else len(fold["gallery_record_indices"]))
        initial, training, retrieval = actual["initialization"], actual["training"], actual["retrieval"]
        assert initial == preflight["initialization"]
        assert initial["source_ids"] == fold["source_ids"] and initial["heldout_ids"] == fold["heldout_ids"]
        assert not set(fold["source_ids"]) & set(fold["heldout_ids"])
        assert initial["initial_state_sha256"] == training["initial_state_sha256"]
        assert training["trainable_tensors"] == training["nonzero_gradient_tensors"] == 203
        assert not training["missing_nonzero_gradients"] and training["overflow_events"] == 0
        assert training["frozen_state_before_sha256"] == training["frozen_state_after_sha256"]
        assert training["signal_state_before_sha256"] == training["signal_state_after_sha256"] == initial["signal_state_sha256"]
        assert training["initial_state_sha256"] != training["final_state_sha256"] == actual["strict_reload_state_sha256"]
        assert all(actual["engineering_checks"].values())
        source = set(fold["source_record_indices"])
        gallery = [records[i] for i in fold["gallery_record_indices"]]
        gallery_union.extend(fold["gallery_record_indices"])
        assert not source & set(fold["gallery_record_indices"])
        assert source | set(fold["gallery_record_indices"]) == set(range(1032))
        assert retrieval["gallery_manifest"] == gallery and retrieval["query_rows"] == fold["query_rows"]
        assert retrieval["baseline_features_and_distances_bitwise_equal_to_b0"]
        expected_queries = [(i, r["index"], sum(s["identity"] == r["identity"] and s["scene"] != r["scene"] for s in gallery))
                            for i, r in enumerate(gallery)]
        expected_queries = [row for row in expected_queries if row[2]]
        assert expected_queries == [(q["gallery_position"], q["record_index"], q["valid_positives"]) for q in fold["query_rows"]]
        ranking_path = args.rankings_dir / f"fold_{index}_rankings.json"
        assert hashlib.sha256(ranking_path.read_bytes()).hexdigest() == retrieval["rankings_sha256"]
        rankings = json.loads(ranking_path.read_text())
        fold_metrics, identity_values = {}, defaultdict(lambda: {k: [] for k in OUTPUTS})
        for name in OUTPUTS:
            aps, first = [], []
            recorded = retrieval["outputs"][name]
            for q, order, ap_recorded, first_recorded in zip(fold["query_rows"], rankings[name],
                    recorded["average_precision"], recorded["first_match_rank"], strict=True):
                assert sorted(order) == list(range(len(gallery)))
                matches = [gallery[i]["identity"] == q["identity"] for i in order
                           if not (gallery[i]["identity"] == q["identity"] and gallery[i]["scene"] == q["scene"])]
                positions = [i + 1 for i, match in enumerate(matches) if match]
                assert len(positions) == q["valid_positives"] and first_recorded == positions[0]
                ap = statistics.fmean(j / rank for j, rank in enumerate(positions, 1))
                metric_errors.append(abs(ap - ap_recorded))
                aps.append(ap)
                first.append(positions[0])
                identity_values[q["identity"]][name].append(ap)
            fold_metrics[name] = metrics(aps, first)
            metric_errors.extend(abs(value - recorded["metrics"][key]) for key, value in fold_metrics[name].items())
            assert max(recorded["upstream_metric_difference_pp"].values()) < 1e-5
            all_aps[name].extend(aps)
            all_ranks[name].extend(first)
        all_ids.extend(q["identity"] for q in fold["query_rows"])
        for identity, values in sorted(identity_values.items()):
            per_identity.append({"identity": identity, "query_count": len(values["fused"]),
                                 "map_by_output": {k: statistics.fmean(v) * 100 for k, v in values.items()}})
        assert training["epochs"] == len(training["history"]) == 20
        assert training["optimizer_steps"] == len(training["steps"]) == 260
        by_epoch, exposed, cross_pairs, positive_pairs = defaultdict(list), [], 0, 0
        for number, step in enumerate(training["steps"], 1):
            assert step["step"] == number and step["epoch"] == (number - 1) // 13 + 1
            sampled = step["sampled_record_indices"]
            assert len(sampled) == 64 and set(sampled) <= source
            batch = [records[i] for i in sampled]
            assert sorted(Counter(r["identity"] for r in batch).values()) == [8] * 8
            assert step["amp_scale_after"] >= step["amp_scale_before"]
            c = step["components"]
            value = weights["ID_FUSED"] * c["id_fused"] + weights["TRIPLET_FUSED"] * c["triplet_fused"]
            for expert in OUTPUTS[2:]:
                value += weights["ID_BRANCH"] * c["id_" + expert] + weights["TRIPLET_BRANCH"] * c["triplet_" + expert]
                value += weights["ID_RESIDUAL"] * c["id_residual_" + expert] + weights["TRIPLET_RESIDUAL"] * c["triplet_residual_" + expert]
            assert math.isfinite(value) and math.isfinite(step["loss"])
            loss_errors.append(abs(value - step["loss"]))
            by_epoch[step["epoch"]].append(step["loss"])
            exposed.extend(sampled)
            for i, a in enumerate(batch):
                for b in batch[i + 1:]:
                    if a["identity"] == b["identity"]:
                        positive_pairs += 1
                        cross_pairs += int(a["scene"] != b["scene"])
        for number, epoch in enumerate(training["history"], 1):
            assert epoch["epoch"] == number and epoch["optimizer_steps"] == len(by_epoch[number]) == 13
            factor = number / 5 if number <= 5 else 0.5 * (1 + math.cos(math.pi * (number - 6) / 15))
            assert abs(epoch["learning_rate"] - config["OPTIMIZATION"]["NEW_MODULE_LR"] * factor) < 1e-15
            mean_errors.append(abs(statistics.fmean(by_epoch[number]) - epoch["mean_loss"]))
            epoch_log_rows.append({"event": "msvr310_role_epoch", "fold": index, "mode": "comparison", **epoch})
        history_rows.extend(training["history"])
        fold_rows.append({"fold": index, "metrics": fold_metrics, "source_record_exposures": len(exposed),
                          "unique_source_records_exposed": len(set(exposed)),
                          "source_identities_exposed": len({records[i]["identity"] for i in exposed}),
                          "same_identity_positive_pairs": positive_pairs, "cross_scene_positive_pairs": cross_pairs,
                          "optimizer_steps": 260, "gallery_records": len(gallery), "queries": len(fold["query_rows"])})
    assert sorted(gallery_union) == list(range(1032)) and len(all_ids) == 600 and len(set(all_ids)) == 60
    logs = args.original_log.read_text(encoding="utf-8") + "\n" + args.resume_log.read_text(encoding="utf-8")
    observed_epochs = [json.loads(line) for line in logs.splitlines() if line.startswith('{"event": "msvr310_role_epoch"')]
    assert observed_epochs == epoch_log_rows and len(history_rows) == 60
    computed_metrics = {name: metrics(all_aps[name], all_ranks[name]) for name in OUTPUTS}
    comparison = summary["comparison"]
    for name in OUTPUTS:
        metric_errors.extend(abs(v - comparison["metrics"][name][k]) for k, v in computed_metrics[name].items())
    identities = np.asarray(all_ids)
    unique = np.unique(identities)
    differences = (np.asarray(all_aps["fused"]) - np.asarray(all_aps["baseline_only"])) * 100
    sums = np.asarray([differences[identities == identity].sum() for identity in unique])
    counts = np.asarray([np.sum(identities == identity) for identity in unique])
    draws = np.random.default_rng(42).choice(len(unique), size=(10000, len(unique)), replace=True)
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lower = float(np.percentile(means, 2.5, method="linear"))
    bootstrap_error = abs(lower - comparison["identity_bootstrap"]["lower_bound_pp"])
    assert bootstrap_error < 1e-10 and max(metric_errors) < 1e-10 and max(mean_errors) < 1e-10
    gains = {name: computed_metrics[name]["mAP"] - computed_metrics["baseline_only"]["mAP"] for name in OUTPUTS}
    fold_gains = [r["metrics"]["fused"]["mAP"] - r["metrics"]["baseline_only"]["mAP"] for r in fold_rows]
    assert all(abs(gains[k] - comparison["gains_over_signal_pp"][k]) < 1e-10 for k in OUTPUTS)
    assert all(abs(a - b) < 1e-10 for a, b in zip(fold_gains, comparison["fold_fused_gains_pp"], strict=True))
    for recomputed, recorded in zip(sorted(per_identity, key=lambda r: r["identity"]), comparison["per_identity"], strict=True):
        assert recomputed["identity"] == recorded["identity"] and recomputed["query_count"] == recorded["query_count"]
        assert all(abs(recomputed["map_by_output"][k] - recorded["map_by_output"][k]) < 1e-10 for k in OUTPUTS)
    recorded_ap = {name: np.asarray([v for f in summary["folds"] for v in f["retrieval"]["outputs"][name]["average_precision"]])
                   for name in OUTPUTS}
    for name in OUTPUTS[1:]:
        delta = recorded_ap[name] - recorded_ap["baseline_only"]
        first, baseline_first = np.asarray(all_ranks[name]), np.asarray(all_ranks["baseline_only"])
        changes = {"ap_improved": int(np.sum(delta > 0)), "ap_declined": int(np.sum(delta < 0)), "ap_unchanged": int(np.sum(delta == 0)),
                   "rank1_repaired": int(np.sum((baseline_first > 1) & (first == 1))),
                   "rank1_new_errors": int(np.sum((baseline_first == 1) & (first > 1)))}
        assert changes == comparison["query_changes"][name]
    assert comparison["identity_bootstrap"]["seed"] == 42 and comparison["identity_bootstrap"]["resamples"] == 10000
    assert comparison["identity_bootstrap"]["cluster_count"] == 60
    checks = {"fused_gain_at_least_1pp": gains["fused"] >= 1,
              "all_fold_fused_gains_nonnegative": all(x >= 0 for x in fold_gains),
              "all_full_branches_not_below_signal": all(gains[x] >= 0 for x in OUTPUTS[2:]),
              "identity_bootstrap_lower_positive": lower > 0,
              "fused_strictly_best": all(computed_metrics["fused"]["mAP"] > computed_metrics[x]["mAP"] for x in ("baseline_only", *OUTPUTS[2:]))}
    assert checks == comparison["scientific_checks"] and all(checks.values()) == comparison["scientific_passed"]
    assert summary["status"] == ("COMPLETE_COMPARISON_SUPPORT_PASS" if all(checks.values()) else "COMPLETE_COMPARISON_SUPPORT_FAIL")
    result = {"status": "PASS_COMPLETE_SCALAR_REPLAY", "scope": "Local JSON/stdlib/NumPy only; all3000 query outputs and780 training steps; remote binary facts remain receipted",
              "folds": fold_rows, "metrics": computed_metrics, "gains_over_signal_pp": gains,
              "fold_fused_gains_pp": fold_gains, "identity_bootstrap_lower_pp": lower,
              "bootstrap_recomputation_absolute_difference": bootstrap_error, "scientific_checks": checks,
              "maximum_metric_absolute_difference": max(metric_errors), "maximum_epoch_mean_difference": max(mean_errors),
              "maximum_weighted_loss_difference": max(loss_errors),
              "loss_rounding_scope": "Saved AMP intermediate dtypes not retained; scalar recomposition discrepancy reported without a new gate",
              "optimizer_steps": 780, "new_optimizer_steps": 520, "source_record_exposures": 49920,
              "queries": 600, "query_output_evaluations": 3000, "gallery_records": 1032, "query_identities": 60,
              "per_identity": per_identity, "elapsed_seconds": time.perf_counter() - started}
    result["input_file_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.summary, args.config, args.protocol, args.m0, args.original_log, args.resume_log,
                  *[args.rankings_dir / f"fold_{i}_rankings.json" for i in range(3)]]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_identity", "folds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "protocol", "m0", "rankings-dir", "original-log", "resume-log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
