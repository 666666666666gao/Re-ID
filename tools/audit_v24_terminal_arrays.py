#!/usr/bin/env python3
"""Recompute V24 terminal metrics and label masks without loading any model."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import random
import hashlib
import json
from pathlib import Path

import numpy as np


ENDPOINTS = ("ordinary_two_view", "environment_identity_prototype")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def metrics(ap, rank):
    return {"mAP": float(np.mean(ap) * 100),
            **{f"Rank-{k}": float(np.mean(np.asarray(rank) <= k) * 100)
               for k in (1, 5, 10)}}


def original_total(row, view):
    value = .25 * row[f"mean_{view}_id_fused"] + row[f"mean_{view}_triplet_fused"]
    for expert in OUTPUTS[2:]:
        value += row[f"mean_{view}_id_{expert}"] / 12
        value += .25 * row[f"mean_{view}_triplet_{expert}"]
        value += row[f"mean_{view}_id_residual_{expert}"] / 12
        value += .25 * row[f"mean_{view}_triplet_residual_{expert}"]
    return value


def audit_training(training, binding, fold, trained):
    assert training["prototype_loss_enabled"] is trained
    assert binding["prototype_loss_enabled"] is trained
    assert training["optimizer_steps"] == (580, 560, 540)[fold]
    assert training["view_forward_backward_pairs"] == 2 * training["optimizer_steps"]
    assert (binding["total_parameters"], binding["trainable_parameters"],
            binding["trainable_tensors"]) == (98800141, 7841292, 203)
    assert training["trainable_tensors"] == 203 and binding["role_modules_and_classification_heads_trainable"]
    loss_errors = []
    for row in training["history"]:
        assert row["batches"] == (29, 28, 27)[fold]
        assert row["view_forward_backward_pairs"] == 2 * row["batches"]
        for view in ("weak", "strong"):
            loss_errors.append(abs(original_total(row, view) - row[f"mean_{view}_total"]))
        prototype = (row["mean_prototype_global"] + row["mean_prototype_environment"]) / 2
        loss_errors.append(abs(prototype - row["mean_prototype_total"]))
        loss_errors.append(abs(float(trained) * prototype - row["mean_weighted_prototype"]))
        expected = .5 * row["mean_weak_total"] + .5 * row["mean_strong_total"] + row["mean_weighted_prototype"]
        loss_errors.append(abs(expected - row["mean_total"]))
    assert max(loss_errors) < 1e-6
    return {"fold": fold, "prototype_loss_enabled": trained,
            "optimizer_steps": training["optimizer_steps"],
            "view_forward_backward_pairs": training["view_forward_backward_pairs"],
            "trainable_parameters": 7841292, "trainable_tensors": 203,
            "loss_scope": "All 20 recorded epoch means, both views' 14 components; per-update Q1 loss rows were not stored",
            "max_loss_component_roundoff": max(loss_errors)}


def metadata_sampler(source):
    # Execute only the original label/index sampler class. Omit its torch Sampler base;
    # no constructor/method body changes, data-loader imports, images or tensor runtime.
    tree = ast.parse(source)
    node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CrossCameraIdentitySampler"]
    node.bases = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    namespace = {"random": random, "defaultdict": defaultdict}
    exec(compile(ast.fix_missing_locations(module), "<original-metadata-sampler>", "exec"), namespace)
    return namespace["CrossCameraIdentitySampler"]


def audit_memory_trajectory(training, binding, registry, sampler_class):
    fit_ids = binding["fit_identity_ids"]
    label_map = {identity: i for i, identity in enumerate(fit_ids)}
    source = [row for row in registry if row["identity"] in label_map]
    records = [((row["file"],) * 3, label_map[row["identity"]], row["camera"], 0) for row in source]
    initial, final = training["memory_initial"], training["memory_final"]
    assert len(records) == initial["source_records"]
    assert initial["source_feature_forward_calls"] == (len(records) + 127) // 128
    assert initial["source_only"] and initial["model_state_unchanged"]
    assert hashlib.sha256(json.dumps([r["file"] for r in source]).encode()).hexdigest() == initial["source_file_order_sha256"]
    pairs = sorted({(row[1], row[2]) for row in records})
    assert len(pairs) == 108 and len({p[0] for p in pairs}) == 94
    index = {pair: i for i, pair in enumerate(pairs)}
    for memory in (initial, final):
        assert memory["pairs"] == 108 and memory["dimension"] == 7680
        assert list(zip(memory["pair_labels"], memory["pair_cameras"], strict=True)) == pairs
        assert max(abs(n - 1) for n in memory["prototype_norms"]) < 1e-5
    assert initial["last_update"] == initial["update_count"] == [0] * 108
    cameras_per_id = Counter(p[0] for p in pairs)
    identities_per_camera = Counter(p[1] for p in pairs)
    last, updates, steps = [0] * 108, [0] * 108, 0
    sampler = sampler_class(records, batch_size=64, num_instances=8, seed=42)
    order = hashlib.sha256()
    all_positive = cross_positive = exposure = 0
    group_histogram = Counter()
    errors, epoch_rows = [], []
    for epoch, history in enumerate(training["history"], 1):
        indices = list(iter(sampler))
        assert len(indices) == history["batches"] * 64
        coverage = defaultdict(list)
        before_steps = steps
        for offset in range(0, len(indices), 64):
            selected = indices[offset:offset + 64]
            rows = [records[i] for i in selected]
            paths = [row[0][0] for row in rows]
            order.update((json.dumps(paths) + "\n").encode())
            if steps < 8:
                for view in ("weak", "strong"):
                    raw = training["first_eight_batch_receipts"][steps][view]
                    assert raw["sampler_indices"] == selected and raw["paths"] == paths
            steps += 1
            id_counts = Counter(row[1] for row in rows)
            pair_counts = Counter((row[1], row[2]) for row in rows)
            assert len(id_counts) == 8 and set(id_counts.values()) == {8}
            camera_groups = Counter(pair[0] for pair in pair_counts)
            group_histogram[sum(value > 1 for value in camera_groups.values())] += 1
            all_positive += sum(n * (n - 1) for n in id_counts.values())
            cross_positive += sum(n * (id_counts[identity] - n) for (identity, _), n in pair_counts.items())
            exposure += len(rows)
            ages = [steps - value for value in last]
            values = {
                "global_negative_identities_per_anchor": 93,
                "environment_negative_relations": sum(identities_per_camera[row[2]] - 1 for row in rows),
                "anchors_with_real_other_camera_positive": sum(cameras_per_id[row[1]] > 1 for row in rows),
                "real_other_camera_positive_prototypes": sum(cameras_per_id[row[1]] - 1 for row in rows),
                "prototype_age_max_steps": max(ages),
                "prototype_age_mean_steps": float(np.mean(np.asarray(ages, dtype=np.float32))),
                "prototype_pairs_updated": sum(count > 0 for count in updates),
            }
            for key, value in values.items():
                coverage[key].append(value)
            for pair in pair_counts:
                j = index[pair]
                last[j], updates[j] = steps, updates[j] + 1
        assert steps - before_steps == history["batches"]
        assert [steps - value for value in last] == history["memory_age_steps_at_epoch_end"]
        assert updates == history["memory_update_counts"]
        for key, values in coverage.items():
            errors.append(abs(float(np.mean(values)) - history["mean_" + key]))
        epoch_rows.append({"epoch": epoch, "steps": steps, "updated_pairs": sum(x > 0 for x in updates),
                           "maximum_age_steps": max(steps - x for x in last),
                           "minimum_update_count": min(updates), "maximum_update_count": max(updates)})
    assert steps == training["optimizer_steps"]
    assert last == final["last_update"] and updates == final["update_count"]
    assert order.hexdigest() == training["sample_order_sha256"]
    assert max(errors) < 1e-5, max(errors)
    return {"full_sample_order_sha256": order.hexdigest(), "sample_order_matches_execution": True,
            "all_20_epoch_memory_count_age_arrays_exact": True,
            "maximum_coverage_mean_roundoff": max(errors), "prototype_pairs": len(pairs),
            "source_records": len(records), "source_identities": len(fit_ids), "sample_exposures": exposure,
            "all_directed_positive_pairs": all_positive, "directed_cross_camera_positive_pairs": cross_positive,
            "cross_camera_positive_pair_fraction": cross_positive / all_positive,
            "cross_camera_identity_groups_histogram": dict(group_histogram),
            "all_epoch_metadata": epoch_rows}


def audit_log(summary, log):
    rows = [json.loads(line) for line in log.decode("utf-8").splitlines() if line.startswith("{")]
    epoch_rows = [row for row in rows if "epoch" in row and "endpoint" in row]
    expected_rows = [row for fold in summary["folds"] for endpoint in ENDPOINTS
                     for row in fold["endpoints"][endpoint]["training"]["history"]]
    assert epoch_rows == expected_rows and len(epoch_rows) == 120
    m0_rows = [row for row in rows if row.get("stage") == "M0"]
    assert m0_rows == [{"stage": "M0", **summary["m0"]}]
    finals = [row for row in rows if row.get("stage") == "Q1_final"]
    expected_finals = [{"stage": "Q1_final", "fold": fold["fold"], "endpoint": endpoint,
                        "metrics": {name: output["metrics_percent"] for name, output in fold["endpoints"][endpoint]["outputs"].items()}}
                       for fold in summary["folds"] for endpoint in ENDPOINTS]
    assert finals == expected_finals and len(finals) == 6
    keys = ("status", "aggregate", "matched_gains_mAP", "scientific_checks", "bootstrap")
    assert rows[-1] == {key: summary[key] for key in keys}
    return {"all_120_epoch_rows_exact": True, "all_six_final_metric_events_exact": True,
            "complete_m0_event_exact": True, "last_terminal_event_exact": True}


def audit(summary):
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert summary["repository_commit"] == "6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33"
    assert summary["seed"] == 42 and summary["epochs_per_endpoint"] == 20
    assert summary["m0"]["passed"] and all(summary["m0"]["checks"].values())
    assert summary["new_inference_parameters"] == 0
    assert summary["dev_access_count"] == summary["official_test_access_count"] == 0
    assert not summary["d1_executed"] and summary["source_checkpoint_files_unchanged"]
    assert summary["evaluation_type"] == "real_gt_train_internal_complete_path_oof_reused_development_qualification"
    assert summary["oof_is_reused_development_qualification"]
    training_audits, memory_audits = [], []
    sampler_path = Path(__file__).resolve().parents[1] / "modeling/trifusion/aligned_data.py"
    sampler_raw = sampler_path.read_bytes()
    assert hashlib.sha256(sampler_raw).hexdigest() == summary["source_file_sha256"]["modeling/trifusion/aligned_data.py"]
    sampler_class = metadata_sampler(sampler_raw.decode("utf-8"))
    folds = summary["folds"]
    registry = sorted([row for fold in folds for row in fold["gallery_manifest"]], key=lambda row: row["file"])
    assert len(registry) == len({row["file"] for row in registry}) == 3126
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
        assert a["training"]["memory_initial"]["state_sha256"] == b["training"]["memory_initial"]["state_sha256"]
        for key in ("source", "source_model_state_sha256", "fit_identity_ids", "heldout_identity_ids",
                    "new_inference_parameters", "total_parameters"):
            assert a["binding"][key] == b["binding"][key], key
        for end, receipt in ends.items():
            trained = end == ENDPOINTS[1]
            training, binding = receipt["training"], receipt["binding"]
            preflight = summary["preflight"][fold["fold"]]["endpoints"][int(trained)]
            assert preflight["endpoint"] == end and preflight["binding"] == binding
            assert preflight["initial_state_sha256"] == training["initial_state_sha256"]
            assert preflight["state_unchanged"] and preflight["memory_unchanged"]
            assert preflight["memory_initial"]["state_sha256"] == training["memory_initial"]["state_sha256"]
            assert receipt["strict_reload"] and receipt["read_only_evaluation"]
            assert training["epochs"] == len(training["history"]) == 20
            assert [row["epoch"] for row in training["history"]] == list(range(1, 21))
            assert training["optimizer_steps"] == sum(row["batches"] for row in training["history"])
            training_audits.append(audit_training(training, binding, fold["fold"], trained))
            memory_audits.append({"fold": fold["fold"], "endpoint": end,
                                  **audit_memory_trajectory(training, binding, registry, sampler_class)})
            assert training["overflow_events"] == 0 and training["frozen_state_unchanged"]
            assert not training["missing_nonzero_gradients"]
            assert training["nonzero_gradient_tensors"] == training["trainable_tensors"]
            assert len(set(training["trainable_names"])) == training["trainable_tensors"]
            assert set(binding["heldout_identity_ids"]) == set(counts)
            assert len(binding["fit_identity_ids"]) == 94
            assert set(binding["fit_identity_ids"]).isdisjoint(counts)
            assert binding["baseline_frozen"] and binding["shared_tail_frozen"]
            assert binding["source_state_strictly_loaded"] and binding["new_inference_parameters"] == 0
            assert binding["architecture"] == "signal_preserving_v24_source_prototype"
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
            "view_forward_backward_pairs_recounted": steps * 2,
            "all_six_source_memory_metadata_audits": memory_audits,
            "sampler_source_sha256": hashlib.sha256(sampler_raw).hexdigest(),
            "metadata_replay_scope": "Original sampler class method bodies executed with only its torch base omitted; labels and indices only, no tensors or images",
            "max_absolute_numeric_difference_percent": max(errors), "fold_scopes": scopes,
            "aggregate": aggregate, "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains,
            "bootstrap_lower_bound_95_mAP": lower, "scientific_checks": checks,
            "all_identity_fused_gains": identity_rows, "all_query_paired_changes": paired_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    report = audit(summary)
    log_raw = args.log.read_bytes()
    report["log_verification"] = audit_log(summary, log_raw)
    report["input_log_sha256"] = hashlib.sha256(log_raw).hexdigest()
    report["input_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: report[key] for key in ("verification_passed", "scientific_status",
                                                 "max_absolute_numeric_difference_percent",
                                                 "bootstrap_lower_bound_95_mAP")}))
