#!/usr/bin/env python3
"""Verify every V26 terminal artifact, training row and held-out ranking on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ENDPOINTS = ("control", "responsibility")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_metrics(actual, expected):
    assert set(actual) == set(expected)
    error = max(abs(actual[key] - expected[key]) for key in actual)
    assert error < 1e-10, (actual, expected)
    return error


def metrics(ap, ranks):
    return {
        "mAP": float(np.mean(ap) * 100),
        **{f"Rank-{k}": float(np.mean(np.asarray(ranks) <= k) * 100) for k in (1, 5, 10)},
    }


def verify(args):
    started = time.time()
    repo, run = args.repo.resolve(), args.run_dir.resolve()
    assert int(Path(str(run) + ".exit").read_text()) == 0
    terminal = json.loads(Path(str(run) + ".terminal.json").read_bytes())
    assert terminal["exit_code"] == 0
    assert not (Path("/proc") / str(terminal["original_pid"])).exists()
    summary_path = run / "run_summary.json"
    summary = json.loads(summary_path.read_bytes())
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert len(summary["folds"]) == len(summary["preflight"]) == 3
    commit = "ff18e40b12a62ddee25e36c5bdfe5fe4de0f7d5a"
    assert summary["repository_commit"] == terminal["execution_commit"] == commit
    cfgrel = "configs/RGBNT201/TriFusion-signal-preserving-v26-role-modal-responsibility-rtx3090.json"
    planrel = "refine-logs/trifusion_v26_role_modal_responsibility/EXPERIMENT_PLAN.md"
    cfg = json.loads((repo / cfgrel).read_bytes())
    expected_files = dict(summary["source_file_sha256"])
    expected_files.update({
        cfgrel: summary["config_sha256"], planrel: summary["plan_sha256"],
        "tools/train_signal_preserving_v26.py": summary["runner_sha256"],
        cfg["SUPERVISION_METADATA"]["PATH"]: cfg["SUPERVISION_METADATA"]["SHA256"],
        cfg["SIGNAL"]["CLIP_WEIGHT"]: cfg["SIGNAL"]["CLIP_WEIGHT_SHA256"],
        cfg["INITIALIZATION"]["V12_RUN_SUMMARY"]: cfg["INITIALIZATION"]["V12_RUN_SUMMARY_SHA256"],
    })
    for fold in cfg["INITIALIZATION"]["V12_FOLDS"]:
        for kind in ("SIGNAL", "EXPERT"):
            expected_files[fold[kind + "_CHECKPOINT"]] = fold[kind + "_CHECKPOINT_SHA256"]
    for rel, expected in expected_files.items():
        assert sha(repo / rel) == expected, rel
    preregrel = "evidence/trifusion_v26_role_modal_responsibility_preregistration_20260907.json"
    assert sha(repo / preregrel) == "ce977db9062151cf5ad1fc9030920fe96f160084d9d34004d6bb704698c48778"
    prereg = json.loads((repo / preregrel).read_bytes())
    for rel, expected in prereg["paths_sha256"].items():
        original = subprocess.check_output(["git", "show", commit + ":" + rel], cwd=repo)
        assert hashlib.sha256(original).hexdigest() == expected, rel
        assert sha(repo / rel) == expected, rel
    replay = json.loads((repo / cfg["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    assert replay["status"] == "COMPLETE_METADATA_REPLAY_PASS"
    assert summary["math_check"]["status"] == "PASS_ANALYTICAL_VALUE_GRADIENT_DETACHED_WEIGHTS_CLASS_ZERO_PERMUTATION"
    assert summary["math_check"]["gradient_max_error"] < 2e-8
    assert cfg["LOSS"]["RESPONSIBILITY_WEIGHT"] == 1.0
    assert cfg["LOSS"]["RESPONSIBILITY_TEMPERATURE"] == 0.1
    for fold in replay["folds"]:
        assert fold["arms"]["control"] == fold["arms"]["responsibility"]
    for fold in summary["preflight"]:
        a, b = fold["endpoints"]
        for key in ("initial_state_sha256", "all_output_sha256", "batch_receipts", "relation_diagnostics"):
            assert a[key] == b[key], key
        assert len(a["batch_receipts"]) == len(a["relation_diagnostics"]) == 8
    m0 = summary["m0"]
    assert m0["passed"] and all(m0["checks"].values())
    assert len(m0["capacities"]) == 2
    assert [r["steps"] for r in m0["capacities"]] == [8, 8]
    assert m0["overfit"]["steps"] == len(m0["overfit"]["losses"]) == 100
    for row in (*m0["capacities"], m0["overfit"]):
        assert row["trainable_tensors"] == row["nonzero_gradient_tensors"] == 203
        assert not row["missing_nonzero_gradients"] and row["frozen_state_unchanged"]
        assert row["overflow_events"] == 0 and all(math.isfinite(x) for x in row["losses"])
    gradient_rows = m0["capacities"][1]["same_block_parameter_gradient_diagnostics"]
    assert len(gradient_rows) == 8
    for row in gradient_rows:
        for expert, count in zip(OUTPUTS[2:], (42, 54, 93), strict=True):
            g = row["experts"][expert]
            assert g["parameter_tensors"] == count and g["new_encoder_gradient_nonzero"]
            assert all(math.isfinite(v) for v in g.values())
            assert g["auxiliary_gradient_norm"] > 0 and g["auxiliary_to_base_norm_ratio"] > 0
            assert -1.00001 <= g["fused_role_cosine"] <= 1.00001
            assert -1.00001 <= g["base_auxiliary_cosine"] <= 1.00001
    overfit = m0["overfit"]
    ratio = ((overfit["losses"][-1] - overfit["combined_loss_floor"])
             / (overfit["losses"][0] - overfit["combined_loss_floor"]))
    assert abs(ratio - overfit["excess_loss_ratio"]) < 1e-14 and ratio <= 0.1
    all_ap = {e: {n: [] for n in OUTPUTS} for e in ENDPOINTS}
    all_rank = {e: {n: [] for n in OUTPUTS} for e in ENDPOINTS}
    query_rows, query_ids, endpoint_proofs = [], [], []
    max_distance_error, max_metric_error = 0.0, 0.0
    checked_steps, ranking_positions, distance_entries = 0, 0, 0
    role_support_rows = []
    for fold in summary["folds"]:
        number = fold["fold"]
        manifest = fold["gallery_manifest"]
        identities = np.asarray([r["identity"] for r in manifest])
        cameras = np.asarray([r["camera"] for r in manifest])
        assert len(manifest) == (1000, 1051, 1075)[number]
        eligible = np.flatnonzero([
            np.any((identities == identity) & (cameras != camera))
            for identity, camera in zip(identities, cameras, strict=True)
        ])
        assert len(eligible) == (190, 179, 202)[number]
        baseline_arrays, recomputed, top1 = None, {}, {}
        for endpoint in ENDPOINTS:
            saved = fold["endpoints"][endpoint]
            receipt_path = run / f"fold_{number}_{endpoint}_receipt.json"
            assert json.loads(receipt_path.read_bytes()) == saved
            assert sha(saved["checkpoint"]) == saved["checkpoint_sha256"]
            assert saved["strict_reload"] and saved["read_only_evaluation"]
            train = saved["training"]
            assert train["optimizer_steps"] == (580, 560, 540)[number]
            assert train["epochs"] == len(train["history"]) == 20
            assert train["overflow_events"] == 0 and not train["missing_nonzero_gradients"]
            assert train["frozen_state_unchanged"] and train["trainable_tensors"] == 203
            assert train["nonzero_gradient_tensors"] == 203
            arm = replay["folds"][number]["arms"][endpoint]
            log_path = Path(train["all_training_steps_path"])
            assert sha(log_path) == train["all_training_steps_sha256"]
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            assert len(rows) == train["optimizer_steps"] == arm["batch_count"]
            exposure, cross_pairs, order = Counter(), 0, hashlib.sha256()
            for row, wanted in zip(rows, arm["batches"], strict=True):
                assert row["fold"] == number and row["endpoint"] == endpoint
                for key in ("epoch", "step", "sampler_indices", "sample_order_sha256",
                            "cross_camera_identity_groups", "directed_cross_camera_positive_pairs",
                            "all_directed_positive_pairs"):
                    assert row[key] == wanted[key], (number, endpoint, key)
                paths = [replay["folds"][number]["source_manifest"][i]["file"]
                         for i in row["sampler_indices"]]
                assert row["paths"] == paths
                assert hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest() == row["sample_order_sha256"]
                assert not row["overflow"] and all(math.isfinite(v) for v in row["losses"].values())
                loss = row["losses"]
                assert loss["responsibility_enabled"] == int(endpoint == "responsibility")
                assert loss["responsibility_triplets"] == 25088
                assert loss["responsibility_slot_triplets"] == 225792
                assert row["unique_source_records_in_batch"] == len(set(paths))
                assert row["same_record_positive_pair_exposures"] == sum(v * (v - 1) for v in Counter(paths).values())
                assert row["cross_camera_triplet_exposures"] == row["directed_cross_camera_positive_pairs"] * 56
                assert 0 <= loss["responsibility_fused_nonpositive"] <= 25088
                assert 0 <= loss["responsibility_raw_slot_norm_max_error"] < 0.005
                assert 0 <= loss["responsibility_fused_decomposition_max_error"] < 0.005
                assert abs(loss["original_total"] - loss["common_identity_and_branch_triplet"]
                           - loss["ordinary_residual_triplet"]) < 2e-6
                expected_total = loss["original_total"] + int(endpoint == "responsibility") * loss["responsibility_loss"]
                assert abs(loss["total"] - expected_total) < 2e-6
                slot_losses = []
                for expert in OUTPUTS[2:]:
                    for modality in ("RGB", "NI", "TI"):
                        prefix = f"responsibility_{expert}_{modality}_"
                        assert 0 <= loss[prefix + "weight_min"] <= loss[prefix + "weight_mean"] <= loss[prefix + "weight_max"] <= 1
                        assert 0 < loss[prefix + "effective_sample_size"] <= 25088.05
                        assert 0 <= loss[prefix + "joint_nonpositive"] <= min(loss["responsibility_fused_nonpositive"], loss[prefix + "slot_nonpositive"]) <= 25088
                        assert loss[prefix + "gradient_norm"] >= 0
                        slot_losses.append(loss[prefix + "weighted_loss"])
                assert abs(float(np.mean(slot_losses)) - loss["responsibility_loss"]) < 1e-6
                exposure.update(row["sampler_indices"])
                cross_pairs += row["directed_cross_camera_positive_pairs"]
                order.update((json.dumps(paths) + "\n").encode())
            assert order.hexdigest() == train["sample_order_sha256"]
            assert sum(exposure.values()) == train["sample_exposures"] == arm["sample_exposures"]
            assert cross_pairs == train["directed_cross_camera_positive_pairs"] == arm["directed_cross_camera_positive_pairs"]
            for row in arm["identities"]:
                assert [exposure[i] for i in row["record_indices"]] == row["record_exposures"]
            for history in train["history"]:
                epoch_rows = [row for row in rows if row["epoch"] == history["epoch"]]
                assert len(epoch_rows) == history["batches"]
                assert all(row["learning_rate"] == history["learning_rate"] for row in epoch_rows)
                for key in epoch_rows[0]["losses"]:
                    mean = float(np.mean([row["losses"][key] for row in epoch_rows]))
                    assert abs(mean - history["mean_" + key]) < 1e-12
            checked_steps += len(rows)
            role_support_rows.append({
                "fold": number, "endpoint": endpoint, "optimizer_steps": len(rows),
                "ordered_triplet_exposures": sum(r["losses"]["responsibility_triplets"] for r in rows),
                "same_record_positive_pair_exposures": sum(r["same_record_positive_pair_exposures"] for r in rows),
                "fused_nonpositive_triplet_exposures": sum(r["losses"]["responsibility_fused_nonpositive"] for r in rows),
                "mean_responsibility_loss": float(np.mean([r["losses"]["responsibility_loss"] for r in rows])),
                "mean_original_total": float(np.mean([r["losses"]["original_total"] for r in rows])),
                "maximum_decomposition_error": max(r["losses"]["responsibility_fused_decomposition_max_error"] for r in rows),
                "slots": {expert + "_" + modality: {
                    "joint_nonpositive_triplet_exposures": sum(r["losses"][f"responsibility_{expert}_{modality}_joint_nonpositive"] for r in rows),
                    "mean_weight": float(np.mean([r["losses"][f"responsibility_{expert}_{modality}_weight_mean"] for r in rows])),
                    "mean_effective_sample_size": float(np.mean([r["losses"][f"responsibility_{expert}_{modality}_effective_sample_size"] for r in rows])),
                    "mean_auxiliary_gradient_norm": float(np.mean([r["losses"][f"responsibility_{expert}_{modality}_gradient_norm"] for r in rows])),
                    "nonzero_auxiliary_gradient_steps": sum(r["losses"][f"responsibility_{expert}_{modality}_gradient_norm"] > 0 for r in rows),
                } for expert in OUTPUTS[2:] for modality in ("RGB", "NI", "TI")},
            })
            arrays_path = Path(saved["retrieval_arrays"]["path"])
            assert arrays_path.stat().st_size == saved["retrieval_arrays"]["bytes"]
            assert sha(arrays_path) == saved["retrieval_arrays"]["sha256"]
            arrays = torch.load(arrays_path, map_location="cpu", weights_only=True)
            assert arrays["model_state_sha256"] == train["final_state_sha256"]
            assert np.array_equal(arrays["identities"].numpy(), identities)
            assert np.array_equal(arrays["cameras"].numpy(), cameras)
            assert set(arrays["outputs"]) == set(OUTPUTS)
            if endpoint == "control":
                baseline_arrays = arrays["outputs"]["baseline_only"]
            else:
                for key, value in baseline_arrays.items():
                    assert torch.equal(value, arrays["outputs"]["baseline_only"][key]), key
            recomputed[endpoint], top1[endpoint] = {}, {}
            for name in OUTPUTS:
                data = arrays["outputs"][name]
                features, distances = data["features"], data["distances"]
                assert features.shape[0] == len(manifest)
                assert tuple(distances.shape) == (len(manifest), len(manifest))
                assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(distances).all())
                assert float((torch.linalg.vector_norm(features, dim=1) - 1).abs().max()) < 2e-6
                recomputed_distances = torch.cdist(features, features)
                error = float((recomputed_distances - distances).abs().max())
                assert error < 2e-6, (number, endpoint, name, error)
                max_distance_error = max(max_distance_error, error)
                ranks = np.argsort(distances.numpy()[eligible], axis=1, kind="stable")
                assert np.array_equal(ranks, data["rankings_full_gallery"].numpy())
                ranking_positions += ranks.size
                distance_entries += distances.numel()
                ap, first_ranks, first_gallery = [], [], []
                for query, order_row in zip(eligible, ranks, strict=True):
                    valid = order_row[~((identities[order_row] == identities[query])
                                        & (cameras[order_row] == cameras[query]))]
                    positive_positions = np.flatnonzero(identities[valid] == identities[query]) + 1
                    assert positive_positions.size > 0
                    ap.append(float(np.mean(np.arange(1, len(positive_positions) + 1) / positive_positions)))
                    first_ranks.append(int(positive_positions[0]))
                    first_gallery.append(int(valid[0]))
                score = saved["outputs"][name]
                assert score["query_indices"] == eligible.tolist()
                assert score["excluded_no_cross_camera_positive"] == np.flatnonzero(
                    ~np.isin(np.arange(len(manifest)), eligible)
                ).tolist()
                assert np.max(np.abs(np.asarray(ap) - score["average_precision"])) < 1e-12
                assert first_ranks == score["first_match_rank"]
                current_metrics = metrics(ap, first_ranks)
                max_metric_error = max(max_metric_error, compare_metrics(current_metrics, score["metrics_percent"]))
                recomputed[endpoint][name] = {"ap": ap, "ranks": first_ranks}
                top1[endpoint][name] = first_gallery
                all_ap[endpoint][name].extend(ap)
                all_rank[endpoint][name].extend(first_ranks)
            endpoint_proofs.append({
                "fold": number, "endpoint": endpoint,
                "checkpoint_sha256": saved["checkpoint_sha256"],
                "all_training_steps_sha256": train["all_training_steps_sha256"],
                "array_sha256": saved["retrieval_arrays"]["sha256"],
                "training_steps_checked": len(rows), "all_source_exposures_match": True,
                "all_gallery_records": len(manifest), "eligible_queries": len(eligible),
                "all_five_output_rankings_and_scores_match": True,
            })
            del arrays
        control_train = fold["endpoints"]["control"]["training"]
        candidate_train = fold["endpoints"]["responsibility"]["training"]
        for key in ("sample_order_sha256", "first_eight_batch_receipts", "initial_state_sha256",
                    "optimizer_steps", "sample_exposures", "directed_cross_camera_positive_pairs"):
            assert control_train[key] == candidate_train[key], key
        for position, query in enumerate(eligible):
            row = {
                "fold": number, "query_index": int(query), **manifest[query], "outputs": {},
            }
            for name in OUTPUTS:
                old, new = (recomputed[e][name] for e in ENDPOINTS)
                top = top1["responsibility"][name][position]
                row["outputs"][name] = {
                    "control_ap": old["ap"][position], "candidate_ap": new["ap"][position],
                    "ap_gain_pp": (new["ap"][position] - old["ap"][position]) * 100,
                    "control_first_match_rank": old["ranks"][position],
                    "candidate_first_match_rank": new["ranks"][position],
                    "candidate_first_gallery_index": top,
                    "candidate_first_gallery": manifest[top],
                }
            query_rows.append(row)
            query_ids.append(int(identities[query]))
    assert checked_steps == 3360 and len(query_rows) == len(query_ids) == 571
    assert distance_entries == 32602260 and ranking_positions == 5952790
    aggregate = {
        e: {n: metrics(all_ap[e][n], all_rank[e][n]) for n in OUTPUTS} for e in ENDPOINTS
    }
    for e in ENDPOINTS:
        for n in OUTPUTS:
            max_metric_error = max(max_metric_error, compare_metrics(aggregate[e][n], summary["aggregate"][e][n]))
    gains = {n: aggregate["responsibility"][n]["mAP"] - aggregate["control"][n]["mAP"] for n in OUTPUTS}
    compare_metrics(gains, summary["matched_gains_mAP"])
    labels = np.asarray(query_ids)
    clusters = np.unique(labels)
    assert len(clusters) == 21
    delta = np.asarray(all_ap["responsibility"]["fused"]) - np.asarray(all_ap["control"]["fused"])
    sums = np.asarray([delta[labels == identity].sum() for identity in clusters])
    counts = np.asarray([(labels == identity).sum() for identity in clusters])
    sampled = np.random.default_rng(42).choice(np.arange(21), size=(10000, 21), replace=True)
    bootstrap_means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lower = float(np.quantile(bootstrap_means, 0.025) * 100)
    assert abs(lower - summary["bootstrap"]["lower_bound_95_mAP"]) < 1e-10
    fold_gains = [
        fold["endpoints"]["responsibility"]["outputs"]["fused"]["metrics_percent"]["mAP"]
        - fold["endpoints"]["control"]["outputs"]["fused"]["metrics_percent"]["mAP"]
        for fold in summary["folds"]
    ]
    checks = {
        "aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.0,
        "all_fold_fused_nonnegative": all(gain >= 0 for gain in fold_gains),
        "all_expert_aggregate_nonnegative": all(gains[n] >= 0 for n in OUTPUTS[2:]),
        "fused_bootstrap_lower_positive": lower > 0,
        "fused_beats_baseline_and_experts": all(
            aggregate["responsibility"]["fused"]["mAP"] > aggregate["responsibility"][n]["mAP"]
            for n in ("baseline_only", *OUTPUTS[2:])
        ),
    }
    assert checks == summary["scientific_checks"]
    assert summary["status"] == ("Q1_PASS" if all(checks.values()) else "Q1_FAIL")
    identity_rows = []
    for identity in clusters:
        selected = [row for row in query_rows if row["identity"] == int(identity)]
        entry = {"identity": int(identity), "query_count": len(selected), "outputs": {}}
        for name in OUTPUTS:
            old = np.asarray([row["outputs"][name]["control_ap"] for row in selected])
            new = np.asarray([row["outputs"][name]["candidate_ap"] for row in selected])
            old_r = np.asarray([row["outputs"][name]["control_first_match_rank"] for row in selected])
            new_r = np.asarray([row["outputs"][name]["candidate_first_match_rank"] for row in selected])
            entry["outputs"][name] = {
                "control_mAP": float(old.mean() * 100), "candidate_mAP": float(new.mean() * 100),
                "mAP_gain_pp": float((new - old).mean() * 100),
                "rank1_repaired": int(((old_r > 1) & (new_r == 1)).sum()),
                "rank1_new_errors": int(((old_r == 1) & (new_r > 1)).sum()),
            }
        identity_rows.append(entry)
    proof = {
        "status": "PASS_COMPLETE_V26_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES",
        "verified_at": datetime.now().astimezone().isoformat(),
        "execution_commit": commit, "run_summary_sha256": sha(summary_path),
        "verifier_sha256": sha(__file__), "scientific_status": summary["status"],
        "scientific_checks": checks, "aggregate": aggregate, "matched_gains_mAP": gains,
        "bootstrap_lower_95_mAP": lower, "checked_training_steps": checked_steps,
        "checked_distance_entries": distance_entries, "checked_full_ranking_positions": ranking_positions,
        "eligible_queries": len(query_rows), "identity_clusters": len(clusters),
        "maximum_distance_recompute_error": max_distance_error,
        "maximum_metric_error_pp": max_metric_error,
        "all_frozen_Signal_features_distances_rankings_equal_between_arms": True,
        "endpoints": endpoint_proofs, "all_queries": query_rows, "all_identities": identity_rows,
        "role_support_full_training": role_support_rows,
        "all_training_objective_and_relation_receipts_match": True,
        "both_arms_identical_registered_sampling": True,
        "new_model_forwards": 0, "new_optimizer_steps": 0, "image_reads": 0,
        "checkpoint_tensor_loads": 0, "retrieval_array_loads": 6,
        "independent_audit": False, "elapsed_seconds": time.time() - started,
    }
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(proof, indent=2) + "\n").encode())
    print(json.dumps({k: v for k, v in proof.items() if k not in ("all_queries", "all_identities", "endpoints", "role_support_full_training")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
