#!/usr/bin/env python3
"""Recompute every saved official distance, ranking, query score and identity gate."""

import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_rgbnt100_trifusion_m0_files import sha, state_sha

WIDTHS = {"baseline_only": 3072, "fused": 7680, "cnn": 4608, "transformer": 4608, "mamba": 4608}


def main(args):
    import numpy as np
    import torch

    started = time.perf_counter()
    assert not args.output.exists()
    config, summary = [json.loads(p.read_text()) for p in (args.config, args.summary)]
    assert summary["schema"] == config["schema"] == "rgbnt100-fixed-official-evaluation-v1"
    assert summary["config_sha256"] == sha(args.config)
    assert summary["runner_sha256"] == sha(ROOT / "tools/evaluate_rgbnt100_official_main.py")
    for path, expected in config["source_files_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for key in ("roles_config", "roles_summary", "roles_verification"):
        assert sha(ROOT / config[key]) == config[key + "_sha256"], key
    role_config = json.loads((ROOT / config["roles_config"]).read_text())
    roles = json.loads((ROOT / config["roles_summary"]).read_text())
    role_verification = json.loads((ROOT / config["roles_verification"]).read_text())
    assert role_verification["status"] == "PASS_FULL50_ROLES_FILES_AND_ALL_UPDATES" and role_verification["engineering_passed"]
    assert role_verification["mode"] == "main" and role_verification["summary_sha256"] == config["roles_summary_sha256"]
    base = role_config["BASELINE"]
    for key in ("CONFIG", "SUMMARY", "VERIFICATION"):
        assert sha(ROOT / base[key]) == base[key + "_SHA256"], key
    baseline = json.loads((ROOT / base["SUMMARY"]).read_text())
    base_verification = json.loads((ROOT / base["VERIFICATION"]).read_text())
    assert base_verification["status"] == "PASS_FULL50_SIGNAL_FILES_ALL_UPDATES_AND_AUTHOR_LR"
    assert base_verification["mode"] == "baseline" and base_verification["summary_sha256"] == base["SUMMARY_SHA256"]
    assert summary["roles_summary_sha256"] == config["roles_summary_sha256"]
    assert summary["baseline_summary_sha256"] == base["SUMMARY_SHA256"] == roles["baseline_summary_sha256"]
    assert roles["training"]["epochs"] == 20 and baseline["training"]["epochs"] == 30
    for endpoint in (baseline, roles):
        assert sha(endpoint["checkpoint"]) == endpoint["checkpoint_sha256"]
        payload = torch.load(endpoint["checkpoint"], map_location="cpu", weights_only=True)
        assert state_sha(payload["model_state_dict"]) == endpoint["training"]["final_state_sha256"] == endpoint["strict_reload_state_sha256"]
        assert payload["source_ids"] == list(range(501, 600, 2)) and payload["heldout_ids"] == list(range(502, 601, 2))
        assert endpoint["official_model_record_forwards"] == 0
        del payload
    protocol_path = ROOT / role_config["DATA"]["PROTOCOL"]
    assert sha(protocol_path) == role_config["DATA"]["PROTOCOL_SHA256"] == summary["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    query, gallery = protocol["records"]["query"], protocol["records"]["gallery"]
    assert len(query) == summary["query_count"] == 1715 and len(gallery) == summary["gallery_count"] == 8575
    assert summary["seed"] == 42 and summary["query_identities"] == 50
    assert summary["signal_official_record_forwards"] == summary["roles_official_record_forwards"] == 10290
    assert summary["signal_features_bitwise_equal"] and summary["signal_distances_bitwise_equal"]
    assert summary["training_updates"] == summary["rgbnt201_dev_record_forwards"] == 0 and not summary["reranking"]
    assert summary["checkpoint_selection"] == {"signal": "fixed_epoch30", "roles": "fixed_epoch20"}
    ids = np.asarray([r["identity"] for r in gallery])
    cameras = np.asarray([r["camera"] for r in gallery])
    masks = []
    for index, (row, frozen) in enumerate(zip(query, protocol["query_rows"], strict=True)):
        assert frozen["query_index"] == index and frozen["identity"] == row["identity"]
        excluded = (ids == row["identity"]) & (cameras == row["camera"])
        positives = (ids == row["identity"]) & ~excluded
        assert np.flatnonzero(excluded).tolist() == frozen["excluded_gallery_positions"]
        assert np.flatnonzero(positives).tolist() == frozen["positive_gallery_positions"]
        assert int(positives.sum()) == frozen["valid_positive_count"] > 0
        masks.append((excluded, positives))
    retrieval = summary["retrieval"]
    assert retrieval["feature_widths"] == WIDTHS
    assert sha(retrieval["retrieval_arrays"]) == retrieval["retrieval_arrays_sha256"]
    arrays = torch.load(retrieval["retrieval_arrays"], map_location="cpu", weights_only=True)
    assert arrays["protocol_sha256"] == summary["protocol_sha256"]
    assert arrays["query_record_indices"] == list(range(1715)) and arrays["gallery_record_indices"] == list(range(8575))
    assert arrays["record_order"] == "all1715 query followed by all8575 gallery"
    assert set(arrays["features"]) == set(arrays["distances"]) == set(WIDTHS)
    assert torch.equal(arrays["features"]["baseline_only"], arrays["independent_signal_features"])
    assert torch.equal(arrays["distances"]["baseline_only"], arrays["independent_signal_distances"])
    all_aps, all_first, metric_errors, rows_verified = {}, {}, [], 0
    census = [{"query_index": i, "identity": r["identity"], "camera": r["camera"], "path": r["path"], "outputs": {}}
              for i, r in enumerate(query)]
    for name, width in WIDTHS.items():
        values = arrays["features"][name]
        assert values.shape == (10290, width) and torch.isfinite(values).all().item()
        normalized = torch.nn.functional.normalize(values.float(), dim=1)
        qf, gf = normalized[:1715], normalized[1715:]
        matrix = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None]
        matrix.addmm_(qf, gf.T, beta=1, alpha=-2)
        assert matrix.shape == (1715, 8575) and torch.isfinite(matrix).all().item()
        assert torch.equal(matrix, arrays["distances"][name])
        expected_order = np.argsort(matrix.numpy(), axis=1)
        ranking = retrieval["rankings"][name]
        assert sha(ranking["path"]) == ranking["sha256"] and Path(ranking["path"]).stat().st_size == ranking["bytes"]
        assert ranking["query_rows"] == 1715 and ranking["positions_per_query"] == 8575
        aps, first = [], []
        with gzip.open(ranking["path"], "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                item = json.loads(line)
                assert item["query_index"] == index and np.array_equal(item["order"], expected_order[index])
                order = expected_order[index]
                excluded, positives = masks[index]
                retained = order[~excluded[order]]
                matches = positives[retained]
                positions = np.flatnonzero(matches) + 1
                assert len(positions) == protocol["query_rows"][index]["valid_positive_count"]
                ap = statistics.fmean(j / int(rank) for j, rank in enumerate(positions, 1))
                recorded = retrieval["outputs"][name]
                metric_errors.append(abs(ap - recorded["average_precision"][index]))
                assert int(positions[0]) == recorded["first_match_rank"][index]
                aps.append(ap)
                first.append(int(positions[0]))
                top = int(retained[0])
                positive = int(retained[positions[0] - 1])
                negative = int(retained[np.flatnonzero(~matches)[0]])
                census[index]["outputs"][name] = {
                    "average_precision": recorded["average_precision"][index], "first_match_rank": int(positions[0]),
                    "top_gallery_index": top, "top_identity": int(ids[top]), "top_camera": int(cameras[top]),
                    "top_matches_identity": bool(matches[0]), "top_same_camera": int(cameras[top]) == query[index]["camera"],
                    "nearest_positive_gallery_index": positive, "nearest_negative_gallery_index": negative,
                    "nearest_positive_distance": float(matrix[index, positive]),
                    "nearest_negative_distance": float(matrix[index, negative]),
                    "negative_minus_positive_distance": float(matrix[index, negative] - matrix[index, positive])}
                rows_verified += 1
        assert len(aps) == len(first) == 1715
        all_aps[name], all_first[name] = np.asarray(aps), np.asarray(first)
        assert max(retrieval["outputs"][name]["upstream_metric_difference_pp"].values()) < 1e-5
        del matrix, normalized, expected_order
    assert rows_verified == 8575 and max(metric_errors) < 1e-12
    computed = {name: {"mAP": float(aps.mean() * 100),
                        **{f"Rank-{k}": float(np.mean(all_first[name] <= k) * 100) for k in (1, 5, 10)}}
                for name, aps in all_aps.items()}
    comparison = summary["comparison"]
    for name, metrics in computed.items():
        for key, value in metrics.items():
            metric_errors.append(abs(value - comparison["metrics"][name][key]))
            metric_errors.append(abs(value - retrieval["outputs"][name]["metrics"][key]))
    identities = np.asarray([r["identity"] for r in query])
    unique = np.unique(identities)
    assert unique.tolist() == list(range(502, 601, 2))
    differences = (all_aps["fused"] - all_aps["baseline_only"]) * 100
    sums = np.asarray([differences[identities == i].sum() for i in unique])
    counts = np.asarray([np.sum(identities == i) for i in unique])
    draws = np.random.default_rng(42).choice(50, size=(10000, 50), replace=True)
    lower = float(np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), 2.5, method="linear"))
    bootstrap = comparison["identity_bootstrap"]
    assert bootstrap["seed"] == 42 and bootstrap["resamples"] == 10000 and bootstrap["cluster_count"] == 50
    assert bootstrap["percentile"] == 2.5 and bootstrap["quantile_method"] == "linear"
    assert abs(lower - bootstrap["lower_bound_pp"]) < 1e-10
    per_identity = []
    for identity, reported in zip(unique, comparison["per_identity"], strict=True):
        selected = identities == identity
        row = {"identity": int(identity), "query_count": int(selected.sum()),
               "map_by_output": {name: float(values[selected].mean() * 100) for name, values in all_aps.items()}}
        assert row["identity"] == reported["identity"] and row["query_count"] == reported["query_count"]
        metric_errors.extend(abs(value - reported["map_by_output"][name]) for name, value in row["map_by_output"].items())
        per_identity.append(row)
    for name in ("fused", "cnn", "transformer", "mamba"):
        recorded_ap = np.asarray(retrieval["outputs"][name]["average_precision"])
        delta = recorded_ap - np.asarray(retrieval["outputs"]["baseline_only"]["average_precision"])
        repaired = (all_first["baseline_only"] > 1) & (all_first[name] == 1)
        new_errors = (all_first["baseline_only"] == 1) & (all_first[name] > 1)
        changes = {"ap_improved": int(np.sum(delta > 0)), "ap_declined": int(np.sum(delta < 0)),
                   "ap_unchanged": int(np.sum(delta == 0)), "rank1_repaired": int(repaired.sum()),
                   "rank1_new_errors": int(new_errors.sum())}
        assert changes == comparison["query_changes"][name]
        for index, row in enumerate(census):
            row["outputs"][name].update({"ap_delta_over_signal_pp": float(delta[index] * 100),
                                         "rank1_repaired": bool(repaired[index]), "rank1_new_error": bool(new_errors[index])})
    gains = {name: metrics["mAP"] - computed["baseline_only"]["mAP"] for name, metrics in computed.items()}
    assert all(abs(v - comparison["gains_over_signal_pp"][k]) < 1e-10 for k, v in gains.items())
    checks = {"fused_gain_at_least_1pp": gains["fused"] >= 1.0,
              "all_full_branches_not_below_signal": all(gains[k] >= 0 for k in ("cnn", "transformer", "mamba")),
              "identity_bootstrap_lower_positive": lower > 0,
              "fused_strictly_best": all(computed["fused"]["mAP"] > computed[k]["mAP"]
                                           for k in ("baseline_only", "cnn", "transformer", "mamba"))}
    assert checks == comparison["scientific_checks"] and all(checks.values()) == comparison["scientific_passed"]
    assert summary["status"] == ("COMPLETE_OFFICIAL_COMPARISON_SUPPORT_PASS" if all(checks.values())
                                  else "COMPLETE_OFFICIAL_COMPARISON_SUPPORT_FAIL")
    assert max(metric_errors) < 1e-10
    census_path = args.summary.parent / "query_error_census.json"
    assert not census_path.exists()
    census_path.write_text(json.dumps({"scope": "All1715 official queries and all5 outputs; descriptive only",
        "summary_sha256": sha(args.summary), "queries": census, "per_identity": per_identity}, indent=2) + "\n")
    result = {"status": "PASS_COMPLETE_OFFICIAL_FILES_ARRAYS_RANKINGS_AND_SCORES", "verified_at": datetime.now().astimezone().isoformat(),
              "summary_sha256": sha(args.summary), "verifier_sha256": sha(__file__), "metrics": computed,
              "gains_over_signal_pp": gains, "scientific_checks": checks, "scientific_passed": all(checks.values()),
              "identity_bootstrap_lower_pp": lower, "maximum_metric_absolute_difference": max(metric_errors),
              "bootstrap_absolute_difference": abs(lower - bootstrap["lower_bound_pp"]),
              "queries": 1715, "gallery_records": 8575, "identities": 50, "query_output_scores_verified": rows_verified,
              "distance_entries_verified": 1715 * 8575 * 5, "complete_ranking_positions_verified": 1715 * 8575 * 5,
              "signal_features_and_distances_bitwise_equal": True, "all5_saved_distance_matrices_bitwise_recomputed": True,
              "error_census": str(census_path), "error_census_sha256": sha(census_path),
              "model_forwards": 0, "optimizer_updates": 0,
              "scope": "Remote CPU whole saved endpoints/arrays and complete label/ranking/metric replay; no model or image forward",
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
