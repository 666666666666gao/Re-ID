#!/usr/bin/env python3
"""Evaluate fixed full50 endpoints on every official RGBNT100 query and gallery."""

import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_rgbnt100_signal_oof import camera_scores, extract as extract_signal, new_model, sha256, write_json
from tools.train_rgbnt100_trifusion_main import load_inputs
from tools.train_rgbnt100_trifusion_oof import EXPERTS, OUTPUT_WIDTHS, build_model, extract


def distance_matrix(values):
    import torch

    values = torch.nn.functional.normalize(values.float(), dim=1)
    qf, gf = values[:1715], values[1715:]
    matrix = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None]
    matrix.addmm_(qf, gf.T, beta=1, alpha=-2)
    assert matrix.shape == (1715, 8575) and torch.isfinite(matrix).all().item()
    return matrix


def comparison_summary(scores, query_rows):
    import numpy as np
    import torch
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound

    identities = np.asarray([r["identity"] for r in query_rows])
    assert len(identities) == 1715 and len(np.unique(identities)) == 50
    aps = {name: np.asarray(row["average_precision"]) for name, row in scores.items()}
    first = {name: np.asarray(row["first_match_rank"]) for name, row in scores.items()}
    metrics = {name: row["metrics"] for name, row in scores.items()}
    differences = {name: (values - aps["baseline_only"]) * 100 for name, values in aps.items()}
    bootstrap = identity_cluster_bootstrap_lower_bound(torch.from_numpy(differences["fused"]),
                                                       torch.from_numpy(identities), seed=42, resamples=10000)
    gains = {name: metrics[name]["mAP"] - metrics["baseline_only"]["mAP"] for name in OUTPUT_WIDTHS}
    checks = {"fused_gain_at_least_1pp": gains["fused"] >= 1.0,
              "all_full_branches_not_below_signal": all(gains[k] >= 0 for k in EXPERTS),
              "identity_bootstrap_lower_positive": bootstrap.lower_bound > 0,
              "fused_strictly_best": all(metrics["fused"]["mAP"] > metrics[k]["mAP"]
                                           for k in ("baseline_only", *EXPERTS))}
    identities_report = [{"identity": int(identity), "query_count": int(np.sum(identities == identity)),
                          "map_by_output": {name: float(values[identities == identity].mean() * 100)
                                            for name, values in aps.items()}}
                         for identity in np.unique(identities)]
    changes = {name: {"ap_improved": int(np.sum(differences[name] > 0)),
                       "ap_declined": int(np.sum(differences[name] < 0)),
                       "ap_unchanged": int(np.sum(differences[name] == 0)),
                       "rank1_repaired": int(np.sum((first["baseline_only"] > 1) & (first[name] == 1))),
                       "rank1_new_errors": int(np.sum((first["baseline_only"] == 1) & (first[name] > 1)))}
               for name in ("fused", *EXPERTS)}
    return {"metrics": metrics, "gains_over_signal_pp": gains, "scientific_checks": checks,
            "scientific_passed": all(checks.values()), "per_identity": identities_report, "query_changes": changes,
            "identity_bootstrap": {"lower_bound_pp": bootstrap.lower_bound, "seed": 42, "resamples": 10000,
                                   "cluster_count": bootstrap.cluster_count, "percentile": 2.5,
                                   "quantile_method": "linear", "weighting": "whole identity resampling with query weights"}}


def run(args):
    import numpy as np
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256

    started = time.perf_counter()
    assert not args.output_dir.exists()
    config = json.loads(args.config.read_text())
    assert config["schema"] == "rgbnt100-fixed-official-evaluation-v1"
    for name, expected in config["source_files_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    for name in ("roles_config", "roles_summary", "roles_verification"):
        assert sha256(ROOT / config[name]) == config[name + "_sha256"], name
    role_config, baseline, protocol, cfg, binding, fold, _ = load_inputs(ROOT / config["roles_config"])
    roles = json.loads((ROOT / config["roles_summary"]).read_text())
    verified = json.loads((ROOT / config["roles_verification"]).read_text())
    assert roles["status"] == "COMPLETE_FULL50_ROLES_FIXED_EPOCH20"
    assert roles["training"]["epochs"] == 20 and all(roles["engineering_checks"].values())
    assert roles["config_sha256"] == config["roles_config_sha256"]
    assert roles["baseline_summary_sha256"] == role_config["BASELINE"]["SUMMARY_SHA256"]
    assert verified["status"] == "PASS_FULL50_ROLES_FILES_AND_ALL_UPDATES" and verified["engineering_passed"]
    assert verified["mode"] == "main" and verified["summary_sha256"] == config["roles_summary_sha256"]
    assert sha256(roles["checkpoint"]) == roles["checkpoint_sha256"]
    assert roles["official_model_record_forwards"] == baseline["official_model_record_forwards"] == 0
    query, gallery = protocol["records"]["query"], protocol["records"]["gallery"]
    assert len(query) == 1715 and len(gallery) == 8575
    records = [(str(Path(protocol["dataset_root"]) / r["path"]), r["identity"], r["camera"], r["view"])
               for r in query + gallery]
    assert not set(fold["source_ids"]) & {r["identity"] for r in query + gallery}
    args.output_dir.mkdir(parents=True)
    summary = {"schema": config["schema"], "status": "RUNNING", "started_at": datetime.now().astimezone().isoformat(),
               "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "config_sha256": sha256(args.config), "runner_sha256": sha256(__file__),
               "protocol_sha256": role_config["DATA"]["PROTOCOL_SHA256"],
               "roles_summary_sha256": config["roles_summary_sha256"],
               "baseline_summary_sha256": role_config["BASELINE"]["SUMMARY_SHA256"], **binding,
               "query_count": 1715, "gallery_count": 8575, "query_identities": 50, "seed": 42,
               "checkpoint_selection": {"signal": "fixed_epoch30", "roles": "fixed_epoch20"},
               "training_updates": 0, "reranking": False, "rgbnt201_dev_record_forwards": 0,
               "signal_official_record_forwards": 0, "roles_official_record_forwards": 0,
               "filter": "exclude same identity AND same camera; keep every different identity"}
    write_json(args.output_dir / "summary.json", summary)
    torch.cuda.reset_peak_memory_stats()
    for row in query + gallery:
        assert sha256(Path(protocol["dataset_root"]) / row["path"]) == row["sha256"], row["path"]
    signal = new_model(cfg, fold)
    payload = torch.load(baseline["checkpoint"], map_location="cpu", weights_only=True)
    signal.load_state_dict(payload["model_state_dict"], strict=True)
    assert _module_state_sha256(signal) == baseline["training"]["final_state_sha256"]
    phase_started = time.perf_counter()
    baseline_features = extract_signal(signal, records, cfg)
    assert baseline_features.shape == (10290, 3072)
    assert _module_state_sha256(signal) == baseline["training"]["final_state_sha256"]
    summary.update({"signal_official_record_forwards": 10290,
                    "signal_extraction_seconds": time.perf_counter() - phase_started})
    write_json(args.output_dir / "summary.json", summary)
    del signal, payload
    torch.cuda.empty_cache()
    model, initial = build_model(role_config, cfg, fold, baseline)
    assert initial == roles["initialization"]
    payload = torch.load(roles["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    assert _module_state_sha256(model) == roles["training"]["final_state_sha256"]
    phase_started = time.perf_counter()
    features = extract(model, records, cfg)
    assert all(values.shape == (10290, OUTPUT_WIDTHS[name]) for name, values in features.items())
    assert torch.equal(features["baseline_only"], baseline_features)
    assert _module_state_sha256(model) == roles["training"]["final_state_sha256"]
    summary.update({"roles_official_record_forwards": 10290,
                    "roles_extraction_seconds": time.perf_counter() - phase_started,
                    "signal_features_bitwise_equal": True,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2})
    write_json(args.output_dir / "summary.json", summary)
    del model, payload
    torch.cuda.empty_cache()
    from utils.metrics import eval_func

    query_ids, gallery_ids = (np.asarray([r["identity"] for r in rows]) for rows in (query, gallery))
    query_cameras, gallery_cameras = (np.asarray([r["camera"] for r in rows]) for rows in (query, gallery))
    baseline_distances = distance_matrix(baseline_features)
    scores, distances, rankings = {}, {}, {}
    for name, values in features.items():
        matrix = distance_matrix(values)
        if name == "baseline_only":
            assert torch.equal(matrix, baseline_distances)
        scores[name] = camera_scores(matrix.numpy(), query_ids, gallery_ids, query_cameras, gallery_cameras)
        cmc, mean_ap = eval_func(matrix.numpy(), query_ids, gallery_ids, query_cameras, gallery_cameras)
        difference = {"mAP": abs(scores[name]["metrics"]["mAP"] - float(mean_ap * 100))}
        difference.update({f"Rank-{k}": abs(scores[name]["metrics"][f"Rank-{k}"] - float(cmc[k - 1] * 100))
                           for k in (1, 5, 10)})
        assert difference["mAP"] < 1e-10 and max(difference.values()) < 1e-5
        scores[name]["upstream_metric_difference_pp"] = difference
        order = np.argsort(matrix.numpy(), axis=1)
        ranking_path = args.output_dir / ("rankings_" + name + ".jsonl.gz")
        with gzip.open(ranking_path, "wt", encoding="utf-8") as handle:
            for index, row in enumerate(order):
                handle.write(json.dumps({"query_index": index, "order": row.tolist()}, separators=(",", ":")) + "\n")
        rankings[name] = {"path": str(ranking_path), "bytes": ranking_path.stat().st_size,
                          "sha256": sha256(ranking_path), "query_rows": 1715, "positions_per_query": 8575}
        distances[name] = matrix
    array_path = args.output_dir / "retrieval_arrays.pt"
    torch.save({"features": features, "distances": distances,
                "independent_signal_features": baseline_features, "independent_signal_distances": baseline_distances,
                "query_record_indices": list(range(1715)), "gallery_record_indices": list(range(8575)),
                "record_order": "all1715 query followed by all8575 gallery", "protocol_sha256": summary["protocol_sha256"]}, array_path)
    summary.update({"retrieval": {"outputs": scores, "feature_widths": OUTPUT_WIDTHS, "rankings": rankings,
                                  "retrieval_arrays": str(array_path), "retrieval_arrays_bytes": array_path.stat().st_size,
                                  "retrieval_arrays_sha256": sha256(array_path)},
                    "signal_distances_bitwise_equal": True, "comparison": comparison_summary(scores, query),
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "elapsed_seconds": time.perf_counter() - started})
    summary["status"] = ("COMPLETE_OFFICIAL_COMPARISON_SUPPORT_PASS" if summary["comparison"]["scientific_passed"]
                         else "COMPLETE_OFFICIAL_COMPARISON_SUPPORT_FAIL")
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "comparison": summary["comparison"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
