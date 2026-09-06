#!/usr/bin/env python3
"""Describe every RGBNT100 query error from completed, verified saved rankings."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    started = time.perf_counter()
    assert not args.output.exists()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    verified = json.loads(args.scalar_verification.read_text(encoding="utf-8"))
    assert verified["status"] == "PASS_COMPLETE_SCALAR_REPLAY"
    assert verified["queries"] == 8675 and verified["query_output_evaluations"] == 43375
    assert summary["status"] in ("COMPLETE_COMPARISON_SUPPORT_PASS", "COMPLETE_COMPARISON_SUPPORT_FAIL")
    assert verified["input_file_sha256"][str(args.summary)] == sha(args.summary)
    assert verified["input_file_sha256"][str(args.protocol)] == sha(args.protocol)
    rows, inputs = [], [args.summary, args.protocol, args.scalar_verification]
    records = protocol["records"]
    for fold, actual in zip(protocol["folds"], summary["folds"], strict=True):
        index = fold["fold"]
        assert actual["fold"] == index
        rank_path = args.rankings_dir / f"fold_{index}" / "rankings.json.gz"
        assert verified["input_file_sha256"][str(rank_path)] == sha(rank_path)
        inputs.append(rank_path)
        with gzip.open(rank_path, "rt", encoding="utf-8") as handle:
            orders = json.load(handle)
        gallery = [records[i] for i in fold["gallery_record_indices"]]
        identities = Counter(r["identity"] for r in gallery)
        cameras = Counter(r["camera"] for r in gallery)
        members = Counter((r["identity"], r["camera"]) for r in gallery)
        for query_index, query in enumerate(fold["query_rows"]):
            query_record = records[query["record_index"]]
            identity, camera = query_record["identity"], query_record["camera"]
            negative_count = len(gallery) - identities[identity]
            same_camera_count = cameras[camera] - members[identity, camera]
            row = {"fold": index, "record_index": query["record_index"],
                   "identity": identity, "camera": camera, "path": query_record["path"],
                   "negative_candidates": negative_count,
                   "same_camera_negative_candidates": same_camera_count,
                   "same_camera_negative_candidate_fraction": same_camera_count / negative_count,
                   "outputs": {}}
            for name in OUTPUTS:
                order = orders[name][query_index]
                first = next(gallery[i] for i in order
                             if not (gallery[i]["identity"] == identity and gallery[i]["camera"] == camera))
                negative = next(gallery[i] for i in order if gallery[i]["identity"] != identity)
                values = actual["retrieval"]["outputs"][name]
                rank = values["first_match_rank"][query_index]
                correct = first["identity"] == identity
                assert (rank == 1) == correct
                row["outputs"][name] = {
                    "average_precision": values["average_precision"][query_index],
                    "first_positive_rank": rank, "rank1_correct": correct,
                    "first_retained_record_index": first["index"],
                    "first_retained_identity": first["identity"],
                    "first_retained_camera": first["camera"],
                    "nearest_negative_record_index": negative["index"],
                    "nearest_negative_identity": negative["identity"],
                    "nearest_negative_camera": negative["camera"],
                    "nearest_negative_same_camera": negative["camera"] == camera}
            rows.append(row)
        del orders
    assert len(rows) == 8675 and len({r["identity"] for r in rows}) == 50
    assert sorted(r["record_index"] for r in rows) == list(range(8675))
    aggregates, per_identity = {}, []
    by_identity = defaultdict(list)
    for row in rows:
        by_identity[row["identity"]].append(row)
    for identity, group in sorted(by_identity.items()):
        maps = {name: statistics.fmean(r["outputs"][name]["average_precision"] for r in group) * 100
                for name in OUTPUTS}
        per_identity.append({"identity": identity, "queries": len(group), "mAP": maps,
                             "mAP_gain_over_signal_pp": {name: maps[name] - maps["baseline_only"] for name in OUTPUTS}})
    for name in OUTPUTS:
        values = [r["outputs"][name] for r in rows]
        errors = [r for r in rows if not r["outputs"][name]["rank1_correct"]]
        new_errors = [r for r in errors if r["outputs"]["baseline_only"]["rank1_correct"]]
        repaired = [r for r in rows if r["outputs"][name]["rank1_correct"]
                    and not r["outputs"]["baseline_only"]["rank1_correct"]]
        deltas = [r["outputs"][name]["average_precision"] - r["outputs"]["baseline_only"]["average_precision"]
                  for r in rows]
        changes = {"ap_improved": sum(d > 0 for d in deltas),
                   "ap_declined": sum(d < 0 for d in deltas), "ap_unchanged": sum(d == 0 for d in deltas),
                   "rank1_repaired": len(repaired), "rank1_new_errors": len(new_errors)}
        if name != "baseline_only":
            assert changes == summary["comparison"]["query_changes"][name]
        aggregates[name] = {
            "all_queries": len(rows), **changes, "all_rank1_errors": len(errors),
            "all_rank1_errors_same_camera": sum(r["outputs"][name]["nearest_negative_same_camera"] for r in errors),
            "new_errors_same_camera": sum(r["outputs"][name]["nearest_negative_same_camera"] for r in new_errors),
            "repaired_errors_original_same_camera": sum(r["outputs"]["baseline_only"]["nearest_negative_same_camera"] for r in repaired),
            "all_queries_nearest_negative_same_camera": sum(v["nearest_negative_same_camera"] for v in values),
            "all_query_negative_candidate_same_camera_fraction_mean": statistics.fmean(
                r["same_camera_negative_candidate_fraction"] for r in rows),
            "new_error_query_negative_candidate_same_camera_fraction_sum": sum(
                r["same_camera_negative_candidate_fraction"] for r in new_errors),
            "mAP": statistics.fmean(v["average_precision"] for v in values) * 100,
            "Rank-1": (len(rows) - len(errors)) / len(rows) * 100}
        assert abs(aggregates[name]["mAP"] - verified["metrics"][name]["mAP"]) < 1e-10
        assert abs(aggregates[name]["Rank-1"] - verified["metrics"][name]["Rank-1"]) < 1e-10
    result = {
        "observed_at": datetime.now().astimezone().isoformat(), "status": "COMPLETE_ALL8675_QUERY_ERROR_CENSUS",
        "scope": "Descriptive census of all8675 queries/all5 outputs/50 identities using verified saved rankings. No new scientific gate, model selection, inference or training.",
        "interpretation": "Candidate camera composition is a label count, not a random-retrieval null model or causal proof. Error-camera counts do not identify visual background, occlusion or view mechanisms.",
        "inputs_sha256": {str(p): sha(p) for p in inputs}, "script_sha256": sha(Path(__file__)),
        "summary": aggregates, "per_identity": per_identity, "all_queries": rows,
        "training": {"optimizer_steps": verified["optimizer_steps"],
                     "source_record_exposures": verified["source_record_exposures"],
                     "same_identity_positive_pairs": sum(f["same_identity_positive_pairs"] for f in verified["folds"]),
                     "cross_camera_positive_pairs": sum(f["cross_camera_positive_pairs"] for f in verified["folds"])},
        "local_model_tensor_image_calls": 0, "optimizer_updates": 0, "new_retrieval_forwards": 0,
        "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("all_queries", "per_identity")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "protocol", "scalar-verification", "rankings-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
