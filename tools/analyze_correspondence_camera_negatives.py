#!/usr/bin/env python3
"""Compare RGBNT201 negative-camera availability and verified retrieval errors."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import camera_scores, sha


def summarize(indices, availability, nearest_same, top_same):
    return {
        "queries": len(indices),
        "negative_pool_same_camera_fraction_mean": float(availability[indices].mean()),
        "negative_pool_same_camera_fraction_min": float(availability[indices].min()),
        "negative_pool_same_camera_fraction_max": float(availability[indices].max()),
        "nearest_negative_same_camera": int(nearest_same[indices].sum()),
        "top_gallery_same_camera": int(top_same[indices].sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT / "logs/official_three_dataset_protocols_20260923/RGBNT201.json"
    protocol = json.loads(protocol_path.read_text())
    records = protocol["records"]
    matrix = json.loads(args.matrix.read_text())
    rows = {row["variant"]: row for row in matrix["rows"]
            if row["dataset"] == "RGBNT201" and row["status"] == "VERIFIED_COMPLETE"}
    qids = np.asarray([row["identity"] for row in records["query"]])
    gids = np.asarray([row["identity"] for row in records["gallery"]])
    qcam = np.asarray([row["camera"] for row in records["query"]])
    gcam = np.asarray([row["camera"] for row in records["gallery"]])
    negative = qids[:, None] != gids[None, :]
    same_camera = qcam[:, None] == gcam[None, :]
    positive = (~negative) & (~same_camera)
    assert negative.any(axis=1).all() and positive.any(axis=1).all()
    same_count = (negative & same_camera).sum(axis=1)
    total_count = negative.sum(axis=1)
    availability = same_count / total_count
    report = {
        "status": "CPU_NEGATIVE_CAMERA_AVAILABILITY_AND_RETRIEVAL_PARITY_COMPLETE",
        "at": datetime.now().astimezone().isoformat(),
        "dataset": "RGBNT201", "protocol_sha256": sha(protocol_path),
        "source_sha256": sha(Path(__file__)), "new_inference": False,
        "training_modified": False,
        "reference_definition": "Fraction among all different-identity gallery records; uniform negative selection reference, not a calibrated error probability.",
        "limits": "Post-selection descriptive diagnosis. Camera coincidence does not establish background dependence or a causal module effect. No gallery exclusions beyond the author protocol.",
        "gallery_camera_counts": {str(cam): int((gcam == cam).sum()) for cam in np.unique(gcam)},
        "negative_pairs": int(negative.sum()),
        "same_camera_negative_pairs": int((negative & same_camera).sum()),
        "all_query_availability_mean": float(availability.mean()),
        "outputs": {},
    }
    first_ranks, camera_results = {}, {}
    for bits in ("000", "111"):
        row = rows[bits]
        run = Path(row["run_dir"])
        assert sha(run / "official_metrics.json") == row["receipt_sha256"]
        assert sha(run / "official_distances.pt") == row["distance_sha256"]
        receipt = json.loads((run / "official_metrics.json").read_text())
        assert receipt["protocol_sha256"] == report["protocol_sha256"]
        saved = torch.load(run / "official_distances.pt", map_location="cpu", weights_only=False)
        for key, expected in (("query_ids", qids), ("gallery_ids", gids),
                              ("query_cameras", qcam), ("gallery_cameras", gcam)):
            assert np.array_equal(saved[key], expected)
        distance = saved["fused"].numpy()
        assert distance.shape == negative.shape and np.isfinite(distance).all()
        scores = camera_scores(distance, qids, gids, qcam, gcam)
        assert all(abs(scores["metrics"][key] - row["metrics"][key]) < 1e-5
                   for key in row["metrics"])
        first = np.asarray(scores["first_match_rank"])
        nearest_negative = np.argmin(np.where(negative, distance, np.inf), axis=1)
        nearest_same = gcam[nearest_negative] == qcam
        top = np.asarray([order[(negative[i, order] | positive[i, order])][0]
                          for i, order in enumerate(np.argsort(distance, axis=1))])
        top_same = gcam[top] == qcam
        errors = np.flatnonzero(first != 1)
        assert np.array_equal(top[errors], nearest_negative[errors])
        result = {
            "best_epoch": row["best_epoch"], "metrics": scores["metrics"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "distance_sha256": row["distance_sha256"],
            "receipt_sha256": row["receipt_sha256"],
            "all_queries": summarize(np.arange(len(qids)), availability, nearest_same, top_same),
            "rank1_errors": summarize(errors, availability, nearest_same, top_same),
            "error_camera_breakdown": {},
        }
        for cam in np.unique(qcam[errors]):
            indices = errors[qcam[errors] == cam]
            result["error_camera_breakdown"][str(cam)] = summarize(
                indices, availability, nearest_same, top_same)
        report["outputs"][bits] = result
        first_ranks[bits] = first
        camera_results[bits] = (nearest_same, top_same)
    new_errors = np.flatnonzero((first_ranks["000"] == 1) & (first_ranks["111"] != 1))
    repairs = np.flatnonzero((first_ranks["000"] != 1) & (first_ranks["111"] == 1))
    report["111_new_errors"] = summarize(new_errors, availability, *camera_results["111"])
    report["000_repaired_errors"] = summarize(repairs, availability, *camera_results["000"])
    report["per_query_availability"] = [
        {"query_index": i, "identity": int(qids[i]), "camera": int(qcam[i]),
         "negative_gallery_count": int(total_count[i]),
         "same_camera_negative_count": int(same_count[i]),
         "same_camera_negative_fraction": float(availability[i])} for i in range(len(qids))]
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "per_query_availability"}, indent=2))


if __name__ == "__main__":
    main()
