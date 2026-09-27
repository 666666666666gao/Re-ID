"""Evaluate a fixed equal-weight average of two completed official distance arrays."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-receipt", type=Path, required=True)
    parser.add_argument("--plain-receipt", type=Path, required=True)
    parser.add_argument("--author-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    signal = json.loads(args.signal_receipt.read_text(encoding="utf-8"))
    plain = json.loads(args.plain_receipt.read_text(encoding="utf-8"))
    assert signal["status"] == plain["status"] == "COMPLETE"
    assert plain["method"] == "PLAIN_V8"
    for key in ("dataset", "protocol_sha256", "filter", "reranking", "query_count", "gallery_count"):
        assert signal[key] == plain[key], key

    arrays = []
    for receipt in (signal, plain):
        path = Path(receipt["distance_arrays"])
        assert sha256(path) == receipt["distance_arrays_sha256"]
        arrays.append(torch.load(path, map_location="cpu", weights_only=False))
    reference, student = arrays
    for key in ("query_ids", "gallery_ids", "query_cameras", "gallery_cameras",
                "query_scenes", "gallery_scenes"):
        assert np.array_equal(reference[key], student[key]), key
    assert reference["protocol_sha256"] == signal["protocol_sha256"]
    assert student["protocol_sha256"] == plain["protocol_sha256"]

    sys.path.insert(0, str(args.author_metrics.parents[1]))
    spec = importlib.util.spec_from_file_location("signal_author_metrics", args.author_metrics)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    qids, gids = reference["query_ids"], reference["gallery_ids"]
    qcameras, gcameras = reference["query_cameras"], reference["gallery_cameras"]
    qscenes, gscenes = reference["query_scenes"], reference["gallery_scenes"]

    def evaluate(distance):
        if signal["dataset"] == "MSVR310":
            cmc, mean_ap = metrics.eval_func_msrv(
                distance, qids, gids, qcameras, gcameras, qscenes, gscenes)
        else:
            cmc, mean_ap = metrics.eval_func(distance, qids, gids, qcameras, gcameras)
        return {"mAP": float(mean_ap * 100), "Rank-1": float(cmc[0] * 100),
                "Rank-5": float(cmc[4] * 100), "Rank-10": float(cmc[9] * 100)}

    signal_distance = reference["distances"]["baseline_only"].numpy()
    plain_distance = student["distances"]["fused"].numpy()
    assert signal_distance.shape == plain_distance.shape == (len(qids), len(gids))
    signal_scores = evaluate(signal_distance)
    plain_scores = evaluate(plain_distance)
    for name, calculated, receipt in (("baseline_only", signal_scores, signal),
                                      ("fused", plain_scores, plain)):
        saved = receipt["outputs"][name]["metrics"]
        assert all(abs(calculated[key] - saved[key]) < 1e-5 for key in calculated)
    combined_scores = evaluate(0.5 * (signal_distance + plain_distance))
    report = {
        "status": "COMPLETE_FIXED_CROSS_BASELINE_DIAGNOSIS",
        "dataset": signal["dataset"],
        "scope": "Read-only official-test diagnostic, component checkpoints already selected on test",
        "fusion": "0.5 * author Signal squared Euclidean distance + 0.5 * plain V8 squared Euclidean distance",
        "signal_receipt_sha256": sha256(args.signal_receipt),
        "plain_receipt_sha256": sha256(args.plain_receipt),
        "signal_distance_sha256": signal["distance_arrays_sha256"],
        "plain_distance_sha256": plain["distance_arrays_sha256"],
        "protocol_sha256": signal["protocol_sha256"],
        "query_count": len(qids), "gallery_count": len(gids),
        "signal_metrics": signal_scores,
        "plain_v8_metrics": plain_scores,
        "equal_average_metrics": combined_scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
