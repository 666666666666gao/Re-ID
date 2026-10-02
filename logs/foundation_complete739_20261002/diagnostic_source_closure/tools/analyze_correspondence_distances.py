#!/usr/bin/env python3
"""Compare verified ablations using saved distances, without GPU inference."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(matrix, dataset, control, candidate):
    rows = {row["variant"]: row for row in matrix["rows"]
            if row["dataset"] == dataset and row["status"] == "VERIFIED_COMPLETE"}
    scores, arrays, bindings = {}, {}, {}
    for bits in (control, candidate):
        row = rows[bits]
        run = Path(row["run_dir"])
        assert sha(run / "official_metrics.json") == row["receipt_sha256"]
        assert sha(run / "official_distances.pt") == row["distance_sha256"]
        arrays[bits] = torch.load(run / "official_distances.pt", map_location="cpu", weights_only=False)
        data = arrays[bits]
        environment = "scenes" if dataset == "MSVR310" else "cameras"
        scorer = scene_scores if dataset == "MSVR310" else camera_scores
        scores[bits] = scorer(data["fused"].numpy(), data["query_ids"], data["gallery_ids"],
                              data[f"query_{environment}"], data[f"gallery_{environment}"])
        assert all(abs(scores[bits]["metrics"][name] - row["metrics"][name]) < 1e-5
                   for name in row["metrics"])
        bindings[bits] = {name: row[name] for name in
                          ("best_epoch", "checkpoint_sha256", "distance_sha256", "receipt_sha256")}
    for name in ("query_ids", "gallery_ids", "query_cameras", "gallery_cameras", "query_scenes", "gallery_scenes"):
        assert np.array_equal(arrays[control][name], arrays[candidate][name])
    assert arrays[control]["fused"].shape == arrays[candidate]["fused"].shape
    first, second = scores[control], scores[candidate]
    control_rank, candidate_rank = (np.asarray(row["first_match_rank"]) for row in (first, second))
    delta_ap = np.asarray(second["average_precision"]) - np.asarray(first["average_precision"])
    repairs = int(((control_rank != 1) & (candidate_rank == 1)).sum())
    new_errors = int(((control_rank == 1) & (candidate_rank != 1)).sum())
    delta_metrics = {name: second["metrics"][name] - first["metrics"][name] for name in first["metrics"]}
    assert abs(delta_metrics["Rank-1"] - (repairs - new_errors) * 100 / len(delta_ap)) < 1e-5
    qids = arrays[control]["query_ids"]
    identities = [{"identity": int(identity), "queries": int((qids == identity).sum()),
                   "mean_delta_ap_points": float(delta_ap[qids == identity].mean() * 100)}
                  for identity in np.unique(qids)]
    changes = np.asarray([row["mean_delta_ap_points"] for row in identities])
    bootstrap = np.random.default_rng(42).choice(changes, size=(2000, len(changes)), replace=True).mean(axis=1)
    return {"status": "CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE", "dataset": dataset,
            "control": control, "candidate": candidate,
            "time": datetime.now().astimezone().isoformat(), "bindings": bindings,
            "metrics": {control: first["metrics"], candidate: second["metrics"]},
            "delta_metrics": delta_metrics, "rank1_repairs": repairs, "rank1_new_errors": new_errors,
            "query_ap_improved": int((delta_ap > 1e-8).sum()),
            "query_ap_worsened": int((delta_ap < -1e-8).sum()),
            "identity_ap_improved": int((changes > 1e-6).sum()),
            "identity_ap_worsened": int((changes < -1e-6).sum()),
            "identity_macro_mean_delta_ap_points": float(changes.mean()),
            "identity_bootstrap_95_percentile_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "identity_changes": identities,
            "boundary": "Official post-selection diagnosis, not new inference or selection. Different capacity and possibly random-number consumption. Bootstrap resamples fixed-model identities, not training seeds."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.matrix.read_text()), args.dataset, args.control, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "identity_changes"}))


if __name__ == "__main__":
    main()
