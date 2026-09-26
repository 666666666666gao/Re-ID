#!/usr/bin/env python3
"""Reconstruct the checkpoint's baseline/residual-bank split from saved distances."""

import argparse
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


ROLES = ("cnn", "transformer", "mamba")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE"
    assert receipt["dataset"] in ("RGBNT201", "RGBNT100", "MSVR310")
    assert receipt["method"] in ("PLAIN_V8", "SIGNAL_V8", "SIGNAL_V8_BRANCH_ONLY", "SIGNAL_SIM_FEEDBACK", "SIGNAL_SIM_JOINT_STAGED")
    distance_path = Path(receipt["distance_arrays"])
    assert digest(distance_path) == receipt["distance_arrays_sha256"]
    saved = torch.load(distance_path, map_location="cpu", weights_only=False)
    assert saved["protocol_sha256"] == receipt["protocol_sha256"]
    distances = {name: value.numpy() for name, value in saved["distances"].items()}
    baseline = distances["baseline_only"]
    fused = distances["fused"]
    role_distances = [distances[name] for name in ROLES]
    fusion_error = float(np.max(np.abs(fused - sum(role_distances) / len(ROLES))))
    assert fusion_error < 1e-5

    # Both the checkpoint's baseline and the normalized three-role bank have half of
    # the final squared-Euclidean distance: d_fused^2 = (d_base^2 + d_bank^2)/2.
    bank = 2 * fused - baseline
    residuals = {name: 2 * distances[name] - baseline for name in ROLES}
    bank_error = float(np.max(np.abs(bank - sum(residuals.values()) / len(ROLES))))
    assert bank_error < 1e-5

    def scores(matrix):
        if receipt["dataset"] == "MSVR310":
            return scene_scores(matrix, saved["query_ids"], saved["gallery_ids"],
                                saved["query_scenes"], saved["gallery_scenes"])
        else:
            return camera_scores(matrix, saved["query_ids"], saved["gallery_ids"],
                                 saved["query_cameras"], saved["gallery_cameras"])

    def metrics(values):
        names = (("mAP", "Rank-1", "Rank-5", "Rank-10") if receipt["dataset"] == "RGBNT201"
                 else ("mAP", "Rank-1"))
        return {name: values["metrics"][name] for name in names}

    rankings = {"baseline_only": scores(baseline), "residual_bank": scores(bank),
                "fused": scores(fused)}
    top1 = {name: np.asarray(value["first_match_rank"]) == 1
            for name, value in rankings.items()}
    base, residual, combined = (top1[name] for name in rankings)

    for name, matrix in (("baseline_only", baseline), ("fused", fused)):
        for metric, value in metrics(rankings[name]).items():
            assert abs(value - receipt["outputs"][name]["metrics"][metric]) < 1e-4

    result = {
        "schema": "trifusion-official-residual-bank-posthoc-v2",
        "scope": "posthoc official test diagnosis; not a model-selection metric",
        "dataset": receipt["dataset"], "method": receipt["method"], "seed": receipt["seed"],
        "receipt_sha256": digest(args.receipt),
        "distance_arrays_sha256": receipt["distance_arrays_sha256"],
        "fusion_identity_max_error": fusion_error,
        "bank_identity_max_error": bank_error,
        "baseline": metrics(rankings["baseline_only"]),
        "residual_bank": metrics(rankings["residual_bank"]),
        "fused": metrics(rankings["fused"]),
        "role_residuals": {name: metrics(scores(matrix)) for name, matrix in residuals.items()},
        "top1_query_counts": {
            "total": int(len(base)),
            "baseline_correct": int(base.sum()),
            "bank_correct": int(residual.sum()),
            "fused_correct": int(combined.sum()),
            "baseline_correct_bank_wrong": int((base & ~residual).sum()),
            "baseline_wrong_bank_correct": int((~base & residual).sum()),
            "baseline_correct_fused_wrong": int((base & ~combined).sum()),
            "baseline_wrong_fused_correct": int((~base & combined).sum()),
            "bank_wrong_fused_correct": int((~residual & combined).sum()),
            "bank_correct_fused_wrong": int((residual & ~combined).sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
