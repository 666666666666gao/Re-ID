#!/usr/bin/env python3
"""Reconstruct the frozen-baseline/residual-bank split from saved official distances."""

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
    assert receipt["status"] == "COMPLETE" and receipt["dataset"] == "MSVR310"
    assert receipt["method"] in ("PLAIN_V8", "SIGNAL_V8")
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

    # Both the frozen baseline and the normalized three-role bank have half of
    # the final squared-Euclidean distance: d_fused^2 = (d_base^2 + d_bank^2)/2.
    bank = 2 * fused - baseline
    residuals = {name: 2 * distances[name] - baseline for name in ROLES}
    bank_error = float(np.max(np.abs(bank - sum(residuals.values()) / len(ROLES))))
    assert bank_error < 1e-5

    def metrics(matrix):
        values = scene_scores(matrix, saved["query_ids"], saved["gallery_ids"],
                              saved["query_scenes"], saved["gallery_scenes"])["metrics"]
        return {name: values[name] for name in ("mAP", "Rank-1")}

    for name, matrix in (("baseline_only", baseline), ("fused", fused)):
        for metric, value in metrics(matrix).items():
            assert abs(value - receipt["outputs"][name]["metrics"][metric]) < 1e-4

    result = {
        "scope": "posthoc official test diagnosis; not a model-selection metric",
        "dataset": receipt["dataset"], "method": receipt["method"], "seed": receipt["seed"],
        "receipt_sha256": digest(args.receipt),
        "distance_arrays_sha256": receipt["distance_arrays_sha256"],
        "fusion_identity_max_error": fusion_error,
        "bank_identity_max_error": bank_error,
        "baseline": metrics(baseline), "residual_bank": metrics(bank), "fused": metrics(fused),
        "role_residuals": {name: metrics(matrix) for name, matrix in residuals.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
