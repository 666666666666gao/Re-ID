#!/usr/bin/env python3
"""Reconstruct logged task scalars for verified, full 50-epoch endpoints."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(row):
    run = Path(row["run_dir"])
    receipt = run / "official_metrics.json"
    assert sha(receipt) == row["receipt_sha256"]
    training_path = run / "training.json"
    training = json.loads(training_path.read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert [item["epoch"] for item in training["history"]] == list(range(1, 51))
    weight = training["initializer"]["prediction_weight"]
    enabled = training["initializer"]["m3"]
    steps_path = run / "training_steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    grouped = {epoch: [] for epoch in range(1, 51)}
    errors = []
    for step in steps:
        prediction = step["prediction"]
        assert set(prediction) == ({"local", "relation", "cross_modal"} if enabled else set())
        values = [step["loss"], step["id"], step["triplet"], *prediction.values()]
        assert all(math.isfinite(value) for value in values)
        reconstructed = step["id"] + step["triplet"] + weight * sum(prediction.values())
        assert math.isclose(reconstructed, step["loss"], rel_tol=1e-5, abs_tol=1e-5)
        errors.append(abs(reconstructed - step["loss"]))
        grouped[step["epoch"]].append(step)
    history = []
    for item in training["history"]:
        epoch_steps = grouped[item["epoch"]]
        assert [step["batch"] for step in epoch_steps] == list(range(item["steps"]))
        count = len(epoch_steps)
        averages = {name: math.fsum(step[name] for step in epoch_steps) / count
                    for name in ("loss", "id", "triplet")}
        assert math.isclose(averages["loss"], item["mean_loss"], rel_tol=1e-8, abs_tol=1e-8)
        prediction = {name: math.fsum(step["prediction"][name] for step in epoch_steps) / count
                      for name in ("local", "relation", "cross_modal") if enabled}
        history.append({"epoch": item["epoch"], "steps": count, **averages,
                        "prediction": prediction, "weighted_prediction": weight * sum(prediction.values()),
                        "official_fused": item["official_fused"]})
    assert len(steps) == sum(item["steps"] for item in training["history"])
    return {"dataset": row["dataset"], "variant": row["variant"],
            "best_epoch": row["best_epoch"], "m3_enabled": enabled,
            "prediction_weight": weight, "logged_steps": len(steps),
            "nonzero_triplet_steps": sum(step["triplet"] > 0 for step in steps),
            "nonzero_prediction_steps": {name: sum(step["prediction"][name] > 0 for step in steps)
                                         for name in ("local", "relation", "cross_modal") if enabled},
            "maximum_loss_reconstruction_error": max(errors),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "receipt_sha256": row["receipt_sha256"],
            "training_sha256": sha(training_path), "steps_sha256": sha(steps_path),
            "first": history[0], "best": history[row["best_epoch"] - 1],
            "last": history[-1], "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows = [audit(row) for row in matrix["rows"] if row["status"] == "VERIFIED_COMPLETE"]
    assert len(rows) == matrix["verified_complete"]
    report = {"status": "FULL_LOG_TASK_SCALAR_RECONSTRUCTION_COMPLETE",
              "at": datetime.now().astimezone().isoformat(), "source_sha256": sha(Path(__file__)),
              "matrix_sha256": sha(args.matrix), "verified_endpoints": len(rows), "rows": rows,
              "boundary": "Read-only post-selection scalar audit. No inference, training, gradient measurement or estimate of optimizer update shares; scalar magnitudes do not establish causality or task usefulness."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "verified_endpoints": len(rows),
                      "logged_steps": sum(row["logged_steps"] for row in rows),
                      "maximum_loss_reconstruction_error": max(row["maximum_loss_reconstruction_error"] for row in rows),
                      "task_support": [{key: row[key] for key in
                                        ("dataset", "variant", "logged_steps", "nonzero_triplet_steps", "nonzero_prediction_steps")}
                                       for row in rows]}))


if __name__ == "__main__":
    main()
