#!/usr/bin/env python3
"""Reconstruct fused ID, triplet and auxiliary ID for complete context runs."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def audit(run, auxiliary_target):
    training = json.loads((run / "training.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert training["initializer"]["condition"]["auxiliary_target"] == auxiliary_target
    assert training["initializer"]["auxiliary_id_weight"] == 1.0
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    steps = [json.loads(line) for line in (run / "training_steps.jsonl").read_text().splitlines()]
    groups = {epoch: [] for epoch in range(1, 51)}
    errors = []
    for step in steps:
        assert step["auxiliary_target"] == auxiliary_target
        assert all(math.isfinite(step[key]) for key in ("loss", "id", "triplet", "auxiliary_id"))
        if auxiliary_target == "none":
            assert step["auxiliary_id"] == 0.0
        reconstructed = step["id"] + step["triplet"] + step["auxiliary_id"]
        assert math.isclose(reconstructed, step["loss"], rel_tol=1e-5, abs_tol=1e-5)
        errors.append(abs(reconstructed - step["loss"]))
        groups[step["epoch"]].append(step)
    history = []
    for item in training["history"]:
        group = groups[item["epoch"]]
        assert [step["batch"] for step in group] == list(range(item["steps"]))
        means = {key: math.fsum(step[key] for step in group) / len(group)
                 for key in ("loss", "id", "triplet", "auxiliary_id")}
        assert math.isclose(means["loss"], item["mean_loss"], rel_tol=1e-8, abs_tol=1e-8)
        history.append({"epoch": item["epoch"], "steps": len(group), **means,
                        "official_fused": item["official_fused"]})
    assert len(steps) == sum(row["steps"] for row in training["history"])
    return {"dataset": training["dataset"], "condition": training["initializer"]["condition"],
            "logged_steps": len(steps), "maximum_loss_reconstruction_error": max(errors),
            "nonzero_triplet_steps": sum(step["triplet"] > 0 for step in steps),
            "nonzero_auxiliary_id_steps": sum(step["auxiliary_id"] > 0 for step in steps),
            "first": history[0], "best": history[training["best_epoch"] - 1], "last": history[-1],
            "history": history,
            "boundary": "Scalar support and exact task accounting; no gradient/update-share or causal claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--auxiliary-target", choices=("none", "local", "global"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = {"status": "FULL50_CONTEXT_TASK_SCALARS_VERIFIED",
              "at": datetime.now().astimezone().isoformat(), **audit(args.run, args.auxiliary_target)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}))
