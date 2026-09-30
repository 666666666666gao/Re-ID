"""Audit finished training/checkpoint bytes; this is not a retrieval acceptance."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    training = json.loads((args.run / "training.json").read_text())
    m0 = json.loads((args.m0 / "training.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert m0["status"] == "M0_PASS" and m0["history"][0]["steps"] == 8
    assert training["initializer"] == m0["initializer"]
    assert training["seed"] == 42 and training["epochs"] == 50
    assert training["checkpoint_policy"] == "best_official_map"
    assert [r["epoch"] for r in training["history"]] == list(range(1, 51))
    assert all(sha(args.root / name) == digest for name, digest in manifest["source_sha256"].items())
    steps = [json.loads(line) for line in (args.run / "training_steps.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for step in steps:
        assert np.isfinite([step[k] for k in ("loss", "id", "triplet", "auxiliary_id")]).all()
        assert step["auxiliary_target"] == "none" and step["auxiliary_id"] == 0
        assert abs(step["loss"] - step["id"] - step["triplet"]) < 1e-5
        grouped[step["epoch"]].append(step)
    assert sorted(grouped) == list(range(1, 51))
    for row in training["history"]:
        batch_rows = grouped[row["epoch"]]
        assert [r["batch"] for r in batch_rows] == list(range(row["steps"]))
        assert abs(np.mean([r["loss"] for r in batch_rows]) - row["mean_loss"]) < 1e-9
    best = max(training["history"], key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    checkpoint = args.run / "best_map.pth"
    assert training["checkpoint"] == str(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-patch-memory-roles-v1"
    assert payload["dataset"] == training["dataset"] and payload["seed"] == 42
    assert payload["epoch"] == training["best_epoch"] == best["epoch"]
    assert payload["metrics"] == best["official_fused"]
    assert payload["protocol_sha256"] == training["protocol_sha256"]
    assert payload["baseline_sha256"] == training["initializer"]["author_checkpoint_sha256"]
    assert payload["memory_mode"] == training["initializer"]["memory_mode"]
    assert payload["variants"] == {"m1": True, "m2": True, "m3": False}
    assert payload["condition"] == {"query_mode": "context", "auxiliary_target": "none"}
    assert payload["state"] and all(torch.isfinite(tensor).all() for tensor in payload["state"].values())
    report = {"schema": "trifusion-patch-memory-training-only-check-v1",
              "status": "TRAINING_CHECKPOINT_BYTES_PASS_PENDING_STRICT_RETRIEVAL",
              "checked_at": datetime.now().astimezone().isoformat(),
              "dataset": training["dataset"], "memory_mode": payload["memory_mode"],
              "completed_epochs": 50, "logged_steps": len(steps),
              "selected_epoch": best["epoch"], "finite_saved_state_tensors": len(payload["state"]),
              "checkpoint_sha256": sha(checkpoint), "training_receipt_sha256": sha(args.run / "training.json"),
              "training_steps_sha256": sha(args.run / "training_steps.jsonl"),
              "source_files_checked": len(manifest["source_sha256"]),
              "boundary": "No fresh neural forward, strict model reload or full-gallery replay performed; official endpoint remains unaccepted."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
