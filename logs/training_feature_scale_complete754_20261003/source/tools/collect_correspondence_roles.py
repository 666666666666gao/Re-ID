#!/usr/bin/env python3
"""Collect the registered 24 endpoints without selecting unfinished results."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, WEIGHTS

GPUS = {"000": 0, "001": 3, "010": 2, "011": 2,
        "100": 1, "101": 3, "110": 1}
METRICS = ("mAP", "Rank-1", "Rank-5", "Rank-10")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(run, m0_run, dataset, variant, protocol_sha, baseline_sha):
    import torch

    training, official, m0 = (read(path) for path in
                             (run / "training.json", run / "official_metrics.json",
                              m0_run / "training.json"))
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert official["status"] == "COMPLETE" and m0["status"] == "M0_PASS"
    flags = {f"m{index}": bit == "1" for index, bit in enumerate(variant, 1)}
    for receipt in (training, m0):
        assert receipt["dataset"] == dataset and receipt["seed"] == 42
        assert receipt["epochs"] == 50 and receipt["checkpoint_policy"] == "best_official_map"
        assert receipt["protocol_sha256"] == protocol_sha
        assert receipt["initializer"]["author_checkpoint_sha256"] == baseline_sha
        assert all(receipt["initializer"][name] == value for name, value in flags.items())
    assert m0["initializer"] == training["initializer"]
    assert m0["m0"]["frozen_signal_unchanged"]
    assert m0["m0"]["nonzero_gradient_parameters"] == m0["m0"]["trainable_parameters"]
    assert m0["m0"]["reload_max_abs_difference"] <= 1e-5
    history = training["history"]
    assert [row["epoch"] for row in history] == list(range(1, 51))
    best = max(history, key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    checkpoint = run / "best_map.pth"
    assert str(checkpoint) == training["checkpoint"]
    assert sha(checkpoint) == official["checkpoint_sha256"]
    assert sha(run / "official_distances.pt") == official["distance_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-correspondence-roles-v1"
    assert payload["dataset"] == official["dataset"] == dataset
    assert payload["seed"] == official["seed"] == 42
    assert payload["variants"] == flags
    assert payload["protocol_sha256"] == official["protocol_sha256"] == protocol_sha
    assert payload["baseline_sha256"] == official["baseline_sha256"] == baseline_sha
    assert training["best_epoch"] == official["selected_epoch"] == payload["epoch"] == best["epoch"]
    assert official["training_epochs"] == 50 and not official["reranking"]
    assert official["independent_upstream_metrics_equal"]
    for name in METRICS:
        assert abs(payload["metrics"][name] - official["metrics"][name]) < 1e-5
        assert abs(best["official_fused"][name] - official["metrics"][name]) < 1e-5
    return {"status": "VERIFIED_COMPLETE", "best_epoch": best["epoch"],
            "metrics": official["metrics"],
            "trainable_parameters": training["initializer"]["trainable_parameters"],
            "training_and_epoch_eval_seconds":
                (datetime.fromisoformat(training["completed_at"]) -
                 datetime.fromisoformat(training["started_at"])).total_seconds(),
            "checkpoint_sha256": official["checkpoint_sha256"],
            "distance_sha256": official["distance_sha256"],
            "receipt_sha256": sha(run / "official_metrics.json")}


def collect():
    rows = []
    for dataset, (weight_name, baseline_sha) in BASELINES.items():
        assert sha(WEIGHTS / weight_name) == baseline_sha
        protocol_sha = sha(PROTOCOLS / f"{dataset}.json")
        for variant in (*GPUS, "111"):
            if variant == "111":
                prefix = f"correspondence_roles_v1_{dataset}_seed42"
                run = ROOT / f"trained-model/{prefix}_full_20260928"
                m0_run = ROOT / f"trained-model/{prefix}_m0_20260928"
            else:
                prefix = f"correspondence_roles_ablation_gpu{GPUS[variant]}_20260928_{dataset}_m{variant}_seed42"
                run = ROOT / f"trained-model/{prefix}_full"
                m0_run = ROOT / f"trained-model/{prefix}_m0"
            row = {"dataset": dataset, "variant": variant, "run_dir": str(run)}
            if (run / "official_metrics.json").exists():
                row.update(verify(run, m0_run, dataset, variant, protocol_sha, baseline_sha))
            elif (run / "training.json").exists():
                training = read(run / "training.json")
                row.update(status="UNACCEPTED", training_status=training["status"],
                           completed_epochs=len(training["history"]))
            else:
                row["status"] = "PENDING"
            rows.append(row)
    return {"schema": "trifusion-correspondence-roles-matrix-verification-v1",
            "collected_at": datetime.now().astimezone().isoformat(), "seed": 42,
            "expected_endpoints": 24,
            "verified_complete": sum(row["status"] == "VERIFIED_COMPLETE" for row in rows),
            "scope": "artifact binding, full 50-epoch selection and reload receipt; no new inference",
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("collected_at", "expected_endpoints", "verified_complete")}))
