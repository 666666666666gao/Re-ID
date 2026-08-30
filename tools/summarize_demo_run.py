#!/usr/bin/env python3
"""Parse an official DeMo training log into an auditable JSON run summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


EPOCH_DONE = re.compile(
    r"Epoch (?P<epoch>\d+) done\. Time per batch: (?P<seconds>[0-9.]+)\[s\] "
    r"Speed: (?P<speed>[0-9.]+)\[samples/s\]"
)
EVAL_EPOCH = re.compile(r"Validation Results - Epoch: (?P<epoch>\d+)")
MAP_LINE = re.compile(r"mAP: (?P<value>[0-9.]+)%")
RANK_LINE = re.compile(r"CMC curve, Rank-(?P<rank>\d+)\s*:(?P<value>[0-9.]+)%")
LOG_TIME = re.compile(r"^(?P<value>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError:",
)
FEATURE_MODES = ("ori", "moe", "joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_time(line: str) -> datetime | None:
    match = LOG_TIME.match(line)
    return (
        datetime.strptime(match.group("value"), "%Y-%m-%d %H:%M:%S,%f")
        if match
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-seconds", type=float, default=300.0)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "train_log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    text = log_path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()

    train_epochs = []
    evaluations = []
    mode = None
    mode_started_at = None
    pending = None
    for line in lines:
        match = EPOCH_DONE.search(line)
        if match:
            train_epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "seconds_per_batch": float(match.group("seconds")),
                    "samples_per_second": float(match.group("speed")),
                }
            )
        if "Current is the ori feature testing!" in line:
            mode = "ori"
            mode_started_at = log_time(line)
        elif "Current is the moe feature testing!" in line:
            mode = "moe"
            mode_started_at = log_time(line)
        elif "Current is the [moe,ori] feature testing!" in line:
            mode = "joint"
            mode_started_at = log_time(line)

        match = EVAL_EPOCH.search(line)
        if match:
            pending = {
                "epoch": int(match.group("epoch")),
                "mode": mode,
                "started_at": mode_started_at.isoformat()
                if mode_started_at
                else None,
            }
            continue
        if pending is None:
            continue
        match = MAP_LINE.search(line)
        if match:
            pending["mAP_percent"] = float(match.group("value"))
            continue
        match = RANK_LINE.search(line)
        if match:
            pending[f"rank{match.group('rank')}_percent"] = float(
                match.group("value")
            )
            if match.group("rank") == "10":
                completed_at = log_time(line)
                pending["completed_at"] = (
                    completed_at.isoformat() if completed_at else None
                )
                pending["elapsed_seconds"] = (
                    (completed_at - mode_started_at).total_seconds()
                    if completed_at and mode_started_at
                    else None
                )
                evaluations.append(pending)
                pending = None

    by_mode = {
        name: [record for record in evaluations if record["mode"] == name]
        for name in FEATURE_MODES
    }
    best_by_mode = {
        name: max(records, key=lambda record: record["mAP_percent"])
        if records
        else None
        for name, records in by_mode.items()
    }
    fatal_hits = [pattern for pattern in FATAL_PATTERNS if pattern in text]
    nonfinite_hits = [
        line for line in lines if re.search(r"\b(?:loss|mAP).*\b(?:nan|inf)\b", line, re.I)
    ]
    expected_epoch_set = set(range(1, args.expected_epochs + 1))
    trained_epoch_set = {record["epoch"] for record in train_epochs}
    latest_epoch = max(trained_epoch_set) if trained_epoch_set else 0
    expected_live_epoch_set = set(range(1, latest_epoch + 1))
    expected_evaluation_keys = {
        (epoch, mode)
        for epoch in expected_live_epoch_set
        for mode in FEATURE_MODES
    }
    observed_evaluation_keys = [
        (record["epoch"], record["mode"]) for record in evaluations
    ]
    complete = trained_epoch_set == expected_epoch_set and all(
        {record["epoch"] for record in records} == expected_epoch_set
        for records in by_mode.values()
    )
    checkpoints = []
    for checkpoint in sorted(run_dir.glob("*.pth")):
        item = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size}
        if args.hash_checkpoints:
            item["sha256"] = sha256(checkpoint)
        checkpoints.append(item)

    checks = {
        "no_fatal_log_pattern": not fatal_hits,
        "no_nonfinite_metric_or_loss": not nonfinite_hits,
        "has_completed_training_epoch": bool(trained_epoch_set),
        "epochs_are_unique": len(train_epochs) == len(trained_epoch_set),
        "trained_epochs_form_expected_prefix": (
            trained_epoch_set == expected_live_epoch_set
            and trained_epoch_set <= expected_epoch_set
        ),
        "evaluation_records_cover_trained_epochs_exactly": (
            len(observed_evaluation_keys) == len(expected_evaluation_keys)
            and set(observed_evaluation_keys) == expected_evaluation_keys
        ),
        "evaluation_records_complete": all(
            all(
                key in record
                for key in (
                    "mAP_percent",
                    "rank1_percent",
                    "rank5_percent",
                    "rank10_percent",
                )
            )
            for record in evaluations
        ),
        "evaluation_latency_within_limit": all(
            record["elapsed_seconds"] is not None
            and record["elapsed_seconds"] <= args.max_eval_seconds
            for record in evaluations
        ),
    }
    if args.require_complete:
        checks["run_complete"] = complete
    report = {
        "valid": all(checks.values()),
        "complete": complete,
        "checks": checks,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "protocol": {
            "dataset": "RGBNT201/train_171 -> test query/gallery",
            "baseline": "DeMo",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_instances": args.num_instances,
            "expected_epochs": args.expected_epochs,
            "max_eval_seconds": args.max_eval_seconds,
            "feature_modes": list(FEATURE_MODES),
        },
        "epochs_completed": len(trained_epoch_set),
        "latest_epoch": latest_epoch,
        "train_epochs": train_epochs,
        "evaluations": evaluations,
        "best_by_mode": best_by_mode,
        "fatal_hits": fatal_hits,
        "nonfinite_hits": nonfinite_hits,
        "checkpoints": checkpoints,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    output_path = args.json_out or (run_dir / "summary.json")
    output_path.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
