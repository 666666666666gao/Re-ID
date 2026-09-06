#!/usr/bin/env python3
"""Replay the fixed RGBNT100 original-role M0 using stored JSON only."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def weighted_loss_f32(pieces, weights):
    experts = ("cnn", "transformer", "mamba")
    sums = {}
    for prefix in ("id_", "triplet_", "id_residual_", "triplet_residual_"):
        value = 0.0
        for expert in experts:
            value = f32(value + pieces[prefix + expert])
        sums[prefix] = value
    value = f32(f32(pieces["id_fused"] * f32(weights["ID_FUSED"]))
                + f32(pieces["triplet_fused"] * f32(weights["TRIPLET_FUSED"])))
    for prefix, weight in (("id_", "ID_BRANCH"), ("triplet_", "TRIPLET_BRANCH"),
                           ("id_residual_", "ID_RESIDUAL"), ("triplet_residual_", "TRIPLET_RESIDUAL")):
        value = f32(value + f32(sums[prefix] * f32(weights[weight])))
    return value


def verify(summary, config, protocol, receipts_dir):
    assert summary["mode"] == "m0" and len(summary["folds"]) == 3
    assert len(protocol["records"]) == 8675
    assert {r["identity"] for r in protocol["records"]} == set(range(501, 600, 2))
    assert summary["heldout_record_forwards"] == 0
    assert summary["official_test_image_access"] == summary["rgbnt201_dev_image_access"] == 0
    assert summary["source_only_training"] and summary["seed"] == 42
    assert not summary["m0_weights_reused"] and not summary["rgbnt201_role_weights_reused"]
    weights = config["LOSS"]
    maximum_epoch_mean_difference = 0.0
    exact_losses, sections, expected_events = 0, [], []
    entries = [(row, protocol["folds"][i], 8, f"fold_{i}") for i, row in enumerate(summary["folds"])]
    entries.append((summary["overfit"], protocol["folds"][0], 100, "overfit_fold0"))
    for entry, fold, expected_updates, name in entries:
        training, initial = entry["training"], entry["initialization"]
        directory = receipts_dir / name
        assert training == json.loads((directory / "training.json").read_text(encoding="utf-8"))
        steps = training["steps"]
        assert steps == [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        assert training["epochs"] == 1 and len(steps) == training["optimizer_steps"] == expected_updates
        assert len(training["history"]) == 1
        assert not set(fold["source_ids"]) & set(fold["heldout_ids"])
        source_indices = set(fold["source_record_indices"])
        exposed, cross_camera_pairs, same_identity_pairs = [], 0, 0
        for number, step in enumerate(steps, 1):
            assert step["step"] == number and step["epoch"] == 1
            indices = step["sampled_record_indices"]
            assert len(indices) == 64 and set(indices) <= source_indices
            rows = [protocol["records"][i] for i in indices]
            assert sorted(Counter(r["identity"] for r in rows).values()) == [8] * 8
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    if a["identity"] == b["identity"]:
                        same_identity_pairs += 1
                        cross_camera_pairs += int(a["camera"] != b["camera"])
            exposed.extend(indices)
            assert step["optimizer_update_applied"] and step["amp_scale_after"] >= step["amp_scale_before"]
            assert set(step["gradient_finite"]) == set(initial["trainable_names"])
            assert len(step["gradient_finite"]) == 203 and all(step["gradient_finite"].values())
            pieces = step["components"]
            assert math.isfinite(step["loss"]) and all(math.isfinite(x) for x in pieces.values())
            assert weighted_loss_f32(pieces, weights) == step["loss"]
            exact_losses += 1
        mean = statistics.fmean(s["loss"] for s in steps)
        maximum_epoch_mean_difference = max(maximum_epoch_mean_difference,
                                            abs(mean - training["history"][0]["mean_loss"]))
        assert training["history"][0]["optimizer_steps"] == expected_updates
        assert training["history"][0]["learning_rate"] == config["OPTIMIZATION"]["NEW_MODULE_LR"]
        expected_events.append({"event": "rgbnt100_role_epoch", "fold": fold["fold"],
                                "mode": training["mode"], **training["history"][0]})
        assert initial["initial_state_sha256"] == training["initial_state_sha256"]
        assert initial["signal_state_sha256"] == training["signal_state_before_sha256"]
        assert initial["source_ids"] == fold["source_ids"] and initial["heldout_ids"] == fold["heldout_ids"]
        assert not initial["role_weights_loaded"] and initial["role_initialization_seed"] == 42
        assert len(initial["trainable_names"]) == training["trainable_tensors"] == training["nonzero_gradient_tensors"] == 203
        checks = {
            "all_trainable_gradients_live": not training["missing_nonzero_gradients"],
            "overflow_zero": training["overflow_events"] == 0,
            "frozen_state_unchanged": training["frozen_state_before_sha256"] == training["frozen_state_after_sha256"],
            "signal_state_unchanged": training["signal_state_before_sha256"] == training["signal_state_after_sha256"],
            "role_state_updated": training["initial_state_sha256"] != training["final_state_sha256"],
            "capacity_below_24gib": training["peak_reserved_mib"] < 24 * 1024,
        }
        recorded_checks = entry["engineering_checks"] if expected_updates == 8 else entry["checks"]
        assert all(recorded_checks[k] == v for k, v in checks.items())
        if expected_updates == 8:
            assert entry == json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
            assert recorded_checks["expected_training_length"]
            assert entry["strict_reload_state_sha256"] == training["final_state_sha256"]
            assert entry["direct_signal_source_parity"] and entry["strict_reload_all_outputs_bitwise_equal"]
            assert entry["clean_role_source_record_forwards"] == 24
            assert entry["direct_signal_source_record_forwards"] == 8 and entry["heldout_record_forwards"] == 0
        else:
            assert initial == summary["folds"][0]["initialization"]
            assert all(s["sampled_record_indices"] == steps[0]["sampled_record_indices"] for s in steps)
            assert recorded_checks["fixed_100_updates"]
        sections.append({"fold": fold["fold"], "mode": training["mode"],
                         "optimizer_steps": expected_updates, "source_record_exposures": len(exposed),
                         "unique_source_records": len(set(exposed)),
                         "unique_source_identities": len({protocol["records"][i]["identity"] for i in exposed}),
                         "same_identity_unordered_pairs": same_identity_pairs,
                         "cross_camera_positive_pairs": cross_camera_pairs,
                         "trainable_tensors": training["trainable_tensors"],
                         "nonzero_gradient_tensors": training["nonzero_gradient_tensors"],
                         "checks_recomputed": checks})
    classes = len(protocol["folds"][0]["source_ids"])
    smoothing = weights["LABEL_SMOOTHING"]
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    entropy = -correct * math.log(correct) - (classes - 1) * other * math.log(other)
    floor = (weights["ID_FUSED"] + 3 * weights["ID_BRANCH"] + 3 * weights["ID_RESIDUAL"]) * entropy
    losses = [r["loss"] for r in summary["overfit"]["training"]["steps"]]
    ratio = (losses[-1] - floor) / (losses[0] - floor)
    reported = summary["overfit"]["loss_gate"]
    assert floor == reported["minimum_loss"] and ratio == reported["loss_ratio"]
    assert reported["passed"] == (ratio <= 0.1)
    assert summary["overfit"]["checks"]["overfit_excess_ratio_at_most_point1"] == reported["passed"]
    passed = all(v for row in sections for v in row["checks_recomputed"].values()) and ratio <= 0.1
    assert (summary["status"] == "PASS_ENGINEERING_ONLY") == passed
    assert summary["optimizer_steps"] == exact_losses == 124
    assert maximum_epoch_mean_difference < 1e-10
    return {"status": "PASS_COMPLETE_M0_SCALAR_REPLAY", "engineering_gate_passed": passed,
            "optimizer_steps": 124, "training_record_exposures": 7936,
            "clean_role_source_record_forwards": 72, "direct_signal_source_record_forwards": 24,
            "heldout_record_forwards": 0, "sections": sections,
            "entropy_floor": floor, "overfit_excess_ratio": ratio,
            "exact_fp32_loss_recompositions": exact_losses,
            "maximum_epoch_mean_absolute_difference": maximum_epoch_mean_difference,
            "expected_epoch_log_events": expected_events,
            "scope": "All saved JSONL/training rows, FP32 arithmetic and source labels; binary/model facts remain remote receipts"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "protocol", "receipts-dir", "log", "remote-verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.perf_counter()
    summary, config, protocol, remote = [json.loads(p.read_text(encoding="utf-8"))
        for p in (args.summary, args.config, args.protocol, args.remote_verification)]
    assert summary["config_sha256"] == sha(args.config)
    assert summary["protocol_sha256"] == sha(args.protocol)
    assert remote["summary_sha256"] == sha(args.summary) and remote["status"] == "PASS_COMPLETE_M0_FILES_AND_CHECKPOINTS"
    report = verify(summary, config, protocol, args.receipts_dir)
    events = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()
              if line.startswith('{"event": "rgbnt100_role_epoch"')]
    assert events == report["expected_epoch_log_events"]
    report.update({"inputs": {name: {"path": str(getattr(args, name)), "sha256": sha(getattr(args, name))}
                               for name in ("summary", "config", "protocol", "log", "remote_verification")},
                   "elapsed_seconds": time.perf_counter() - started, "verifier_sha256": sha(__file__)})
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sections", "inputs", "expected_epoch_log_events")}, ensure_ascii=False))
