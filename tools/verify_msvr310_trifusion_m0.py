#!/usr/bin/env python3
"""Recompute MSVR310 three-role M0 scalar and source-index accounting only."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(summary, config, protocol):
    assert summary["mode"] == "m0" and len(summary["folds"]) == 3
    assert summary["heldout_record_forwards"] == 0
    assert summary["official_test_image_access"] == summary["rgbnt201_dev_image_access"] == 0
    assert not summary["m0_weights_reused"] and not summary["rgbnt201_role_weights_reused"]
    weights = config["LOSS"]
    maximum_loss_difference = 0.0
    maximum_epoch_mean_difference = 0.0
    sections = []
    entries = [(row, protocol["folds"][i], 8) for i, row in enumerate(summary["folds"])]
    entries.append((summary["overfit"], protocol["folds"][0], 100))
    for entry, fold, expected_updates in entries:
        training = entry["training"]
        steps = training["steps"]
        assert training["epochs"] == 1 and len(steps) == training["optimizer_steps"] == expected_updates
        assert len(training["history"]) == 1
        source_indices = set(fold["source_record_indices"])
        exposed = []
        cross_scene_pairs = 0
        same_identity_pairs = 0
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
                        cross_scene_pairs += int(a["scene"] != b["scene"])
            exposed.extend(indices)
            assert step["amp_scale_after"] >= step["amp_scale_before"]
            pieces = step["components"]
            total = weights["ID_FUSED"] * pieces["id_fused"] + weights["TRIPLET_FUSED"] * pieces["triplet_fused"]
            for expert in ("cnn", "transformer", "mamba"):
                total += weights["ID_BRANCH"] * pieces["id_" + expert]
                total += weights["TRIPLET_BRANCH"] * pieces["triplet_" + expert]
                total += weights["ID_RESIDUAL"] * pieces["id_residual_" + expert]
                total += weights["TRIPLET_RESIDUAL"] * pieces["triplet_residual_" + expert]
            assert math.isfinite(step["loss"]) and all(math.isfinite(x) for x in pieces.values())
            maximum_loss_difference = max(maximum_loss_difference, abs(total - step["loss"]))
        mean = statistics.fmean(s["loss"] for s in steps)
        maximum_epoch_mean_difference = max(maximum_epoch_mean_difference,
                                            abs(mean - training["history"][0]["mean_loss"]))
        assert training["history"][0]["optimizer_steps"] == expected_updates
        assert training["history"][0]["learning_rate"] == config["OPTIMIZATION"]["NEW_MODULE_LR"]
        initial = entry["initialization"]
        assert initial["initial_state_sha256"] == training["initial_state_sha256"]
        assert initial["signal_state_sha256"] == training["signal_state_before_sha256"]
        assert initial["source_ids"] == fold["source_ids"] and initial["heldout_ids"] == fold["heldout_ids"]
        assert not initial["role_weights_loaded"] and initial["role_initialization_seed"] == 42
        assert len(initial["trainable_names"]) == training["trainable_tensors"]
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
            assert entry["strict_reload_state_sha256"] == training["final_state_sha256"]
            assert entry["direct_signal_source_parity"] and entry["strict_reload_all_outputs_bitwise_equal"]
            assert entry["clean_role_source_record_forwards"] == 24
            assert entry["direct_signal_source_record_forwards"] == 8
            assert entry["heldout_record_forwards"] == 0
        else:
            assert all(s["sampled_record_indices"] == steps[0]["sampled_record_indices"] for s in steps)
        sections.append({"fold": fold["fold"], "mode": training["mode"],
                         "optimizer_steps": expected_updates, "source_record_exposures": len(exposed),
                         "unique_source_records": len(set(exposed)),
                         "unique_source_identities": len({protocol["records"][i]["identity"] for i in exposed}),
                         "same_identity_unordered_pairs": same_identity_pairs,
                         "cross_scene_positive_pairs": cross_scene_pairs,
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
    passed = all(v for row in sections for v in row["checks_recomputed"].values()) and ratio <= 0.1
    assert (summary["status"] == "PASS_ENGINEERING_ONLY") == passed
    assert summary["optimizer_steps"] == 124
    return {"status": "PASS_SCALAR_ACCOUNTING", "engineering_gate_passed": passed,
            "optimizer_steps": 124, "training_record_exposures": 7936,
            "clean_role_source_record_forwards": 72, "direct_signal_source_record_forwards": 24,
            "heldout_record_forwards": 0, "sections": sections,
            "entropy_floor": floor, "overfit_excess_ratio": ratio,
            "maximum_weighted_loss_absolute_difference": maximum_loss_difference,
            "maximum_epoch_mean_absolute_difference": maximum_epoch_mean_difference,
            "loss_difference_note": "Saved scalar components reweighted in Python double; AMP intermediate dtype not saved; no new tolerance gate imposed",
            "scope": "stdlib JSON, source labels and arithmetic; binary/model/gradient facts remain remote receipts"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.perf_counter()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assert summary["config_sha256"] == sha(args.config)
    assert summary["protocol_sha256"] == sha(args.protocol)
    report = verify(summary, config, protocol)
    report.update({"inputs": {name: {"path": str(getattr(args, name)), "sha256": sha(getattr(args, name))}
                               for name in ("summary", "config", "protocol")},
                   "elapsed_seconds": time.perf_counter() - started,
                   "verifier_sha256": sha(__file__)})
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sections", "inputs")}, ensure_ascii=False))
