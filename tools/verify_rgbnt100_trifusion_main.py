#!/usr/bin/env python3
"""Verify every full50 role update and the entire retained checkpoint on CPU."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_rgbnt100_trifusion_m0 import weighted_loss_f32
from verify_rgbnt100_trifusion_m0_files import sha, state_sha


def check_training(entry, directory, records, config, mode):
    initial, training = entry["initialization"], entry["training"]
    assert training == json.loads((directory / "training.json").read_text())
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    assert steps == training["steps"] and len(steps) == training["optimizer_steps"]
    assert training["mode"] == mode
    assert initial["initial_state_sha256"] == training["initial_state_sha256"] != training["final_state_sha256"]
    assert training["frozen_state_before_sha256"] == training["frozen_state_after_sha256"]
    assert initial["signal_state_sha256"] == training["signal_state_before_sha256"] == training["signal_state_after_sha256"]
    assert training["trainable_tensors"] == training["nonzero_gradient_tensors"] == len(initial["trainable_names"]) == 203
    assert not training["missing_nonzero_gradients"] and training["overflow_events"] == 0
    assert training["peak_reserved_mib"] < 24 * 1024
    epochs = 20 if mode == "comparison" else 1
    assert training["epochs"] == len(training["history"]) == epochs
    if mode != "comparison":
        assert len(steps) == (8 if mode == "capacity" else 100)
    by_epoch, seen = defaultdict(list), set()
    positive, cross = 0, 0
    for number, step in enumerate(steps, 1):
        assert step["step"] == number and 1 <= step["epoch"] <= epochs
        assert step["optimizer_update_applied"] and step["amp_scale_after"] >= step["amp_scale_before"]
        assert set(step["gradient_finite"]) == set(initial["trainable_names"])
        assert all(step["gradient_finite"].values())
        indices = step["sampled_record_indices"]
        assert len(indices) == 64 and all(0 <= i < 8675 for i in indices)
        assert sorted(Counter(records[i]["identity"] for i in indices).values()) == [8] * 8
        if mode == "overfit":
            assert indices == steps[0]["sampled_record_indices"]
        assert all(math.isfinite(v) for v in step["components"].values()) and math.isfinite(step["loss"])
        assert weighted_loss_f32(step["components"], config["LOSS"]) == step["loss"]
        by_epoch[step["epoch"]].append(step["loss"])
        seen.update(indices)
        cameras = defaultdict(Counter)
        for i in indices:
            cameras[records[i]["identity"]][records[i]["camera"]] += 1
        positive += 224
        cross += sum(28 - sum(n * (n - 1) // 2 for n in counts.values()) for counts in cameras.values())
    assert [step["epoch"] for step in steps] == [h["epoch"] for h in training["history"] for _ in range(h["optimizer_steps"])]
    for number, epoch in enumerate(training["history"], 1):
        assert epoch["epoch"] == number and epoch["optimizer_steps"] == len(by_epoch[number]) > 0
        factor = 1.0
        if mode == "comparison":
            factor = number / 5 if number <= 5 else 0.5 * (1 + math.cos(math.pi * (number - 6) / 15))
        assert abs(epoch["learning_rate"] - config["OPTIMIZATION"]["NEW_MODULE_LR"] * factor) < 1e-15
        assert abs(epoch["mean_loss"] - statistics.fmean(by_epoch[number])) < 1e-10
    if mode == "comparison":
        assert seen == set(range(8675))
    return {"mode": mode, "epochs": epochs, "optimizer_steps": len(steps),
            "exact_fp32_loss_recompositions": len(steps), "unique_source_records_seen": len(seen),
            "source_identities_seen": len({records[i]["identity"] for i in seen}),
            "same_identity_positive_pairs": positive, "cross_camera_positive_pairs": cross}


def main(args):
    import torch

    started = time.perf_counter()
    assert not args.output.exists()
    config = json.loads(args.config.read_text())
    summary = json.loads(args.summary.read_text())
    assert config["schema"] == summary["schema"] == "rgbnt100-original-roles-full50-main-v1"
    assert summary["config_sha256"] == sha(args.config)
    assert summary["runner_sha256"] == sha(ROOT / "tools/train_rgbnt100_trifusion_main.py")
    assert summary["project_source_file_sha256"] == config["project_source_file_sha256"]
    for name, expected in config["project_source_file_sha256"].items():
        assert sha(ROOT / name) == expected, name
    base = config["BASELINE"]
    for key in ("CONFIG", "SUMMARY", "VERIFICATION"):
        assert sha(ROOT / base[key]) == base[key + "_SHA256"], key
    baseline = json.loads((ROOT / base["SUMMARY"]).read_text())
    assert baseline["status"] == "COMPLETE_FULL50_SIGNAL_FIXED_EPOCH30"
    assert summary["baseline_summary_sha256"] == base["SUMMARY_SHA256"]
    verified = json.loads((ROOT / base["VERIFICATION"]).read_text())
    assert verified["status"] == "PASS_FULL50_SIGNAL_FILES_ALL_UPDATES_AND_AUTHOR_LR"
    assert verified["mode"] == "baseline" and verified["summary_sha256"] == base["SUMMARY_SHA256"]
    protocol_path = ROOT / config["DATA"]["PROTOCOL"]
    assert sha(protocol_path) == config["DATA"]["PROTOCOL_SHA256"] == summary["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    records = protocol["records"]["train"]
    assert len(records) == summary["source_records"] == 8675
    assert summary["source_identities"] == 50 and summary["seed"] == 42
    assert summary["official_model_record_forwards"] == summary["rgbnt201_dev_record_forwards"] == 0
    assert not summary["m0_weights_reused"] and not summary["oof_role_weights_reused"]
    initial = summary["initialization"]
    assert initial["source_ids"] == protocol["identities"]["train"] == list(range(501, 600, 2))
    assert initial["heldout_ids"] == protocol["identities"]["query"] == list(range(502, 601, 2))
    assert initial["role_initialization_seed"] == 42 and not initial["role_weights_loaded"]
    m0 = summary["mode"] == "m0"
    assert summary["mode"] in ("m0", "main")
    assert all(summary["engineering_checks"].values())
    sections = [check_training(summary, args.summary.parent, records, config, "capacity" if m0 else "comparison")]
    passed, ratio = True, None
    if m0:
        overfit = summary["overfit"]
        assert overfit["initialization"] == initial
        sections.append(check_training(overfit, args.summary.parent / "overfit", records, config, "overfit"))
        smoothing = config["LOSS"]["LABEL_SMOOTHING"]
        correct, other = 1 - smoothing + smoothing / 50, smoothing / 50
        entropy = -correct * math.log(correct) - 49 * other * math.log(other)
        floor = (config["LOSS"]["ID_FUSED"] + 3 * config["LOSS"]["ID_BRANCH"] + 3 * config["LOSS"]["ID_RESIDUAL"]) * entropy
        losses = [s["loss"] for s in overfit["training"]["steps"]]
        ratio = (losses[-1] - floor) / (losses[0] - floor)
        assert floor == overfit["loss_gate"]["minimum_loss"] and ratio == overfit["loss_gate"]["loss_ratio"]
        passed = ratio <= 0.1
        assert overfit["loss_gate"]["passed"] == overfit["checks"]["overfit_excess_ratio_at_most_point1"] == passed
        assert all(v for k, v in overfit["checks"].items() if k != "overfit_excess_ratio_at_most_point1")
        assert summary["status"] == ("PASS_FULL50_ROLES_ENGINEERING" if passed else "FAIL_FULL50_ROLES_ENGINEERING")
        assert summary["optimizer_steps"] == 108
    else:
        assert summary["status"] == "COMPLETE_FULL50_ROLES_FIXED_EPOCH20"
        preflight = json.loads(args.m0_receipt.read_text())
        assert preflight["status"] == "PASS_FULL50_ROLES_ENGINEERING" and preflight["initialization"] == initial
        assert sha(args.m0_receipt) == summary["m0_receipt_sha256"] and preflight["config_sha256"] == summary["config_sha256"]
    assert summary["optimizer_steps"] == sum(s["optimizer_steps"] for s in sections)
    assert summary["strict_reload_all_outputs_bitwise_equal"] and summary["direct_signal_source_parity"]
    assert summary["clean_role_source_record_forwards"] == (24 if m0 else 16)
    assert summary["direct_signal_source_record_forwards"] == (16 if m0 else 8)
    checkpoint = Path(summary["checkpoint"])
    assert sha(checkpoint) == summary["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["source_ids"] == initial["source_ids"] and payload["heldout_ids"] == initial["heldout_ids"]
    assert payload["fold"] == "full_train" and payload["config_sha256"] == summary["config_sha256"]
    state = payload["model_state_dict"]
    assert state_sha(state) == summary["training"]["final_state_sha256"] == summary["strict_reload_state_sha256"]
    assert all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point())
    assert sum(state[n].numel() for n in initial["trainable_names"]) == initial["trainable_parameters"]
    shapes = {"fused": list(state["fused_classifier.weight"].shape)}
    for expert in ("cnn", "transformer", "mamba"):
        shapes["branch_" + expert] = list(state["branch_classifiers." + expert + ".weight"].shape)
        shapes["residual_" + expert] = list(state["residual_classifiers." + expert + ".weight"].shape)
    assert shapes == summary["classifier_weight_shapes"] and all(v[0] == 50 for v in shapes.values())
    assert sha(baseline["checkpoint"]) == initial["signal_checkpoint_sha256"] == baseline["checkpoint_sha256"]
    baseline_state = torch.load(baseline["checkpoint"], map_location="cpu", weights_only=True)["model_state_dict"]
    saved_signal = {n.removeprefix("baseline.signal."): t for n, t in state.items() if n.startswith("baseline.signal.")}
    assert set(saved_signal) == set(baseline_state)
    assert all(torch.equal(t, baseline_state[n]) for n, t in saved_signal.items())
    assert state_sha(baseline_state) == baseline["training"]["final_state_sha256"] == initial["signal_state_sha256"]
    files = [p for p in sorted(args.summary.parent.rglob("*")) if p.is_file()]
    result = {"status": "PASS_FULL50_ROLES_FILES_AND_ALL_UPDATES", "mode": summary["mode"],
              "engineering_passed": passed, "verified_at": datetime.now().astimezone().isoformat(),
              "summary_sha256": sha(args.summary), "verifier_sha256": sha(__file__), "sections": sections,
              "overfit_excess_ratio": ratio, "signal_saved_state_bitwise_equal_b0": True,
              "checkpoint_state_sha256": summary["strict_reload_state_sha256"],
              "files": {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files},
              "scope": "Complete saved checkpoint and all scalar steps on remote CPU; original source-forward parity is a runtime receipt",
              "official_model_record_forwards": 0, "model_forwards": 0, "optimizer_updates": 0,
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--m0-receipt", type=Path)
    main(parser.parse_args())
