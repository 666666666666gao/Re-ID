#!/usr/bin/env python3
"""Verify a complete full50 Signal endpoint, every update, and its saved state."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_rgbnt100_signal_terminal_files import sha, state_sha
from verify_rgbnt100_signal_terminal_scalars import f32


def main(args):
    import torch

    started = time.perf_counter()
    assert not args.output.exists()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol_path = ROOT / config["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    m0 = summary["mode"] == "m0"
    assert summary["mode"] in ("m0", "baseline")
    assert summary["status"] == ("PASS_FULL50_SIGNAL_ENGINEERING" if m0 else "COMPLETE_FULL50_SIGNAL_FIXED_EPOCH30")
    assert summary["config_sha256"] == sha(args.config)
    assert summary["protocol_sha256"] == sha(protocol_path) == config["protocol_sha256"]
    for path, expected in config["source_files_sha256"].items():
        assert sha(ROOT / path) == expected, path
    assert summary["source_files_sha256"] == config["source_files_sha256"]
    assert summary["runner_sha256"] == sha(ROOT / "tools/train_rgbnt100_signal_main.py")
    assert summary["source_ids"] == protocol["identities"]["train"] == list(range(501, 600, 2))
    assert summary["heldout_ids"] == protocol["identities"]["query"] == list(range(502, 601, 2))
    assert summary["model_num_classes"] == summary["source_identities"] == 50
    assert summary["source_records"] == 8675 and summary["seed"] == 42 and summary["fold"] == "full_train"
    assert summary["official_model_record_forwards"] == summary["rgbnt201_dev_record_forwards"] == 0
    assert not summary["official_metrics_read"] and not summary["m0_trained_weights_loaded"]
    assert all(summary["checks"].values()) and summary["strict_reload_source_features_bitwise_equal"]
    baseline_path = ROOT / config["baseline_config"]
    assert sha(baseline_path) == config["baseline_config_sha256"] == summary["baseline_config_sha256"]
    baseline = json.loads(baseline_path.read_text())
    signal = Path(baseline["signal_source"])
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=signal, text=True).strip() == baseline["signal_commit"]
    for name, expected in baseline["signal_source_file_sha256"].items():
        assert sha(signal / name) == expected, name
    assert sha(Path(baseline["clip_weight"])) == baseline["clip_weight_sha256"]
    training_path = args.summary.parent / "training.json"
    step_path = args.summary.parent / "steps.jsonl"
    training = summary["training"]
    assert training == json.loads(training_path.read_text())
    assert training["steps"] == [json.loads(line) for line in step_path.read_text().splitlines()]
    expected_epochs = 1 if m0 else 30
    assert training["epochs"] == len(training["history"]) == expected_epochs
    assert training["optimizer_steps"] == len(training["steps"])
    assert training["trainable_tensors"] == training["gradient_tensors"] == len(training["optimizer_groups"]) == 195
    assert not training["trainable_without_gradient"] and training["overflow_events"] == 0
    assert training["frozen_token_selection_parameters"] == 787968
    lr_path = ROOT / config["author_lr_reference"]
    assert sha(lr_path) == config["author_lr_reference_sha256"]
    lr_reference = json.loads(lr_path.read_text())
    assert lr_reference["status"] == "PASS_COMPLETE_BASELINE_FILES_ARRAYS_RANKS_AND_AUTHOR_LR"
    records = protocol["records"]["train"]
    seen, losses = set(), defaultdict(list)
    same, cross = 0, 0
    for number, step in enumerate(training["steps"], 1):
        assert step["step"] == number and 1 <= step["epoch"] <= expected_epochs
        assert step["optimizer_update_applied"] and step["amp_scale_after"] >= step["amp_scale_before"]
        indices = step["sampled_record_indices"]
        assert len(indices) == 64 and all(0 <= i < 8675 for i in indices)
        assert sorted(Counter(records[i]["identity"] for i in indices).values()) == [8] * 8
        assert len(step["id_triplet_head_losses"]) == 4
        exact = 0.0
        for component in step["id_triplet_head_losses"]:
            exact = f32(exact + component)
        exact = f32(exact + f32(f32(.1) * step["gram_loss"]))
        exact = f32(exact + f32(f32(.1) * step["patch_loss"]))
        assert math.isfinite(exact) and exact == step["loss"]
        losses[step["epoch"]].append(step["loss"])
        seen.update(indices)
        cameras = defaultdict(Counter)
        for index in indices:
            row = records[index]
            cameras[row["identity"]][row["camera"]] += 1
        same += 224
        cross += sum(28 - sum(n * (n - 1) // 2 for n in counts.values()) for counts in cameras.values())
    for number, epoch in enumerate(training["history"], 1):
        assert epoch["epoch"] == number and epoch["optimizer_steps"] == len(losses[number])
        assert abs(statistics.mean(losses[number]) - epoch["mean_loss"]) < 1e-10
        assert epoch["learning_rates"] == lr_reference["author_lr_schedule"][number - 1]["learning_rates"]
    assert {records[i]["identity"] for i in seen} == set(summary["source_ids"])
    assert summary["unique_training_records_seen"] == len(seen)
    assert summary["source_record_exposures"] == len(training["steps"]) * 64
    if not m0:
        assert seen == set(range(8675))
        preflight = json.loads(args.m0_receipt.read_text())
        assert sha(args.m0_receipt) == summary["m0_receipt_sha256"]
        assert preflight["status"] == "PASS_FULL50_SIGNAL_ENGINEERING"
        assert preflight["training"]["initial_state_sha256"] == training["initial_state_sha256"]
        assert preflight["training"]["trainable_parameters"] == training["trainable_parameters"]
    checkpoint = Path(summary["checkpoint"])
    assert sha(checkpoint) == summary["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["source_ids"] == summary["source_ids"] and payload["heldout_ids"] == summary["heldout_ids"]
    assert payload["config_sha256"] == summary["config_sha256"] and payload["fold"] == "full_train"
    state = payload["model_state_dict"]
    assert state_sha(state) == training["final_state_sha256"] == summary["strict_reload_state_sha256"]
    assert training["initial_state_sha256"] != training["final_state_sha256"]
    assert all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point())
    selector = {name.removeprefix("SIM.token_selection."): value for name, value in state.items()
                if name.startswith("SIM.token_selection.")}
    assert state_sha(selector) == training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"]
    inputs = [args.summary, args.config, protocol_path, baseline_path, training_path, step_path, checkpoint, lr_path]
    result = {"verified_at": datetime.now().astimezone().isoformat(),
              "status": "PASS_FULL50_SIGNAL_FILES_ALL_UPDATES_AND_AUTHOR_LR", "mode": summary["mode"],
              "summary_sha256": sha(args.summary), "verifier_sha256": sha(Path(__file__)),
              "input_files": {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in inputs},
              "checkpoint_state_sha256": training["final_state_sha256"],
              "optimizer_steps": len(training["steps"]), "epochs": expected_epochs,
              "exact_fp32_loss_recompositions": len(training["steps"]),
              "unique_source_records_seen": len(seen), "source_identities_seen": 50,
              "same_identity_positive_pairs": same, "cross_camera_positive_pairs": cross,
              "model_forwards": 0, "optimizer_updates": 0,
              "scope": "Remote CPU saved-weight and complete scalar verification; no model construction or image access",
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--m0-receipt", type=Path)
    main(parser.parse_args())
