#!/usr/bin/env python3
"""Train the fixed full50-ID Signal prerequisite without official retrieval."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_rgbnt100_signal_oof import (
    configure, extract, loader_for, new_model, sha256, train_source, write_json,
)


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed

    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert config["schema"] == "rgbnt100-official-fixed-main-v1"
    for name, expected in config["source_files_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    protocol_path = ROOT / config["protocol"]
    assert sha256(protocol_path) == config["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_OFFICIAL_PROTOCOL_NO_MODEL_EVALUATION"
    assert protocol["counts"] == {"train": 8675, "query": 1715, "gallery": 8575}
    baseline_path = ROOT / config["baseline_config"]
    assert sha256(baseline_path) == config["baseline_config_sha256"]
    baseline_config = json.loads(baseline_path.read_text(encoding="utf-8"))
    cfg, binding = configure(baseline_config)
    assert protocol["dataset_root"] == baseline_config["dataset_root"]
    source_ids = protocol["identities"]["train"]
    assert source_ids == list(range(501, 600, 2))
    fold = {"fold": "full_train", "source_ids": source_ids,
            "heldout_ids": protocol["identities"]["query"],
            "source_camera_values": protocol["camera_values"]["train"]}
    source = protocol["records"]["train"]
    records = [(str(Path(protocol["dataset_root"]) / r["path"]),
                protocol["train_label_map"][str(r["identity"])], r["camera"], r["view"])
               for r in source]
    indices = [r["index"] for r in source]
    assert indices == list(range(8675))
    m0 = args.mode == "m0"
    if not m0:
        preflight = json.loads(args.m0_receipt.read_text(encoding="utf-8"))
        assert preflight["status"] == "PASS_FULL50_SIGNAL_ENGINEERING"
        assert preflight["config_sha256"] == sha256(args.config)
        assert preflight["protocol_sha256"] == sha256(protocol_path)
        assert preflight["runner_sha256"] == sha256(Path(__file__))
        assert all(preflight["checks"].values())
        verified = json.loads(args.m0_verification.read_text(encoding="utf-8"))
        assert verified["status"] == "PASS_FULL50_SIGNAL_FILES_ALL_UPDATES_AND_AUTHOR_LR"
        assert verified["mode"] == "m0" and verified["summary_sha256"] == sha256(args.m0_receipt)
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    _set_seed(42)
    model = new_model(cfg, fold)
    assert model.num_classes == 50
    initial = _module_state_sha256(model)
    if not m0:
        assert initial == preflight["training"]["initial_state_sha256"]
    summary = {"schema": config["schema"], "mode": args.mode, "status": "RUNNING",
               "started_at": datetime.now().astimezone().isoformat(),
               "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "config_sha256": sha256(args.config), "protocol_sha256": sha256(protocol_path),
               "runner_sha256": sha256(Path(__file__)), "source_files_sha256": config["source_files_sha256"],
               "baseline_config_sha256": sha256(baseline_path), **binding,
               "source_ids": source_ids, "heldout_ids": fold["heldout_ids"], "fold": "full_train",
               "seed": 42, "source_records": 8675, "source_identities": 50, "model_num_classes": model.num_classes,
               "official_model_record_forwards": 0, "rgbnt201_dev_record_forwards": 0,
               "official_metrics_read": False, "m0_trained_weights_loaded": False}
    write_json(args.output_dir / "summary.json", summary)
    torch.cuda.reset_peak_memory_stats()
    training = train_source(model, loader_for(records, True), cfg, indices, records,
                            preflight=m0, step_log=args.output_dir / "steps.jsonl")
    training["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    training["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    write_json(args.output_dir / "training.json", training)
    seen = {i for row in training["steps"] for i in row["sampled_record_indices"]}
    seen_ids = {source[i]["identity"] for i in seen}
    checks = {"all50_train_identities_seen": seen_ids == set(source_ids),
              "all195_trainable_tensors_have_gradients": training["trainable_tensors"] == training["gradient_tensors"] == 195,
              "no_missing_gradients": not training["trainable_without_gradient"],
              "all_updates_applied": all(r["optimizer_update_applied"] for r in training["steps"]),
              "overflow_zero": training["overflow_events"] == 0,
              "fixed_epoch_count": training["epochs"] == (1 if m0 else 30),
              "state_updated": training["initial_state_sha256"] == initial != training["final_state_sha256"],
              "token_selection_frozen": training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"],
              "capacity_below24GiB": training["peak_reserved_mib"] < 24 * 1024}
    if not m0:
        checks["all8675_training_records_seen"] = seen == set(indices)
    summary.update({"training": training, "checks": checks, "source_record_exposures": training["optimizer_steps"] * 64,
                    "unique_training_records_seen": len(seen), "unique_training_identities_seen": len(seen_ids)})
    summary["status"] = "TRAINING_COMPLETE_CHECKS_PENDING" if all(checks.values()) else "FAIL_FIXED_SIGNAL_ENGINEERING"
    write_json(args.output_dir / "summary.json", summary)
    assert all(checks.values()), checks
    checkpoint = args.output_dir / ("signal_m0.pth" if m0 else "signal_epoch30.pth")
    torch.save({"model_state_dict": model.state_dict(), "fold": "full_train", "source_ids": source_ids,
                "heldout_ids": fold["heldout_ids"], "config_sha256": sha256(args.config)}, checkpoint)
    before = extract(model, records[:8], cfg)
    assert _module_state_sha256(model) == training["final_state_sha256"]
    del model
    torch.cuda.empty_cache()
    model = new_model(cfg, fold)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    assert _module_state_sha256(model) == training["final_state_sha256"]
    after = extract(model, records[:8], cfg)
    assert before.shape == after.shape == (8, 3072) and torch.equal(before, after)
    assert _module_state_sha256(model) == training["final_state_sha256"]
    summary.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                    "strict_reload_state_sha256": _module_state_sha256(model),
                    "strict_reload_source_features_bitwise_equal": True,
                    "clean_source_record_forwards": 16, "completed_at": datetime.now().astimezone().isoformat(),
                    "status": "PASS_FULL50_SIGNAL_ENGINEERING" if m0 else "COMPLETE_FULL50_SIGNAL_FIXED_EPOCH30",
                    "elapsed_seconds": time.perf_counter() - started})
    if not m0:
        summary["m0_receipt_sha256"] = sha256(args.m0_receipt)
        summary["m0_verification_sha256"] = sha256(args.m0_verification)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "optimizer_steps": training["optimizer_steps"],
                      "checkpoint_sha256": summary["checkpoint_sha256"], "official_model_record_forwards": 0,
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("m0", "baseline"), required=True)
    parser.add_argument("--m0-receipt", type=Path)
    parser.add_argument("--m0-verification", type=Path)
    args = parser.parse_args()
    assert args.mode == "m0" or (args.m0_receipt is not None and args.m0_verification is not None)
    run(args)
