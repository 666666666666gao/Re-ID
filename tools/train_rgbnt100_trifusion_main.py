#!/usr/bin/env python3
"""Train the original full50-ID roles at fixed endpoints before official retrieval."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_rgbnt100_signal_oof import configure, sha256, write_json
from tools.train_rgbnt100_trifusion_oof import (
    OUTPUT_WIDTHS, build_model, engineering_checks, extract, train_roles,
)


def load_inputs(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema"] == "rgbnt100-original-roles-full50-main-v1"
    assert config["EXPERIMENT"]["SEED"] == 42
    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    base = config["BASELINE"]
    for name in ("CONFIG", "SUMMARY", "VERIFICATION"):
        assert sha256(ROOT / base[name]) == base[name + "_SHA256"], name
    baseline_config = json.loads((ROOT / base["CONFIG"]).read_text())
    baseline = json.loads((ROOT / base["SUMMARY"]).read_text())
    verified = json.loads((ROOT / base["VERIFICATION"]).read_text())
    assert baseline["status"] == "COMPLETE_FULL50_SIGNAL_FIXED_EPOCH30"
    assert baseline["config_sha256"] == base["CONFIG_SHA256"]
    assert baseline["training"]["epochs"] == 30 and all(baseline["checks"].values())
    assert baseline["model_num_classes"] == 50 and baseline["official_model_record_forwards"] == 0
    assert verified["status"] == "PASS_FULL50_SIGNAL_FILES_ALL_UPDATES_AND_AUTHOR_LR"
    assert verified["mode"] == "baseline" and verified["summary_sha256"] == base["SUMMARY_SHA256"]
    assert sha256(baseline["checkpoint"]) == baseline["checkpoint_sha256"]
    protocol_path = ROOT / config["DATA"]["PROTOCOL"]
    assert sha256(protocol_path) == config["DATA"]["PROTOCOL_SHA256"] == baseline["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    assert protocol["counts"] == {"train": 8675, "query": 1715, "gallery": 8575}
    settings_path = ROOT / baseline_config["baseline_config"]
    assert sha256(settings_path) == baseline_config["baseline_config_sha256"]
    signal_config, binding = configure(json.loads(settings_path.read_text()))
    fold = {"fold": "full_train", "source_ids": protocol["identities"]["train"],
            "heldout_ids": protocol["identities"]["query"],
            "source_camera_values": protocol["camera_values"]["train"]}
    assert fold["source_ids"] == baseline["source_ids"] == list(range(501, 600, 2))
    assert fold["heldout_ids"] == baseline["heldout_ids"] == list(range(502, 601, 2))
    records = [(str(Path(protocol["dataset_root"]) / r["path"]),
                protocol["train_label_map"][str(r["identity"])], r["camera"], r["view"])
               for r in protocol["records"]["train"]]
    return config, baseline, protocol, signal_config, binding, fold, records


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, evaluate_overfit_gate, overfit_loss_floor,
    )

    started = time.perf_counter()
    config, baseline, protocol, cfg, binding, fold, records = load_inputs(args.config)
    m0 = args.mode == "m0"
    assert not args.output_dir.exists()
    if not m0:
        preflight = json.loads(args.m0_receipt.read_text())
        assert preflight["status"] == "PASS_FULL50_ROLES_ENGINEERING"
        assert preflight["config_sha256"] == sha256(args.config)
        verified = json.loads(args.m0_verification.read_text())
        assert verified["status"] == "PASS_FULL50_ROLES_FILES_AND_ALL_UPDATES"
        assert verified["mode"] == "m0" and verified["engineering_passed"]
        assert verified["summary_sha256"] == sha256(args.m0_receipt)
    args.output_dir.mkdir(parents=True)
    model, initial = build_model(config, cfg, fold, baseline)
    assert model.num_classes == model.baseline.signal.num_classes == 50
    classifiers = {"fused": model.fused_classifier,
                   **{"branch_" + k: v for k, v in model.branch_classifiers.items()},
                   **{"residual_" + k: v for k, v in model.residual_classifiers.items()}}
    head_shapes = {k: list(v.weight.shape) for k, v in classifiers.items()}
    assert len(head_shapes) == 7 and all(shape[0] == 50 for shape in head_shapes.values())
    if not m0:
        assert initial == preflight["initialization"]
    summary = {"schema": config["schema"], "mode": args.mode, "status": "RUNNING",
               "started_at": datetime.now().astimezone().isoformat(), "config_sha256": sha256(args.config),
               "runner_sha256": sha256(__file__), "baseline_summary_sha256": config["BASELINE"]["SUMMARY_SHA256"],
               "protocol_sha256": config["DATA"]["PROTOCOL_SHA256"], "project_source_file_sha256": config["project_source_file_sha256"],
               "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               **binding, "initialization": initial, "classifier_weight_shapes": head_shapes, "seed": 42,
               "source_records": 8675, "source_identities": 50, "official_model_record_forwards": 0,
               "rgbnt201_dev_record_forwards": 0, "m0_weights_reused": False, "oof_role_weights_reused": False}
    write_json(args.output_dir / "summary.json", summary)
    if m0:
        extract(model, records[:8], cfg, verify_direct_signal=True)
        assert _module_state_sha256(model) == initial["initial_state_sha256"]
    training = train_roles(model, records, list(range(8675)), config, fold="full_train",
                           mode="capacity" if m0 else "comparison", directory=args.output_dir)
    checks = engineering_checks(training)
    checks["all203_trainable_gradients_live"] = training["trainable_tensors"] == training["nonzero_gradient_tensors"] == 203
    checks["expected_training_length"] = training["optimizer_steps"] == 8 if m0 else training["epochs"] == 20
    if not m0:
        seen = {i for row in training["steps"] for i in row["sampled_record_indices"]}
        checks["all8675_source_records_seen"] = seen == set(range(8675))
    summary.update({"training": training, "engineering_checks": checks,
                    "status": "TRAINING_COMPLETE_CHECKS_PENDING" if all(checks.values()) else "FAIL_FULL50_ROLES_ENGINEERING"})
    write_json(args.output_dir / "summary.json", summary)
    assert all(checks.values()), checks
    checkpoint = args.output_dir / ("roles_m0.pth" if m0 else "roles_epoch20.pth")
    torch.save({"model_state_dict": model.state_dict(), "fold": "full_train", "source_ids": fold["source_ids"],
                "heldout_ids": fold["heldout_ids"], "config_sha256": sha256(args.config)}, checkpoint)
    before = extract(model, records[:8], cfg)
    assert _module_state_sha256(model) == training["final_state_sha256"]
    del classifiers, model
    torch.cuda.empty_cache()
    model, _ = build_model(config, cfg, fold, baseline)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    after = extract(model, records[:8], cfg, verify_direct_signal=True)
    assert all(torch.equal(before[k], after[k]) for k in OUTPUT_WIDTHS)
    assert _module_state_sha256(model) == training["final_state_sha256"]
    summary.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                    "strict_reload_state_sha256": training["final_state_sha256"],
                    "strict_reload_all_outputs_bitwise_equal": True, "direct_signal_source_parity": True,
                    "clean_role_source_record_forwards": 24 if m0 else 16,
                    "direct_signal_source_record_forwards": 16 if m0 else 8})
    del model, payload, before, after
    torch.cuda.empty_cache()
    if m0:
        overfit_dir = args.output_dir / "overfit"
        overfit_dir.mkdir()
        model, overfit_initial = build_model(config, cfg, fold, baseline)
        assert overfit_initial == initial
        overfit = train_roles(model, records, list(range(8675)), config, fold="full_train",
                              mode="overfit", directory=overfit_dir)
        gate = evaluate_overfit_gate([r["loss"] for r in overfit["steps"]], max_ratio=0.1,
                                    minimum_loss=overfit_loss_floor(config, num_classes=50))
        overfit_checks = engineering_checks(overfit)
        overfit_checks.update({"fixed100_updates": overfit["optimizer_steps"] == 100,
                               "all203_gradients_live": overfit["nonzero_gradient_tensors"] == overfit["trainable_tensors"] == 203,
                               "overfit_excess_ratio_at_most_point1": gate["passed"]})
        summary["overfit"] = {"initialization": overfit_initial, "training": overfit,
                              "loss_gate": gate, "checks": overfit_checks}
        summary["status"] = "PASS_FULL50_ROLES_ENGINEERING" if all(overfit_checks.values()) else "FAIL_FULL50_ROLES_ENGINEERING"
        summary["optimizer_steps"] = 108
    else:
        summary["status"] = "COMPLETE_FULL50_ROLES_FIXED_EPOCH20"
        summary["optimizer_steps"] = training["optimizer_steps"]
        summary["m0_receipt_sha256"] = sha256(args.m0_receipt)
        summary["m0_verification_sha256"] = sha256(args.m0_verification)
    summary.update({"completed_at": datetime.now().astimezone().isoformat(),
                    "elapsed_seconds": time.perf_counter() - started})
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "optimizer_steps": summary["optimizer_steps"],
                      "elapsed_seconds": summary["elapsed_seconds"], "official_model_record_forwards": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("m0", "main"), required=True)
    parser.add_argument("--m0-receipt", type=Path)
    parser.add_argument("--m0-verification", type=Path)
    args = parser.parse_args()
    assert args.mode == "m0" or (args.m0_receipt is not None and args.m0_verification is not None)
    run(args)
