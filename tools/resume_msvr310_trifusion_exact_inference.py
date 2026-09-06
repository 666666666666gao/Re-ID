#!/usr/bin/env python3
"""Finish the fixed MSVR310 comparison after verified inference-only SIM repair."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    import torch
    from tools.train_msvr310_signal_oof import configure, loader_for, records_for, sha256, write_json
    from tools.train_msvr310_trifusion_oof import (
        build_model, train_roles, engineering_checks, output_mapping, evaluate,
        comparison_summary, frozen_state_sha,
    )
    from tools.run_signal_preserving_v5 import _module_state_sha256, _training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward

    runroot = args.run_root.resolve()
    directory = runroot / "comparison_resume_r3"
    assert not directory.exists()
    plan_path = ROOT / "evidence/trifusion_msvr310_trifusion_v1_resume_r3_plan_20260906.json"
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in plan["original_run_file_sha256"].items():
        assert sha256(runroot / name) == expected, name
    config_path = ROOT / "configs/MSVR310/TriFusion-source-oof-v1.json"
    assert sha256(config_path) == "49d82696fbf5d0d71431ed8d7f1e5f80d7c13215a8bec6cafe77969ee8e276a5"
    config = json.loads(config_path.read_text())
    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    baseline_config_path = ROOT / config["BASELINE"]["CONFIG"]
    assert sha256(baseline_config_path) == config["BASELINE"]["CONFIG_SHA256"]
    base = json.loads(baseline_config_path.read_text())
    assert sha256(config["BASELINE"]["SUMMARY"]) == config["BASELINE"]["SUMMARY_SHA256"]
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text())
    cfg, binding = configure(base)
    assert sha256(ROOT / base["protocol"]) == base["protocol_sha256"]
    protocol = json.loads((ROOT / base["protocol"]).read_text())
    preflight = json.loads((runroot / "m0/summary.json").read_text())
    assert preflight["status"] == "PASS_ENGINEERING_ONLY"
    assert preflight["config_sha256"] == sha256(config_path)
    assert preflight["runner_sha256"] == sha256(ROOT / "tools/train_msvr310_trifusion_oof.py")
    verification = json.loads((runroot / "exact_signal_inference_verification/summary.json").read_text())
    assert verification["status"] == "PASS_EXACT_INFERENCE_ENGINEERING"
    assert all(verification["checks"].values())
    assert verification["helper_sha256"] == sha256(ROOT / "tools/msvr310_exact_signal_inference.py")
    verified_arrays_path = runroot / "exact_signal_inference_verification/verification_arrays.pt"
    assert sha256(verified_arrays_path) == verification["arrays_sha256"]
    assert (runroot / "comparison_exit.txt").read_text().strip() == "1"
    partial_path = runroot / "comparison/summary.json"
    partial = json.loads(partial_path.read_text())
    assert partial["status"] == "RUNNING" and len(partial["folds"]) == 1
    assert partial["project_commit"] == "1c444cdf72e13fd041afd0c641dc8f522faa5844"
    assert partial["folds"][0]["training"]["optimizer_steps"] == 260
    assert partial["folds"][0]["training"] == json.loads((runroot / "comparison/fold_0/training.json").read_text())
    assert partial["folds"][0] == json.loads((runroot / "comparison/fold_0/receipt.json").read_text())
    assert all(not (runroot / f"comparison/fold_{i}").exists() for i in (1, 2))
    for row in baseline["folds"]:
        assert sha256(row["checkpoint"]) == row["checkpoint_sha256"]
        assert sha256(Path(row["checkpoint"]).parent / "retrieval_arrays.pt") == row["retrieval"]["retrieval_arrays_sha256"]
    started = time.perf_counter()
    execution = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    directory.mkdir()
    summary = {"schema": "msvr310-trifusion-source-oof-v1-resume-r3", "mode": "comparison_resume_r3",
               "status": "RUNNING", "project_commit": execution, "seed": 42,
               "config_sha256": sha256(config_path), "runner_sha256": sha256(__file__),
               "original_training_runner_sha256": sha256(ROOT / "tools/train_msvr310_trifusion_oof.py"),
               "resume_plan_sha256": sha256(plan_path), "inference_helper_sha256": verification["helper_sha256"],
               "baseline_summary_sha256": config["BASELINE"]["SUMMARY_SHA256"],
               "protocol_sha256": base["protocol_sha256"], **binding,
               "m0_receipt_sha256": sha256(runroot / "m0/summary.json"),
               "exact_inference_verification_sha256": sha256(runroot / "exact_signal_inference_verification/summary.json"),
               "original_failed_partial_summary_sha256": sha256(partial_path),
               "original_training_updates_reused": 260, "new_training_updates_budget": 520,
               "new_optimizer_steps": 0, "folds": [], "official_test_image_access": 0,
               "rgbnt201_dev_image_access": 0, "source_only_training": True,
               "evaluation_type": "train_internal_identity_oof_cross_dataset_comparison",
               "checkpoint_selection": "fixed_epoch20", "m0_weights_reused": False,
               "rgbnt201_role_weights_reused": False, "fold0_retrained": False,
               "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__)}
    write_json(directory / "summary.json", summary)
    for fold, b0 in zip(protocol["folds"], baseline["folds"], strict=True):
        index = fold["fold"]
        assert index == b0["fold"] and not set(fold["source_ids"]) & set(fold["heldout_ids"])
        fold_dir = directory / f"fold_{index}"
        fold_dir.mkdir()
        if index == 0:
            row = copy.deepcopy(partial["folds"][0])
            checkpoint = runroot / "comparison/fold_0/roles_epoch20.pth"
            assert sha256(checkpoint) == verification["role_checkpoint_sha256"]
            verified_arrays = torch.load(verified_arrays_path, map_location="cpu", weights_only=True)
            assert verified_arrays["gallery_record_indices"] == fold["gallery_record_indices"]
            features = verified_arrays["repaired_features"]
            row.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "strict_reload_state_sha256": row["training"]["final_state_sha256"],
                        "strict_reload_proof": str(runroot / "exact_signal_inference_verification/summary.json"),
                        "training_execution_commit": partial["project_commit"], "training_reused_from_original_run": True,
                        "feature_extraction_source": str(verified_arrays_path), "new_heldout_record_forwards": 0})
            del verified_arrays
        else:
            source = records_for(base, protocol, fold, True)
            model, initialization = build_model(config, cfg, fold, b0)
            assert initialization["initial_state_sha256"] == preflight["folds"][index]["initialization"]["initial_state_sha256"]
            training = train_roles(model, source, fold["source_record_indices"], config,
                                   fold=index, mode="comparison", directory=fold_dir)
            checks = engineering_checks(training)
            checks["expected_training_length"] = training["epochs"] == 20 and training["optimizer_steps"] == 260
            row = {"fold": index, "counts": fold["counts"], "initialization": initialization,
                   "training": training, "engineering_checks": checks,
                   "training_execution_commit": execution, "training_reused_from_original_run": False}
            write_json(fold_dir / "receipt.json", row)
            assert all(checks.values()), checks
            checkpoint = fold_dir / "roles_epoch20.pth"
            torch.save({"model_state_dict": model.state_dict(), "fold": index,
                        "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
                        "config_sha256": sha256(config_path)}, checkpoint)
            del model
            torch.cuda.empty_cache()
            model, _ = build_model(config, cfg, fold, b0)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            assert _module_state_sha256(model) == training["final_state_sha256"]
            model.eval()
            row.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "strict_reload_state_sha256": _module_state_sha256(model)})
            parts = {}
            gallery = records_for(base, protocol, fold, False)
            with torch.no_grad():
                for raw in loader_for(gallery, False):
                    batch, _ = _training_batch(raw)
                    for name, value in output_mapping(exact_signal_forward(model, batch)).items():
                        assert torch.isfinite(value).all().item()
                        parts.setdefault(name, []).append(value.float().cpu())
            features = {name: torch.cat(values) for name, values in parts.items()}
            assert _module_state_sha256(model) == training["final_state_sha256"]
            assert frozen_state_sha(model) == training["frozen_state_after_sha256"]
            assert all(not p.requires_grad for p in model.baseline.signal.parameters())
            row["new_heldout_record_forwards"] = len(gallery)
            summary["new_optimizer_steps"] += training["optimizer_steps"]
            del model, payload
            torch.cuda.empty_cache()
        row["retrieval"] = evaluate(features, protocol, fold, fold_dir, b0)
        row["heldout_record_forwards"] = len(fold["gallery_record_indices"])
        write_json(fold_dir / "receipt.json", row)
        summary["folds"].append(row)
        write_json(directory / "summary.json", summary)
        print(json.dumps({"event": "fold_complete", "fold": index,
                          "training_reused": index == 0, "new_optimizer_steps": summary["new_optimizer_steps"]}), flush=True)
    assert summary["new_optimizer_steps"] == 520
    summary["comparison"] = comparison_summary(summary["folds"])
    summary["status"] = "COMPLETE_COMPARISON_SUPPORT_PASS" if summary["comparison"]["scientific_passed"] else "COMPLETE_COMPARISON_SUPPORT_FAIL"
    summary["optimizer_steps"] = sum(f["training"]["optimizer_steps"] for f in summary["folds"])
    assert summary["optimizer_steps"] == 780
    summary["new_heldout_record_forwards"] = sum(f["new_heldout_record_forwards"] for f in summary["folds"])
    summary["evaluated_gallery_records"] = sum(len(f["retrieval"]["gallery_manifest"]) for f in summary["folds"])
    assert summary["evaluated_gallery_records"] == 1032 and summary["new_heldout_record_forwards"] == 672
    for name, expected in plan["original_run_file_sha256"].items():
        assert sha256(runroot / name) == expected, name
    summary["original_failed_run_files_unchanged"] = True
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["completed_at"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "complete", "status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    run(parser.parse_args())
