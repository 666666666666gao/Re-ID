#!/usr/bin/env python3
"""Measure all three fixed V22 initializations on the full Q1 gallery without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from tools.train_signal_preserving_v22 import (
    ENDPOINTS, OUTPUTS, _configure_signal, _load_records, _model_state_sha256,
    _sha256, build_complete_path_fold_records, build_model, evaluate,
    load_contract, source_bindings,
)


def run(args):
    started = time.time()
    assert not args.output.exists()
    assert _sha256(Path(__file__)) == args.script_sha256
    assert _sha256(args.plan) == args.plan_sha256
    assert _sha256(args.summary) == args.summary_sha256 == "b8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3"
    terminal = json.loads(args.summary.read_bytes())
    assert terminal["status"] == "Q1_FAIL" and len(terminal["folds"]) == 3
    assert terminal["repository_commit"] == "5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36"
    assert _sha256(args.config) == terminal["config_sha256"]
    expected = {Path(path): value for path, value in terminal["source_file_sha256"].items()}
    expected[Path("tools/train_signal_preserving_v22.py")] = terminal["runner_sha256"]
    expected[Path("refine-logs/v22/EXPERIMENT_PLAN.md")] = terminal["plan_sha256"]
    config, sources = load_contract(args.config)
    expected.update({Path(path): value for path, value in source_bindings(config).items()})
    expected[Path(config["SIGNAL"]["CLIP_WEIGHT"])] = config["SIGNAL"]["CLIP_WEIGHT_SHA256"]
    expected[Path(config["INITIALIZATION"]["V12_RUN_SUMMARY"])] = config["INITIALIZATION"]["V12_RUN_SUMMARY_SHA256"]
    expected[Path(config["SUPERVISION_METADATA"]["PATH"])] = config["SUPERVISION_METADATA"]["SHA256"]
    for path, value in expected.items():
        assert _sha256(path) == value, str(path)
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    assert signal_commit == terminal["signal_commit"] and signal_diff == terminal["signal_diff_sha256"]
    records = _load_records(config)
    assert len(records) == 3126
    splits = [build_complete_path_fold_records(records, heldout_ids=set(row["heldout_identity_ids"]))
              for row in sources["fold_receipts"]]
    assert len(splits) == 3 and all(not split["identity_overlap"] for split in splits)
    report = {
        "schema_version": "v22-fixed-initialization-full-gallery-diagnostic-v1",
        "status": "RUNNING",
        "evaluation_type": "read_only_fixed_source_initialization_on_reused_train_internal_complete_path_oof_full_gallery",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": args.script_sha256, "plan_sha256": args.plan_sha256,
        "terminal_summary_sha256": args.summary_sha256,
        "terminal_execution_commit": terminal["repository_commit"],
        "config_sha256": terminal["config_sha256"],
        "source_file_sha256": {str(path): value for path, value in expected.items()},
        "signal_commit": signal_commit, "signal_diff_sha256": signal_diff,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__, "gpu": torch.cuda.get_device_name(0),
        "seed": 42, "optimizer_steps": 0, "checkpoint_writes": 0,
        "dev_access_count": 0, "official_test_access_count": 0,
        "checkpoint_selection": "none_fixed_common_initialization_only",
        "changes_q1_qualification": False, "d1_authorized": False,
        "folds": [],
    }

    def save():
        report["elapsed_seconds"] = time.time() - started
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))

    save()
    for fold, split in enumerate(splits):
        paired = terminal["folds"][fold]
        gallery = [{"file": Path(row[0][0]).name, "identity": row[1], "camera": row[2]}
                   for row in split["heldout_records"]]
        assert gallery == paired["gallery_manifest"]
        model, binding = build_model(config, signal_cfg, fold, split)
        initial = _model_state_sha256(model)
        for endpoint in ENDPOINTS:
            assert binding == paired["endpoints"][endpoint]["binding"]
            assert initial == paired["endpoints"][endpoint]["training"]["initial_state_sha256"]
        counts = {"forward_calls": 0, "triplets": 0}

        def count_forward(_module, arguments):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            counts["forward_calls"] += 1
            counts["triplets"] += len(arguments[0]["camera_ids"])

        hook = model.register_forward_pre_hook(count_forward)
        scores = evaluate(model, split["heldout_records"], config)
        hook.remove()
        assert counts["triplets"] == len(gallery)
        assert counts["forward_calls"] == (len(gallery) + 127) // 128
        assert initial == _model_state_sha256(model)
        assert all(parameter.grad is None for parameter in model.parameters())
        for endpoint in ENDPOINTS:
            assert scores["baseline_only"] == paired["endpoints"][endpoint]["outputs"]["baseline_only"]
        row = {
            "fold": fold, "gallery_manifest": gallery, "binding": binding,
            "initial_state_sha256": initial, "matches_both_terminal_initial_states": True,
            "model_state_unchanged": True, "no_parameter_gradients": True,
            "forward_counts": counts, "outputs": scores,
            "terminal_minus_initial_mAP": {
                endpoint: {name: paired["endpoints"][endpoint]["outputs"][name]["metrics_percent"]["mAP"]
                            - scores[name]["metrics_percent"]["mAP"] for name in OUTPUTS}
                for endpoint in ENDPOINTS},
        }
        report["folds"].append(row)
        save()
        print(json.dumps({"stage": "fixed_initialization_evaluated", "fold": fold, "counts": counts,
                          "metrics": {name: result["metrics_percent"] for name, result in scores.items()},
                          "terminal_minus_initial_mAP": row["terminal_minus_initial_mAP"]}), flush=True)
        del model
        torch.cuda.empty_cache()

    aggregate = {}
    for name in OUTPUTS:
        ap = np.concatenate([np.asarray(fold["outputs"][name]["average_precision"], dtype=np.float64)
                             for fold in report["folds"]])
        ranks = np.concatenate([np.asarray(fold["outputs"][name]["first_match_rank"])
                                for fold in report["folds"]])
        assert len(ap) == len(ranks) == 571
        aggregate[name] = {"mAP": float(ap.mean() * 100),
                           **{f"Rank-{k}": float(np.mean(ranks <= k) * 100) for k in (1, 5, 10)}}
    assert sum(row["forward_counts"]["triplets"] for row in report["folds"]) == 3126
    for path, value in expected.items():
        assert _sha256(path) == value, str(path)
    assert _sha256(args.summary) == args.summary_sha256
    report.update({
        "status": "COMPLETE_READONLY_DIAGNOSTIC",
        "initialization_aggregate": aggregate,
        "terminal_aggregate": terminal["aggregate"],
        "terminal_minus_initial_mAP": {
            endpoint: {name: terminal["aggregate"][endpoint][name]["mAP"] - aggregate[name]["mAP"]
                       for name in OUTPUTS} for endpoint in ENDPOINTS},
        "source_files_unchanged": True, "terminal_summary_unchanged": True,
        "total_triplet_forwards": 3126,
        "total_model_forward_calls": sum(row["forward_counts"]["forward_calls"] for row in report["folds"]),
        "total_gallery_records": 3126, "total_eligible_queries": 571,
    })
    save()
    print(json.dumps({key: report[key] for key in ("status", "initialization_aggregate",
                       "terminal_minus_initial_mAP", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
