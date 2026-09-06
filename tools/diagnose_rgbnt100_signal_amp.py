#!/usr/bin/env python3
"""Capture the first RGBNT100 AMP stop in the unchanged registered train loop."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run(args):
    import torch
    from tools import train_rgbnt100_signal_oof as runner
    from tools.run_signal_preserving_v5 import _set_seed

    assert runner.sha256(runner.__file__) == "514a2c86634b61ef8b6de32b6ff1990cdf18e208a91673fb192534b52e5a1357"
    assert runner.sha256(args.config) == "7270e2bf95c5f5a60e1dc6d6b047f043dce667d508783b36bc4734aecbc4c15b"
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    protocol = json.loads((ROOT / config["protocol"]).read_text())
    assert runner.sha256(ROOT / config["protocol"]) == config["protocol_sha256"]
    assert runner.sha256(args.m0_receipt) == "7e9f6214efdbf14611fd5bb0ce2f4d7e6c9ae68c9549af06659d018b39e09820"
    m0 = json.loads(args.m0_receipt.read_text())["folds"][0]["training"]
    cfg, binding = runner.configure(config)
    fold = protocol["folds"][0]
    _set_seed(42)
    records = runner.records_for(config, protocol, fold, True)
    loader = runner.loader_for(records, True)
    model = runner.new_model(cfg, fold)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_rows = []
    before = {}
    target_code = runner.train_source.__code__
    launch = {
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "scope": "fold0 source only, unchanged train_source via line tracing, stop at first overflow or epoch1 boundary",
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diagnostic_sha256": runner.sha256(__file__), "runner_sha256": runner.sha256(runner.__file__),
        "config_sha256": runner.sha256(args.config), "protocol_sha256": config["protocol_sha256"],
        "m0_receipt_sha256": runner.sha256(args.m0_receipt), **binding,
        "heldout_image_forwards": 0, "official_test_image_access": 0,
        "new_method_training": False, "original_failed_successful_steps": "UNKNOWN_NOT_PERSISTED",
    }
    runner.write_json(args.output_dir / "launch.json", launch)

    def trace(frame, event, arg):
        if frame.f_code is not target_code:
            return None
        if event != "line":
            return trace
        state = frame.f_locals
        if frame.f_lineno == 164 and state["epoch"] == 2:
            runner.write_json(args.output_dir / "diagnosis.json", {
                **launch, "status": "NOT_REPRODUCED_IN_FIXED_FIRST_EPOCH",
                "attempted_steps": len(step_rows), "successful_optimizer_steps": len(step_rows),
                "source_record_exposures": len(step_rows) * 64,
                "elapsed_seconds": time.perf_counter() - started})
            raise RuntimeError("Fixed diagnostic boundary reached: no second-epoch update")
        if frame.f_lineno == 175:
            before.clear()
            before.update({
                "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
                "cpu_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_state": torch.cuda.get_rng_state().cpu().clone(),
            })
        if frame.f_lineno == 201:
            scale_after = state["scaler"].get_scale()
            overflow = scale_after < state["scale_before"]
            row = {
                "step": len(step_rows) + 1, "epoch": state["epoch"],
                "loss": float(state["loss"].detach()),
                "id_triplet_head_losses": state["components"],
                "gram_loss": float(state["output"][-2].detach()),
                "patch_loss": float(state["output"][-1].detach()),
                "sampled_record_indices": [state["name_to_index"][p] for p in state["paths"]],
                "amp_scale_before": state["scale_before"], "amp_scale_after": scale_after,
                "optimizer_update_applied": not overflow,
            }
            step_rows.append(row)
            with (args.output_dir / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if overflow:
                gradient_rows = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        gradient = parameter.grad.detach()
                        gradient_rows.append({
                            "name": name, "elements": gradient.numel(),
                            "nan_count": int(torch.isnan(gradient).sum()),
                            "positive_inf_count": int(torch.isposinf(gradient).sum()),
                            "negative_inf_count": int(torch.isneginf(gradient).sum()),
                        })
                model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                model_state.update(before["buffers"])
                fixture = args.output_dir / "failed_batch_preforward.pt"
                torch.save({
                    "model_state_dict": model_state,
                    "images": {name: value.detach().cpu() for name, value in state["images"].items()},
                    "labels": state["labels"].detach().cpu(),
                    "cameras": state["cameras"].detach().cpu(),
                    "scenes": state["scenes"].detach().cpu(),
                    "cpu_rng_state": before["cpu_rng_state"], "cuda_rng_state": before["cuda_rng_state"],
                    "amp_scale": state["scale_before"], "step": row["step"], "fold": 0,
                    "source_ids": fold["source_ids"], "source_record_indices": row["sampled_record_indices"],
                }, fixture)
                overlap = min(8, len(step_rows))
                comparisons = []
                keys = ["loss", "id_triplet_head_losses", "gram_loss", "patch_loss",
                        "sampled_record_indices", "amp_scale_before", "amp_scale_after"]
                for index in range(overlap):
                    comparisons.append({
                        "step": index + 1,
                        "all_recorded_values_exactly_match_m0": all(
                            step_rows[index][key] == m0["steps"][index][key] for key in keys),
                        "loss_difference": step_rows[index]["loss"] - m0["steps"][index]["loss"],
                        "sample_indices_exact": step_rows[index]["sampled_record_indices"] ==
                                                m0["steps"][index]["sampled_record_indices"],
                    })
                runner.write_json(args.output_dir / "diagnosis.json", {
                    **launch, "status": "REPRODUCED_FIRST_AMP_OVERFLOW",
                    "attempted_steps": len(step_rows), "successful_optimizer_steps": len(step_rows) - 1,
                    "source_record_exposures": len(step_rows) * 64, "failed_step": row,
                    "all_parameter_gradient_counts": gradient_rows, "m0_first8_comparison": comparisons,
                    "diagnostic_initial_state_matches_m0": state["initial_state"] == m0["initial_state_sha256"],
                    "fixture_sha256": runner.sha256(fixture), "fixture_bytes": fixture.stat().st_size,
                    "fixture_semantics": "weights before skipped step, all buffers and RNG restored to before failed forward; exact augmented source batch",
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "original_failed_successful_steps": "UNKNOWN_NOT_PERSISTED",
                })
                print(json.dumps({"event": "diagnostic_overflow_capture", "step": row["step"],
                                  "scale_before": row["amp_scale_before"], "scale_after": scale_after}), flush=True)
        return trace

    sys.settrace(trace)
    runner.train_source(model, loader, cfg, fold["source_record_indices"], records, preflight=False)
    raise AssertionError("Diagnostic must stop before the second epoch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m0-receipt", type=Path, required=True)
    run(parser.parse_args())
