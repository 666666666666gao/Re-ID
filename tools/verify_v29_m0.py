#!/usr/bin/env python3
"""Recompute V29's complete M0 scalar/file receipts without reading images/models."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.verify_v29_complete_terminal import expected_style_plan, sha, verify_fp32_contract


def verify(args):
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    precision = verify_fp32_contract(summary, args.repo)
    m0 = summary["m0"]
    config_path = args.repo / "configs/RGBNT201/TriFusion-signal-preserving-v29-bounded-joint-rtx3090.json"
    config = json.loads(config_path.read_bytes())
    assert sha(config_path) == summary["config_sha256"]
    plan = args.repo / "refine-logs/trifusion_v29_bounded_joint/EXPERIMENT_PLAN.md"
    assert sha(plan) == summary["plan_sha256"]
    for name, expected in config["SOURCE_FILE_SHA256"].items():
        assert sha(args.repo / name) == expected, name
    metadata = config["SUPERVISION_METADATA"]
    assert sha(args.repo / metadata["PATH"]) == metadata["SHA256"]
    replay = json.loads((args.repo / metadata["PATH"]).read_bytes())
    assert len(summary["preflight"]) == 3
    for fold in summary["preflight"]:
        control, candidate = fold["endpoints"]
        for key in ("common_initial_state_sha256", "all_output_sha256", "batch_receipts"):
            assert control[key] == candidate[key]
        for end in (control, candidate):
            assert end["state_unchanged"] and len(end["batch_receipts"]) == 8
            assert len(end["original_v8_forward_parity"]) == len(end["field_diagnostics"]) == 8
            for row in end["original_v8_forward_parity"]:
                assert row["independent_outputs_and_logits_exact"]
                assert row["extra_slot_normalization_fused_max_difference"] < 1e-5
    for fold in summary["preflight"]:
        number = fold["fold"]
        for end in fold["endpoints"]:
            assert end["evaluation_style_disabled"] and end["all_eight_batches_match_own_registered_sampler"]
            for index, field in enumerate(end["field_diagnostics"]):
                indices = replay["folds"][number]["arms"]["control"]["batches"][index]["sampler_indices"]
                cameras = [replay["folds"][number]["source_manifest"][i]["camera"] for i in indices]
                assert field["plan"] == expected_style_plan(cameras, number, index, True)
                assert field["baseline_exact"] and field["role_reference_both_frozen"]
                assert field["additional_visual_passes"] == 3
    stages = [*m0["capacities"], m0["overfit"]]
    assert [row["steps"] for row in stages] == [8, 8, 100]
    all_steps = []
    maximum_loss_error = 0.0
    for stage in stages:
        path = Path(stage["all_steps_path"])
        assert sha(path) == stage["all_steps_sha256"]
        steps = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert len(steps) == len(stage["losses"]) == len(stage["components"]) == stage["steps"]
        for index, step in enumerate(steps):
            assert step["losses"] == stage["components"][index]
            assert step["losses"]["total"] == stage["losses"][index]
            assert step["step"] == index + 1 and step["endpoint"] == stage["endpoint"]
            losses = step["losses"]
            assert all(math.isfinite(value) for value in losses.values())
            number = 0 if stage["fixed_batch"] else index
            indices = replay["folds"][0]["arms"]["control"]["batches"][number]["sampler_indices"]
            cameras = [replay["folds"][0]["source_manifest"][i]["camera"] for i in indices]
            plan = expected_style_plan(cameras, 0, number, stage["fixed_batch"])
            assert plan == step["style_plan"] == stage["style_plans"][index]
            assert losses["style_active"] == int(plan["active"])
            assert losses["joint_enabled"] == int(stage["endpoint"] == "bounded_joint")
            weight = config["LOSS"]
            total = weight["ID_FUSED"] * losses["id_fused"] + weight["TRIPLET_FUSED"] * losses["triplet_fused"]
            for expert in ("cnn", "transformer", "mamba"):
                total += weight["ID_BRANCH"] * losses["id_" + expert]
                total += weight["TRIPLET_BRANCH"] * losses["triplet_" + expert]
                total += weight["ID_RESIDUAL"] * losses["id_residual_" + expert]
                total += weight["TRIPLET_RESIDUAL"] * losses["triplet_residual_" + expert]
            maximum_loss_error = max(maximum_loss_error, abs(total - losses["total"]))
        assert stage["overflow_events"] == sum(int(step["overflow"]) for step in steps)
        all_steps.append(steps)
    assert maximum_loss_error < 2e-6
    smoothing = config["LOSS"]["LABEL_SMOOTHING"]
    classes = 94
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    floor = 0.75 * (-correct * math.log(correct) - (classes - 1) * other * math.log(other))
    overfit = m0["overfit"]
    ratio = (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)
    assert abs(floor - overfit["combined_loss_floor"]) < 1e-14
    assert abs(ratio - overfit["excess_loss_ratio"]) < 1e-14
    capacities = m0["capacities"]
    checks = {
        "all_gradients_live": all(not row["missing_nonzero_gradients"] for row in stages),
        "all_frozen_states_unchanged": all(row["frozen_state_unchanged"] for row in stages),
        "overflow_zero": all(row["overflow_events"] == 0 for row in stages),
        "capacity_within_limit": all(row["peak_reserved_mib"] < 24576 for row in capacities),
        "overfit_excess_ratio_at_most_point1": ratio <= 0.1,
        "style_alters_both_role_fields": all(row["style_anchor_max_change"] > 0 and row["style_reference_max_change"] > 0
            for fold in summary["preflight"] for row in fold["endpoints"][1]["field_diagnostics"]),
        "all_training_signal_prefixes_exact": all(row["baseline_exact"] for fold in summary["preflight"]
            for end in fold["endpoints"] for row in end["field_diagnostics"]),
        "candidate_has_active_capacity_batches": any(row["style_active"] for row in capacities[1]["components"]),
        "all_overfit_updates_style_active": all(row["style_active"] for row in overfit["components"]),
        "matched_capacity_plans": capacities[0]["style_plans"] == capacities[1]["style_plans"],
        "joint_correction_becomes_nonzero": any(row["joint_correction_abs_mean"] > 0 for row in capacities[1]["components"]),
        "both_capacity_arms_same_style_activation": [row["style_active"] for row in capacities[0]["components"]]
            == [row["style_active"] for row in capacities[1]["components"]],
    }
    checks["all_actual_slot_updates_within_geometry_bound"] = all(
        row["geometry_update_ratio_max"] <= 0.5 + 2e-6 and row["geometry_cosine_min"] >= 1 / math.sqrt(1.25) - 2e-6
        for stage in stages for row in stage["components"])
    assert checks == m0["checks"] and all(checks.values()) == m0["passed"]
    report = {
        "status": "PASS_COMPLETE_V29_BOUNDED_GEOMETRY_M0_RECEIPT_RECOMPUTATION",
        "precision_fixture": precision,
        "execution_commit": summary["repository_commit"],
        "m0_passed": m0["passed"], "checks": checks, "fixed_overfit_excess_ratio": ratio,
        "original_summary_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "verifier_sha256": sha(__file__), "checked_real_optimizer_updates": 116,
        "checked_preflight_batches": 48, "checked_original_v8_parity_forwards": 48,
        "checked_backbone_interface_forwards": 48, "maximum_total_loss_recompute_error": maximum_loss_error,
        "stage_gradient_tensor_ranges": [
            {"endpoint": stage["endpoint"], "fixed": stage["fixed_batch"],
             "trainable": stage["trainable_tensors"], "phase_nonzero": stage["nonzero_gradient_tensors"],
             "minimum_per_step": min(row["live_gradient_tensors"] for row in steps),
             "maximum_per_step": max(row["live_gradient_tensors"] for row in steps)}
            for stage, steps in zip(stages, all_steps, strict=True)],
        "source_image_reads": 0, "new_optimizer_updates": 0, "model_tensor_loads": 0,
        "independent_audit": False,
    }
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
