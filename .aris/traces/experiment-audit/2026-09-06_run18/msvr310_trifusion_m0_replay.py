#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN_DIR = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = RUN_DIR / "input_manifest.json"

CONFIG_REL = pathlib.Path("configs/MSVR310/TriFusion-source-oof-v1.json")
PRELAUNCH_REL = pathlib.Path("refine-logs/msvr310_trifusion_v1/CONFIG_PRELAUNCH_R1_20260906.json")
PROTOCOL_REL = pathlib.Path("protocols/msvr310_train_oof_v1.json")
M0_SUMMARY_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json")
M0_EXIT_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_m0_exit_20260906.txt")
M0_LOG_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_m0_run_20260906.log")
BASELINE_REL = pathlib.Path("evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json")
SIGNAL_CLOSURE_REL = pathlib.Path("evidence/trifusion_msvr310_signal_v1_audit_closure_20260906.json")
SOURCE_BINDING_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json")
PRELAUNCH_BYTES_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_prelaunch_source_bytes_20260906.json")
FILE_VERIFICATION_REL = pathlib.Path("evidence/trifusion_msvr310_trifusion_v1_m0_file_verification_20260906.json")
REQUEST_REL = pathlib.Path(".aris/traces/experiment-audit/2026-09-06_run18/001-m0.request.json")
RUN_META_REL = pathlib.Path(".aris/traces/experiment-audit/2026-09-06_run18/run.meta.json")

TRAINING_RELS = {
    "capacity_fold0": pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_0_training.json"),
    "capacity_fold1": pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_1_training.json"),
    "capacity_fold2": pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_2_training.json"),
    "overfit_fold0": pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/overfit_fold0_training.json"),
}
RECEIPT_RELS = {
    0: pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_0_receipt.json"),
    1: pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_1_receipt.json"),
    2: pathlib.Path("evidence/msvr310_trifusion_v1_m0_receipts/fold_2_receipt.json"),
}

EXPERTS = ("cnn", "transformer", "mamba")
EXPECTED_COMPONENTS = (
    "id_fused",
    "triplet_fused",
    "id_cnn",
    "triplet_cnn",
    "id_residual_cnn",
    "triplet_residual_cnn",
    "id_transformer",
    "triplet_transformer",
    "id_residual_transformer",
    "triplet_residual_transformer",
    "id_mamba",
    "triplet_mamba",
    "id_residual_mamba",
    "triplet_residual_mamba",
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: pathlib.Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def verify_manifest() -> dict:
    manifest = load_json(MANIFEST_PATH)
    checks = []
    ok = True
    for entry in manifest["files"]:
        path = pathlib.Path(entry["path"])
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = sha256_path(path) if exists else None
        row = {
            "path": path.as_posix(),
            "expected_bytes": entry["bytes"],
            "actual_bytes": size,
            "expected_sha256": entry["sha256"],
            "actual_sha256": digest,
            "ok": bool(exists and size == entry["bytes"] and digest == entry["sha256"]),
        }
        ok = ok and row["ok"]
        checks.append(row)
    return {
        "manifest": rel(MANIFEST_PATH),
        "file_count": len(checks),
        "all_ok": ok,
        "checks": checks,
    }


def weighted_loss(components: dict[str, float], config: dict) -> float:
    weights = config["LOSS"]
    branch_id = sum(float(components[f"id_{expert}"]) for expert in EXPERTS)
    branch_triplet = sum(float(components[f"triplet_{expert}"]) for expert in EXPERTS)
    residual_id = sum(float(components[f"id_residual_{expert}"]) for expert in EXPERTS)
    residual_triplet = sum(float(components[f"triplet_residual_{expert}"]) for expert in EXPERTS)
    return (
        float(components["id_fused"]) * float(weights["ID_FUSED"])
        + float(components["triplet_fused"]) * float(weights["TRIPLET_FUSED"])
        + branch_id * float(weights["ID_BRANCH"])
        + branch_triplet * float(weights["TRIPLET_BRANCH"])
        + residual_id * float(weights["ID_RESIDUAL"])
        + residual_triplet * float(weights["TRIPLET_RESIDUAL"])
    )


def label_smoothing_floor(config: dict, num_classes: int) -> dict:
    smoothing = float(config["LOSS"]["LABEL_SMOOTHING"])
    correct = 1.0 - smoothing + smoothing / num_classes
    other = smoothing / num_classes
    entropy = -correct * math.log(correct)
    entropy -= (num_classes - 1) * other * math.log(other)
    identity_weight = (
        float(config["LOSS"]["ID_FUSED"])
        + 3.0 * float(config["LOSS"]["ID_BRANCH"])
        + 3.0 * float(config["LOSS"]["ID_RESIDUAL"])
    )
    return {
        "num_classes": int(num_classes),
        "label_smoothing": smoothing,
        "correct_probability": correct,
        "other_probability": other,
        "entropy": entropy,
        "identity_weight": identity_weight,
        "weighted_entropy_floor": identity_weight * entropy,
    }


def sample_stats(training: dict, fold: dict, protocol: dict, *, fixed_batch: bool) -> dict:
    records = {int(row["index"]): row for row in protocol["records"]}
    source_indices = set(map(int, fold["source_record_indices"]))
    source_ids = set(map(int, fold["source_ids"]))
    heldout_ids = set(map(int, fold["heldout_ids"]))
    all_indices = []
    per_step = []
    identity_counter = Counter()
    unique_indices = set()
    same_pairs = 0
    cross_scene_pairs = 0
    first_indices = None
    fixed_equal = True

    for step in training["steps"]:
        indices = [int(x) for x in step["sampled_record_indices"]]
        if first_indices is None:
            first_indices = indices
        elif indices != first_indices:
            fixed_equal = False
        all_indices.extend(indices)
        unique_indices.update(indices)
        rows = [records[i] for i in indices]
        ids = [int(r["identity"]) for r in rows]
        scenes = [int(r["scene"]) for r in rows]
        id_counts = Counter(ids)
        identity_counter.update(ids)
        unordered_same = 0
        unordered_cross_scene = 0
        for positions in _positions_by_value(ids).values():
            n = len(positions)
            unordered_same += n * (n - 1) // 2
            for a_pos, a in enumerate(positions):
                for b in positions[a_pos + 1:]:
                    if scenes[a] != scenes[b]:
                        unordered_cross_scene += 1
        same_pairs += unordered_same
        cross_scene_pairs += unordered_cross_scene
        per_step.append(
            {
                "step": int(step["step"]),
                "sample_count": len(indices),
                "unique_identity_count": len(id_counts),
                "identity_counts": sorted(id_counts.values()),
                "all_indices_in_source_records": all(i in source_indices for i in indices),
                "all_identities_in_source_ids": all(i in source_ids for i in ids),
                "any_identity_in_heldout_ids": any(i in heldout_ids for i in ids),
                "unordered_same_identity_pairs": unordered_same,
                "unordered_cross_scene_positive_pairs": unordered_cross_scene,
            }
        )

    sampled_identity_set = sorted(identity_counter)
    return {
        "steps": len(training["steps"]),
        "exposures": len(all_indices),
        "unique_source_identities": len(sampled_identity_set),
        "unique_source_identity_values": sampled_identity_set,
        "unique_source_records": len(unique_indices),
        "all_steps_len_64": all(row["sample_count"] == 64 for row in per_step),
        "all_steps_8_identities_x8": all(
            row["unique_identity_count"] == 8 and row["identity_counts"] == [8] * 8
            for row in per_step
        ),
        "all_sampled_indices_in_source_records": all(row["all_indices_in_source_records"] for row in per_step),
        "all_sampled_identities_in_source_ids": all(row["all_identities_in_source_ids"] for row in per_step),
        "no_sampled_identity_in_heldout_ids": not any(row["any_identity_in_heldout_ids"] for row in per_step),
        "fixed_batch_requested": fixed_batch,
        "fixed_batch_repeated_after_step1": fixed_equal if fixed_batch else None,
        "unordered_same_identity_pairs": same_pairs,
        "unordered_cross_scene_positive_pairs": cross_scene_pairs,
    }


def _positions_by_value(values: list[int]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        positions[value].append(index)
    return positions


def training_checks(name: str, training: dict, config: dict, fold: dict, protocol: dict, *, fixed_batch: bool) -> dict:
    diffs = []
    missing_or_extra = []
    component_values = {key: [] for key in EXPECTED_COMPONENTS}
    for step in training["steps"]:
        components = {key: float(value) for key, value in step["components"].items()}
        missing = sorted(set(EXPECTED_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(EXPECTED_COMPONENTS))
        if missing or extra:
            missing_or_extra.append({"step": step["step"], "missing": missing, "extra": extra})
        recomputed = weighted_loss(components, config)
        diff = abs(recomputed - float(step["loss"]))
        diffs.append((diff, int(step["step"]), recomputed, float(step["loss"])))
        for key in EXPECTED_COMPONENTS:
            if key in components:
                component_values[key].append(components[key])

    history_rechecks = []
    for row in training["history"]:
        epoch = int(row["epoch"])
        epoch_losses = [float(step["loss"]) for step in training["steps"] if int(step["epoch"]) == epoch]
        mean = statistics.fmean(epoch_losses)
        history_rechecks.append(
            {
                "epoch": epoch,
                "recorded_mean_loss": float(row["mean_loss"]),
                "recomputed_mean_loss": mean,
                "abs_diff": abs(mean - float(row["mean_loss"])),
                "recorded_optimizer_steps": int(row["optimizer_steps"]),
                "recomputed_optimizer_steps": len(epoch_losses),
            }
        )

    component_summary = {}
    for key, values in component_values.items():
        if values:
            component_summary[key] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }

    worst = max(diffs, key=lambda row: row[0]) if diffs else (0.0, None, None, None)
    return {
        "name": name,
        "mode": training["mode"],
        "steps": len(training["steps"]),
        "optimizer_steps_recorded": int(training["optimizer_steps"]),
        "component_key_problems": missing_or_extra,
        "max_weighted_loss_abs_diff": worst[0],
        "worst_weighted_loss_step": worst[1],
        "worst_weighted_loss_recomputed": worst[2],
        "worst_weighted_loss_recorded": worst[3],
        "epoch_rechecks": history_rechecks,
        "component_summary": component_summary,
        "sample_stats": sample_stats(training, fold, protocol, fixed_batch=fixed_batch),
    }


def protocol_checks(protocol: dict, summary: dict, baseline: dict) -> dict:
    all_record_ids = {int(row["index"]) for row in protocol["records"]}
    fold_rows = []
    all_heldout = []
    all_gallery_indices = []
    ok = True
    for fold, m0_fold, base_fold in zip(protocol["folds"], summary["folds"], baseline["folds"], strict=True):
        source_ids = set(map(int, fold["source_ids"]))
        heldout_ids = set(map(int, fold["heldout_ids"]))
        source_indices = set(map(int, fold["source_record_indices"]))
        gallery_indices = set(map(int, fold["gallery_record_indices"]))
        all_heldout.extend(heldout_ids)
        all_gallery_indices.extend(gallery_indices)
        source_record_ids = {int(protocol["records"][i]["identity"]) for i in source_indices}
        gallery_record_ids = {int(protocol["records"][i]["identity"]) for i in gallery_indices}
        rows_by_index = {int(row["index"]): row for row in protocol["records"]}
        query_ok = True
        for query in fold["query_rows"]:
            q_idx = int(query["record_index"])
            q_id = int(query["identity"])
            if q_idx not in gallery_indices or q_id not in heldout_ids:
                query_ok = False
                break
            q_scene = int(rows_by_index[q_idx]["scene"])
            has_cross_scene = any(
                int(rows_by_index[g]["identity"]) == q_id and int(rows_by_index[g]["scene"]) != q_scene
                for g in gallery_indices
            )
            if not has_cross_scene:
                query_ok = False
                break
        row = {
            "fold": int(fold["fold"]),
            "source_id_count": len(source_ids),
            "heldout_id_count": len(heldout_ids),
            "source_record_count": len(source_indices),
            "gallery_record_count": len(gallery_indices),
            "query_count": len(fold["query_rows"]),
            "source_heldout_disjoint": not (source_ids & heldout_ids),
            "source_indices_valid": source_indices <= all_record_ids,
            "gallery_indices_valid": gallery_indices <= all_record_ids,
            "source_records_match_source_ids": source_record_ids == source_ids,
            "gallery_records_match_heldout_ids": gallery_record_ids == heldout_ids,
            "query_rows_match_gallery_heldout_and_have_cross_scene_positive": query_ok,
            "m0_counts_match_protocol_counts": m0_fold["counts"] == fold["counts"],
            "baseline_fold_binding_match": (
                base_fold["fold"] == fold["fold"]
                and base_fold["source_ids"] == fold["source_ids"]
                and base_fold["heldout_ids"] == fold["heldout_ids"]
                and m0_fold["initialization"]["signal_checkpoint_sha256"] == base_fold["checkpoint_sha256"]
                and m0_fold["initialization"]["signal_state_sha256"] == base_fold["training"]["final_state_sha256"]
            ),
        }
        ok = ok and all(
            row[key]
            for key in (
                "source_heldout_disjoint",
                "source_indices_valid",
                "gallery_indices_valid",
                "source_records_match_source_ids",
                "gallery_records_match_heldout_ids",
                "query_rows_match_gallery_heldout_and_have_cross_scene_positive",
                "m0_counts_match_protocol_counts",
                "baseline_fold_binding_match",
            )
        )
        fold_rows.append(row)
    return {
        "folds": fold_rows,
        "all_ok": ok,
        "total_protocol_records": len(protocol["records"]),
        "unique_protocol_identities": len({int(row["identity"]) for row in protocol["records"]}),
        "heldout_identity_union_count": len(set(all_heldout)),
        "heldout_identity_assignments": len(all_heldout),
        "heldout_identity_assignments_unique": len(set(all_heldout)) == len(all_heldout),
        "gallery_record_union_count": len(set(all_gallery_indices)),
        "gallery_record_assignments": len(all_gallery_indices),
        "gallery_record_assignments_unique": len(set(all_gallery_indices)) == len(all_gallery_indices),
    }


def source_binding_checks(manifest: dict, binding: dict, prelaunch_bytes: dict) -> dict:
    manifest_snapshots = manifest.get("actual_remote_source_snapshots", {})
    rows = []
    ok = True
    for entry in binding["changed_remote_byte_bindings"]:
        snapshot = pathlib.Path(manifest_snapshots[entry["path"]])
        digest = sha256_path(snapshot)
        row = {
            "path": entry["path"],
            "registered_matches_remote": bool(entry["registered_matches_remote"]),
            "local_lf_bytes_match_remote": bool(entry["local_lf_bytes_match_remote"]),
            "remote_matches_git_blob": bool(entry["remote_matches_git_blob"]),
            "ast_equal": bool(entry["ast_equal"]),
            "snapshot_sha256": digest,
            "snapshot_matches_remote_sha256": digest == entry["remote_sha256"],
            "local_sha256": entry["local_sha256"],
            "remote_sha256": entry["remote_sha256"],
        }
        ok = ok and row["local_lf_bytes_match_remote"] and row["remote_matches_git_blob"] and row["ast_equal"] and row["snapshot_matches_remote_sha256"]
        rows.append(row)
    return {
        "status": binding["status"],
        "original_config_sha256": binding["original_config_sha256"],
        "revised_config_sha256": binding["config_sha256"],
        "model_or_optimizer_or_gate_or_protocol_changes": bool(binding["model_or_optimizer_or_gate_or_protocol_changes"]),
        "runtime_fallback_added": bool(binding["runtime_fallback_added"]),
        "prelaunch_fail_before_process_creation": prelaunch_bytes["status"] == "PRELAUNCH_SOURCE_BINDING_FAIL_BEFORE_PROCESS_CREATION",
        "prelaunch_training_model_tensor_image_calls": int(prelaunch_bytes["training_model_tensor_image_calls"]),
        "prelaunch_optimizer_updates": int(prelaunch_bytes["optimizer_updates"]),
        "changed_rows": rows,
        "all_rows_lf_ast_remote_snapshot_ok": ok,
    }


def initialization_checks(summary: dict, baseline: dict) -> dict:
    rows = []
    ok = True
    for m0_fold, base_fold in zip(summary["folds"], baseline["folds"], strict=True):
        init = m0_fold["initialization"]
        train = m0_fold["training"]
        row = {
            "fold": int(m0_fold["fold"]),
            "role_weights_loaded": bool(init["role_weights_loaded"]),
            "role_initialization_seed": int(init["role_initialization_seed"]),
            "signal_checkpoint_matches_baseline": init["signal_checkpoint_sha256"] == base_fold["checkpoint_sha256"],
            "signal_state_matches_baseline_final": init["signal_state_sha256"] == base_fold["training"]["final_state_sha256"],
            "training_initial_matches_initialization": train["initial_state_sha256"] == init["initial_state_sha256"],
            "role_state_updated": train["initial_state_sha256"] != train["final_state_sha256"],
            "frozen_state_unchanged": train["frozen_state_before_sha256"] == train["frozen_state_after_sha256"],
            "signal_state_unchanged": train["signal_state_before_sha256"] == train["signal_state_after_sha256"],
        }
        ok = ok and (not row["role_weights_loaded"]) and all(
            row[k] for k in row if k not in ("fold", "role_weights_loaded")
        )
        rows.append(row)
    overfit = summary["overfit"]
    overfit_row = {
        "role_weights_loaded": bool(overfit["initialization"]["role_weights_loaded"]),
        "role_initialization_seed": int(overfit["initialization"]["role_initialization_seed"]),
        "initial_state_equals_fold0_capacity_initial": overfit["initialization"]["initial_state_sha256"] == summary["folds"][0]["initialization"]["initial_state_sha256"],
        "final_state_differs_from_fold0_capacity_final": overfit["training"]["final_state_sha256"] != summary["folds"][0]["training"]["final_state_sha256"],
        "training_initial_matches_initialization": overfit["training"]["initial_state_sha256"] == overfit["initialization"]["initial_state_sha256"],
        "frozen_state_unchanged": overfit["training"]["frozen_state_before_sha256"] == overfit["training"]["frozen_state_after_sha256"],
        "signal_state_unchanged": overfit["training"]["signal_state_before_sha256"] == overfit["training"]["signal_state_after_sha256"],
    }
    ok = ok and (not overfit_row["role_weights_loaded"]) and all(
        value for key, value in overfit_row.items() if key != "role_weights_loaded"
    )
    return {
        "folds": rows,
        "overfit": overfit_row,
        "summary_m0_weights_reused": bool(summary["m0_weights_reused"]),
        "summary_rgbnt201_role_weights_reused": bool(summary["rgbnt201_role_weights_reused"]),
        "summary_checkpoint_selection": summary["checkpoint_selection"],
        "all_ok": ok and not summary["m0_weights_reused"] and not summary["rgbnt201_role_weights_reused"] and summary["checkpoint_selection"] == "discarded_m0",
    }


def receipt_checks(summary: dict, trainings: dict, receipts: dict) -> dict:
    rows = []
    ok = True
    for fold_index, receipt in receipts.items():
        summary_fold = summary["folds"][fold_index]
        training = trainings[f"capacity_fold{fold_index}"]
        row = {
            "fold": fold_index,
            "receipt_training_equals_independent_training_file": receipt["training"] == training,
            "summary_training_equals_independent_training_file": summary_fold["training"] == training,
            "receipt_counts_equal_protocol_summary": receipt["counts"] == summary_fold["counts"],
            "strict_reload_state_matches_final_state": receipt["strict_reload_state_sha256"] == training["final_state_sha256"],
            "direct_signal_source_parity": bool(receipt["direct_signal_source_parity"]),
            "strict_reload_all_outputs_bitwise_equal": bool(receipt["strict_reload_all_outputs_bitwise_equal"]),
            "clean_role_source_record_forwards": int(receipt["clean_role_source_record_forwards"]),
            "direct_signal_source_record_forwards": int(receipt["direct_signal_source_record_forwards"]),
            "heldout_record_forwards": int(receipt["heldout_record_forwards"]),
        }
        ok = ok and all(
            row[key]
            for key in (
                "receipt_training_equals_independent_training_file",
                "summary_training_equals_independent_training_file",
                "receipt_counts_equal_protocol_summary",
                "strict_reload_state_matches_final_state",
                "direct_signal_source_parity",
                "strict_reload_all_outputs_bitwise_equal",
            )
        ) and row["clean_role_source_record_forwards"] == 24 and row["direct_signal_source_record_forwards"] == 8 and row["heldout_record_forwards"] == 0
        rows.append(row)
    return {
        "folds": rows,
        "total_clean_role_source_record_forwards": sum(row["clean_role_source_record_forwards"] for row in rows),
        "total_direct_signal_source_record_forwards": sum(row["direct_signal_source_record_forwards"] for row in rows),
        "total_heldout_record_forwards": sum(row["heldout_record_forwards"] for row in rows),
        "all_ok": ok,
    }


def engineering_arithmetic(summary: dict, trainings: dict, config: dict, file_verification: dict) -> dict:
    capacity_steps = sum(int(trainings[f"capacity_fold{i}"]["optimizer_steps"]) for i in range(3))
    overfit_steps = int(trainings["overfit_fold0"]["optimizer_steps"])
    total_steps = capacity_steps + overfit_steps
    capacity_exposures = {str(i): int(trainings[f"capacity_fold{i}"]["optimizer_steps"]) * 64 for i in range(3)}
    overfit_exposures = overfit_steps * 64
    elapsed_training_sum = sum(float(trainings[f"capacity_fold{i}"]["history"][0]["elapsed_seconds"]) for i in range(3))
    elapsed_training_sum += float(trainings["overfit_fold0"]["history"][0]["elapsed_seconds"])
    peak_allocated = max(float(t["peak_allocated_mib"]) for t in trainings.values())
    peak_reserved = max(float(t["peak_reserved_mib"]) for t in trainings.values())
    all_gradient_live = all(int(t["trainable_tensors"]) == int(t["nonzero_gradient_tensors"]) and not t["missing_nonzero_gradients"] for t in trainings.values())
    overflow_events = sum(int(t["overflow_events"]) for t in trainings.values())
    return {
        "configured_capacity_steps_per_fold": int(config["OPTIMIZATION"]["CAPACITY_STEPS_PER_FOLD"]),
        "configured_overfit_steps": int(config["OPTIMIZATION"]["OVERFIT_STEPS"]),
        "capacity_steps": capacity_steps,
        "overfit_steps": overfit_steps,
        "total_optimizer_steps": total_steps,
        "summary_optimizer_steps": int(summary["optimizer_steps"]),
        "total_exposures": total_steps * 64,
        "capacity_exposures": capacity_exposures,
        "overfit_exposures": overfit_exposures,
        "summary_elapsed_seconds": float(summary["elapsed_seconds"]),
        "sum_training_epoch_elapsed_seconds": elapsed_training_sum,
        "max_peak_allocated_mib": peak_allocated,
        "max_peak_reserved_mib": peak_reserved,
        "under_24gib_reserved": peak_reserved < 24 * 1024,
        "all_trainable_gradients_live": all_gradient_live,
        "overflow_events": overflow_events,
        "summary_heldout_record_forwards": int(summary["heldout_record_forwards"]),
        "file_verification_checkpoint_count": sum(1 for name in file_verification["files"] if name.endswith("roles_m0.pth")),
        "file_verification_all_three_checkpoints_hashed": bool(file_verification["all_three_full_checkpoints_hashed"]),
        "file_verification_independent_fold_jsons_equal_summary": bool(file_verification["independent_fold_jsons_equal_summary"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=pathlib.Path, default=RUN_DIR / "msvr310_trifusion_m0_replay_result.json")
    args = parser.parse_args()

    started = time.perf_counter()
    command = [sys.executable, pathlib.Path(__file__).as_posix(), "--output-json", args.output_json.as_posix()]

    manifest_obj = load_json(MANIFEST_PATH)
    manifest_check = verify_manifest()
    config = load_json(PROJECT_ROOT / CONFIG_REL)
    prelaunch = load_json(PROJECT_ROOT / PRELAUNCH_REL)
    protocol = load_json(PROJECT_ROOT / PROTOCOL_REL)
    summary = load_json(PROJECT_ROOT / M0_SUMMARY_REL)
    baseline = load_json(PROJECT_ROOT / BASELINE_REL)
    signal_closure = load_json(PROJECT_ROOT / SIGNAL_CLOSURE_REL)
    source_binding = load_json(PROJECT_ROOT / SOURCE_BINDING_REL)
    prelaunch_bytes = load_json(PROJECT_ROOT / PRELAUNCH_BYTES_REL)
    file_verification = load_json(PROJECT_ROOT / FILE_VERIFICATION_REL)
    run_meta = load_json(PROJECT_ROOT / RUN_META_REL)
    request = load_json(PROJECT_ROOT / REQUEST_REL)
    m0_exit = (PROJECT_ROOT / M0_EXIT_REL).read_text(encoding="utf-8").strip()
    m0_log_lines = (PROJECT_ROOT / M0_LOG_REL).read_text(encoding="utf-8").splitlines()
    trainings = {name: load_json(PROJECT_ROOT / path) for name, path in TRAINING_RELS.items()}
    receipts = {fold: load_json(PROJECT_ROOT / path) for fold, path in RECEIPT_RELS.items()}

    training_results = {}
    for name, training in trainings.items():
        fold_index = 0 if name.endswith("fold0") else 1 if name.endswith("fold1") else 2
        fixed_batch = name == "overfit_fold0"
        training_results[name] = training_checks(name, training, config, protocol["folds"][fold_index], protocol, fixed_batch=fixed_batch)

    floor = label_smoothing_floor(config, len(protocol["folds"][0]["source_ids"]))
    overfit_losses = [float(step["loss"]) for step in trainings["overfit_fold0"]["steps"]]
    initial_loss = overfit_losses[0]
    final_loss = overfit_losses[-1]
    initial_excess = initial_loss - floor["weighted_entropy_floor"]
    final_excess = final_loss - floor["weighted_entropy_floor"]
    loss_ratio = final_excess / initial_excess
    overfit_gate = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "minimum_loss": floor["weighted_entropy_floor"],
        "initial_excess_loss": initial_excess,
        "final_excess_loss": final_excess,
        "loss_ratio": loss_ratio,
        "maximum_loss_ratio": float(config["GATES"]["OVERFIT_MAX_EXCESS_RATIO"]),
        "passed": loss_ratio <= float(config["GATES"]["OVERFIT_MAX_EXCESS_RATIO"]),
        "matches_summary_loss_gate": all(
            abs(float(summary["overfit"]["loss_gate"][key]) - value) < 1e-12
            for key, value in {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "minimum_loss": floor["weighted_entropy_floor"],
                "initial_excess_loss": initial_excess,
                "final_excess_loss": final_excess,
                "loss_ratio": loss_ratio,
                "maximum_loss_ratio": float(config["GATES"]["OVERFIT_MAX_EXCESS_RATIO"]),
            }.items()
        ) and bool(summary["overfit"]["loss_gate"]["passed"]) == (loss_ratio <= float(config["GATES"]["OVERFIT_MAX_EXCESS_RATIO"])),
    }

    max_loss_diff = max(row["max_weighted_loss_abs_diff"] for row in training_results.values())
    max_epoch_diff = max(
        max(epoch["abs_diff"] for epoch in row["epoch_rechecks"])
        for row in training_results.values()
    )

    source_binding = source_binding_checks(manifest_obj, source_binding, prelaunch_bytes)
    protocols = protocol_checks(protocol, summary, baseline)
    initialization = initialization_checks(summary, baseline)
    receipts_checked = receipt_checks(summary, trainings, receipts)
    arithmetic = engineering_arithmetic(summary, trainings, config, file_verification)

    result = {
        "schema": "independent-msvr310-trifusion-m0-replay-v1",
        "generated_at": now_iso(),
        "python_executable": sys.executable,
        "command": command,
        "working_directory": os.getcwd(),
        "project_root": PROJECT_ROOT.as_posix(),
        "elapsed_seconds": time.perf_counter() - started,
        "input_manifest_sha256": sha256_path(MANIFEST_PATH),
        "manifest_check": manifest_check,
        "m0_exit_text": m0_exit,
        "m0_log_epoch_events": [line for line in m0_log_lines if '"event": "msvr310_role_epoch"' in line or '"event": "complete"' in line],
        "config_summary": {
            "seed": int(config["EXPERIMENT"]["SEED"]),
            "capacity_steps_per_fold": int(config["OPTIMIZATION"]["CAPACITY_STEPS_PER_FOLD"]),
            "overfit_steps": int(config["OPTIMIZATION"]["OVERFIT_STEPS"]),
            "retrieval_outputs": config["MODEL"]["RETRIEVAL_OUTPUTS"],
            "label_smoothing": float(config["LOSS"]["LABEL_SMOOTHING"]),
            "protocol_official_test_access": bool(config["PROTOCOL"]["OFFICIAL_TEST_ACCESS"]),
            "protocol_rgbnt201_dev_access": bool(config["PROTOCOL"]["RGBNT201_DEV_ACCESS"]),
            "protocol_m0_weights_reused": bool(config["PROTOCOL"]["M0_WEIGHTS_REUSED"]),
            "model_selection": config["PROTOCOL"]["MODEL_SELECTION"],
        },
        "prelaunch_config_sha256": sha256_path(PROJECT_ROOT / PRELAUNCH_REL),
        "revised_config_sha256": sha256_path(PROJECT_ROOT / CONFIG_REL),
        "m0_summary_status": summary["status"],
        "m0_summary_mode": summary["mode"],
        "m0_summary_evaluation_type": summary["evaluation_type"],
        "m0_summary_source_only_training": bool(summary["source_only_training"]),
        "m0_summary_official_test_image_access": int(summary["official_test_image_access"]),
        "m0_summary_rgbnt201_dev_image_access": int(summary["rgbnt201_dev_image_access"]),
        "training_recomputations": training_results,
        "label_smoothing_entropy_floor": floor,
        "overfit_gate_recomputed": overfit_gate,
        "protocol_checks": protocols,
        "source_binding_checks": source_binding,
        "initialization_checks": initialization,
        "receipt_checks": receipts_checked,
        "engineering_arithmetic": arithmetic,
        "scalar_recompute_summary": {
            "total_recomputed_training_steps": sum(row["steps"] for row in training_results.values()),
            "max_weighted_loss_abs_diff": max_loss_diff,
            "max_epoch_mean_abs_diff": max_epoch_diff,
            "all_component_keys_expected": all(not row["component_key_problems"] for row in training_results.values()),
        },
        "reviewer_provenance": {
            "run_meta_executor": run_meta.get("executor"),
            "run_meta_requested_reviewer_model": run_meta.get("requested_reviewer_model"),
            "run_meta_requested_reasoning_effort": run_meta.get("requested_reasoning_effort"),
            "request_tool": request.get("tool"),
            "request_model": request.get("arguments", {}).get("model"),
            "request_reasoning_effort": request.get("arguments", {}).get("reasoning_effort"),
            "dispatch_observation_present": (RUN_DIR / "dispatch_observation.json").is_file(),
            "same_family_type_a_only": True,
            "backend_independently_attested": False,
            "no_further_delegation_by_this_reviewer": True,
        },
        "signal_b0_prerequisite": {
            "status": baseline["status"],
            "audit_closure_status": signal_closure["status"],
            "audit_closure_engineering": signal_closure["engineering"],
            "baseline_optimizer_steps": int(baseline["optimizer_steps"]),
            "fold_checkpoint_sha256": [fold["checkpoint_sha256"] for fold in baseline["folds"]],
            "fold_final_state_sha256": [fold["training"]["final_state_sha256"] for fold in baseline["folds"]],
            "not_reaudited_here": True,
        },
        "limitations": {
            "model_tensor_image_execution": False,
            "remote_binary_and_image_direct_inspection": False,
            "official_test_or_unknown_identity_retrieval_result": False,
            "comparison_20_epoch_terminal_result_in_scope": False,
        },
    }

    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if manifest_check["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
