"""Additional complete stage/secondary-number checks; no artifact code execution."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
T = Path(__file__).resolve().parent
manifest = json.loads((T / "input_manifest.json").read_text(encoding="utf-8-sig"))
E = manifest["entries"]
D = {i: json.loads(Path(e["path"]).read_text(encoding="utf-8-sig")) for i, e in enumerate(E) if e["source_relative_path"].endswith(".json")}
read = lambda name: json.loads((T / name).read_text(encoding="utf-8"))
checks = []
def check(name, value):
    checks.append({"check": name, "pass": bool(value)})
def log_json(i):
    return [json.loads(line) for line in Path(E[i]["path"]).read_text(encoding="utf-8").splitlines() if line.startswith("{")]

m0 = read("audit_M0_replay.json")
cap = read("audit_capture_replay.json")
grads = read("audit_gradient_replay.json")
chain = read("audit_numerical_chain_replay.json")
for actual, derived in zip(D[126]["folds"], m0["folds"], strict=True):
    expect = {"fold": derived["fold"], "updates": len(derived["steps"]), "source_exposures": derived["records_forwarded"], "source_identity_count_exposed": derived["unique_identities"], "source_record_count_exposed": derived["unique_records"], "same_identity_pairs": derived["same_identity_pairs"], "cross_camera_pairs": derived["cross_camera_pairs"], "total_parameters": derived["total_parameters"], "trainable_parameters": derived["trainable_parameters"], "peak_allocated_mib": derived["peak_allocated_mib"], "training_8_steps_seconds": derived["training_seconds"], "mean_loss": derived["mean_loss"], "initial_parameter_group_lrs": derived["initial_lrs"], "epoch1_actual_lrs": derived["actual_epoch1_lrs"], "epoch1_common_noise_factor": derived["common_multipliers"][0]}
    check(f"M0_secondary_fold{derived['fold']}_every_reported_quantity", actual == expect)
check("M0_secondary_max_double_error", D[126]["max_loss_composition_difference"] == max(f["max_double_recomposition_error"] for f in m0["folds"]))
check("M0_secondary_zero_mean_error", D[126]["max_epoch_mean_difference"] == 0)
check("capture_secondary_update_forward_counts", D[105]["attempted_steps"] == cap["attempts"] and D[105]["successful_updates"] == cap["successful_updates"] and D[105]["source_record_exposures"] == cap["exposure"]["records_forwarded"])
check("capture_secondary_M0_scalar_index_comparisons", D[105]["m0_first8_equal_scalars_count"] == sum(r["all_recorded_values_exactly_match_m0"] for r in cap["first8_comparison"]) and D[105]["m0_first8_equal_indices"] == all(r["sample_indices_exact"] for r in cap["first8_comparison"]))
capture_rows = [json.loads(line) for line in Path(E[44]["path"]).read_text().splitlines()]
double_grouping = [{"step": r["step"], "sum_then_subtract_loss": abs(sum(r["id_triplet_head_losses"]) + .1 * r["gram_loss"] + .1 * r["patch_loss"] - r["loss"]), "successively_subtract_components": abs(r["loss"] - sum(r["id_triplet_head_losses"]) - .1 * r["gram_loss"] - .1 * r["patch_loss"])} for r in capture_rows]
reported_error = D[105]["maximum_loss_recomposition_error"]
alternate_error = max(r["successively_subtract_components"] for r in double_grouping)
check("capture_secondary_error_matches_successive_subtraction", reported_error == alternate_error)
for g in grads:
    if g["source"] == E[42]["source_relative_path"]:
        check("capture_secondary_all_gradient_counts", g["tensors"] == D[105]["gradient_tensors"] and g["nonfinite_tensors"] == D[105]["nonfinite_gradient_tensors"] and g["nonfinite_only_clip_encoder"] == D[105]["nonfinite_only_clip_encoder"])
m0_events = [r for r in log_json(72) if r.get("event") == "signal_source_epoch"]
check("all_three_M0_epoch_log_lines_match_full_training_JSON", [{k: v for k, v in r.items() if k != "event"} for r in m0_events] == [D[65 + 2 * f]["history"][0] for f in range(3)])
check("M0_complete_log_fields", log_json(72)[-1] == {"event": "complete", "status": D[71]["status"], "elapsed_seconds": D[71]["elapsed_seconds"]})
check("baseline_has_no_complete_epoch_log_event", not any(r.get("event") == "signal_source_epoch" for r in log_json(52)))
check("T0_entire_json_log_exact", log_json(77)[-1] == D[76])
check("capture_overflow_log_exact", log_json(45)[-1] == {"event": "diagnostic_overflow_capture", "step": 34, "scale_before": 256.0, "scale_after": 128.0})
for txt, term in ((53, 55), (59, 61), (46, 48), (80, 84), (87, 94)):
    check(f"exit_file_{txt}_terminal_{term}_agreement", int(Path(E[txt]["path"]).read_text()) == D[term]["exit_code"])
check("t0_m0_exit_success", all(int(Path(E[i]["path"]).read_text()) == 0 for i in (73, 78)))

stages = [
 {"stage": "R1_T0", "classification": "DATA_PROTOCOL_CHECK_PLUS_SYNTHETIC_FIXTURES", "status": D[76]["status"], "project_commit": D[76]["project_commit"], "elapsed_seconds": D[76]["elapsed_seconds"], "optimizer_updates": 0, "source_record_model_forwards": 0, "image_decodes_reported": 17350, "real_retrieval": False, "evidence": [f"{E[76]['source_relative_path']}:1", f"{E[152]['source_relative_path']}:33"]},
 {"stage": "R1_M0", "classification": "THREE_PARTIAL_EPOCH_ENGINEERING_PREFLIGHTS", "status": D[71]["status"], "project_commit": D[71]["project_commit"], "elapsed_seconds": D[71]["elapsed_seconds"], "optimizer_updates": 24, "training_record_forwards": 1536, "clean_source_record_model_forwards": m0["clean_source_feature_forwards"], "heldout_record_model_forwards": 0, "real_retrieval": False, "evidence": [f"{E[71]['source_relative_path']}:1", f"{E[17]['source_relative_path']}:209"]},
 {"stage": "R1_M0_remote_checkpoint_verification", "classification": "REMOTE_SAVED_FILE_CONTENT_RECEIPT", "status": D[123]["status"], "project_commit": D[123]["project_commit"], "elapsed_seconds": D[123]["elapsed_seconds"], "optimizer_updates": 0, "source_record_model_forwards": 0, "remote_checkpoint_files_claimed": 3, "locally_loaded_checkpoints": 0, "evidence": [f"{E[123]['source_relative_path']}:102", f"{E[74]['source_relative_path']}:31"]},
 {"stage": "R1_B0", "classification": "FORMAL_BASELINE_ATTEMPT_WITH_ENGINEERING_STOP", "status": "ENGINEERING_STOP_AMP_OVERFLOW", "project_commit": D[54]["project_commit"], "exit_code": D[55]["exit_code"], "elapsed_seconds": D[55]["elapsed_seconds"], "completed_epochs": 0, "optimizer_updates": "UNKNOWN_NOT_PERSISTED", "record_model_forwards": "UNKNOWN_NOT_PERSISTED", "real_retrieval": False, "evidence": [f"{E[52]['source_relative_path']}:18", f"{E[108]['source_relative_path']}:4"]},
 {"stage": "source_capture", "classification": "SEPARATE_DIAGNOSTIC_TRAINING_TRAJECTORY", "status": D[42]["status"], "project_commit": D[42]["project_commit"], "exit_code": D[48]["exit_code"], "elapsed_seconds": D[48]["elapsed_seconds"], "attempted_steps": cap["attempts"], "optimizer_updates": cap["successful_updates"], "training_record_forwards": cap["exposure"]["records_forwarded"], "real_retrieval": False, "evidence": [f"{E[44]['source_relative_path']}:1", f"{E[44]['source_relative_path']}:34"]},
 {"stage": "saved_batch_probes", "classification": "THREE_SINGLE_BATCH_NUMERICAL_DIAGNOSTICS", "status": D[39]["status"], "project_commit": D[39]["project_commit"], "exit_codes": [r["exit_code"] for r in D[38]], "elapsed_seconds": D[39]["elapsed_seconds"], "optimizer_updates": 0, "source_record_model_forwards": sum(D[i]["source_record_forwards"] for i in (29, 32, 34)), "image_decodes": 0, "real_retrieval": False, "evidence": [f"{E[39]['source_relative_path']}:13", f"{E[33]['source_relative_path']}:18"]},
 {"stage": "local_Gram_FP32", "classification": "REAL_INPUT_OPERATOR_REGRESSION_FAILED_BEFORE_MODEL_GATE", "status": "FAIL_OPERATOR_FINITE_GATE", "project_commit": D[82]["project_commit"], "exit_code": D[84]["exit_code"], "elapsed_seconds": D[84]["elapsed_seconds"], "optimizer_updates": 0, "source_record_model_forwards": 0, "operator_backwards": 2, "real_retrieval": False, "evidence": [f"{E[82]['source_relative_path']}:87", f"{E[150]['source_relative_path']}:102"]},
 {"stage": "stable_Gram_floor", "classification": "ONE_REAL_INPUT_OPERATOR_AND_SAVED_BATCH_ENGINEERING_REGRESSION", "status": D[92]["status"], "project_commit": D[92]["project_commit"], "exit_code": D[94]["exit_code"], "elapsed_seconds": D[94]["elapsed_seconds"], "optimizer_updates": 0, "source_record_model_forwards": D[92]["full_batch"]["source_record_forwards"], "operator_backwards": D[92]["operator_backward_passes"], "full_model_backwards": D[92]["full_model_backward_passes"], "real_retrieval": False, "evidence": [f"{E[92]['source_relative_path']}:153", f"{E[151]['source_relative_path']}:126"]},
 {"stage": "R2_T0_then_complete_epoch_M0", "classification": "REGISTERED_ENGINEERING_WRAPPER_WITH_LAUNCH_RECEIPT_ONLY", "status": "LAUNCH_OBSERVED_RESULTS_NOT_IN_SNAPSHOT", "project_commit": D[130]["project_commit"], "observed_at": D[130]["observed_at"], "wrapper_pid_observed": D[130]["wrapper_pid"], "individual_T0_and_M0_status": "UNKNOWN_FROM_SNAPSHOT", "optimizer_updates": "UNKNOWN_FROM_SNAPSHOT", "real_retrieval": False, "evidence": [f"{E[130]['source_relative_path']}:1", f"{E[131]['source_relative_path']}:15"]},
 {"stage": "full_terminal_verifiers", "classification": "SOURCE_PREPARATION_RECEIPT_ONLY", "status": D[134]["status"], "executed": False, "source_files_in_manifest": False, "evidence": [f"{E[134]['source_relative_path']}:3"]}
]
timings = []
for launch, terminal in ((60, 61), (54, 55), (47, 48), (37, 39), (81, 84), (88, 94)):
    seconds = (datetime.fromisoformat(D[terminal]["completed_at"]) - datetime.fromisoformat(D[launch]["launched_at"])).total_seconds()
    timings.append({"launch": E[launch]["source_relative_path"], "terminal": E[terminal]["source_relative_path"], "wall_clock_interval_seconds": seconds, "recorded_perf_counter_seconds": D[terminal]["elapsed_seconds"], "wall_minus_perf_seconds": seconds - D[terminal]["elapsed_seconds"]})
check("R2_launch_precedes_request_snapshot", datetime.fromisoformat(D[130]["observed_at"]) < datetime.fromisoformat(json.loads((T / "001-rgbnt100-gram-engineering.request.json").read_text(encoding="utf-8"))["created_at"]))
result = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks, "all_checks_pass": all(c["pass"] for c in checks), "stage_ledger": stages, "wall_vs_perf_timing_arithmetic": timings, "original_B0_attempt_does_not_inherit_diagnostic_steps": True, "all_capture_double_error_parenthesizations": double_grouping, "capture_secondary_reported_error": reported_error, "capture_sum_then_subtract_loss_max_error": cap["max_double_recomposition_error"], "capture_successive_subtraction_max_error": alternate_error}
(T / "audit_stage_replay.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
out = json.dumps({"checks": len(checks), "all_checks_pass": result["all_checks_pass"], "failed_checks": [c for c in checks if not c["pass"]], "stage_count": len(stages), "capture_double_error_parenthesizations": {"reported": reported_error, "sum_then_subtract_loss": cap["max_double_recomposition_error"], "successive_subtraction": alternate_error}, "timings": timings}, indent=2)
(T / "audit_stage_replay.stdout.txt").write_text(out + "\n", encoding="utf-8")
print(out)
