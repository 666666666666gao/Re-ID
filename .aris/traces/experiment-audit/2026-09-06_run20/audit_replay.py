"""Independent immutable-artifact replay using only the Python standard library.

This script does not import any input module, access remote systems, open images,
load tensors, run models, or evaluate saved retrieval features/distances.
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8")
TRACE = Path(__file__).resolve().parent
MANIFEST = TRACE / "input_manifest.json"
EXPECTED = "b83e0121166effe4e6e84940ade7ab593610846f93634f9bca1b195a9c837e69"
entries = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["entries"]
source = {i: e["source_relative_path"] for i, e in enumerate(entries)}
raw = {i: Path(e["path"]).read_bytes() for i, e in enumerate(entries)}
texts = {i: b.decode("utf-8-sig") for i, b in raw.items()}
sha = lambda b: hashlib.sha256(b).hexdigest()
hashes = {i: sha(b) for i, b in raw.items()}
data = {i: json.loads(t) for i, t in texts.items() if source[i].endswith(".json")}
data[44] = [json.loads(line) for line in texts[44].splitlines()]
trees = {i: ast.parse(t, filename=source[i]) for i, t in texts.items() if source[i].endswith(".py")}
checks = []
def check(label, value, detail=None):
    checks.append({"check": label, "pass": bool(value), "detail": detail})

def save(name, obj):
    (TRACE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def walk(obj, pointer=""):
    yield pointer, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, pointer + "/" + str(k).replace("~", "~0").replace("/", "~1"))
    elif isinstance(obj, list):
        for k, v in enumerate(obj):
            yield from walk(v, pointer + "/" + str(k))

check("manifest_sha256", sha(MANIFEST.read_bytes()) == EXPECTED)
check("all_153_initial_hashes_and_sizes", all(hashes[i] == e["sha256"] and len(raw[i]) == e["bytes"] for i, e in enumerate(entries)))
numeric_inventory = []
hash_references = []
loss_occurrences = []
gradient_occurrences = []
for i, obj in data.items():
    numbers = [(p, v) for p, v in walk(obj) if type(v) in (int, float)]
    nonfinite = [p for p, v in numbers if not math.isfinite(v)]
    numeric_inventory.append({"source": source[i], "numeric_leaves": len(numbers), "nonfinite_json_numbers": nonfinite})
    for pointer, value in walk(obj):
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            hash_references.append({"source": source[i], "pointer": pointer, "sha256": value,
                                    "matching_manifest_sources": [source[j] for j, h in hashes.items() if h == value]})
        if isinstance(value, dict) and "id_triplet_head_losses" in value:
            loss_occurrences.append((i, pointer, value))
        if isinstance(value, list) and value and isinstance(value[0], dict) and "nan_count" in value[0]:
            gradient_occurrences.append((i, pointer, value))
check("all_JSON_numeric_values_finite", all(not r["nonfinite_json_numbers"] for r in numeric_inventory))
save("audit_numeric_inventory.json", numeric_inventory)
save("audit_hash_references.json", hash_references)

# Full protocol replay: every file-name record, identity-camera count, partition,
# source label, query row, positive count, and exclusion count. No images/distances.
protocol = data[136]
records = protocol["records"]
ids = sorted({r["identity"] for r in records})
by_id = Counter(r["identity"] for r in records)
by_camera = Counter((r["identity"], r["camera"]) for r in records)
record_checks = []
for idx, r in enumerate(records):
    match = re.fullmatch(r"rgbir/bounding_box_train/(\d{4})_c(\d{4})_(\d{3})\.jpg", r["path"])
    parsed = tuple(map(int, match.groups())) if match else None
    good = bool(parsed and r["index"] == idx and parsed == (r["identity"], r["camera"] + 1, r["frame"]) and r["view"] == -1 and r["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", r["sha256"]))
    record_checks.append({"index": idx, "pass": good, "identity": r["identity"], "camera": r["camera"]})
check("all_8675_record_metadata", all(r["pass"] for r in record_checks))
check("record_paths_unique", len({r["path"] for r in records}) == len(records))
check("identity_camera_counts_exact", protocol["identity_camera_counts"] == {str(pid): {str(c): n for (p, c), n in sorted(by_camera.items()) if p == pid} for pid in ids})
computed_counts = {"identities": len(ids), "cross_camera_identities": sum(len({c for p, c in by_camera if p == pid}) > 1 for pid in ids), "single_camera_identities": sum(len({c for p, c in by_camera if p == pid}) == 1 for pid in ids), "gallery_records": len(records), "query_records": len(records)}
check("protocol_top_counts", computed_counts == protocol["counts"])
fold_details, query_checks = [], []
all_galleries, all_sources = [], []
for f in protocol["folds"]:
    fid = f["fold"]
    held_ids = ids[fid::3]
    source_ids = [pid for pid in ids if pid not in held_ids]
    src = [r["index"] for r in records if r["identity"] in source_ids]
    gallery = [r["index"] for r in records if r["identity"] in held_ids]
    check(f"fold{fid}_round_robin_ids", f["heldout_ids"] == held_ids and f["source_ids"] == source_ids)
    check(f"fold{fid}_partition", f["source_record_indices"] == src and f["gallery_record_indices"] == gallery and set(src).isdisjoint(gallery) and set(src) | set(gallery) == set(range(len(records))))
    check(f"fold{fid}_label_and_camera_maps", f["source_label_map"] == {str(p): j for j, p in enumerate(source_ids)} and f["source_camera_values"] == sorted({records[j]["camera"] for j in src}))
    expected_queries = []
    for position, idx in enumerate(gallery):
        r = records[idx]
        positives = by_id[r["identity"]] - by_camera[(r["identity"], r["camera"])]
        excluded = by_camera[(r["identity"], r["camera"])]
        retained_same_camera_negatives = sum(by_camera[(p, r["camera"])] for p in held_ids if p != r["identity"])
        if positives:
            expected_queries.append({"record_index": idx, "gallery_position": position, "identity": r["identity"], "valid_positive_count": positives})
            query_checks.append({"fold": fid, "record_index": idx, "positive_count": positives, "excluded_same_id_and_camera": excluded, "retained_same_camera_negatives": retained_same_camera_negatives, "valid_gallery_count": len(gallery) - excluded})
    check(f"fold{fid}_every_query_exact", f["query_rows"] == expected_queries)
    counts = {"source_identities": len(source_ids), "source_records": len(src), "heldout_identities": len(held_ids), "gallery_records": len(gallery), "query_records": len(expected_queries), "query_identities": len({q["identity"] for q in expected_queries})}
    check(f"fold{fid}_counts", counts == f["counts"])
    fold_details.append({"fold": fid, **counts})
    all_galleries.extend(gallery)
    all_sources.extend(src)
check("all_gallery_queries_exactly_once", sorted(all_galleries) == list(range(len(records))))
check("each_record_source_in_two_folds", set(Counter(all_sources).values()) == {2} and len(Counter(all_sources)) == len(records))
save("audit_protocol_replay.json", {"counts": computed_counts, "folds": fold_details, "record_checks": record_checks, "query_checks": query_checks, "image_byte_checks_performed": 0})

# Historical selected-source bindings; the R1 runner must resolve to its archive.
bindings = []
for cfg_idx in (19, 18):
    cfg = data[cfg_idx]
    for family in ("signal_source_file_sha256", "project_source_file_sha256"):
        for name, expected in cfg[family].items():
            candidates = [i for i, path in source.items() if (path == name or path.endswith("/" + name))]
            if cfg_idx == 19 and name == "tools/train_rgbnt100_signal_oof.py":
                candidates = [17]
            matching = [i for i in candidates if hashes[i] == expected]
            status = "MATCH" if matching else "ABSENT_FROM_MANIFEST" if not candidates else "HASH_MISMATCH"
            bindings.append({"config": source[cfg_idx], "family": family, "bound_path": name, "expected_sha256": expected, "status": status, "resolved": [source[i] for i in matching]})
check("all_present_selected_sources_match_historical_binding", all(x["status"] != "HASH_MISMATCH" for x in bindings))
check("protocol_binds_same_hash_in_both_configs", all(data[i]["protocol_sha256"] == hashes[136] for i in (18, 19)))
save("audit_source_bindings.json", bindings)
trees_by_name = {i: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))} for i, tree in trees.items()}
registered_ast = data[132]["data_and_retrieval_functions_ast_identical_to_r1"]
ast_details = [{"name": name, "equal": trees_by_name[17][name] == trees_by_name[149][name]} for name in registered_ast]
check("registered_AST_comparisons", all(r["equal"] for r in ast_details))
all_functions = sorted(set(trees_by_name[17]) | set(trees_by_name[149]))
save("audit_AST_comparison.json", {"registered_count": len(registered_ast), "registered": ast_details, "all_top_level_definitions": [{"name": name, "equal": trees_by_name[17].get(name) == trees_by_name[149].get(name)} for name in all_functions], "parsed_python_files": len(trees)})

f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
def loss_recompose(row):
    heads = row["id_triplet_head_losses"]
    total = 0.0
    for v in heads:
        total = f32(total + v)
    total = f32(total + f32(f32(.1) * row["gram_loss"]))
    total = f32(total + f32(f32(.1) * row["patch_loss"]))
    double = sum(heads) + .1 * row["gram_loss"] + .1 * row["patch_loss"]
    return {"recorded_loss": row["loss"], "components": heads, "gram_loss": row["gram_loss"], "patch_loss": row["patch_loss"], "double_recomposition": double, "double_minus_recorded": double - row["loss"], "sequential_float32_recomposition": total, "float32_exact": total == row["loss"]}

loss_results = [{"source": source[i], "pointer": pointer, **loss_recompose(row)} for i, pointer, row in loss_occurrences]
check("all_recorded_head_counts_four", all(len(r["id_triplet_head_losses"]) == 4 for _, _, r in loss_occurrences))
check("all_saved_loss_occurrences_f32_exact", all(r["float32_exact"] for r in loss_results))
save("audit_loss_replay.json", loss_results)

def exposures(rows, fold):
    details, indices = [], []
    for row in rows:
        ix = row["sampled_record_indices"]
        r = [records[j] for j in ix]
        same, cross = 0, 0
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                if r[a]["identity"] == r[b]["identity"]:
                    same += 1
                    cross += r[a]["camera"] != r[b]["camera"]
        good = len(ix) == 64 and set(ix) <= set(fold["source_record_indices"]) and sorted(Counter(x["identity"] for x in r).values()) == [8] * 8
        details.append({"step": row["step"], "pass": good, "unique_records": len(set(ix)), "identity_counts": dict(Counter(x["identity"] for x in r)), "same_identity_pairs": same, "cross_camera_pairs": cross})
        indices.extend(ix)
    return {"records_forwarded": len(indices), "unique_records": len(set(indices)), "unique_identities": len({records[j]["identity"] for j in indices}), "same_identity_pairs": sum(x["same_identity_pairs"] for x in details), "cross_camera_pairs": sum(x["cross_camera_pairs"] for x in details), "steps": details}

m0_folds, optimizer_details = [], []
for fid in range(3):
    receipt, training = data[64 + fid * 2], data[65 + fid * 2]
    fold = protocol["folds"][fid]
    check(f"m0_fold{fid}_receipt_training_summary", receipt == data[71]["folds"][fid] and receipt["training"] == training)
    check(f"m0_fold{fid}_partition_receipt", all(receipt[k] == fold[k] for k in ("fold", "source_ids", "heldout_ids", "counts")))
    check(f"m0_fold{fid}_8steps_1partialepoch", len(training["steps"]) == training["optimizer_steps"] == 8 and len(training["history"]) == training["epochs"] == 1 and all(s["step"] == j and s["epoch"] == 1 for j, s in enumerate(training["steps"], 1)))
    check(f"m0_fold{fid}_scale_success", all(s["amp_scale_before"] == s["amp_scale_after"] == 256 for s in training["steps"]) and training["overflow_events"] == 0)
    check(f"m0_fold{fid}_parameter_accounting", training["total_parameters"] - training["trainable_parameters"] == training["frozen_token_selection_parameters"] + 4 * 768 and training["frozen_token_selection_parameters"] == 3 * (512 * 512 + 512))
    check(f"m0_fold{fid}_gradient_names", training["trainable_tensors"] == training["gradient_tensors"] == len(training["optimizer_groups"]) == 195 and not training["trainable_without_gradient"])
    check(f"m0_fold{fid}_state_receipts", training["initial_state_sha256"] != training["final_state_sha256"] and training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"])
    exposure = exposures(training["steps"], fold)
    check(f"m0_fold{fid}_every_PK_source_batch", all(x["pass"] for x in exposure["steps"]))
    initial_lrs = set()
    for group in training["optimizer_groups"]:
        expected_lr = .0007 * (2 if "bias" in group["name"] else 1)
        if "base" in group["name"] and "adapter" not in group["name"]:
            expected_lr = .000005
        expected_decay = .0001
        optimizer_details.append({"fold": fid, **group, "lr_rule_exact": group["initial_lr"] == expected_lr, "warmup_initial_exact": group["lr"] == .1 * .0007, "decay_exact": group["weight_decay"] == expected_decay})
        initial_lrs.add(expected_lr)
    warm = .1 * .0007
    clean = sorted(warm + (v - warm) / 5 for v in initial_lrs)
    actual = training["history"][0]["learning_rates"]
    factors = [a / b for a, b in zip(actual, clean, strict=True)]
    # Reconstruct the stored author equation, not torch's unexecuted RNG.
    inferred_noise = (actual[0] - clean[0]) / clean[0]
    lr_equation = [b + b * inferred_noise for b in clean]
    check(f"m0_fold{fid}_lr_equation_exact", lr_equation == actual and abs(inferred_noise) < .67)
    average = mean(s["loss"] for s in training["steps"])
    check(f"m0_fold{fid}_mean_exact", average == training["history"][0]["mean_loss"])
    m0_folds.append({"fold": fid, **exposure, "mean_loss": average, "max_double_recomposition_error": max(abs(loss_recompose(s)["double_minus_recorded"]) for s in training["steps"]), "total_parameters": training["total_parameters"], "trainable_parameters": training["trainable_parameters"], "initial_lrs": sorted(initial_lrs), "warmup_clean_lrs": clean, "actual_epoch1_lrs": actual, "inferred_common_noise": inferred_noise, "common_multipliers": factors, "training_seconds": training["history"][0]["elapsed_seconds"], "peak_allocated_mib": receipt["peak_allocated_mib"]})
check("all_585_optimizer_groups", all(r["lr_rule_exact"] and r["warmup_initial_exact"] and r["decay_exact"] for r in optimizer_details))
check("M0_total_steps_and_forwards", data[71]["optimizer_steps"] == sum(len(x["steps"]) for x in m0_folds) == 24 and sum(x["records_forwarded"] for x in m0_folds) == 1536)
save("audit_M0_replay.json", {"folds": m0_folds, "optimizer_groups": optimizer_details, "clean_source_feature_forwards": sum(data[64 + j * 2]["clean_source_feature_forwards"] for j in range(3))})

capture = data[42]
steps = data[44]
check("capture_sequential_steps", [s["step"] for s in steps] == list(range(1, 35)) and all(s["epoch"] == 1 for s in steps))
check("capture_34attempt_33updates", len(steps) == capture["attempted_steps"] == 34 and sum(s["optimizer_update_applied"] for s in steps) == capture["successful_optimizer_steps"] == 33 and capture["failed_step"] == steps[-1])
check("capture_AMP_update_flags", all(s["optimizer_update_applied"] == (s["amp_scale_after"] >= s["amp_scale_before"]) for s in steps) and all(s["amp_scale_before"] == 256 for s in steps) and [s["amp_scale_after"] for s in steps] == [256] * 33 + [128])
capture_exposure = exposures(steps, protocol["folds"][0])
check("capture_all_PK_source_batches", all(x["pass"] for x in capture_exposure["steps"]))
check("capture_exposure_total", capture_exposure["records_forwarded"] == capture["source_record_exposures"] == 2176)
keys = ("loss", "id_triplet_head_losses", "gram_loss", "patch_loss", "sampled_record_indices", "amp_scale_before", "amp_scale_after")
comparison = [{"step": j + 1, "all_recorded_values_exactly_match_m0": all(s[k] == data[65]["steps"][j][k] for k in keys), "loss_difference": s["loss"] - data[65]["steps"][j]["loss"], "sample_indices_exact": s["sampled_record_indices"] == data[65]["steps"][j]["sampled_record_indices"]} for j, s in enumerate(steps[:8])]
check("capture_M0_comparison_fields_exact", comparison == capture["m0_first8_comparison"])
save("audit_capture_replay.json", {"attempts": len(steps), "successful_updates": sum(s["optimizer_update_applied"] for s in steps), "exposure": capture_exposure, "first8_comparison": comparison, "max_double_recomposition_error": max(abs(loss_recompose(s)["double_minus_recorded"]) for s in steps), "original_failed_successful_updates": "UNKNOWN_NOT_PERSISTED"})

# Every available gradient-count list, including duplicate embedded copies.
gradient_results = []
for i, pointer, rows in gradient_occurrences:
    counts, names, nonfinite_names = Counter(), [], []
    for r in rows:
        counts["elements"] += r["elements"]
        counts["nan"] += r["nan_count"]
        inf = r.get("inf_count", 0) + r.get("positive_inf_count", 0) + r.get("negative_inf_count", 0)
        counts["inf"] += inf
        name = r.get("name", r.get("input"))
        names.append(name)
        if r["nan_count"] + inf:
            nonfinite_names.append(name)
        check(f"gradient_count_bounds_{i}{pointer}/{name}", 0 <= r["nan_count"] + inf <= r["elements"])
    check(f"gradient_names_unique_{i}{pointer}", len(names) == len(set(names)))
    gradient_results.append({"source": source[i], "pointer": pointer, "tensors": len(rows), **dict(counts), "nonfinite_tensors": len(nonfinite_names), "nonfinite_names": nonfinite_names, "nonfinite_only_clip_encoder": bool(nonfinite_names) and all(n.startswith("clip_vision_encoder.") for n in nonfinite_names)})
check("capture_and_probe_all195_gradients_exact", data[42]["all_parameter_gradient_counts"] == data[30]["all_gradient_counts"])
for i in (30, 35):
    result = next(x for x in gradient_results if x["source"] == source[i])
    check(f"probe{i}_gradient_summary_fields", result["tensors"] == data[i]["gradient_tensors"] and result["nonfinite_names"] == data[i]["nonfinite_gradient_names"])
for i in (35, 90):
    result = next(x for x in gradient_results if x["source"] == source[i])
    check(f"finite_gradient_total_matches_M0_parameters_{i}", result["elements"] == data[65]["trainable_parameters"] and result["nan"] == result["inf"] == 0)
check("all_195_gradient_shapes_agree_across_modes", [(r["name"], r["elements"]) for r in data[30]["all_gradient_counts"]] == [(r["name"], r["elements"]) for r in data[35]["all_gradient_counts"]] == [(r["name"], r["elements"]) for r in data[90]])
save("audit_gradient_replay.json", gradient_results)

probe_keys = ("loss", "id_triplet_head_losses", "gram_loss", "patch_loss")
for i in (29, 30, 32):
    check(f"saved_FP16_forward_{i}_exact_capture", all(data[i][k] == steps[-1][k] for k in probe_keys))
check("full_FP32_forward_result_agree", all(data[35][k] == v for k, v in data[34].items()))
check("FP16_forward_result_agree", all(data[30][k] == v for k, v in data[29].items()))
check("probe_progress_terminal_agree", data[38] == data[39]["probes"])
check("probe_fixed_exit_codes", [(r["mode"], r["exit_code"]) for r in data[38]] == [("fp16", 0), ("fp16_anomaly", 1), ("fp32", 0)])
for i in (29, 30, 32, 34, 35):
    r = data[i]
    check(f"probe{i}_capture_difference_exact", r["capture_loss_difference"] == r["loss"] - steps[-1]["loss"] and r["capture_loss_exact"] == (r["loss"] == steps[-1]["loss"]))
    check(f"probe{i}_paired_volume_calls_identical", r["volume_calls"][0] == r["volume_calls"][1])
    for j, v in enumerate(r["volume_calls"]):
        check(f"probe{i}_volume{j}_coordinate_counts", v["shape"] == [64, 64] and len(v["zero_coordinates"]) == v["exact_zero_count"] and len(set(map(tuple, v["zero_coordinates"]))) == v["exact_zero_count"] and all(0 <= k < 64 for xy in v["zero_coordinates"] for k in xy) and 0 <= v["exact_zero_count"] + v["negative_count"] <= 4096)
check("stable_embedded_operator_results_exact", data[91]["operators"] == data[92]["operators"])
check("stable_fullbatch_embedded_exact", all(data[89][k] == v for k, v in data[92]["full_batch"].items()))
check("stable_same_fourheads_patch", all(data[89][k] == data[29][k] for k in ("id_triplet_head_losses", "patch_loss")))
check("stable_and_local_FP32_Gram_scalar_equal", data[91]["operators"][1]["gram_loss"] == data[82]["operators"][1]["gram_loss"] == data[89]["gram_loss"])
for i in (82, 91):
    check(f"operator{i}_AMP_control_loss", data[i]["operators"][0]["gram_loss"] == data[29]["gram_loss"])
    for op in data[i]["operators"]:
        check(f"operator{i}_{op['mode']}_paired_calls_exact", op["volume_calls"][0] == op["volume_calls"][1])
        for v in op["volume_calls"]:
            check(f"operator{i}_{op['mode']}_zero_count", len(v["zero_coordinates"]) == v["zero_determinants"] and len(set(map(tuple, v["zero_coordinates"]))) == v["zero_determinants"])
check("floor_float32_min_volume_exact", f32(math.sqrt(f32(1e-12))) == data[91]["operators"][1]["volume_calls"][0]["minimum_volume"])
save("audit_numerical_chain_replay.json", {"original_amp_loss": data[29]["loss"], "full_FP32_loss": data[34]["loss"], "full_FP32_loss_delta": data[34]["loss"] - data[29]["loss"], "stable_amp_loss": data[89]["loss"], "stable_amp_loss_delta": data[89]["loss"] - data[29]["loss"], "stable_Gram_delta": data[89]["gram_loss"] - data[29]["gram_loss"], "stable_weighted_Gram_delta": .1 * (data[89]["gram_loss"] - data[29]["gram_loss"]), "floor_min_volume_float32": f32(math.sqrt(f32(1e-12))), "operator_records": [{"source": source[i], "operators": data[i]["operators"]} for i in (82, 91)], "tensor_arrays_recomputed": False, "new_source_record_model_forwards": 0})

# Hand arithmetic for the two already-preserved synthetic ranking fixtures only.
# No data distances/features, artifact functions, or actual retrieval evaluation.
fixture_assign = next(n for n in ast.walk(trees[152]) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fixtures" for t in n.targets))
def literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return literal(node.left) / literal(node.right)
    if isinstance(node, ast.List):
        return [literal(n) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values, strict=True)}
    return ast.literal_eval(node)
fixtures = literal(fixture_assign.value)
synthetic = []
for f, saved in zip(fixtures, data[76]["camera_ranking_synthetic_fixtures"], strict=True):
    order = sorted(range(len(f["d"])), key=lambda j: f["d"][j])
    kept = [j for j in order if not (f["ids"][j] == 1 and f["cams"][j] == 0)]
    positions = [j + 1 for j, idx in enumerate(kept) if f["ids"][idx] == 1]
    ap = sum(k / pos for k, pos in enumerate(positions, 1)) / len(positions)
    check("synthetic_fixture_saved_AP_and_rank", ap == saved["actual_ap"] and positions[0] == saved["first_match_rank"] and f["ap"] == saved["expected_ap"])
    synthetic.append({"positive_ranks": positions, "derived_AP": ap, "expected_rational_as_float": f["ap"], "expected_minus_actual": f["ap"] - ap, "real_retrieval_result": False})
save("audit_synthetic_fixture_arithmetic.json", synthetic)

# All observed intake files and M0 file receipts checked where bytes are present.
intake_matches = []
for i, prefix in ((97, "evidence/rgbnt100_signal_v1_amp_batch_probe/"), (102, "evidence/rgbnt100_signal_v1_amp_capture/"), (108, "evidence/rgbnt100_signal_v1_baseline_failure/"), (114, "evidence/rgbnt100_signal_v1_gram_fp32_regression/"), (118, "evidence/rgbnt100_signal_v1_gram_stable_regression/"), (123, "evidence/rgbnt100_signal_v1_engineering/"), (125, "evidence/rgbnt100_signal_v1_engineering/")):
    file_entries = data[i]["copied_files"] if i == 108 else data[i]["files"]
    if i == 125:
        file_entries = [{"path": r["remote"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in file_entries]
    if isinstance(file_entries, dict):
        file_entries = [{"path": path, **record} for path, record in file_entries.items()]
    for item in file_entries:
        candidates = [j for j, path in source.items() if path == prefix + item["path"]]
        status = "ABSENT_FROM_MANIFEST"
        if candidates:
            j = candidates[0]
            status = "MATCH" if hashes[j] == item["sha256"] and len(raw[j]) == item["bytes"] else "MISMATCH"
        intake_matches.append({"intake": source[i], "path": item["path"], "status": status})
check("all_available_intake_files_exact", all(r["status"] != "MISMATCH" for r in intake_matches))
save("audit_intake_hash_replay.json", intake_matches)

# Explicit provenance references where target meaning is known (not just hash lookup).
references = [(76, "config_sha256", 19), (76, "protocol_sha256", 136), (76, "runner_sha256", 17), (76, "verifier_sha256", 152), (71, "config_sha256", 19), (71, "protocol_sha256", 136), (71, "protocol_receipt_sha256", 76), (42, "diagnostic_sha256", 143), (42, "runner_sha256", 17), (42, "config_sha256", 19), (42, "protocol_sha256", 136), (42, "m0_receipt_sha256", 71), (29, "probe_sha256", 144), (29, "plan_sha256", 99), (82, "plan_sha256", 116), (92, "plan_sha256", 120), (132, "config_sha256", 18), (132, "runner_sha256", 149), (132, "helper_sha256", 148), (132, "plan_sha256", 137), (132, "original_runner_sha256", 17), (132, "prior_stable_regression_summary_sha256", 92), (130, "wrapper_sha256", 131), (123, "verifier_sha256", 74), (123, "summary_sha256", 71), (133, "config_sha256", 19), (133, "runner_sha256", 17), (133, "protocol_verifier_sha256", 152), (133, "plan_sha256", 138)]
for a, key, b in references:
    check(f"specific_hash_reference_{a}_{key}_{b}", data[a][key] == hashes[b])
wrapper_pairs = [(41, 101), (50, 106), (57, 111), (63, 113), (74, 124), (86, 117), (96, 121)]
for a, b in wrapper_pairs:
    check(f"wrapper_copy_exact_{a}_{b}", raw[a] == raw[b])

final_hash_rows = [{"source": e["source_relative_path"], "bytes": len(Path(e["path"]).read_bytes()), "sha256": sha(Path(e["path"]).read_bytes()), "expected_sha256": e["sha256"]} for e in entries]
final_hash_pass = sha(MANIFEST.read_bytes()) == EXPECTED and all(r["sha256"] == e["sha256"] and r["bytes"] == e["bytes"] for r, e in zip(final_hash_rows, entries, strict=True))
check("all_manifest_hashes_after_replay", final_hash_pass)
save("audit_hash_recheck.json", {"checked_at": datetime.now().astimezone().isoformat(), "manifest_sha256": sha(MANIFEST.read_bytes()), "files": len(final_hash_rows), "total_bytes": sum(r["bytes"] for r in final_hash_rows), "pass": final_hash_pass, "entries": final_hash_rows})
save("audit_replay_checks.json", checks)
overview = {"script_sha256": sha(Path(__file__).read_bytes()), "files_read": len(raw), "bytes_read": sum(map(len, raw.values())), "JSON_files": len(data) - 1, "JSONL_rows": len(data[44]), "AST_parsed_python_files": len(trees), "numeric_JSON_leaves": sum(r["numeric_leaves"] for r in numeric_inventory), "loss_occurrences_replayed_including_duplicates": len(loss_results), "all_float32_loss_recompositions_exact": all(r["float32_exact"] for r in loss_results), "independent_training_steps": 24 + len(steps), "gradient_record_lists_including_duplicates": len(gradient_results), "gradient_rows_examined_including_duplicates": sum(r["tensors"] for r in gradient_results), "selected_source_bindings": dict(Counter(r["status"] for r in bindings)), "intake_file_bindings": dict(Counter(r["status"] for r in intake_matches)), "registered_AST_count": len(registered_ast), "checks": len(checks), "passed_checks": sum(r["pass"] for r in checks), "failed_checks": [r for r in checks if not r["pass"]], "hash_recheck_pass": final_hash_pass, "local_torch_imports_tensor_loads_image_decodes_model_training_retrieval_SSH_calls": 0}
save("audit_replay_summary.json", overview)
out = json.dumps(overview, indent=2, ensure_ascii=False)
(TRACE / "audit_replay.stdout.txt").write_text(out + "\n", encoding="utf-8")
print(out)
