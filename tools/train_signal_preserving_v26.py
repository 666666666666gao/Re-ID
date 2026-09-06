#!/usr/bin/env python3
"""Run the fixed V26 role-modal ranking responsibility M0 and complete paired Q1."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.audit_v17_full_gallery import full_gallery_scores
from tools.check_v26_responsibility_math import run as check_responsibility_math
from tools.build_v12_complete_path_oof_targets import (
    _configure_signal, _load_records, build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _training_batch, learning_rate_multiplier, load_raw_config
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256,
    _tensor_mapping_sha256, _trainable_names,
)
from tools.train_signal_preserving_v18 import image_batch, loader_for
from tools.train_signal_preserving_v19 import criterion_for, frozen_state_sha, seed_everything, source_bindings
from tools.train_signal_preserving_v22 import build_model
from tools.train_signal_preserving_v23 import new_optimizer
from trifusion.aligned_data import (
    AlignedTripletImageDataset, CrossCameraIdentitySampler, SharedGeometryTripletTransform,
)
from trifusion.role_modal_responsibility_v26 import (
    MODALITIES, responsibility_loss, same_block_gradient_diagnostics,
)
from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("control", "responsibility")


def load_contract(path):
    config = load_raw_config(path)
    assert config["MODEL"]["ARCHITECTURE"] == "signal_preserving_v26_role_modal_responsibility"
    assert config["EXPERIMENT"]["SEED"] == 42
    assert (config["DATA"]["TRAIN_BATCH_SIZE"], config["DATA"]["NUM_INSTANCES"]) == (64, 8)
    assert config["OPTIMIZATION"]["MAX_EPOCHS"] == 20
    assert config["PROTOCOL"]["Q1_ENDPOINTS"] == list(ENDPOINTS)
    assert not config["PROTOCOL"]["DEV_ACCESS_DURING_Q1"]
    assert not config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"]
    assert not config["PROTOCOL"]["RERANKING"]
    for source, expected in {**source_bindings(config), **config["SOURCE_FILE_SHA256"]}.items():
        assert _sha256(Path(source)) == expected, source
    assert _sha256(Path(config["SIGNAL"]["CLIP_WEIGHT"])) == config["SIGNAL"]["CLIP_WEIGHT_SHA256"]
    initial = config["INITIALIZATION"]
    summary = Path(initial["V12_RUN_SUMMARY"])
    assert _sha256(summary) == initial["V12_RUN_SUMMARY_SHA256"]
    sources = json.loads(summary.read_bytes())
    assert sources["status"] == "PASS" and len(sources["fold_receipts"]) == 3
    metadata = config["SUPERVISION_METADATA"]
    assert _sha256(Path(metadata["PATH"])) == metadata["SHA256"]
    replay = json.loads(Path(metadata["PATH"]).read_bytes())
    assert replay["status"] == "COMPLETE_METADATA_REPLAY_PASS" and replay["all_gates_pass"]
    assert replay["total_batches"] == 3360
    assert config["LOSS"]["RESPONSIBILITY_TEMPERATURE"] == 0.1
    assert config["LOSS"]["RESPONSIBILITY_WEIGHT"] == 1.0
    for fold in replay["folds"]:
        assert fold["arms"]["control"] == fold["arms"]["responsibility"]
    return config, sources


def training_loader(records, config, endpoint):
    from data.datasets.make_dataloader import train_collate_fn
    sampler_type = {
        "control": CrossCameraIdentitySampler,
        "responsibility": CrossCameraIdentitySampler,
    }[endpoint]
    return torch.utils.data.DataLoader(
        AlignedTripletImageDataset(records, transform=SharedGeometryTripletTransform()),
        batch_size=64,
        sampler=sampler_type(records, batch_size=64, num_instances=8, seed=42),
        num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def check_batch(raw, index, expected):
    paths = list(raw[-1])
    actual_indices = [index[path] for path in paths]
    assert actual_indices == expected["sampler_indices"]
    order_hash = hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest()
    assert order_hash == expected["sample_order_sha256"]
    labels, cameras = raw[1].tolist(), raw[2].tolist()
    assert sorted(Counter(labels).values()) == [8] * 8
    cross_pairs, cross_groups = 0, 0
    for offset, group in zip(range(0, 64, 8), expected["groups"], strict=True):
        assert labels[offset:offset + 8] == [group["identity"]] * 8
        counts = Counter(cameras[offset:offset + 8])
        assert counts == {int(k): v for k, v in group["camera_counts"].items()}
        cross_pairs += 64 - sum(value * value for value in counts.values())
        cross_groups += int(len(counts) > 1)
    assert cross_pairs == expected["directed_cross_camera_positive_pairs"]
    assert cross_groups == expected["cross_camera_identity_groups"]
    return {
        "sampler_indices": actual_indices, "paths": paths,
        "sample_order_sha256": order_hash,
        "cross_camera_identity_groups": cross_groups,
        "directed_cross_camera_positive_pairs": cross_pairs,
        "all_directed_positive_pairs": 448,
        "unique_source_records_in_batch": len(set(paths)),
        "same_record_positive_pair_exposures": sum(v * (v - 1) for v in Counter(paths).values()),
        "cross_camera_triplet_exposures": cross_pairs * 56,
    }



def step(model, criterion, raw, optimizer, scaler, config, *, enabled, diagnose_parameters=False):
    optimizer.zero_grad(set_to_none=True)
    batch, labels = _training_batch(raw)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
        assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
        parts = criterion(output, labels)
        weights = config["LOSS"]
        fused_objective = weights["ID_FUSED"] * parts["id_fused"] + weights["TRIPLET_FUSED"] * parts["triplet_fused"]
        common = fused_objective
        for expert in EXPERTS:
            common = common + weights["ID_BRANCH"] * parts[f"id_{expert}"] + weights["TRIPLET_BRANCH"] * parts[f"triplet_{expert}"]
            common = common + weights["ID_RESIDUAL"] * parts[f"id_residual_{expert}"]
        ordinary = weights["TRIPLET_RESIDUAL"] * sum(parts[f"triplet_residual_{expert}"] for expert in EXPERTS)
    base = common.float() + ordinary.float()
    auxiliary, stats = responsibility_loss(output, labels, temperature=weights["RESPONSIBILITY_TEMPERATURE"])
    assert stats["responsibility_triplets"] == 25088
    assert stats["responsibility_slot_triplets"] == 225792
    assert stats["responsibility_raw_slot_norm_max_error"] < 0.005
    assert stats["responsibility_fused_decomposition_max_error"] < 0.005
    weighted = weights["RESPONSIBILITY_WEIGHT"] * auxiliary
    total = base + weighted if enabled else base
    assert bool(torch.isfinite(total)) and bool(torch.isfinite(auxiliary))
    slot_gradients = torch.autograd.grad(
        scaler.scale(weighted), [output.modal_residual_embeddings[e] for e in EXPERTS], retain_graph=True
    )
    for expert, gradient in zip(EXPERTS, slot_gradients, strict=True):
        gradient = gradient.float() / scaler.get_scale()
        assert bool(torch.isfinite(gradient).all())
        for index, modality in enumerate(MODALITIES):
            stats[f"responsibility_{expert}_{modality}_gradient_norm"] = float(gradient[:, index].norm())
    parameter_diagnostics = {}
    if diagnose_parameters:
        parameter_diagnostics = same_block_gradient_diagnostics(
            model, fused_objective.float(), base - fused_objective.float(), weighted, scaler
        )
    values = {
        "total": float(total.detach()), "original_total": float(base.detach()),
        "common_identity_and_branch_triplet": float(common.detach()),
        "ordinary_residual_triplet": float(ordinary.detach()),
        "responsibility_loss": float(auxiliary.detach()), "responsibility_enabled": int(enabled),
        **{name: float(value.detach()) for name, value in parts.items()}, **stats,
    }
    scale = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    live = set()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
            if bool(parameter.grad.abs().sum() > 0):
                live.add(name)
    scaler.step(optimizer)
    scaler.update()
    return values, scaler.get_scale() < scale, live, parameter_diagnostics


def preflight(model, records, config, endpoint, expected):
    seed_everything()
    model.eval()
    before = _model_state_sha256(model)
    index = _record_index_by_path(records)
    receipts, outputs, relation_diagnostics = [], [], []
    iterator = iter(training_loader(records, config, endpoint))
    for number in range(8):
        raw = next(iterator)
        check_batch(raw, index, expected["batches"][number])
        receipts.append(_raw_batch_receipt(raw, record_index_by_path=index))
        batch, labels = _training_batch(raw)
        with torch.no_grad():
            output = model(batch, return_aux=True)
            auxiliary, stats = responsibility_loss(output, labels, temperature=config["LOSS"]["RESPONSIBILITY_TEMPERATURE"])
            assert bool(torch.isfinite(auxiliary))
            assert stats["responsibility_triplets"] == 25088
            assert stats["responsibility_raw_slot_norm_max_error"] < 0.005
            assert stats["responsibility_fused_decomposition_max_error"] < 0.005
            relation_diagnostics.append({"responsibility_loss": float(auxiliary), **stats})
        assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
        outputs.append(_tensor_mapping_sha256({
            "baseline": output.baseline_embedding, "fused": output.fused_embedding,
            **dict(output.branch_embeddings),
        }))
    assert before == _model_state_sha256(model)
    return {
        "initial_state_sha256": before, "batch_receipts": receipts,
        "all_output_sha256": outputs, "state_unchanged": True,
        "relation_diagnostics": relation_diagnostics,
        "all_eight_batches_match_own_registered_sampler": True,
    }


def fixed_steps(model, records, config, *, enabled, count, fixed, expected):
    seed_everything()
    model.train()
    frozen = frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    iterator = iter(training_loader(records, config, ENDPOINTS[int(enabled)]))
    index = _record_index_by_path(records)
    fixed_batch = next(iterator) if fixed else None
    losses, components, receipts, live, overflow = [], [], [], set(), 0
    parameter_diagnostics = []
    for number in range(count):
        raw = fixed_batch if fixed else next(iterator)
        if not fixed or number == 0:
            check_batch(raw, index, expected["batches"][0 if fixed else number])
            receipts.append(_raw_batch_receipt(raw, record_index_by_path=index))
        values, bad, names, gradients = step(model, criterion, raw, optimizer, scaler, config,
                                            enabled=enabled, diagnose_parameters=enabled and not fixed)
        if gradients:
            parameter_diagnostics.append({"step": number + 1, "experts": gradients})
        losses.append(values["total"])
        components.append(values)
        overflow += int(bad)
        live.update(names)
    return {
        "steps": count, "endpoint": ENDPOINTS[int(enabled)], "fixed_batch": fixed,
        "losses": losses, "components": components, "batch_receipts": receipts,
        "same_block_parameter_gradient_diagnostics": parameter_diagnostics,
        "trainable_tensors": len(_trainable_names(model)),
        "nonzero_gradient_tensors": len(live),
        "missing_nonzero_gradients": sorted(set(_trainable_names(model)) - live),
        "frozen_state_unchanged": frozen == frozen_state_sha(model),
        "overflow_events": overflow,
    }


def fit_endpoint(model, records, config, *, fold, endpoint, enabled, expected, directory):
    assert endpoint == ENDPOINTS[int(enabled)]
    seed_everything()
    model.train()
    initial, frozen = _model_state_sha256(model), frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    loader = training_loader(records, config, endpoint)
    index = _record_index_by_path(records)
    order = hashlib.sha256()
    history, batches, live, overflow, steps = [], [], set(), 0, 0
    exposures = Counter()
    cross_pairs = 0
    log_path = directory / f"fold_{fold}_{endpoint}_all_training_steps.jsonl"
    assert not log_path.exists()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(1, 21):
            lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * learning_rate_multiplier(
                epoch, max_epochs=20, warmup_epochs=5
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            rows = []
            for within_epoch, raw in enumerate(loader, 1):
                wanted = expected["batches"][steps]
                assert (wanted["epoch"], wanted["step"]) == (epoch, within_epoch)
                batch_receipt = check_batch(raw, index, wanted)
                order.update((json.dumps(list(raw[-1])) + "\n").encode())
                exposures.update(batch_receipt["sampler_indices"])
                cross_pairs += batch_receipt["directed_cross_camera_positive_pairs"]
                if steps < 8:
                    batches.append(_raw_batch_receipt(raw, record_index_by_path=index))
                values, bad, names, _ = step(model, criterion, raw, optimizer, scaler, config, enabled=enabled)
                rows.append(values)
                overflow += int(bad)
                live.update(names)
                steps += 1
                log.write(json.dumps({
                    "fold": fold, "endpoint": endpoint, "epoch": epoch, "step": within_epoch,
                    "optimizer_step": steps, "learning_rate": lr,
                    "overflow": bool(bad), **batch_receipt, "losses": values,
                }) + "\n")
            log.flush()
            row = {
                "fold": fold, "endpoint": endpoint, "epoch": epoch, "batches": len(rows),
                "learning_rate": lr,
                **{"mean_" + key: float(np.mean([r[key] for r in rows])) for key in rows[0]},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    missing = sorted(set(_trainable_names(model)) - live)
    assert overflow == 0 and not missing and frozen == frozen_state_sha(model)
    assert steps == expected["batch_count"] == (580, 560, 540)[fold]
    assert cross_pairs == expected["directed_cross_camera_positive_pairs"]
    for identity in expected["identities"]:
        assert [exposures[i] for i in identity["record_indices"]] == identity["record_exposures"]
    return {
        "epochs": 20, "optimizer_steps": steps, "sample_exposures": sum(exposures.values()),
        "overflow_events": overflow, "history": history, "endpoint": endpoint,
        "initial_state_sha256": initial, "final_state_sha256": _model_state_sha256(model),
        "sample_order_sha256": order.hexdigest(), "first_eight_batch_receipts": batches,
        "all_training_steps_path": str(log_path), "all_training_steps_sha256": _sha256(log_path),
        "all_registered_batches_match": True, "all_record_exposure_counts_match": True,
        "directed_cross_camera_positive_pairs": cross_pairs,
        "all_directed_positive_pairs": 448 * steps,
        "cross_camera_positive_pair_fraction": cross_pairs / (448 * steps),
        "trainable_tensors": len(_trainable_names(model)),
        "trainable_names": list(_trainable_names(model)),
        "nonzero_gradient_tensors": len(live), "missing_nonzero_gradients": missing,
        "frozen_state_unchanged": True, "elapsed_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def evaluate(model, records, config, directory, tag):
    model.eval()
    before = _model_state_sha256(model)
    features = {name: [] for name in OUTPUTS}
    with torch.inference_mode():
        for images, _, _, camera_labels, _, _ in loader_for(records, config):
            output = model(image_batch(images, camera_labels), return_aux=True)
            assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
            features["baseline_only"].append(output.baseline_embedding.float().cpu())
            features["fused"].append(output.fused_embedding.float().cpu())
            for name in EXPERTS:
                features[name].append(output.branch_embeddings[name].float().cpu())
    identities = np.array([r[1] for r in records])
    cameras = np.array([r[2] for r in records])
    scores, arrays = {}, {}
    for name in OUTPUTS:
        value = F.normalize(torch.cat(features[name]), dim=1)
        distances = torch.cdist(value, value)
        scores[name] = full_gallery_scores(distances.numpy(), identities, cameras)
        rankings = np.argsort(
            distances.numpy()[scores[name]["query_indices"]], axis=1, kind="stable"
        )
        arrays[name] = {
            "features": value, "distances": distances,
            "rankings_full_gallery": torch.from_numpy(rankings),
        }
    assert before == _model_state_sha256(model)
    path = directory / f"{tag}_retrieval_arrays.pt"
    assert not path.exists()
    torch.save({
        "outputs": arrays, "identities": torch.from_numpy(identities),
        "cameras": torch.from_numpy(cameras), "model_state_sha256": before,
    }, path)
    return scores, {
        "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path),
        "all_gallery_records": len(records), "output_count": 5,
        "query_count": len(scores["fused"]["query_indices"]),
        "contains_all_features_distances_and_full_gallery_rankings": True,
    }


def run(args):
    started = time.time()
    assert _sha256(args.config) == args.config_sha256 and _sha256(args.plan) == args.plan_sha256
    config, sources = load_contract(args.config)
    math_check = check_responsibility_math()
    replay = json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    splits = [build_complete_path_fold_records(records, heldout_ids=set(r["heldout_identity_ids"]))
              for r in sources["fold_receipts"]]
    assert len(records) == 3126 and all(not split["identity_overlap"] for split in splits)
    for fold, split in enumerate(splits):
        actual = [{"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]}
                  for r in split["train_records"]]
        assert actual == replay["folds"][fold]["source_manifest"]
        assert list(split["fit_identity_ids"]) == replay["folds"][fold]["source_identity_mapping"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dependencies = config["SOURCE_FILE_SHA256"]
    report = {"schema_version": "v26-role-modal-responsibility-main-v1", "status": "RUNNING", "math_check": math_check,
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": _sha256(Path(__file__)), "config_sha256": args.config_sha256,
              "plan_sha256": args.plan_sha256,
              "metadata_replay_sha256": config["SUPERVISION_METADATA"]["SHA256"],
              "source_file_sha256": {p: _sha256(Path(p)) for p in sorted(dependencies)},
              "signal_commit": signal_commit, "signal_diff_sha256": signal_diff, "seed": 42,
              "evaluation_type": "source_only_engineering_m0_real_train_source_batches",
              "planned_evaluation_type": "real_gt_train_internal_complete_path_oof_reused_development_qualification",
              "oof_is_reused_development_qualification": True, "epochs_per_endpoint": 20,
              "model_selection": "none_final_epoch_only", "new_inference_parameters": 0,
              "cudnn_deterministic": True, "cudnn_benchmark": False, "gradient_accumulation": False,
              "dev_access_count": 0, "official_test_access_count": 0, "d1_executed": False,
              "next_phase_qualified": False, "preflight": [], "folds": []}

    def save():
        report["elapsed_seconds"] = time.time() - started
        (args.output_dir / "run_summary.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())

    save()
    for fold, split in enumerate(splits):
        paired = []
        for endpoint, enabled in zip(ENDPOINTS, (False, True), strict=True):
            model, binding = build_model(config, signal_cfg, fold, split)
            paired.append({"endpoint": endpoint, "binding": binding, **preflight(model, split["train_records"], config, endpoint, replay["folds"][fold]["arms"][endpoint])})
            del model
            torch.cuda.empty_cache()
        for key in ("initial_state_sha256", "state_unchanged", "all_eight_batches_match_own_registered_sampler", "all_output_sha256", "batch_receipts", "relation_diagnostics"):
            assert paired[0][key] == paired[1][key], key
        for key in ("source", "fit_identity_ids", "heldout_identity_ids",
                    "new_inference_parameters", "total_parameters"):
            assert paired[0]["binding"][key] == paired[1]["binding"][key], key
        assert [e["binding"]["total_parameters"] for e in paired] == [98800141, 98800141]
        assert [e["binding"]["trainable_parameters"] for e in paired] == [7841292, 7841292]
        assert [e["binding"]["trainable_tensors"] for e in paired] == [203, 203]
        report["preflight"].append({"fold": fold, "paired": True, "endpoints": paired})
        save()
        print(json.dumps({"stage": "preflight", "fold": fold, "paired": True, "binding": binding}), flush=True)
    enabled = True
    capacities = []
    for coefficient in (False, True):
        model, _ = build_model(config, signal_cfg, 0, splits[0])
        torch.cuda.reset_peak_memory_stats()
        capacity = fixed_steps(model, splits[0]["train_records"], config, enabled=coefficient, count=8, fixed=False,
                               expected=replay["folds"][0]["arms"][ENDPOINTS[int(coefficient)]])
        capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        capacities.append(capacity)
        del model
        torch.cuda.empty_cache()
    model, _ = build_model(config, signal_cfg, 0, splits[0])
    overfit = fixed_steps(model, splits[0]["train_records"], config, enabled=enabled, count=100, fixed=True,
                          expected=replay["folds"][0]["arms"]["responsibility"])
    classes, smoothing = len(splits[0]["fit_identity_ids"]), config["LOSS"]["LABEL_SMOOTHING"]
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    entropy = -correct * math.log(correct) - (classes - 1) * other * math.log(other)
    weights = config["LOSS"]
    id_weight = weights["ID_FUSED"] + 3 * (weights["ID_BRANCH"] + weights["ID_RESIDUAL"])
    floor = id_weight * entropy
    overfit.update({"identity_weight_sum": id_weight, "identity_entropy_floor": id_weight * entropy,
                   "triplet_lower_bound": 0.0, "responsibility_lower_bound": 0.0, "combined_loss_floor": floor,
                   "excess_loss_ratio": (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)})
    del model
    torch.cuda.empty_cache()
    checks = {"all_gradients_live": all(not row["missing_nonzero_gradients"] for row in (*capacities, overfit)),
              "all_frozen_states_unchanged": all(row["frozen_state_unchanged"] for row in (*capacities, overfit)),
              "overflow_zero": all(row["overflow_events"] == 0 for row in (*capacities, overfit)),
              "capacity_within_limit": all(row["peak_reserved_mib"] < 24576 for row in capacities),
              "overfit_excess_ratio_at_most_point1": overfit["excess_loss_ratio"] <= 0.1,
              "all_nine_auxiliary_slot_gradients_live": all(
                  any(row[f"responsibility_{e}_{m}_gradient_norm"] > 0 for row in capacities[1]["components"])
                  for e in EXPERTS for m in MODALITIES),
              "all_three_new_encoder_gradients_live": len(capacities[1]["same_block_parameter_gradient_diagnostics"]) == 8
                  and all(row["experts"][e]["new_encoder_gradient_nonzero"]
                          for row in capacities[1]["same_block_parameter_gradient_diagnostics"] for e in EXPERTS)}
    report["m0"] = {"passed": all(checks.values()), "checks": checks, "capacities": capacities, "overfit": overfit}
    save()
    print(json.dumps({"stage": "M0", **report["m0"]}), flush=True)
    if not report["m0"]["passed"]:
        report["status"] = "M0_FAIL"
        save()
        return
    aps = {end: {name: [] for name in OUTPUTS} for end in ENDPOINTS}
    ranks = {end: {name: [] for name in OUTPUTS} for end in ENDPOINTS}
    identities = []
    for fold, split in enumerate(splits):
        result = {"fold": fold, "gallery_manifest": [
            {"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in split["heldout_records"]], "endpoints": {}}
        for endpoint, coefficient in zip(ENDPOINTS, (False, True), strict=True):
            model, binding = build_model(config, signal_cfg, fold, split)
            assert _model_state_sha256(model) == report["preflight"][fold]["endpoints"][int(coefficient)]["initial_state_sha256"]
            training = fit_endpoint(model, split["train_records"], config, fold=fold, endpoint=endpoint,
                                    enabled=coefficient, expected=replay["folds"][fold]["arms"][endpoint],
                                    directory=args.output_dir)
            checkpoint = args.output_dir / f"fold_{fold}_{endpoint}_final.pth"
            torch.save({"v26_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("baseline.")},
                        "binding": binding, "responsibility_enabled": coefficient, "plan_sha256": args.plan_sha256,
                        "config_sha256": args.config_sha256}, checkpoint)
            checkpoint_sha = _sha256(checkpoint)
            del model
            torch.cuda.empty_cache()
            model, reload_binding = build_model(config, signal_cfg, fold, split)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["binding"] == binding == reload_binding
            assert payload["responsibility_enabled"] == coefficient and payload["plan_sha256"] == args.plan_sha256
            assert payload["config_sha256"] == args.config_sha256
            state = model.state_dict()
            assert set(payload["v26_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
            state.update(payload["v26_state_dict"])
            model.load_state_dict(state, strict=True)
            del payload, state
            assert _model_state_sha256(model) == training["final_state_sha256"]
            scores, array_receipt = evaluate(model, split["heldout_records"], config, args.output_dir, f"fold_{fold}_{endpoint}")
            assert _sha256(checkpoint) == checkpoint_sha
            receipt = {"binding": binding, "training": training, "outputs": scores, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": checkpoint_sha, "strict_reload": True, "read_only_evaluation": True,
                       "retrieval_arrays": array_receipt}
            result["endpoints"][endpoint] = receipt
            (args.output_dir / f"fold_{fold}_{endpoint}_receipt.json").write_bytes((json.dumps(receipt, indent=2) + "\n").encode())
            for name in OUTPUTS:
                aps[endpoint][name].extend(scores[name]["average_precision"])
                ranks[endpoint][name].extend(scores[name]["first_match_rank"])
            print(json.dumps({"stage": "Q1_final", "fold": fold, "endpoint": endpoint,
                              "metrics": {name: row["metrics_percent"] for name, row in scores.items()}}), flush=True)
            del model
            torch.cuda.empty_cache()
        a, b = [result["endpoints"][end] for end in ENDPOINTS]
        for key in ("initial_state_sha256", "optimizer_steps", "sample_exposures", "all_registered_batches_match", "all_record_exposure_counts_match"):
            assert a["training"][key] == b["training"][key], key
        assert a["outputs"]["baseline_only"] == b["outputs"]["baseline_only"]
        assert a["training"]["sample_order_sha256"] == b["training"]["sample_order_sha256"]
        assert a["training"]["first_eight_batch_receipts"] == b["training"]["first_eight_batch_receipts"]
        identities.extend(split["heldout_records"][i][1] for i in scores["fused"]["query_indices"])
        report["folds"].append(result)
        report["evaluation_type"] = "real_gt_train_internal_complete_path_oof_reused_development_qualification"
        save()
    aggregate = {end: {name: {"mAP": float(np.mean(aps[end][name]) * 100),
                              **{f"Rank-{k}": float(np.mean(np.array(ranks[end][name]) <= k) * 100) for k in (1, 5, 10)}}
                       for name in OUTPUTS} for end in ENDPOINTS}
    control, candidate = ENDPOINTS
    gains = {name: aggregate[candidate][name]["mAP"] - aggregate[control][name]["mAP"] for name in OUTPUTS}
    fold_gains = [fold["endpoints"][candidate]["outputs"]["fused"]["metrics_percent"]["mAP"]
                  - fold["endpoints"][control]["outputs"]["fused"]["metrics_percent"]["mAP"] for fold in report["folds"]]
    bootstrap = identity_cluster_bootstrap_lower_bound(
        torch.tensor(aps[candidate]["fused"], dtype=torch.float64) - torch.tensor(aps[control]["fused"], dtype=torch.float64),
        torch.tensor(identities), seed=42, resamples=10000)
    scientific = {"aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.0,
                  "all_fold_fused_nonnegative": all(gain >= 0 for gain in fold_gains),
                  "all_expert_aggregate_nonnegative": all(gains[expert] >= 0 for expert in EXPERTS),
                  "fused_bootstrap_lower_positive": bootstrap.lower_bound > 0,
                  "fused_beats_baseline_and_experts": all(aggregate[candidate]["fused"]["mAP"] > aggregate[candidate][name]["mAP"]
                                                         for name in ("baseline_only", *EXPERTS))}
    assert len(identities) == 571 and sum(len(f["gallery_manifest"]) for f in report["folds"]) == 3126
    assert sum(e["training"]["optimizer_steps"] for f in report["folds"] for e in f["endpoints"].values()) == 3360
    assert all(_sha256(Path(path)) == expected for path, expected in source_bindings(config).items())
    report.update({"status": "Q1_PASS" if all(scientific.values()) else "Q1_FAIL", "aggregate": aggregate,
                   "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains, "scientific_checks": scientific,
                   "bootstrap": {"lower_bound_95_mAP": bootstrap.lower_bound * 100,
                                 "clusters": bootstrap.cluster_count, "resamples": bootstrap.resamples},
                   "next_phase_qualified": all(scientific.values()), "source_checkpoint_files_unchanged": True,
                   "total_gallery_records": 3126, "total_eligible_queries": 571})
    save()
    print(json.dumps({k: report[k] for k in ("status", "aggregate", "matched_gains_mAP", "scientific_checks", "bootstrap")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
