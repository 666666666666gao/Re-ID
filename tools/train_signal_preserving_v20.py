#!/usr/bin/env python3
"""Run the fixed V20 cross-modal identity objective and matched complete Q1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from tools.build_v12_complete_path_oof_targets import (
    _build_signal_teacher, _build_v8_experts, _configure_signal, _load_records,
    build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _training_batch, learning_rate_multiplier, load_raw_config
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256,
    _tensor_mapping_sha256, _trainable_names, _training_loader,
)
from tools.train_signal_preserving_v18 import evaluate
from tools.train_signal_preserving_v19 import criterion_for, frozen_state_sha, seed_everything, source_bindings
from trifusion.cross_modal_identity_v20 import cross_modal_identity_loss
from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("identity_concat", "cross_modal_identity")


def load_contract(path):
    config = load_raw_config(path)
    assert config["MODEL"]["ARCHITECTURE"] == "signal_preserving_v20_cross_modal_identity"
    assert config["EXPERIMENT"]["SEED"] == 42
    assert (config["DATA"]["TRAIN_BATCH_SIZE"], config["DATA"]["NUM_INSTANCES"]) == (64, 8)
    assert config["OPTIMIZATION"]["MAX_EPOCHS"] == 20
    assert config["PROTOCOL"]["Q1_ENDPOINTS"] == list(ENDPOINTS)
    assert config["LOSS"]["CROSS_MODAL_IDENTITY_WEIGHT"] == 0.25
    assert config["LOSS"]["CROSS_MODAL_TEMPERATURE"] == 0.07
    assert not config["PROTOCOL"]["DEV_ACCESS_DURING_Q1"]
    assert not config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"]
    assert not config["PROTOCOL"]["RERANKING"]
    for source, expected in source_bindings(config).items():
        assert _sha256(Path(source)) == expected
    assert _sha256(Path(config["SIGNAL"]["CLIP_WEIGHT"])) == config["SIGNAL"]["CLIP_WEIGHT_SHA256"]
    initial = config["INITIALIZATION"]
    summary = Path(initial["V12_RUN_SUMMARY"])
    assert _sha256(summary) == initial["V12_RUN_SUMMARY_SHA256"]
    sources = json.loads(summary.read_text())
    assert sources["status"] == "PASS" and len(sources["fold_receipts"]) == 3
    return config, sources


def build_model(config, signal_cfg, fold, split):
    seed_everything()
    source = config["INITIALIZATION"]["V12_FOLDS"][fold]
    signal_state = torch.load(source["SIGNAL_CHECKPOINT"], map_location="cpu", weights_only=True)
    expert_state = torch.load(source["EXPERT_CHECKPOINT"], map_location="cpu", weights_only=True)
    for state in (signal_state, expert_state):
        assert tuple(state["fit_identity_ids"]) == split["fit_identity_ids"]
        assert tuple(state["heldout_identity_ids"]) == split["heldout_identity_ids"]
    signal = _build_signal_teacher(
        signal_cfg, num_classes=len(split["fit_identity_ids"]),
        camera_num=len({r[2] for r in split["train_records"]}),
        view_num=len({r[3] for r in split["train_records"]}),
    )
    signal.load_state_dict(signal_state["model_state_dict"], strict=True)
    model = _build_v8_experts(signal, config, signal_checkpoint_sha256=source["SIGNAL_CHECKPOINT_SHA256"],
                              num_classes=len(split["fit_identity_ids"]))
    state = model.state_dict()
    assert set(expert_state["expert_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
    state.update(expert_state["expert_state_dict"])
    model.load_state_dict(state, strict=True)
    assert all(not p.requires_grad for p in model.baseline.parameters())
    binding = {
        "architecture": config["MODEL"]["ARCHITECTURE"], "source": dict(source),
        "fit_identity_ids": list(split["fit_identity_ids"]), "heldout_identity_ids": list(split["heldout_identity_ids"]),
        "source_state_strictly_loaded": True, "baseline_frozen": True, "shared_tail_frozen": True,
        "new_inference_parameters": 0, "router_enabled": False, "hfer_enabled": False,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_tensors": len(_trainable_names(model)),
    }
    return model, binding


def new_optimizer(model, config):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["OPTIMIZATION"]["AMP_INIT_SCALE"])
    return optimizer, scaler


def loss_terms(model, criterion, raw, config, weight):
    batch, labels = _training_batch(raw)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
        assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
        parts = criterion(output, labels)
        weights = config["LOSS"]
        base = weights["ID_FUSED"] * parts["id_fused"] + weights["TRIPLET_FUSED"] * parts["triplet_fused"]
        for expert in EXPERTS:
            base = base + weights["ID_BRANCH"] * parts[f"id_{expert}"] + weights["TRIPLET_BRANCH"] * parts[f"triplet_{expert}"]
            base = base + weights["ID_RESIDUAL"] * parts[f"id_residual_{expert}"] + weights["TRIPLET_RESIDUAL"] * parts[f"triplet_residual_{expert}"]
    with torch.autocast("cuda", enabled=False):
        alignment = cross_modal_identity_loss(output.modal_residual_embeddings, labels,
                                               temperature=weights["CROSS_MODAL_TEMPERATURE"])
    total = base.float() + weight * alignment
    return total, {"total": float(total.detach()), "base_identity_triplet": float(base.detach()),
                   "cross_modal_identity": float(alignment.detach())}


def step(model, criterion, raw, optimizer, scaler, config, weight):
    optimizer.zero_grad(set_to_none=True)
    total, values = loss_terms(model, criterion, raw, config, weight)
    assert bool(torch.isfinite(total))
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
    return values, scaler.get_scale() < scale, live


def preflight(model, records, config):
    seed_everything()
    model.eval()
    before = _model_state_sha256(model)
    index = _record_index_by_path(records)
    receipts, outputs, pairs = [], [], []
    iterator = iter(_training_loader(records, config))
    for _ in range(8):
        raw = next(iterator)
        receipts.append(_raw_batch_receipt(raw, record_index_by_path=index))
        batch, labels = _training_batch(raw)
        same = labels[:, None].eq(labels[None, :])
        assert bool((same.sum(dim=1) == 8).all())
        cross = batch["camera_ids"][:, None] != batch["camera_ids"][None, :]
        assert bool((same & cross).any())
        pairs.append({"positives_per_anchor_per_directed_modality_pair": 8,
                      "cross_camera_identity_pairs_per_directed_modality_pair": int((same & cross).sum()),
                      "directed_modality_pairs_per_expert": 6})
        with torch.no_grad():
            output = model(batch, return_aux=True)
        assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
        outputs.append(_tensor_mapping_sha256({"baseline": output.baseline_embedding,
                                               "fused": output.fused_embedding, **dict(output.branch_embeddings)}))
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
    with torch.autocast("cuda", enabled=False):
        alignment = cross_modal_identity_loss(output.modal_residual_embeddings, labels,
                                               temperature=config["LOSS"]["CROSS_MODAL_TEMPERATURE"])
    alignment.backward()
    gradients = {expert: [] for expert in EXPERTS}
    for name, parameter in model.encoder.named_parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all()), name
            if bool(parameter.grad.abs().sum() > 0):
                for expert in EXPERTS:
                    if name.startswith(expert + "_"):
                        gradients[expert].append(name)
    assert all(gradients.values())
    assert all(p.grad is None for p in model.baseline.parameters())
    model.zero_grad(set_to_none=True)
    assert before == _model_state_sha256(model)
    return {"initial_state_sha256": before, "batch_receipts": receipts, "all_output_sha256": outputs,
            "positive_pair_counts": pairs, "alignment_nonzero_encoder_gradients": gradients,
            "baseline_no_gradient": True, "state_unchanged": True}


def fixed_steps(model, records, config, *, weight, count, fixed):
    seed_everything()
    model.train()
    frozen = frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    iterator = iter(_training_loader(records, config))
    fixed_batch = next(iterator) if fixed else None
    losses, components, live, overflow = [], [], set(), 0
    for _ in range(count):
        values, bad, names = step(model, criterion, fixed_batch if fixed else next(iterator),
                                  optimizer, scaler, config, weight)
        losses.append(values["total"])
        components.append(values)
        overflow += int(bad)
        live.update(names)
    return {"steps": count, "alignment_weight": weight, "losses": losses, "components": components,
            "trainable_tensors": len(_trainable_names(model)), "nonzero_gradient_tensors": len(live),
            "missing_nonzero_gradients": sorted(set(_trainable_names(model)) - live),
            "frozen_state_unchanged": frozen == frozen_state_sha(model), "overflow_events": overflow}


def fit_endpoint(model, records, config, *, fold, endpoint, weight):
    seed_everything()
    model.train()
    initial, frozen = _model_state_sha256(model), frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    loader = _training_loader(records, config)
    index = _record_index_by_path(records)
    order = hashlib.sha256()
    history, batches, live, overflow, steps = [], [], set(), 0, 0
    for epoch in range(1, 21):
        lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * learning_rate_multiplier(epoch, max_epochs=20, warmup_epochs=5)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = []
        for raw in loader:
            order.update((json.dumps(list(raw[-1])) + "\n").encode())
            if steps < 8:
                batches.append(_raw_batch_receipt(raw, record_index_by_path=index))
            values, bad, names = step(model, criterion, raw, optimizer, scaler, config, weight)
            rows.append(values)
            overflow += int(bad)
            live.update(names)
            steps += 1
        row = {"fold": fold, "endpoint": endpoint, "epoch": epoch, "batches": len(rows), "learning_rate": lr,
               **{"mean_" + key: float(np.mean([r[key] for r in rows])) for key in rows[0]}}
        history.append(row)
        print(json.dumps(row), flush=True)
    missing = sorted(set(_trainable_names(model)) - live)
    assert overflow == 0 and not missing and frozen == frozen_state_sha(model)
    return {"epochs": 20, "optimizer_steps": steps, "overflow_events": overflow, "history": history,
            "alignment_weight": weight, "initial_state_sha256": initial, "final_state_sha256": _model_state_sha256(model),
            "sample_order_sha256": order.hexdigest(), "first_eight_batch_receipts": batches,
            "trainable_tensors": len(_trainable_names(model)), "trainable_names": list(_trainable_names(model)),
            "nonzero_gradient_tensors": len(live), "missing_nonzero_gradients": missing, "frozen_state_unchanged": True}


def run(args):
    started = time.time()
    assert _sha256(args.config) == args.config_sha256 and _sha256(args.plan) == args.plan_sha256
    config, sources = load_contract(args.config)
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    splits = [build_complete_path_fold_records(records, heldout_ids=set(r["heldout_identity_ids"]))
              for r in sources["fold_receipts"]]
    assert len(records) == 3126 and all(not split["identity_overlap"] for split in splits)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dependencies = (
        "modeling/trifusion/cross_modal_identity_v20.py",
        "modeling/trifusion/signal_preserving_v8.py",
        "modeling/trifusion/signal_preserving_v8_builder.py",
        "modeling/trifusion/signal_preserving_v13.py",
        "modeling/trifusion/signal_preserving_v19.py",
        "modeling/trifusion/aligned_data.py", "modeling/trifusion/criterion.py",
        "tools/train_signal_preserving_v17.py", "tools/train_signal_preserving_v18.py",
        "tools/train_signal_preserving_v19.py", "tools/build_v12_complete_path_oof_targets.py",
        "tools/run_signal_preserving_v5.py", "tools/audit_v17_full_gallery.py",
        "tools/diagnose_v6_oracle_complementarity.py", "protocols/rgbnt201_dev_v1.json",
    )
    report = {"schema_version": "v20-cross-modal-identity-main-v1", "status": "RUNNING",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": _sha256(Path(__file__)), "config_sha256": args.config_sha256,
              "plan_sha256": args.plan_sha256, "source_file_sha256": {p: _sha256(Path(p)) for p in sorted(dependencies)},
              "signal_commit": signal_commit, "signal_diff_sha256": signal_diff, "seed": 42,
              "evaluation_type": "real_gt_train_internal_complete_path_oof",
              "oof_is_reused_development_qualification": True, "epochs_per_endpoint": 20,
              "model_selection": "none_final_epoch_only", "new_inference_parameters": 0,
              "cudnn_deterministic": True, "cudnn_benchmark": False, "gradient_accumulation": False,
              "dev_access_count": 0, "official_test_access_count": 0, "d1_executed": False,
              "next_phase_qualified": False, "preflight": [], "folds": []}

    def save():
        report["elapsed_seconds"] = time.time() - started
        (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for fold, split in enumerate(splits):
        paired = []
        for endpoint in ENDPOINTS:
            model, binding = build_model(config, signal_cfg, fold, split)
            paired.append({"endpoint": endpoint, "binding": binding, **preflight(model, split["train_records"], config)})
            del model
            torch.cuda.empty_cache()
        for key in ("initial_state_sha256", "batch_receipts", "all_output_sha256", "binding", "positive_pair_counts"):
            assert paired[0][key] == paired[1][key], key
        report["preflight"].append({"fold": fold, "paired": True, "endpoints": paired})
        save()
        print(json.dumps({"stage": "preflight", "fold": fold, "paired": True, "binding": binding}), flush=True)
    weight = config["LOSS"]["CROSS_MODAL_IDENTITY_WEIGHT"]
    capacities = []
    for coefficient in (0.0, weight):
        model, _ = build_model(config, signal_cfg, 0, splits[0])
        torch.cuda.reset_peak_memory_stats()
        capacity = fixed_steps(model, splits[0]["train_records"], config, weight=coefficient, count=8, fixed=False)
        capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        capacities.append(capacity)
        del model
        torch.cuda.empty_cache()
    model, _ = build_model(config, signal_cfg, 0, splits[0])
    overfit = fixed_steps(model, splits[0]["train_records"], config, weight=weight, count=100, fixed=True)
    classes, smoothing = len(splits[0]["fit_identity_ids"]), config["LOSS"]["LABEL_SMOOTHING"]
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    entropy = -correct * math.log(correct) - (classes - 1) * other * math.log(other)
    weights = config["LOSS"]
    id_weight = weights["ID_FUSED"] + 3 * (weights["ID_BRANCH"] + weights["ID_RESIDUAL"])
    floor = id_weight * entropy + weight * math.log(config["DATA"]["NUM_INSTANCES"])
    overfit.update({"identity_weight_sum": id_weight, "identity_entropy_floor": id_weight * entropy,
                   "alignment_entropy_floor": math.log(8), "combined_loss_floor": floor,
                   "excess_loss_ratio": (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)})
    del model
    torch.cuda.empty_cache()
    checks = {"all_gradients_live": all(not row["missing_nonzero_gradients"] for row in (*capacities, overfit)),
              "all_frozen_states_unchanged": all(row["frozen_state_unchanged"] for row in (*capacities, overfit)),
              "overflow_zero": all(row["overflow_events"] == 0 for row in (*capacities, overfit)),
              "capacity_within_limit": all(row["peak_reserved_mib"] < 24576 for row in capacities),
              "overfit_excess_ratio_at_most_point1": overfit["excess_loss_ratio"] <= 0.1}
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
        for endpoint, coefficient in zip(ENDPOINTS, (0.0, weight), strict=True):
            model, binding = build_model(config, signal_cfg, fold, split)
            training = fit_endpoint(model, split["train_records"], config, fold=fold, endpoint=endpoint, weight=coefficient)
            checkpoint = args.output_dir / f"fold_{fold}_{endpoint}_final.pth"
            torch.save({"v20_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("baseline.")},
                        "binding": binding, "alignment_weight": coefficient, "plan_sha256": args.plan_sha256,
                        "config_sha256": args.config_sha256}, checkpoint)
            checkpoint_sha = _sha256(checkpoint)
            del model
            torch.cuda.empty_cache()
            model, reload_binding = build_model(config, signal_cfg, fold, split)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["binding"] == binding == reload_binding
            assert payload["alignment_weight"] == coefficient and payload["plan_sha256"] == args.plan_sha256
            assert payload["config_sha256"] == args.config_sha256
            state = model.state_dict()
            assert set(payload["v20_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
            state.update(payload["v20_state_dict"])
            model.load_state_dict(state, strict=True)
            del payload, state
            assert _model_state_sha256(model) == training["final_state_sha256"]
            scores = evaluate(model, split["heldout_records"], config)
            assert _sha256(checkpoint) == checkpoint_sha
            receipt = {"binding": binding, "training": training, "outputs": scores, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": checkpoint_sha, "strict_reload": True, "read_only_evaluation": True}
            result["endpoints"][endpoint] = receipt
            (args.output_dir / f"fold_{fold}_{endpoint}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            for name in OUTPUTS:
                aps[endpoint][name].extend(scores[name]["average_precision"])
                ranks[endpoint][name].extend(scores[name]["first_match_rank"])
            print(json.dumps({"stage": "Q1_final", "fold": fold, "endpoint": endpoint,
                              "metrics": {name: row["metrics_percent"] for name, row in scores.items()}}), flush=True)
            del model
            torch.cuda.empty_cache()
        a, b = [result["endpoints"][end] for end in ENDPOINTS]
        for key in ("initial_state_sha256", "sample_order_sha256", "first_eight_batch_receipts"):
            assert a["training"][key] == b["training"][key], key
        assert a["outputs"]["baseline_only"] == b["outputs"]["baseline_only"]
        identities.extend(split["heldout_records"][i][1] for i in scores["fused"]["query_indices"])
        report["folds"].append(result)
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
