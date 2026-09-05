#!/usr/bin/env python3
"""Run the fixed V21 SAM/AdamW comparison at equal forward-backward budgets."""
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
    _configure_signal, _load_records,
    build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _training_batch, learning_rate_multiplier, load_raw_config
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256,
    _tensor_mapping_sha256, _trainable_names, _training_loader,
)
from tools.train_signal_preserving_v18 import evaluate
from tools.train_signal_preserving_v19 import criterion_for, frozen_state_sha, seed_everything, source_bindings
from tools.train_signal_preserving_v20 import build_model, new_optimizer
from trifusion.sam_training_v21 import training_step
from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("adamw_40", "sam_20")


def load_contract(path):
    config = load_raw_config(path)
    assert config["MODEL"]["ARCHITECTURE"] == "signal_preserving_v21_sam"
    assert config["EXPERIMENT"]["SEED"] == 42
    assert (config["DATA"]["TRAIN_BATCH_SIZE"], config["DATA"]["NUM_INSTANCES"]) == (64, 8)
    assert config["PROTOCOL"]["Q1_ENDPOINTS"] == list(ENDPOINTS)
    assert config["OPTIMIZATION"]["ENDPOINT_EPOCHS"] == dict(zip(ENDPOINTS, (40, 20), strict=True))
    assert config["OPTIMIZATION"]["ENDPOINT_WARMUP_EPOCHS"] == dict(zip(ENDPOINTS, (10, 5), strict=True))
    assert config["OPTIMIZATION"]["SAM_RHO"] == 0.05
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


def loss_terms(model, criterion, batch, labels, config):
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
        assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
        parts = criterion(output, labels)
        weights = config["LOSS"]
        total = weights["ID_FUSED"] * parts["id_fused"] + weights["TRIPLET_FUSED"] * parts["triplet_fused"]
        for expert in EXPERTS:
            total = total + weights["ID_BRANCH"] * parts[f"id_{expert}"] + weights["TRIPLET_BRANCH"] * parts[f"triplet_{expert}"]
            total = total + weights["ID_RESIDUAL"] * parts[f"id_residual_{expert}"] + weights["TRIPLET_RESIDUAL"] * parts[f"triplet_residual_{expert}"]
    return total.float()


def step(model, criterion, raw, optimizer, scaler, config, rho):
    batch, labels = _training_batch(raw)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    necks = [(name, module) for name, module in model.named_modules()
             if isinstance(module, torch.nn.BatchNorm1d) and module.training]
    assert len(necks) == 7
    before = {name: int(module.num_batches_tracked) for name, module in necks}
    values = training_step(parameters, [module for _, module in necks], optimizer, scaler,
                           lambda: loss_terms(model, criterion, batch, labels, config), rho=rho)
    assert all(int(module.num_batches_tracked) == before[name] + 1 for name, module in necks)
    assert abs(values["actual_perturbation_norm"] - rho) < 1e-5
    first = set(values.pop("first_nonzero_gradient_names"))
    update = set(values.pop("update_nonzero_gradient_names"))
    return values, first, update


def preflight(model, records, config):
    seed_everything()
    model.eval()
    before = _model_state_sha256(model)
    index = _record_index_by_path(records)
    receipts, outputs = [], []
    iterator = iter(_training_loader(records, config))
    for _ in range(8):
        raw = next(iterator)
        receipts.append(_raw_batch_receipt(raw, record_index_by_path=index))
        batch, labels = _training_batch(raw)
        same = labels[:, None].eq(labels[None, :])
        assert bool((same.sum(dim=1) == 8).all())
        cross = batch["camera_ids"][:, None] != batch["camera_ids"][None, :]
        assert bool((same & cross).any())
        with torch.no_grad():
            output = model(batch, return_aux=True)
        assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
        outputs.append(_tensor_mapping_sha256({"baseline": output.baseline_embedding,
                                               "fused": output.fused_embedding, **dict(output.branch_embeddings)}))
    assert before == _model_state_sha256(model)
    return {"initial_state_sha256": before, "batch_receipts": receipts,
            "all_output_sha256": outputs, "state_unchanged": True}


def fixed_steps(model, records, config, *, rho, count, fixed):
    seed_everything()
    model.train()
    frozen = frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    iterator = iter(_training_loader(records, config))
    fixed_batch = next(iterator) if fixed else None
    rows, first_live, update_live = [], set(), set()
    for _ in range(count):
        values, first, update = step(model, criterion, fixed_batch if fixed else next(iterator),
                                     optimizer, scaler, config, rho)
        rows.append(values)
        first_live.update(first)
        update_live.update(update)
    names = set(_trainable_names(model))
    return {"optimizer_steps": count, "rho": rho, "components": rows,
            "losses": [row["loss_at_parameters"] for row in rows],
            "forward_backward_passes": sum(row["forward_backward_passes"] for row in rows),
            "trainable_tensors": len(names),
            "first_nonzero_gradient_tensors": len(first_live), "update_nonzero_gradient_tensors": len(update_live),
            "missing_first_gradients": sorted(names - first_live), "missing_update_gradients": sorted(names - update_live),
            "frozen_state_unchanged": frozen == frozen_state_sha(model),
            "overflow_events": sum(int(row["overflow"]) for row in rows),
            "batchnorm_updates_per_step": 1,
            "sam_restore_batches": sum(row["sam_restore_batches"] for row in rows)}


def fit_endpoint(model, records, config, *, fold, endpoint, rho):
    started = time.perf_counter()
    seed_everything()
    model.train()
    initial, frozen = _model_state_sha256(model), frozen_state_sha(model)
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    loader = _training_loader(records, config)
    index = _record_index_by_path(records)
    epochs = config["OPTIMIZATION"]["ENDPOINT_EPOCHS"][endpoint]
    warmup = config["OPTIMIZATION"]["ENDPOINT_WARMUP_EPOCHS"][endpoint]
    order = hashlib.sha256()
    history, batches, epoch_orders, first_live, update_live = [], [], [], set(), set()
    overflow, steps, passes, restores = 0, 0, 0, 0
    for epoch in range(1, epochs + 1):
        lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * learning_rate_multiplier(epoch, max_epochs=epochs, warmup_epochs=warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows, epoch_order = [], hashlib.sha256()
        for raw in loader:
            sample = (json.dumps(list(raw[-1])) + "\n").encode()
            order.update(sample)
            epoch_order.update(sample)
            if steps < 8:
                batches.append(_raw_batch_receipt(raw, record_index_by_path=index))
            values, first, update = step(model, criterion, raw, optimizer, scaler, config, rho)
            rows.append(values)
            overflow += int(values["overflow"])
            passes += values["forward_backward_passes"]
            restores += values["sam_restore_batches"]
            first_live.update(first)
            update_live.update(update)
            steps += 1
        row = {"fold": fold, "endpoint": endpoint, "epoch": epoch, "batches": len(rows), "learning_rate": lr,
               **{"mean_" + key: float(np.mean([r[key] for r in rows]))
                  for key in ("loss_at_parameters", "loss_for_update_gradient", "first_gradient_norm",
                              "update_gradient_norm", "actual_perturbation_norm")}}
        history.append(row)
        epoch_orders.append(epoch_order.hexdigest())
        if epoch == 20:
            first20 = order.hexdigest()
        print(json.dumps(row), flush=True)
    names = set(_trainable_names(model))
    assert not (names - first_live) and not (names - update_live)
    assert overflow == 0 and frozen == frozen_state_sha(model)
    return {"epochs": epochs, "warmup_epochs": warmup, "optimizer_steps": steps,
            "training_elapsed_seconds": time.perf_counter() - started,
            "forward_backward_passes": passes, "overflow_events": overflow, "history": history, "rho": rho,
            "initial_state_sha256": initial, "final_state_sha256": _model_state_sha256(model),
            "sample_order_sha256": order.hexdigest(), "sample_order_first20_sha256": first20,
            "epoch_sample_order_sha256": epoch_orders, "first_eight_batch_receipts": batches,
            "trainable_tensors": len(names), "trainable_names": sorted(names),
            "first_nonzero_gradient_tensors": len(first_live), "update_nonzero_gradient_tensors": len(update_live),
            "missing_first_gradients": [], "missing_update_gradients": [], "frozen_state_unchanged": True,
            "batchnorm_updates_per_step": 1, "sam_restore_batches": restores}


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
        "modeling/trifusion/sam_training_v21.py",
        "tools/train_signal_preserving_v20.py",
        "modeling/trifusion/experts/semantic_residual.py", "modeling/trifusion/experts/mamba.py",
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
    report = {"schema_version": "v21-sam-main-v1", "status": "RUNNING",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": _sha256(Path(__file__)), "config_sha256": args.config_sha256,
              "plan_sha256": args.plan_sha256, "source_file_sha256": {p: _sha256(Path(p)) for p in sorted(dependencies)},
              "signal_commit": signal_commit, "signal_diff_sha256": signal_diff, "seed": 42,
              "evaluation_type": "real_gt_train_internal_complete_path_oof",
              "oof_is_reused_development_qualification": True, "epochs_per_endpoint": config["OPTIMIZATION"]["ENDPOINT_EPOCHS"],
              "comparison_basis": "equal_forward_backward_passes_not_equal_updates_or_data_exposures",
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
        for key in ("initial_state_sha256", "batch_receipts", "all_output_sha256", "binding"):
            assert paired[0][key] == paired[1][key], key
        report["preflight"].append({"fold": fold, "paired": True, "endpoints": paired})
        save()
        print(json.dumps({"stage": "preflight", "fold": fold, "paired": True, "binding": binding}), flush=True)
    rho = config["OPTIMIZATION"]["SAM_RHO"]
    capacities = []
    for radius in (0.0, rho):
        model, _ = build_model(config, signal_cfg, 0, splits[0])
        torch.cuda.reset_peak_memory_stats()
        capacity = fixed_steps(model, splits[0]["train_records"], config, rho=radius, count=8, fixed=False)
        capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        capacities.append(capacity)
        del model
        torch.cuda.empty_cache()
    model, _ = build_model(config, signal_cfg, 0, splits[0])
    overfit = fixed_steps(model, splits[0]["train_records"], config, rho=rho, count=100, fixed=True)
    classes, smoothing = len(splits[0]["fit_identity_ids"]), config["LOSS"]["LABEL_SMOOTHING"]
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    entropy = -correct * math.log(correct) - (classes - 1) * other * math.log(other)
    weights = config["LOSS"]
    id_weight = weights["ID_FUSED"] + 3 * (weights["ID_BRANCH"] + weights["ID_RESIDUAL"])
    floor = id_weight * entropy
    overfit.update({"identity_weight_sum": id_weight, "identity_entropy_floor": id_weight * entropy,
                   "combined_loss_floor": floor,
                   "excess_loss_ratio": (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)})
    del model
    torch.cuda.empty_cache()
    checks = {"all_gradients_live": all(not row["missing_first_gradients"] and not row["missing_update_gradients"] for row in (*capacities, overfit)),
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
        for endpoint, radius in zip(ENDPOINTS, (0.0, rho), strict=True):
            model, binding = build_model(config, signal_cfg, fold, split)
            training = fit_endpoint(model, split["train_records"], config, fold=fold, endpoint=endpoint, rho=radius)
            checkpoint = args.output_dir / f"fold_{fold}_{endpoint}_final.pth"
            torch.save({"v21_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("baseline.")},
                        "binding": binding, "rho": radius, "plan_sha256": args.plan_sha256,
                        "config_sha256": args.config_sha256}, checkpoint)
            checkpoint_sha = _sha256(checkpoint)
            del model
            torch.cuda.empty_cache()
            model, reload_binding = build_model(config, signal_cfg, fold, split)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["binding"] == binding == reload_binding
            assert payload["rho"] == radius and payload["plan_sha256"] == args.plan_sha256
            assert payload["config_sha256"] == args.config_sha256
            state = model.state_dict()
            assert set(payload["v21_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
            state.update(payload["v21_state_dict"])
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
        for key in ("initial_state_sha256", "sample_order_first20_sha256", "first_eight_batch_receipts"):
            assert a["training"][key] == b["training"][key], key
        assert a["training"]["epoch_sample_order_sha256"][:20] == b["training"]["epoch_sample_order_sha256"]
        assert a["training"]["forward_backward_passes"] == b["training"]["forward_backward_passes"]
        assert a["training"]["optimizer_steps"] == 2 * b["training"]["optimizer_steps"]
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
    assert sum(e["training"]["optimizer_steps"] for f in report["folds"] for e in f["endpoints"].values()) == 5040
    assert sum(e["training"]["forward_backward_passes"] for f in report["folds"] for e in f["endpoints"].values()) == 6720
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
