#!/usr/bin/env python3
"""Run V19 private semantic tails: paired source-only M0 and full-gallery Q1."""
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
from tools.run_signal_preserving_v5 import (
    _set_seed, _training_batch, learning_rate_multiplier, load_raw_config,
)
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path,
    _sha256, _tensor_mapping_sha256, _trainable_names, _training_loader,
)
from tools.train_signal_preserving_v18 import evaluate
from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
from trifusion.signal_preserving_v19 import (
    PrivateSemanticTailEncoderV19, optimizer_parameter_groups,
    private_tail_storage_is_disjoint,
)

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("frozen_private_tail", "trained_private_tail")


def seed_everything():
    _set_seed(42)
    torch.backends.cudnn.benchmark = False


def source_bindings(config):
    return {
        paths[f"{kind}_CHECKPOINT"]: paths[f"{kind}_CHECKPOINT_SHA256"]
        for paths in config["INITIALIZATION"]["V12_FOLDS"]
        for kind in ("SIGNAL", "EXPERT")
    }


def load_contract(config_path):
    config = load_raw_config(config_path)
    assert config["MODEL"]["ARCHITECTURE"] == "signal_preserving_v19_private_semantic_tail"
    assert config["EXPERIMENT"]["SEED"] == 42
    assert (config["DATA"]["TRAIN_BATCH_SIZE"], config["DATA"]["NUM_INSTANCES"]) == (64, 8)
    assert config["OPTIMIZATION"]["MAX_EPOCHS"] == 20
    assert config["PROTOCOL"]["Q1_ENDPOINTS"] == list(ENDPOINTS)
    assert config["PROTOCOL"]["DEV_ACCESS_DURING_Q1"] is False
    assert config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"] is False
    assert config["PROTOCOL"]["RERANKING"] is False
    for path, expected in source_bindings(config).items():
        assert _sha256(Path(path)) == expected
    initial = config["INITIALIZATION"]
    summary_path = Path(initial["V12_RUN_SUMMARY"])
    assert _sha256(summary_path) == initial["V12_RUN_SUMMARY_SHA256"]
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "PASS" and len(summary["fold_receipts"]) == 3
    return config, summary


def build_model(config, signal_cfg, fold, split, train_tail):
    seed_everything()
    paths = config["INITIALIZATION"]["V12_FOLDS"][fold]
    signal_payload = torch.load(paths["SIGNAL_CHECKPOINT"], map_location="cpu", weights_only=True)
    expert_payload = torch.load(paths["EXPERT_CHECKPOINT"], map_location="cpu", weights_only=True)
    for payload in (signal_payload, expert_payload):
        assert tuple(payload["fit_identity_ids"]) == split["fit_identity_ids"]
        assert tuple(payload["heldout_identity_ids"]) == split["heldout_identity_ids"]
    signal = _build_signal_teacher(
        signal_cfg, num_classes=len(split["fit_identity_ids"]),
        camera_num=len({r[2] for r in split["train_records"]}),
        view_num=len({r[3] for r in split["train_records"]}),
    )
    signal.load_state_dict(signal_payload["model_state_dict"], strict=True)
    model = _build_v8_experts(
        signal, config, signal_checkpoint_sha256=paths["SIGNAL_CHECKPOINT_SHA256"],
        num_classes=len(split["fit_identity_ids"]),
    )
    state = model.state_dict()
    expert_state = expert_payload["expert_state_dict"]
    assert set(expert_state) == {k for k in state if not k.startswith("baseline.")}
    state.update(expert_state)
    model.load_state_dict(state, strict=True)
    model.encoder = PrivateSemanticTailEncoderV19(model.encoder, train_private_tail=train_tail).cuda()
    assert private_tail_storage_is_disjoint(model.encoder)
    assert sum(p.numel() for p in model.encoder.private_tails.parameters()) == 63790848
    assert len(list(model.encoder.private_tails.parameters())) == 108
    binding = {
        "architecture": config["MODEL"]["ARCHITECTURE"],
        "train_private_tail": train_tail, "tail_indices": [9, 10, 11],
        "private_tail_parameters": 63790848, "private_tail_tensors": 108,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_tensors": len(_trainable_names(model)),
        "fit_identity_ids": list(split["fit_identity_ids"]),
        "heldout_identity_ids": list(split["heldout_identity_ids"]),
        "source": dict(paths), "source_state_strictly_loaded": True,
        "private_storage_disjoint": True, "baseline_frozen": True,
        "role_modules_and_classification_heads_trainable": True,
        "router_enabled": False, "hfer_enabled": False, "projection_enabled": False,
        "envelope_enabled": False,
    }
    return model, binding


def frozen_state_sha(model):
    frozen_names = {name for name, p in model.named_parameters() if not p.requires_grad}
    return _tensor_mapping_sha256({
        name: value for name, value in model.state_dict().items()
        if name.startswith("baseline.") or name in frozen_names
    })


def new_optimizer(model, config):
    opt = config["OPTIMIZATION"]
    groups = optimizer_parameter_groups(
        model, role_lr=opt["NEW_MODULE_LR"], tail_lr=opt["PRIVATE_TAIL_LR"],
    )
    actual = [id(p) for group in groups for p in group["params"]]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=opt["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=opt["AMP_INIT_SCALE"])
    return optimizer, scaler


def criterion_for(config):
    return ExpertFormationV8Criterion(
        triplet_margin=config["LOSS"]["TRIPLET_MARGIN"],
        label_smoothing=config["LOSS"]["LABEL_SMOOTHING"],
    ).cuda()


def optimization_step(model, criterion, raw_batch, optimizer, scaler, config):
    batch, labels = _training_batch(raw_batch)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output = model(batch, return_aux=True)
        assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
        parts = criterion(output, labels)
        weights = config["LOSS"]
        total = weights["ID_FUSED"] * parts["id_fused"] + weights["TRIPLET_FUSED"] * parts["triplet_fused"]
        for expert in EXPERTS:
            total = total + (
                weights["ID_BRANCH"] * parts[f"id_{expert}"]
                + weights["TRIPLET_BRANCH"] * parts[f"triplet_{expert}"]
                + weights["ID_RESIDUAL"] * parts[f"id_residual_{expert}"]
                + weights["TRIPLET_RESIDUAL"] * parts[f"triplet_residual_{expert}"]
            )
    assert bool(torch.isfinite(total))
    scale_before = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    nonzero = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        if bool(p.grad.abs().sum() > 0):
            nonzero.add(name)
    scaler.step(optimizer)
    scaler.update()
    return float(total.detach()), scaler.get_scale() < scale_before, nonzero


def preflight(model, records, config):
    seed_everything()
    before = _model_state_sha256(model)
    loader = _training_loader(records, config)
    index = _record_index_by_path(records)
    batches = []
    outputs = []
    pair_counts = []
    model.eval()
    for raw_batch in list_batches(loader, config["GATES"]["PREFLIGHT_BATCHES"]):
        batches.append(_raw_batch_receipt(raw_batch, record_index_by_path=index))
        batch, labels = _training_batch(raw_batch)
        with torch.no_grad():
            output = model(batch, return_aux=True)
            field = model.baseline(batch)
            reference = model.fusion(
                field.baseline_embedding,
                model.encoder.roles(field.anchor_sequence, field.reference_sequence),
            )
        assert output.diagnostics["baseline_exact_prefix"]
        assert torch.equal(output.baseline_embedding, field.baseline_embedding)
        assert torch.equal(output.fused_embedding, reference.fused_embedding)
        assert all(torch.equal(output.branch_embeddings[e], reference.branch_embeddings[e]) for e in EXPERTS)
        outputs.append(_tensor_mapping_sha256({
            "baseline": output.baseline_embedding, "fused": output.fused_embedding,
            **dict(output.branch_embeddings),
        }))
        same = labels[:, None] == labels[None, :]
        cross_camera = batch["camera_ids"][:, None] != batch["camera_ids"][None, :]
        pair_counts.append({"cross_camera_positive": int((same & cross_camera).sum()),
                            "negative": int((~same).sum())})
    assert before == _model_state_sha256(model)
    assert len(batches) == config["GATES"]["PREFLIGHT_BATCHES"]
    assert all(p["cross_camera_positive"] > 0 and p["negative"] > 0 for p in pair_counts)
    return {
        "initial_state_sha256": before, "batch_receipts": batches,
        "all_output_sha256": outputs, "pair_counts": pair_counts,
        "original_v8_all_output_exact_parity": True, "signal_exact_prefix": True,
        "state_unchanged": True,
    }


def list_batches(loader, count):
    iterator = iter(loader)
    for _ in range(count):
        yield next(iterator)


def fixed_steps(model, records, config, *, steps, fixed_batch):
    seed_everything()
    loader = _training_loader(records, config)
    iterator = iter(loader)
    fixed = next(iterator) if fixed_batch else None
    criterion = criterion_for(config)
    optimizer, scaler = new_optimizer(model, config)
    frozen = frozen_state_sha(model)
    tail_before = _model_state_sha256(model.encoder.private_tails)
    losses, gradient_names, overflow = [], set(), 0
    started = time.time()
    model.train()
    for _ in range(steps):
        loss, bad, names = optimization_step(
            model, criterion, fixed if fixed_batch else next(iterator), optimizer, scaler, config,
        )
        losses.append(loss)
        overflow += int(bad)
        gradient_names.update(names)
    return {
        "steps": steps, "losses": losses, "overflow_events": overflow,
        "trainable_tensors": len(_trainable_names(model)),
        "nonzero_gradient_tensors": len(gradient_names),
        "missing_nonzero_gradients": sorted(set(_trainable_names(model)) - gradient_names),
        "frozen_state_unchanged": frozen == frozen_state_sha(model),
        "private_tail_changed": tail_before != _model_state_sha256(model.encoder.private_tails),
        "elapsed_seconds": time.time() - started,
        "optimizer_groups": [{"name": g["name"], "lr": g["lr"],
                              "parameters": sum(p.numel() for p in g["params"]),
                              "tensors": len(g["params"])} for g in optimizer.param_groups],
    }


def fit_endpoint(model, records, config, *, fold, endpoint):
    seed_everything()
    loader = _training_loader(records, config)
    criterion = criterion_for(config)
    optimizer, scaler = new_optimizer(model, config)
    index = _record_index_by_path(records)
    frozen = frozen_state_sha(model)
    initial = _model_state_sha256(model)
    tail_before = _model_state_sha256(model.encoder.private_tails)
    history, first_batches, gradient_names = [], [], set()
    sample_order = hashlib.sha256()
    overflow, steps = 0, 0
    model.train()
    opt = config["OPTIMIZATION"]
    started = time.time()
    for epoch in range(1, opt["MAX_EPOCHS"] + 1):
        multiplier = learning_rate_multiplier(
            epoch, max_epochs=opt["MAX_EPOCHS"], warmup_epochs=opt["WARMUP_EPOCHS"],
        )
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        losses = []
        for raw_batch in loader:
            receipt = _raw_batch_receipt(raw_batch, record_index_by_path=index)
            sample_order.update(json.dumps(receipt["sampler_indices"], separators=(",", ":")).encode())
            if len(first_batches) < 8:
                first_batches.append(receipt)
            loss, bad, names = optimization_step(model, criterion, raw_batch, optimizer, scaler, config)
            losses.append(loss)
            overflow += int(bad)
            steps += int(not bad)
            gradient_names.update(names)
        row = {"fold": fold, "endpoint": endpoint, "epoch": epoch, "batches": len(losses),
               "mean_training_loss": sum(losses) / len(losses),
               "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
               "elapsed_seconds": time.time() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "epochs": len(history), "optimizer_steps": steps, "overflow_events": overflow,
        "history": history, "initial_state_sha256": initial,
        "final_state_sha256": _model_state_sha256(model),
        "trainable_tensors": len(_trainable_names(model)), "trainable_names": list(_trainable_names(model)),
        "nonzero_gradient_tensors": len(gradient_names),
        "missing_nonzero_gradients": sorted(set(_trainable_names(model)) - gradient_names),
        "first_eight_batch_receipts": first_batches, "sample_order_sha256": sample_order.hexdigest(),
        "frozen_state_unchanged": frozen == frozen_state_sha(model),
        "private_tail_changed": tail_before != _model_state_sha256(model.encoder.private_tails),
    }
    assert result["frozen_state_unchanged"] and not result["missing_nonzero_gradients"]
    assert overflow == 0 and len(history) == 20
    assert result["private_tail_changed"] == (endpoint == ENDPOINTS[1])
    return result


def run(args):
    started = time.time()
    assert _sha256(args.plan) == args.plan_sha256
    assert _sha256(args.config) == args.config_sha256
    config, sources = load_contract(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    splits = [
        build_complete_path_fold_records(records, heldout_ids=set(r["heldout_identity_ids"]))
        for r in sources["fold_receipts"]
    ]
    assert all(not split["identity_overlap"] for split in splits)
    report = {
        "schema_version": "v19-private-semantic-tail-main-v1", "status": "RUNNING",
        "architecture": config["MODEL"]["ARCHITECTURE"], "seed": 42,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": _sha256(Path(__file__)), "config_sha256": args.config_sha256,
        "plan_sha256": args.plan_sha256, "signal_commit": signal_commit, "signal_diff_sha256": signal_diff,
        "source_file_sha256": {path: _sha256(Path(path)) for path in (
            "modeling/trifusion/signal_preserving_v19.py",
            "modeling/trifusion/signal_preserving_v8.py",
            "modeling/trifusion/signal_preserving_v8_builder.py",
            "modeling/trifusion/aligned_data.py", "modeling/trifusion/criterion.py",
            "tools/train_signal_preserving_v17.py", "tools/train_signal_preserving_v18.py",
            "tools/build_v12_complete_path_oof_targets.py", "tools/run_signal_preserving_v5.py",
            "tools/audit_v17_full_gallery.py", "tools/diagnose_v6_oracle_complementarity.py",
            "protocols/rgbnt201_dev_v1.json",
        )},
        "evaluation_type": "real_gt_train_internal_complete_path_oof",
        "oof_is_reused_development_qualification": True, "epochs_per_endpoint": 20,
        "model_selection": "none_final_epoch_only", "cudnn_deterministic": True,
        "cudnn_benchmark": False, "gradient_accumulation": False,
        "dev_access_count": 0, "official_test_access_count": 0,
        "d1_executed": False, "next_phase_qualified": False, "preflight": [], "folds": [],
    }

    def save():
        report["elapsed_seconds"] = time.time() - started
        (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for fold, split in enumerate(splits):
        receipts = []
        for train_tail in (False, True):
            model, binding = build_model(config, signal_cfg, fold, split, train_tail)
            receipts.append({"binding": binding, **preflight(model, split["train_records"], config)})
            del model
            torch.cuda.empty_cache()
        a, b = receipts
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a["batch_receipts"] == b["batch_receipts"]
        assert a["all_output_sha256"] == b["all_output_sha256"]
        assert b["binding"]["trainable_tensors"] - a["binding"]["trainable_tensors"] == 108
        report["preflight"].append({"fold": fold, "paired": True, "endpoints": receipts})
        save()
        print(json.dumps({"stage": "preflight", "fold": fold, "paired": True,
                          "bindings": [r["binding"] for r in receipts]}), flush=True)
    capacities = []
    for train_tail in (False, True):
        model, binding = build_model(config, signal_cfg, 0, splits[0], train_tail)
        torch.cuda.reset_peak_memory_stats()
        capacity = fixed_steps(model, splits[0]["train_records"], config, steps=8, fixed_batch=False)
        capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        capacity["train_private_tail"] = train_tail
        capacities.append(capacity)
        del model
        torch.cuda.empty_cache()
    model, _ = build_model(config, signal_cfg, 0, splits[0], True)
    overfit = fixed_steps(model, splits[0]["train_records"], config, steps=100, fixed_batch=True)
    classes, smoothing = len(splits[0]["fit_identity_ids"]), config["LOSS"]["LABEL_SMOOTHING"]
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    weights = config["LOSS"]
    id_weight = weights["ID_FUSED"] + 3 * (weights["ID_BRANCH"] + weights["ID_RESIDUAL"])
    floor = id_weight * (-correct * math.log(correct) - (classes - 1) * other * math.log(other))
    overfit["identity_weight_sum"] = id_weight
    overfit["analytic_label_smoothing_floor"] = floor
    overfit["excess_loss_ratio"] = (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)
    del model
    torch.cuda.empty_cache()
    checks = {
        "all_gradients_live": all(not r["missing_nonzero_gradients"] for r in (*capacities, overfit)),
        "all_frozen_states_unchanged": all(r["frozen_state_unchanged"] for r in (*capacities, overfit)),
        "overflow_zero": all(r["overflow_events"] == 0 for r in (*capacities, overfit)),
        "capacity_within_limit": all(r["peak_reserved_mib"] < 24576 for r in capacities),
        "tail_update_contract": not capacities[0]["private_tail_changed"] and capacities[1]["private_tail_changed"] and overfit["private_tail_changed"],
        "overfit_excess_ratio_at_most_point1": overfit["excess_loss_ratio"] <= 0.1,
    }
    report["m0"] = {"passed": all(checks.values()), "checks": checks,
                    "capacities": capacities, "overfit": overfit}
    save()
    print(json.dumps({"stage": "M0", **report["m0"]}), flush=True)
    if not report["m0"]["passed"]:
        report["status"] = "M0_FAIL"
        save()
        return

    all_ap = {endpoint: {name: [] for name in OUTPUTS} for endpoint in ENDPOINTS}
    all_ranks = {endpoint: {name: [] for name in OUTPUTS} for endpoint in ENDPOINTS}
    all_ids = []
    for fold, split in enumerate(splits):
        result = {"fold": fold, "gallery_manifest": [
            {"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in split["heldout_records"]
        ], "endpoints": {}}
        for endpoint in ENDPOINTS:
            train_tail = endpoint == ENDPOINTS[1]
            model, binding = build_model(config, signal_cfg, fold, split, train_tail)
            training = fit_endpoint(model, split["train_records"], config, fold=fold, endpoint=endpoint)
            checkpoint = args.output_dir / f"fold_{fold}_{endpoint}_final.pth"
            torch.save({
                "v19_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("baseline.")},
                "train_private_tail": train_tail, "fit_identity_ids": list(split["fit_identity_ids"]),
                "heldout_identity_ids": list(split["heldout_identity_ids"]), "source_binding": binding["source"],
                "plan_sha256": args.plan_sha256, "config_sha256": args.config_sha256,
            }, checkpoint)
            checkpoint_sha = _sha256(checkpoint)
            del model
            torch.cuda.empty_cache()
            model, _ = build_model(config, signal_cfg, fold, split, train_tail)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = model.state_dict()
            assert set(payload["v19_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
            assert payload["train_private_tail"] == train_tail and payload["source_binding"] == binding["source"]
            assert payload["plan_sha256"] == args.plan_sha256 and payload["config_sha256"] == args.config_sha256
            assert payload["fit_identity_ids"] == list(split["fit_identity_ids"])
            assert payload["heldout_identity_ids"] == list(split["heldout_identity_ids"])
            state.update(payload["v19_state_dict"])
            model.load_state_dict(state, strict=True)
            del state, payload
            assert _model_state_sha256(model) == training["final_state_sha256"]
            scores = evaluate(model, split["heldout_records"], config)
            assert _sha256(checkpoint) == checkpoint_sha
            result["endpoints"][endpoint] = {
                "binding": binding, "training": training, "outputs": scores,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                "strict_reload": True, "read_only_evaluation": True,
            }
            for name in OUTPUTS:
                all_ap[endpoint][name].extend(scores[name]["average_precision"])
                all_ranks[endpoint][name].extend(scores[name]["first_match_rank"])
            (args.output_dir / f"fold_{fold}_{endpoint}_receipt.json").write_text(
                json.dumps(result["endpoints"][endpoint], indent=2) + "\n")
            print(json.dumps({"stage": "Q1_final", "fold": fold, "endpoint": endpoint,
                              "metrics": {k: v["metrics_percent"] for k, v in scores.items()}}), flush=True)
            del model
            torch.cuda.empty_cache()
        a, b = [result["endpoints"][endpoint]["training"] for endpoint in ENDPOINTS]
        assert a["sample_order_sha256"] == b["sample_order_sha256"]
        assert a["first_eight_batch_receipts"] == b["first_eight_batch_receipts"]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert result["endpoints"][ENDPOINTS[0]]["outputs"]["baseline_only"] == result["endpoints"][ENDPOINTS[1]]["outputs"]["baseline_only"]
        all_ids.extend(split["heldout_records"][i][1] for i in scores["fused"]["query_indices"])
        report["folds"].append(result)
        save()
    aggregate = {
        endpoint: {
            name: {"mAP": float(np.mean(all_ap[endpoint][name]) * 100),
                   **{f"Rank-{k}": float(np.mean(np.array(all_ranks[endpoint][name]) <= k) * 100) for k in (1, 5, 10)}}
            for name in OUTPUTS
        } for endpoint in ENDPOINTS
    }
    a, b = ENDPOINTS
    gains = {name: aggregate[b][name]["mAP"] - aggregate[a][name]["mAP"] for name in OUTPUTS}
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound
    bootstrap = identity_cluster_bootstrap_lower_bound(
        torch.tensor(all_ap[b]["fused"], dtype=torch.float64) - torch.tensor(all_ap[a]["fused"], dtype=torch.float64),
        torch.tensor(all_ids), seed=42, resamples=10000,
    )
    fold_gains = [
        f["endpoints"][b]["outputs"]["fused"]["metrics_percent"]["mAP"]
        - f["endpoints"][a]["outputs"]["fused"]["metrics_percent"]["mAP"] for f in report["folds"]
    ]
    checks = {
        "aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.0,
        "all_fold_fused_nonnegative": all(g >= 0 for g in fold_gains),
        "all_expert_aggregate_nonnegative": all(gains[e] >= 0 for e in EXPERTS),
        "fused_bootstrap_lower_positive": bootstrap.lower_bound > 0,
        "fused_beats_baseline_and_experts": all(aggregate[b]["fused"]["mAP"] > aggregate[b][n]["mAP"] for n in ("baseline_only", *EXPERTS)),
    }
    for path, expected in source_bindings(config).items():
        assert _sha256(Path(path)) == expected
    assert len(all_ids) == 571 and sum(len(f["gallery_manifest"]) for f in report["folds"]) == 3126
    report.update({
        "status": "Q1_PASS" if all(checks.values()) else "Q1_FAIL", "aggregate": aggregate,
        "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains,
        "bootstrap": {"lower_bound_95_mAP": bootstrap.lower_bound * 100,
                      "clusters": bootstrap.cluster_count, "resamples": bootstrap.resamples},
        "scientific_checks": checks, "next_phase_qualified": all(checks.values()),
        "total_gallery_records": 3126, "total_eligible_queries": 571,
        "source_checkpoint_files_unchanged": True,
    })
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
