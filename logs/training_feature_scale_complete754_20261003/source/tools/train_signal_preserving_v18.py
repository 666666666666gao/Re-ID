#!/usr/bin/env python3
"""Run the fixed V18 source-paired projection M0 and complete three-fold Q1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.audit_v17_full_gallery import full_gallery_scores
from tools.build_v12_complete_path_oof_targets import (
    _load_records, _configure_signal, _eval_loader, build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _set_seed
from tools.train_signal_preserving_v17 import (
    _load_contract, _build_fold_model, _model_state_sha256, _frozen_state_sha256,
    _tensor_mapping_sha256, _current_source_hashes, _sha256, _initial_endpoint_receipt,
    _fit_endpoint, _criterion, _new_optimizer, _training_loader, _optimization_step,
    _trainable_names,
)

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("uncentered", "projected")


def loader_for(records, config):
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    template = _eval_loader(records, config)
    return torch.utils.data.DataLoader(
        ImageDataset(records, template.dataset.transform),
        batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]), shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]), collate_fn=val_collate_fn,
    )


def image_batch(images, camera_labels):
    return {
        "images": {name: value.cuda() for name, value in images.items()},
        "camera_ids": camera_labels.cuda(),
        "modality_mask": torch.ones(len(camera_labels), 3, dtype=torch.bool, device="cuda"),
    }


def build_model(config, signal_cfg, fold, split, directions, enabled):
    from trifusion.signal_preserving_v18 import PairedViewProjectionV18
    _set_seed(42)
    model, binding = _build_fold_model(config, signal_cfg, fold_index=fold, split=split)
    model.correction = PairedViewProjectionV18(model.correction, directions, enabled=enabled).cuda()
    binding["architecture"] = "signal_preserving_v18_paired_view_projection"
    binding["projection_enabled"] = bool(enabled)
    binding["projection_rank_per_expert"] = 1
    return model, binding


def calibrate(config, signal_cfg, fold, split, directory):
    from trifusion.signal_preserving_v18 import estimate_paired_direction
    _set_seed(42)
    model, binding = _build_fold_model(config, signal_cfg, fold_index=fold, split=split)
    model.eval()
    frozen = _frozen_state_sha256(model)
    features = {expert: [] for expert in EXPERTS}
    records = split["train_records"]
    with torch.inference_mode():
        for images, _, _, camera_labels, _, _ in loader_for(records, config):
            output = model.base_v8(image_batch(images, camera_labels), return_aux=True)
            for expert in EXPERTS:
                features[expert].append(output.residual_embeddings[expert].float().cpu())
    features = {name: torch.cat(value) for name, value in features.items()}
    identities = torch.tensor([r[1] for r in records])
    cameras = torch.tensor([r[2] for r in records])
    source_cache = directory / f"fold_{fold}_source_residuals.pt"
    torch.save({**features, "identities": identities, "cameras": cameras}, source_cache)
    directions = []
    fits = {}
    for expert in EXPERTS:
        direction, fit = estimate_paired_direction(features[expert], identities, cameras)
        directions.append(direction)
        fits[expert] = fit
    directions = torch.stack(directions)
    assert _frozen_state_sha256(model) == frozen
    assert not split["identity_overlap"]
    receipt = {
        "fold": fold, "fit_identity_ids": list(split["fit_identity_ids"]),
        "heldout_identity_ids": list(split["heldout_identity_ids"]),
        "source_records": len(records), "source_cache": str(source_cache),
        "source_cache_sha256": _sha256(source_cache), "binding": binding,
        "fits": fits, "directions": directions.tolist(),
        "directions_sha256": _tensor_mapping_sha256({"directions": directions}),
        "frozen_state_unchanged": True, "heldout_records_used_for_calibration": 0,
    }
    (directory / f"fold_{fold}_calibration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"stage": "source_calibration", "fold": fold,
                      "pairs": len(fits["cnn"]["identity_camera_pairs"]),
                      "direction_energy": {e: fits[e]["top_direction_energy_fraction"] for e in EXPERTS}}), flush=True)
    del model, features
    torch.cuda.empty_cache()
    return directions, receipt


def fixed_steps(model, records, config, *, steps, fixed_batch):
    _set_seed(42)
    loader = _training_loader(records, config)
    iterator = iter(loader)
    fixed = next(iterator) if fixed_batch else None
    criterion = _criterion(config, envelope_enabled=False)
    optimizer, scaler = _new_optimizer(model, config)
    frozen = _frozen_state_sha256(model)
    direction_before = model.correction.directions.detach().clone()
    losses, gradients, overflow = [], set(), 0
    model.train()
    for _ in range(steps):
        values, bad, names = _optimization_step(model, criterion, fixed if fixed_batch else next(iterator), optimizer, scaler, config)
        losses.append(values["total"])
        gradients.update(names)
        overflow += int(bad)
    return {
        "steps": steps, "losses": losses, "overflow_events": overflow,
        "trainable_tensors": len(_trainable_names(model)), "nonzero_gradient_tensors": len(gradients),
        "missing_nonzero_gradients": sorted(set(_trainable_names(model)) - gradients),
        "frozen_state_unchanged": frozen == _frozen_state_sha256(model),
        "directions_unchanged": torch.equal(direction_before, model.correction.directions),
    }


def evaluate(model, records, config):
    model.eval()
    before = _model_state_sha256(model)
    features = {name: [] for name in OUTPUTS}
    with torch.inference_mode():
        for images, _, _, camera_labels, _, _ in loader_for(records, config):
            output = model(image_batch(images, camera_labels), return_aux=True)
            assert bool(output.diagnostics["baseline_exact_prefix"])
            features["baseline_only"].append(output.baseline_embedding.float().cpu())
            features["fused"].append(output.fused_embedding.float().cpu())
            for name in EXPERTS:
                features[name].append(output.branch_embeddings[name].float().cpu())
    identities = np.array([r[1] for r in records])
    cameras = np.array([r[2] for r in records])
    scores = {}
    for name in OUTPUTS:
        value = F.normalize(torch.cat(features[name]), dim=1)
        scores[name] = full_gallery_scores(torch.cdist(value, value).numpy(), identities, cameras)
    assert before == _model_state_sha256(model)
    return scores


def run(args):
    started = time.time()
    assert _sha256(args.plan) == args.plan_sha256
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = _load_contract(args.base_config.resolve())
    config = contract["config"]
    assert int(config["OPTIMIZATION"]["MAX_EPOCHS"]) == 20
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound
    import trifusion.signal_preserving_v18 as module
    report = {
        "schema_version": "v18-paired-view-projection-main-v1", "status": "RUNNING",
        "architecture": "signal_preserving_v18_paired_view_projection", "seed": 42,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": _sha256(Path(__file__)), "module_sha256": _sha256(Path(module.__file__)),
        "plan_sha256": args.plan_sha256, "base_source_sha256": _current_source_hashes(contract),
        "signal_commit": signal_commit, "signal_diff_sha256": signal_diff,
        "envelope_enabled": False, "rank_per_expert": 1, "epochs_per_endpoint": 20,
        "model_selection": "none_final_epoch_only", "dev_access_count": 0,
        "official_test_access_count": 0, "calibrations": [], "folds": [],
    }

    def save():
        report["elapsed_seconds"] = time.time() - started
        (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    splits, axes = [], []
    for fold, registry in enumerate(contract["v12_summary"]["fold_receipts"]):
        split = build_complete_path_fold_records(records, heldout_ids=set(registry["heldout_identity_ids"]))
        directions, receipt = calibrate(config, signal_cfg, fold, split, args.output_dir)
        splits.append(split)
        axes.append(directions)
        report["calibrations"].append(receipt)
        save()

    preflight = []
    for fold, (split, directions) in enumerate(zip(splits, axes, strict=True)):
        receipts = []
        for enabled in (False, True):
            model, _ = build_model(config, signal_cfg, fold, split, directions, enabled)
            receipts.append(_initial_endpoint_receipt(model, split["train_records"], config))
            del model
            torch.cuda.empty_cache()
        a, b = receipts
        paired = a["initial_state_sha256"] == b["initial_state_sha256"] and a["batch_receipts"] == b["batch_receipts"]
        assert paired and a["exact_signal_prefix"] and b["exact_signal_prefix"]
        assert all(r["all_batches_have_positive_and_negative_pairs"] and r["teacher_outputs_detached"] and r["frozen_state_unchanged"] for r in receipts)
        preflight.append({"fold": fold, "paired": paired, "endpoints": receipts})
    report["preflight"] = preflight
    model, _ = build_model(config, signal_cfg, 0, splits[0], axes[0], True)
    torch.cuda.reset_peak_memory_stats()
    capacity = fixed_steps(model, splits[0]["train_records"], config, steps=8, fixed_batch=False)
    capacity["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    del model
    torch.cuda.empty_cache()
    model, _ = build_model(config, signal_cfg, 0, splits[0], axes[0], True)
    overfit = fixed_steps(model, splits[0]["train_records"], config, steps=100, fixed_batch=True)
    classes, smoothing = len(splits[0]["fit_identity_ids"]), float(config["LOSS"]["LABEL_SMOOTHING"])
    correct, other = 1 - smoothing + smoothing / classes, smoothing / classes
    floor = 2 * (-correct * math.log(correct) - (classes - 1) * other * math.log(other))
    overfit["analytic_label_smoothing_floor"] = floor
    overfit["excess_loss_ratio"] = (overfit["losses"][-1] - floor) / (overfit["losses"][0] - floor)
    del model
    torch.cuda.empty_cache()
    m0 = all(x["trainable_tensors"] == 22 and x["frozen_state_unchanged"] and x["directions_unchanged"] and x["overflow_events"] == 0 and not x["missing_nonzero_gradients"] for x in (capacity, overfit)) and overfit["excess_loss_ratio"] <= 0.1
    report["m0"] = {"passed": m0, "capacity": capacity, "overfit": overfit}
    save()
    print(json.dumps({"stage": "M0", "passed": m0, "overfit_ratio": overfit["excess_loss_ratio"]}), flush=True)
    if not m0:
        report["status"] = "M0_FAIL"
        save()
        return

    all_ap = {endpoint: {name: [] for name in OUTPUTS} for endpoint in ENDPOINTS}
    all_ranks = {endpoint: {name: [] for name in OUTPUTS} for endpoint in ENDPOINTS}
    all_ids = []
    for fold, (split, directions) in enumerate(zip(splits, axes, strict=True)):
        result = {"fold": fold, "gallery_manifest": [{"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in split["heldout_records"]], "endpoints": {}}
        for endpoint in ENDPOINTS:
            enabled = endpoint == "projected"
            model, binding = build_model(config, signal_cfg, fold, split, directions, enabled)
            training = _fit_endpoint(model, split["train_records"], config, envelope_enabled=False)
            training["endpoint"] = endpoint
            assert training["epochs"] == 20 and training["overflow_events"] == 0
            assert not training["missing_nonzero_gradient_tensors"] and training["frozen_state_unchanged"]
            assert torch.equal(model.correction.directions.cpu(), directions)
            checkpoint = args.output_dir / f"fold_{fold}_{endpoint}_final.pth"
            torch.save({"v18_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("base_v8.")},
                        "projection_enabled": enabled, "fit_identity_ids": list(split["fit_identity_ids"]),
                        "heldout_identity_ids": list(split["heldout_identity_ids"]), "plan_sha256": args.plan_sha256}, checkpoint)
            checkpoint_sha = _sha256(checkpoint)
            del model
            torch.cuda.empty_cache()
            model, _ = build_model(config, signal_cfg, fold, split, directions, enabled)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = model.state_dict()
            assert set(payload["v18_state_dict"]) == {k for k in state if not k.startswith("base_v8.")}
            assert payload["projection_enabled"] == enabled and payload["plan_sha256"] == args.plan_sha256
            state.update(payload["v18_state_dict"])
            model.load_state_dict(state, strict=True)
            del state, payload
            assert _model_state_sha256(model) == training["final_state_sha256"]
            scores = evaluate(model, split["heldout_records"], config)
            assert _sha256(checkpoint) == checkpoint_sha
            result["endpoints"][endpoint] = {"binding": binding, "training": training, "outputs": scores,
                                           "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                                           "strict_reload": True, "read_only_evaluation": True}
            for name in OUTPUTS:
                all_ap[endpoint][name].extend(scores[name]["average_precision"])
                all_ranks[endpoint][name].extend(scores[name]["first_match_rank"])
            print(json.dumps({"stage": "Q1_final", "fold": fold, "endpoint": endpoint,
                              "metrics": {k: v["metrics_percent"] for k, v in scores.items()}}), flush=True)
            (args.output_dir / f"fold_{fold}_{endpoint}_receipt.json").write_text(json.dumps(result["endpoints"][endpoint], indent=2) + "\n")
            del model
            torch.cuda.empty_cache()
        a, b = [result["endpoints"][e]["training"] for e in ENDPOINTS]
        assert a["sample_order_sha256"] == b["sample_order_sha256"]
        assert a["first_eight_batch_receipts"] == b["first_eight_batch_receipts"]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert result["endpoints"]["uncentered"]["outputs"]["baseline_only"] == result["endpoints"]["projected"]["outputs"]["baseline_only"]
        all_ids.extend(split["heldout_records"][i][1] for i in scores["fused"]["query_indices"])
        report["folds"].append(result)
        save()
    aggregate = {endpoint: {name: {"mAP": float(np.mean(all_ap[endpoint][name]) * 100),
                                 **{f"Rank-{k}": float(np.mean(np.array(all_ranks[endpoint][name]) <= k) * 100) for k in (1, 5, 10)}}
                            for name in OUTPUTS} for endpoint in ENDPOINTS}
    gains = {name: aggregate["projected"][name]["mAP"] - aggregate["uncentered"][name]["mAP"] for name in OUTPUTS}
    bootstrap = identity_cluster_bootstrap_lower_bound(torch.tensor(all_ap["projected"]["fused"]) - torch.tensor(all_ap["uncentered"]["fused"]), torch.tensor(all_ids), seed=42, resamples=10000)
    fold_gains = [f["endpoints"]["projected"]["outputs"]["fused"]["metrics_percent"]["mAP"] - f["endpoints"]["uncentered"]["outputs"]["fused"]["metrics_percent"]["mAP"] for f in report["folds"]]
    checks = {
        "aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.0,
        "all_fold_fused_nonnegative": all(g >= 0 for g in fold_gains),
        "all_expert_aggregate_nonnegative": all(gains[e] >= 0 for e in EXPERTS),
        "fused_bootstrap_lower_positive": bootstrap.lower_bound > 0,
        "fused_beats_baseline_and_experts": all(aggregate["projected"]["fused"]["mAP"] > aggregate["projected"][name]["mAP"] for name in ("baseline_only", *EXPERTS)),
    }
    report.update({"status": "Q1_PASS" if all(checks.values()) else "Q1_FAIL", "aggregate": aggregate,
                   "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains,
                   "bootstrap": {"lower_bound_95_mAP": bootstrap.lower_bound * 100, "clusters": bootstrap.cluster_count, "resamples": bootstrap.resamples},
                   "scientific_checks": checks, "d1_executed": False,
                   "next_phase_qualified": all(checks.values()),
                   "total_gallery_records": sum(len(f["gallery_manifest"]) for f in report["folds"]),
                   "total_eligible_queries": len(all_ids)})
    save()
    print(json.dumps({k: report[k] for k in ("status", "aggregate", "matched_gains_mAP", "scientific_checks", "bootstrap")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
