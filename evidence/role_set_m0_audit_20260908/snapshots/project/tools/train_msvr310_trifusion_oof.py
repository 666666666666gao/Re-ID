#!/usr/bin/env python3
"""Fixed original three-role training on the audited MSVR310 Signal folds."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_msvr310_signal_oof import (
    configure, loader_for, new_model, records_for, scene_scores, sha256, write_json,
)

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUT_WIDTHS = {"baseline_only": 3072, "fused": 7680,
                 "cnn": 4608, "transformer": 4608, "mamba": 4608}


def build_model(config, signal_config, fold, baseline_fold):
    import torch
    from tools.build_v12_complete_path_oof_targets import _build_v8_experts
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed

    _set_seed(42)
    signal = new_model(signal_config, fold)
    payload = torch.load(baseline_fold["checkpoint"], map_location="cpu", weights_only=True)
    assert payload["source_ids"] == fold["source_ids"]
    assert payload["heldout_ids"] == fold["heldout_ids"]
    assert payload["fold"] == fold["fold"]
    signal.load_state_dict(payload["model_state_dict"], strict=True)
    signal_state = _module_state_sha256(signal)
    assert signal_state == baseline_fold["training"]["final_state_sha256"]
    model = _build_v8_experts(
        signal, config, signal_checkpoint_sha256=baseline_fold["checkpoint_sha256"],
        num_classes=len(fold["source_ids"]),
    )
    assert all(not p.requires_grad for p in model.baseline.parameters())
    return model, {
        "initial_state_sha256": _module_state_sha256(model),
        "signal_state_sha256": signal_state,
        "signal_checkpoint_sha256": baseline_fold["checkpoint_sha256"],
        "role_weights_loaded": False, "role_initialization_seed": 42,
        "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
    }


def frozen_state_sha(model):
    from tools.train_signal_preserving_v17 import _tensor_mapping_sha256

    frozen_names = {name for name, p in model.named_parameters() if not p.requires_grad}
    return _tensor_mapping_sha256({
        name: value for name, value in model.state_dict().items()
        if name.startswith("baseline.") or name in frozen_names
    })


def output_mapping(output):
    assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
    values = {"baseline_only": output.baseline_embedding, "fused": output.fused_embedding,
              **dict(output.branch_embeddings)}
    assert {name: value.shape[1] for name, value in values.items()} == OUTPUT_WIDTHS
    return values


def extract(model, records, signal_config, *, verify_direct_signal=False):
    import torch
    from tools.run_signal_preserving_v5 import _training_batch

    model.eval()
    parts = {name: [] for name in OUTPUT_WIDTHS}
    with torch.no_grad():
        for raw in loader_for(records, False):
            batch, _ = _training_batch(raw)
            values = output_mapping(model(batch, return_aux=True))
            if verify_direct_signal:
                direct = model.baseline.signal(
                    batch["images"], cam_label=batch["camera_ids"], view_label=raw[3].cuda(),
                    training=False, sge=signal_config.MODEL.stageName,
                )
                assert torch.equal(direct, values["baseline_only"])
            for name, value in values.items():
                assert torch.isfinite(value).all().item()
                parts[name].append(value.float().cpu())
    return {name: torch.cat(rows) for name, rows in parts.items()}


def train_roles(model, records, record_indices, config, *, fold, mode, directory):
    import numpy as np
    import torch
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier,
        weighted_training_loss,
    )
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion

    _set_seed(42)
    model.train()
    initial, frozen = _module_state_sha256(model), frozen_state_sha(model)
    signal_initial = _module_state_sha256(model.baseline.signal)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    criterion = ExpertFormationV8Criterion(triplet_margin=config["LOSS"]["TRIPLET_MARGIN"],
                                           label_smoothing=config["LOSS"]["LABEL_SMOOTHING"]).cuda()
    loader = loader_for(records, True)
    index_by_name = {Path(row[0][0]).name: index
                     for row, index in zip(records, record_indices, strict=True)}
    assert len(index_by_name) == len(records)
    fixed_raw = next(iter(loader)) if mode == "overfit" else None
    history, steps, live, overflow = [], [], set(), 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, (20 if mode == "comparison" else 1) + 1):
        started = time.perf_counter()
        lr = config["OPTIMIZATION"]["NEW_MODULE_LR"]
        if mode == "comparison":
            lr *= learning_rate_multiplier(epoch, max_epochs=20, warmup_epochs=5)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if mode == "overfit":
            batches = (fixed_raw for _ in range(100))
        elif mode == "capacity":
            batches = itertools.islice(loader, 8)
        else:
            batches = loader
        epoch_rows = []
        for raw in batches:
            assert all(tuple(x.shape) == (64, 3, 128, 256) for x in raw[0].values())
            assert sorted(torch.unique(raw[1], return_counts=True)[1].tolist()) == [8] * 8
            batch, labels = _training_batch(raw)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(batch, return_aux=True)
                output_mapping(output)
                components = criterion(output, labels)
                loss = weighted_training_loss(components, config)
            assert torch.isfinite(loss).item()
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all().item(), name
                    if p.grad.abs().sum().item() > 0:
                        live.add(name)
            scaler.step(optimizer)
            scaler.update()
            overflow += int(scaler.get_scale() < scale)
            row = {"step": len(steps) + 1, "epoch": epoch, "loss": float(loss.detach()),
                   "components": {k: float(v.detach()) for k, v in components.items()},
                   "sampled_record_indices": [index_by_name[p] for p in raw[-1]],
                   "amp_scale_before": scale, "amp_scale_after": scaler.get_scale()}
            steps.append(row)
            epoch_rows.append(row)
        history.append({"epoch": epoch, "optimizer_steps": len(epoch_rows), "learning_rate": lr,
                        "mean_loss": float(np.mean([r["loss"] for r in epoch_rows])),
                        "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"event": "msvr310_role_epoch", "fold": fold, "mode": mode,
                          **history[-1]}), flush=True)
    report = {"mode": mode, "epochs": len(history), "optimizer_steps": len(steps),
              "initial_state_sha256": initial, "final_state_sha256": _module_state_sha256(model),
              "frozen_state_before_sha256": frozen, "frozen_state_after_sha256": frozen_state_sha(model),
              "signal_state_before_sha256": signal_initial,
              "signal_state_after_sha256": _module_state_sha256(model.baseline.signal),
              "trainable_tensors": len(names), "nonzero_gradient_tensors": len(live),
              "missing_nonzero_gradients": sorted(names - live), "overflow_events": overflow,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
              "history": history, "steps": steps}
    # Preserve the real training record before evaluating the engineering gate.
    write_json(directory / "training.json", report)
    return report


def engineering_checks(training):
    return {"all_trainable_gradients_live": not training["missing_nonzero_gradients"],
            "overflow_zero": training["overflow_events"] == 0,
            "frozen_state_unchanged": training["frozen_state_before_sha256"] == training["frozen_state_after_sha256"],
            "signal_state_unchanged": training["signal_state_before_sha256"] == training["signal_state_after_sha256"],
            "role_state_updated": training["initial_state_sha256"] != training["final_state_sha256"],
            "capacity_below_24gib": training["peak_reserved_mib"] < 24 * 1024}


def evaluate(features, protocol, fold, directory, baseline_fold):
    import numpy as np
    import torch
    from utils.metrics import eval_func_msrv

    original = torch.load(Path(baseline_fold["checkpoint"]).parent / "retrieval_arrays.pt",
                          map_location="cpu", weights_only=True)
    assert original["gallery_record_indices"] == fold["gallery_record_indices"]
    assert torch.equal(features["baseline_only"], original["features"])
    rows = [protocol["records"][i] for i in fold["gallery_record_indices"]]
    positions = [q["gallery_position"] for q in fold["query_rows"]]
    assert original["query_gallery_positions"] == positions
    assert all(rows[p]["index"] == q["record_index"]
               for p, q in zip(positions, fold["query_rows"], strict=True))
    ids, cameras, scenes = (np.asarray([r[key] for r in rows]) for key in ("identity", "camera", "scene"))
    scores, distances, rankings = {}, {}, {}
    for name, values in features.items():
        normalized = torch.nn.functional.normalize(values.float(), dim=1)
        qf, gf = normalized[positions], normalized
        matrix = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None]
        matrix.addmm_(qf, gf.T, beta=1, alpha=-2)
        if name == "baseline_only":
            assert torch.equal(matrix, original["distances"])
        scores[name] = scene_scores(matrix.numpy(), ids[positions], ids, scenes[positions], scenes)
        output_dir = directory / name
        output_dir.mkdir()
        os.chdir(output_dir)
        cmc, mean_ap = eval_func_msrv(matrix.numpy(), ids[positions], ids,
                                    cameras[positions], cameras, scenes[positions], scenes)
        difference = {"mAP": abs(scores[name]["metrics"]["mAP"] - float(mean_ap * 100))}
        difference.update({f"Rank-{k}": abs(scores[name]["metrics"][f"Rank-{k}"] - float(cmc[k - 1] * 100))
                           for k in (1, 5, 10)})
        assert difference["mAP"] < 1e-10 and max(difference.values()) < 1e-5
        scores[name]["upstream_metric_difference_pp"] = difference
        rankings[name] = np.argsort(matrix.numpy(), axis=1).tolist()
        distances[name] = matrix
    os.chdir(ROOT)
    assert scores["baseline_only"]["metrics"] == baseline_fold["retrieval"]["metrics"]
    torch.save({"features": features, "distances": distances,
                "gallery_record_indices": fold["gallery_record_indices"],
                "query_gallery_positions": positions}, directory / "retrieval_arrays.pt")
    write_json(directory / "rankings.json", rankings)
    return {"outputs": scores, "query_rows": fold["query_rows"], "gallery_manifest": rows,
            "feature_widths": OUTPUT_WIDTHS, "baseline_features_and_distances_bitwise_equal_to_b0": True,
            "retrieval_arrays_sha256": sha256(directory / "retrieval_arrays.pt"),
            "rankings_sha256": sha256(directory / "rankings.json")}


def comparison_summary(folds):
    import numpy as np
    import torch
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound

    identities = np.asarray([q["identity"] for f in folds for q in f["retrieval"]["query_rows"]])
    aps = {name: np.asarray([v for f in folds for v in f["retrieval"]["outputs"][name]["average_precision"]])
           for name in OUTPUT_WIDTHS}
    first = {name: np.asarray([v for f in folds for v in f["retrieval"]["outputs"][name]["first_match_rank"]])
             for name in OUTPUT_WIDTHS}
    assert len(identities) == 600 and len(np.unique(identities)) == 60
    metrics = {name: {"mAP": float(values.mean() * 100),
                      **{f"Rank-{k}": float(np.mean(first[name] <= k) * 100) for k in (1, 5, 10)}}
               for name, values in aps.items()}
    differences = {name: (values - aps["baseline_only"]) * 100 for name, values in aps.items()}
    bootstrap = identity_cluster_bootstrap_lower_bound(torch.from_numpy(differences["fused"]),
                                                       torch.from_numpy(identities), seed=42, resamples=10000)
    gains = {name: metrics[name]["mAP"] - metrics["baseline_only"]["mAP"] for name in OUTPUT_WIDTHS}
    fold_gains = [f["retrieval"]["outputs"]["fused"]["metrics"]["mAP"] -
                  f["retrieval"]["outputs"]["baseline_only"]["metrics"]["mAP"] for f in folds]
    checks = {"fused_gain_at_least_1pp": gains["fused"] >= 1.0,
              "all_fold_fused_gains_nonnegative": all(x >= 0 for x in fold_gains),
              "all_full_branches_not_below_signal": all(gains[x] >= 0 for x in EXPERTS),
              "identity_bootstrap_lower_positive": bootstrap.lower_bound > 0,
              "fused_strictly_best": all(metrics["fused"]["mAP"] > metrics[x]["mAP"]
                                           for x in ("baseline_only", *EXPERTS))}
    per_identity = [{"identity": int(identity), "query_count": int(np.sum(identities == identity)),
                     "map_by_output": {name: float(values[identities == identity].mean() * 100)
                                       for name, values in aps.items()}}
                    for identity in np.unique(identities)]
    changes = {name: {"ap_improved": int(np.sum(values > 0)), "ap_declined": int(np.sum(values < 0)),
                       "ap_unchanged": int(np.sum(values == 0)),
                       "rank1_repaired": int(np.sum((first["baseline_only"] > 1) & (first[name] == 1))),
                       "rank1_new_errors": int(np.sum((first["baseline_only"] == 1) & (first[name] > 1)))}
               for name, values in differences.items() if name != "baseline_only"}
    return {"metrics": metrics, "gains_over_signal_pp": gains, "fold_fused_gains_pp": fold_gains,
            "identity_bootstrap": {"lower_bound_pp": bootstrap.lower_bound, "seed": 42,
                                   "resamples": 10000, "cluster_count": bootstrap.cluster_count,
                                   "weighting": "resample whole identities with replacement; retain query weights",
                                   "percentile": 2.5, "quantile_method": "linear"},
            "scientific_checks": checks, "scientific_passed": all(checks.values()),
            "per_identity": per_identity, "query_changes": changes}


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import (
        ARCHITECTURE_V8, _module_state_sha256, evaluate_overfit_gate, overfit_loss_floor,
    )

    assert torch.cuda.is_available()
    config_path, directory = args.config.resolve(), args.output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["EXPERIMENT"]["SEED"] == 42 and config["MODEL"]["ARCHITECTURE"] == ARCHITECTURE_V8
    assert config["MODEL"]["GRID_SIZE"] == [8, 16]
    assert (config["OPTIMIZATION"]["MAX_EPOCHS"], config["OPTIMIZATION"]["WARMUP_EPOCHS"]) == (20, 5)
    assert not config["PROTOCOL"]["OFFICIAL_TEST_ACCESS"] and not config["PROTOCOL"]["RGBNT201_DEV_ACCESS"]
    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    baseline_config_path = ROOT / config["BASELINE"]["CONFIG"]
    assert sha256(baseline_config_path) == config["BASELINE"]["CONFIG_SHA256"]
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    assert sha256(config["BASELINE"]["SUMMARY"]) == config["BASELINE"]["SUMMARY_SHA256"]
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text(encoding="utf-8"))
    assert baseline["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert baseline["config_sha256"] == config["BASELINE"]["CONFIG_SHA256"]
    signal_config, binding = configure(baseline_config)
    protocol = json.loads((ROOT / baseline_config["protocol"]).read_text(encoding="utf-8"))
    assert sha256(ROOT / baseline_config["protocol"]) == baseline_config["protocol_sha256"]
    for fold in baseline["folds"]:
        assert sha256(fold["checkpoint"]) == fold["checkpoint_sha256"]
        assert sha256(Path(fold["checkpoint"]).parent / "retrieval_arrays.pt") == fold["retrieval"]["retrieval_arrays_sha256"]
    assert not directory.exists()
    directory.mkdir(parents=True)
    m0 = args.mode == "m0"
    if not m0:
        preflight = json.loads(args.m0_receipt.read_text(encoding="utf-8"))
        assert preflight["status"] == "PASS_ENGINEERING_ONLY"
        assert preflight["config_sha256"] == sha256(config_path)
        assert preflight["runner_sha256"] == sha256(__file__)
    started = time.perf_counter()
    summary = {"schema": "msvr310-trifusion-source-oof-v1", "mode": args.mode, "status": "RUNNING",
               "config_sha256": sha256(config_path), "runner_sha256": sha256(__file__),
               "baseline_summary_sha256": config["BASELINE"]["SUMMARY_SHA256"],
               "protocol_sha256": baseline_config["protocol_sha256"], "project_source_file_sha256": config["project_source_file_sha256"],
               "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               **binding, "folds": [], "seed": 42, "official_test_image_access": 0,
               "rgbnt201_dev_image_access": 0, "source_only_training": True,
               "evaluation_type": "source_only_engineering" if m0 else "train_internal_identity_oof_cross_dataset_comparison",
               "checkpoint_selection": "discarded_m0" if m0 else "fixed_epoch20",
               "m0_weights_reused": False, "rgbnt201_role_weights_reused": False,
               "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__)}
    write_json(directory / "summary.json", summary)
    for fold, baseline_fold in zip(protocol["folds"], baseline["folds"], strict=True):
        assert fold["fold"] == baseline_fold["fold"]
        assert not set(fold["source_ids"]) & set(fold["heldout_ids"])
        fold_dir = directory / f"fold_{fold['fold']}"
        fold_dir.mkdir()
        records = records_for(baseline_config, protocol, fold, True)
        model, model_binding = build_model(config, signal_config, fold, baseline_fold)
        if not m0:
            assert model_binding["initial_state_sha256"] == preflight["folds"][fold["fold"]]["initialization"]["initial_state_sha256"]
        if m0:
            extract(model, records[:8], signal_config, verify_direct_signal=True)
            assert _module_state_sha256(model) == model_binding["initial_state_sha256"]
        training = train_roles(model, records, fold["source_record_indices"], config,
                               fold=fold["fold"], mode="capacity" if m0 else "comparison", directory=fold_dir)
        checks = engineering_checks(training)
        checks["expected_training_length"] = training["optimizer_steps"] == 8 if m0 else training["epochs"] == 20
        row = {"fold": fold["fold"], "counts": fold["counts"], "initialization": model_binding,
               "training": training, "engineering_checks": checks}
        write_json(fold_dir / "receipt.json", row)
        summary["folds"].append(row)
        if not all(checks.values()):
            summary["status"] = "FAIL_FIXED_M0" if m0 else "FAIL_COMPARISON_ENGINEERING"
        write_json(directory / "summary.json", summary)
        assert all(checks.values()), checks
        checkpoint = fold_dir / ("roles_m0.pth" if m0 else "roles_epoch20.pth")
        torch.save({"model_state_dict": model.state_dict(), "fold": fold["fold"],
                    "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
                    "config_sha256": sha256(config_path)}, checkpoint)
        before_reload = extract(model, records[:8], signal_config) if m0 else None
        del model
        torch.cuda.empty_cache()
        model, _ = build_model(config, signal_config, fold, baseline_fold)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        assert _module_state_sha256(model) == training["final_state_sha256"]
        row.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                    "strict_reload_state_sha256": _module_state_sha256(model)})
        if m0:
            after_reload = extract(model, records[:8], signal_config)
            assert all(torch.equal(before_reload[k], after_reload[k]) for k in OUTPUT_WIDTHS)
            row.update({"direct_signal_source_parity": True, "strict_reload_all_outputs_bitwise_equal": True,
                        "clean_role_source_record_forwards": 24, "direct_signal_source_record_forwards": 8,
                        "heldout_record_forwards": 0})
        else:
            gallery = records_for(baseline_config, protocol, fold, False)
            features = extract(model, gallery, signal_config)
            row["retrieval"] = evaluate(features, protocol, fold, fold_dir, baseline_fold)
            row["heldout_record_forwards"] = len(gallery)
        assert _module_state_sha256(model) == training["final_state_sha256"]
        write_json(fold_dir / "receipt.json", row)
        write_json(directory / "summary.json", summary)
        del model, payload
        torch.cuda.empty_cache()
    if m0:
        overfit_dir = directory / "overfit_fold0"
        overfit_dir.mkdir()
        fold = protocol["folds"][0]
        model, initial = build_model(config, signal_config, fold, baseline["folds"][0])
        records = records_for(baseline_config, protocol, fold, True)
        training = train_roles(model, records, fold["source_record_indices"], config,
                               fold=0, mode="overfit", directory=overfit_dir)
        floor = overfit_loss_floor(config, num_classes=len(fold["source_ids"]))
        gate = evaluate_overfit_gate([r["loss"] for r in training["steps"]], max_ratio=0.1, minimum_loss=floor)
        checks = engineering_checks(training)
        checks.update({"fixed_100_updates": training["optimizer_steps"] == 100,
                       "overfit_excess_ratio_at_most_point1": gate["passed"]})
        summary["overfit"] = {"initialization": initial, "training": training, "loss_gate": gate, "checks": checks}
        summary["status"] = "PASS_ENGINEERING_ONLY" if all(checks.values()) else "FAIL_FIXED_M0"
        summary["optimizer_steps"] = 24 + training["optimizer_steps"]
    else:
        summary["comparison"] = comparison_summary(summary["folds"])
        summary["status"] = "COMPLETE_COMPARISON_SUPPORT_PASS" if summary["comparison"]["scientific_passed"] else "COMPLETE_COMPARISON_SUPPORT_FAIL"
        summary["optimizer_steps"] = sum(f["training"]["optimizer_steps"] for f in summary["folds"])
        summary["m0_receipt_sha256"] = sha256(args.m0_receipt)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["heldout_record_forwards"] = sum(f["heldout_record_forwards"] for f in summary["folds"])
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "complete", "status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("m0", "comparison"), required=True)
    parser.add_argument("--m0-receipt", type=Path)
    arguments = parser.parse_args()
    assert arguments.mode == "m0" or arguments.m0_receipt is not None
    run(arguments)
