#!/usr/bin/env python3
"""Read-only clean-source prototype and sample geometry for all V24 endpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from tools.build_v12_complete_path_oof_targets import (
    _configure_signal, _load_records, build_complete_path_fold_records,
)
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _sha256, _tensor_mapping_sha256,
)
from tools.train_signal_preserving_v18 import image_batch, loader_for
from tools.train_signal_preserving_v24 import ENDPOINTS, build_model, load_contract
from trifusion.source_prototype_v24 import CameraIdentityPrototypeMemory


def distribution(value):
    value = value.detach().double().flatten()
    assert len(value) and bool(torch.isfinite(value).all())
    levels = (0, .25, .5, .75, .9, .95, .99, 1)
    quantiles = torch.quantile(value, torch.tensor(levels, dtype=value.dtype, device=value.device))
    return {"count": len(value), "mean": float(value.mean()),
            "quantiles": dict(zip(map(str, levels), quantiles.tolist(), strict=True))}


def describe_columns(columns):
    return {"per_record": {name: value.tolist() for name, value in columns.items()},
            "distributions": {name: distribution(value) for name, value in columns.items()}}


def prototype_scores(features, labels, cameras, memory):
    query = F.normalize(features.float(), dim=1)
    global_similarity = query @ memory.global_prototypes().T
    environment_similarity = query @ memory.prototypes.T
    same_camera = cameras[:, None].eq(memory.pair_cameras[None, :])
    targets = same_camera & labels[:, None].eq(memory.pair_labels[None, :])
    assert bool(targets.sum(dim=1).eq(1).all())
    environment_targets = targets.long().argmax(dim=1)
    global_logits = global_similarity / memory.temperature
    environment_logits = (environment_similarity / memory.temperature).masked_fill(~same_camera, -torch.inf)
    global_ce = F.cross_entropy(global_logits, labels, reduction="none")
    environment_ce = F.cross_entropy(environment_logits, environment_targets, reduction="none")
    total = (global_ce + environment_ce) / 2
    original, _ = memory.loss(features, labels, cameras)
    loss_error = float((total.mean() - original).abs())
    assert loss_error < 1e-6
    rows = torch.arange(len(labels), device=labels.device)
    negative = labels[:, None].ne(torch.arange(len(memory.class_weights), device=labels.device)[None, :])
    negative_similarity, negative_label = global_similarity.masked_fill(~negative, -torch.inf).max(dim=1)
    positive_similarity = global_similarity[rows, labels]
    columns = {
        "global_cross_entropy": global_ce, "environment_cross_entropy": environment_ce,
        "combined_cross_entropy": total,
        "global_true_probability": global_logits.softmax(dim=1)[rows, labels],
        "environment_true_probability": environment_logits.softmax(dim=1)[rows, environment_targets],
        "global_correct": global_logits.argmax(dim=1).eq(labels),
        "environment_correct": environment_logits.argmax(dim=1).eq(environment_targets),
        "environment_candidate_pairs": same_camera.sum(dim=1),
        "global_positive_cosine": positive_similarity,
        "global_nearest_negative_cosine": negative_similarity,
        "global_positive_minus_nearest_negative_cosine": positive_similarity - negative_similarity,
    }
    result = describe_columns(columns)
    result.update({"global_nearest_negative_source_label": negative_label.tolist(),
                   "global_predicted_source_label": global_logits.argmax(dim=1).tolist(),
                   "environment_predicted_pair": environment_logits.argmax(dim=1).tolist(),
                   "environment_true_pair": environment_targets.tolist(),
                   "original_loss_mean_absolute_error": loss_error})
    result["all_source_identity_means"] = [
        {"source_label": int(label), "records": int(labels.eq(label).sum()),
         **{name: float(value[labels == label].double().mean()) for name, value in columns.items()}}
        for label in torch.unique(labels, sorted=True)
    ]
    return result


def sample_geometry(features, labels, cameras, fresh_memory):
    query = F.normalize(features.float(), dim=1)
    similarity = query @ query.T
    same_identity = labels[:, None].eq(labels[None, :])
    positive = same_identity & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same_identity
    assert bool(positive.any(dim=1).all()) and bool(negative.any(dim=1).all())
    closest_negative, negative_index = similarity.masked_fill(~negative, -torch.inf).max(dim=1)
    closest_positive, positive_index = similarity.masked_fill(~positive, -torch.inf).max(dim=1)
    hardest_positive, hardest_positive_index = similarity.masked_fill(~positive, torch.inf).min(dim=1)
    prototype_similarity = query @ fresh_memory.global_prototypes().T
    prototype_negative = labels[:, None].ne(torch.arange(len(fresh_memory.class_weights), device=labels.device)[None, :])
    nearest_negative_prototype = prototype_similarity.masked_fill(~prototype_negative, -torch.inf).max(dim=1).values
    result = describe_columns({
        "nearest_negative_cosine": closest_negative, "nearest_positive_cosine": closest_positive,
        "hardest_positive_cosine": hardest_positive,
        "nearest_positive_minus_nearest_negative_cosine": closest_positive - closest_negative,
        "hardest_positive_minus_nearest_negative_cosine": hardest_positive - closest_negative,
        "sample_negative_minus_fresh_negative_prototype_cosine": closest_negative - nearest_negative_prototype,
        "nearest_negative_same_camera": cameras[negative_index].eq(cameras),
    })
    result.update({"nearest_negative_record_index": negative_index.tolist(),
                   "nearest_positive_record_index": positive_index.tolist(),
                   "hardest_positive_record_index": hardest_positive_index.tolist(),
                   "hardest_positive_margin_negative_records": int((hardest_positive < closest_negative).sum())})
    cross_positive = same_identity & cameras[:, None].ne(cameras[None, :])
    indices = cross_positive.any(dim=1).nonzero().flatten()
    assert len(torch.unique(labels[indices])) == 14
    closest_cross, closest_cross_index = similarity[indices].masked_fill(~cross_positive[indices], -torch.inf).max(dim=1)
    hardest_cross, hardest_cross_index = similarity[indices].masked_fill(~cross_positive[indices], torch.inf).min(dim=1)
    result["real_cross_camera_positive_subset"] = {
        "query_indices": indices.tolist(), "identity_count": 14,
        "nearest_cross_camera_positive_index": closest_cross_index.tolist(),
        "hardest_cross_camera_positive_index": hardest_cross_index.tolist(),
        **describe_columns({"nearest_cross_camera_positive_cosine": closest_cross,
                            "hardest_cross_camera_positive_cosine": hardest_cross,
                            "nearest_cross_positive_minus_nearest_negative_cosine": closest_cross - closest_negative[indices],
                            "hardest_cross_positive_minus_nearest_negative_cosine": hardest_cross - closest_negative[indices]})}
    return result


def extract(model, records, config):
    model.eval()
    before = _model_state_sha256(model)
    batches = []
    with torch.inference_mode():
        for images, _, _, camera_labels, _, _ in loader_for(records, config):
            output = model(image_batch(images, camera_labels), return_aux=True)
            assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
            batches.append(output.fused_embedding.float())
    features = torch.cat(batches)
    assert _model_state_sha256(model) == before
    return features, {"model_state_sha256": before, "state_unchanged": True,
                      "record_forwards": len(records), "batch_forward_calls": len(batches),
                      "fused_features_sha256": _tensor_mapping_sha256({"fused": features})}


def load_memory(fresh, receipt, config):
    path = Path(receipt["file"])
    assert _sha256(path) == receipt["file_sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["temperature"] == config["PROTOTYPE"]["TEMPERATURE"]
    assert payload["momentum"] == config["PROTOTYPE"]["MOMENTUM"]
    memory = copy.deepcopy(fresh)
    memory.load_state_dict(payload["state_dict"], strict=True)
    assert _tensor_mapping_sha256(memory.state_dict()) == receipt["state_sha256"]
    for name in ("pair_labels", "pair_cameras", "class_weights"):
        assert torch.equal(getattr(memory, name), getattr(fresh, name))
    assert _sha256(path) == receipt["file_sha256"]
    return memory


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    assert _sha256(args.q1_summary) == args.q1_sha256
    assert _sha256(args.plan) == args.plan_sha256
    plan = json.loads(args.plan.read_text())
    assert _sha256(Path(__file__)) == plan["script_sha256"]
    q1 = json.loads(args.q1_summary.read_text())
    assert q1["status"] in ("Q1_PASS", "Q1_FAIL") and len(q1["folds"]) == 3
    assert q1["repository_commit"] == plan["q1_execution_commit"]
    assert _sha256(args.config) == q1["config_sha256"] == plan["q1_config_sha256"]
    for path, expected in q1["source_file_sha256"].items():
        assert _sha256(Path(path)) == expected, path
    config, sources = load_contract(args.config)
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    assert (signal_commit, signal_diff) == (q1["signal_commit"], q1["signal_diff_sha256"])
    records = _load_records(config)
    assert len(records) == 3126
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": plan["scope"],
              "evaluation_type": "real_dataset_labels_seen_source_geometry_not_unknown_identity_validation",
              "q1_summary_sha256": args.q1_sha256, "plan_sha256": args.plan_sha256,
              "script_sha256": plan["script_sha256"],
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "q1_execution_commit": q1["repository_commit"], "seed": 42,
              "optimizer_steps": 0, "backward_calls": 0, "checkpoint_writes": 0,
              "heldout_image_forward_calls": 0, "dev_access_count": 0, "official_test_access_count": 0,
              "augmentation": "existing deterministic clean evaluation transform; FP32; no autocast",
              "prototype_self_inclusion": "fresh source prototypes include the scored source records",
              "folds": []}
    for index, registry in enumerate(sources["fold_receipts"]):
        split = build_complete_path_fold_records(records, heldout_ids=set(registry["heldout_identity_ids"]))
        source_records = split["train_records"]
        assert not split["identity_overlap"] and len(source_records) == (2126, 2075, 2051)[index]
        labels = torch.tensor([r[1] for r in source_records], device="cuda")
        cameras = torch.tensor([r[2] for r in source_records], device="cuda")
        assert len(torch.unique(labels)) == 94
        saved_endpoints = q1["folds"][index]["endpoints"]
        for endpoint in ENDPOINTS:
            initial = saved_endpoints[endpoint]["training"]["memory_initial"]
            assert _sha256(Path(initial["file"])) == initial["file_sha256"]
        assert saved_endpoints[ENDPOINTS[0]]["training"]["memory_initial"]["state_sha256"] == saved_endpoints[ENDPOINTS[1]]["training"]["memory_initial"]["state_sha256"]
        fold = {"fold": index, "source_identity_mapping": list(split["fit_identity_ids"]),
                "source_manifest": [{"file": Path(r[0][0]).name, "source_label": r[1], "camera": r[2],
                                     "fit_registry_identity": split["fit_identity_ids"][r[1]]} for r in source_records],
                "models": {}}
        order_sha = hashlib.sha256(json.dumps([Path(r[0][0]).name for r in source_records]).encode()).hexdigest()
        for name in ("initial", *ENDPOINTS):
            enabled = name == ENDPOINTS[1]
            endpoint = ENDPOINTS[0] if name == "initial" else name
            saved = saved_endpoints[endpoint]
            model, binding = build_model(config, signal_cfg, index, split, enabled)
            assert binding == saved["binding"]
            assert binding["source_model_state_sha256"] == saved["training"]["initial_state_sha256"]
            if name != "initial":
                checkpoint = Path(saved["checkpoint"])
                assert _sha256(checkpoint) == saved["checkpoint_sha256"]
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert payload["binding"] == binding and payload["prototype_loss_enabled"] == enabled
                assert payload["plan_sha256"] == q1["plan_sha256"] and payload["config_sha256"] == q1["config_sha256"]
                state = model.state_dict()
                assert set(payload["v24_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
                state.update(payload["v24_state_dict"])
                model.load_state_dict(state, strict=True)
                del state, payload
                assert _model_state_sha256(model) == saved["training"]["final_state_sha256"]
            features, extraction = extract(model, source_records, config)
            fresh = CameraIdentityPrototypeMemory(features, labels, cameras,
                        temperature=config["PROTOTYPE"]["TEMPERATURE"], momentum=config["PROTOTYPE"]["MOMENTUM"])
            assert tuple(fresh.prototypes.shape) == (108, 7680) and not list(fresh.parameters())
            memory_receipt = saved["training"]["memory_initial" if name == "initial" else "memory_final"]
            cached = load_memory(fresh, memory_receipt, config)
            if name == "initial":
                assert order_sha == memory_receipt["source_file_order_sha256"]
                assert _tensor_mapping_sha256(fresh.state_dict()) == memory_receipt["state_sha256"]
            with torch.inference_mode():
                cached_scores = prototype_scores(features, labels, cameras, cached)
                fresh_scores = prototype_scores(features, labels, cameras, fresh)
                geometry = sample_geometry(features, labels, cameras, fresh)
                pair_cosine = F.cosine_similarity(cached.prototypes, fresh.prototypes, dim=1)
                global_cosine = F.cosine_similarity(cached.global_prototypes(), fresh.global_prototypes(), dim=1)
            assert _model_state_sha256(model) == extraction["model_state_sha256"]
            assert _tensor_mapping_sha256(cached.state_dict()) == memory_receipt["state_sha256"]
            if name != "initial":
                assert _sha256(checkpoint) == saved["checkpoint_sha256"]
            fold["models"][name] = {"binding": binding, "extraction": extraction,
                "memory_file": memory_receipt["file"], "memory_file_sha256": memory_receipt["file_sha256"],
                "cached_memory_state_sha256": memory_receipt["state_sha256"],
                "fresh_memory_state_sha256": _tensor_mapping_sha256(fresh.state_dict()),
                "cached_prototype_scores": cached_scores, "fresh_prototype_scores": fresh_scores,
                "sample_geometry": geometry,
                "memory_drift": {"pair_labels": cached.pair_labels.tolist(), "pair_cameras": cached.pair_cameras.tolist(),
                                 "pair_cosine": pair_cosine.tolist(), "pair_cosine_distribution": distribution(pair_cosine),
                                 "global_cosine": global_cosine.tolist(), "global_cosine_distribution": distribution(global_cosine),
                                 "last_update": cached.last_update.tolist(), "update_count": cached.update_count.tolist()}}
            print(json.dumps({"fold": index, "model": name,
                "cached_clean_source_prototype_ce": cached_scores["distributions"]["combined_cross_entropy"]["mean"],
                "sample_hardest_positive_margin_negative_records": geometry["hardest_positive_margin_negative_records"],
                "elapsed_seconds": time.time() - started}), flush=True)
            del model, features, fresh, cached
            torch.cuda.empty_cache()
        report["folds"].append(fold)
        (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    models = [model for fold in report["folds"] for model in fold["models"].values()]
    report.update({"status": "ALL_NINE_SOURCE_MODELS_DIAGNOSED_READ_ONLY",
                   "models_extracted": len(models),
                   "source_record_forwards": sum(model["extraction"]["record_forwards"] for model in models),
                   "source_batch_forward_calls": sum(model["extraction"]["batch_forward_calls"] for model in models),
                   "elapsed_seconds": time.time() - started})
    assert (report["models_extracted"], report["source_record_forwards"], report["source_batch_forward_calls"]) == (9, 18756, 153)
    assert _sha256(args.q1_summary) == args.q1_sha256
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "models_extracted", "source_record_forwards", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--q1-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
