#!/usr/bin/env python3
"""Inspect all V19 final endpoints without updating weights or selecting models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.audit_v17_full_gallery import full_gallery_scores
from tools.build_v12_complete_path_oof_targets import (
    _configure_signal, _load_records, build_complete_path_fold_records,
)
from tools.diagnose_v17_failure_geometry import match_rows
from tools.train_signal_preserving_v17 import _model_state_sha256, _sha256
from tools.train_signal_preserving_v18 import image_batch, loader_for
from tools.train_signal_preserving_v19 import build_model, load_contract

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)
ENDPOINTS = ("frozen_private_tail", "trained_private_tail")
MODALITIES = ("RGB", "NI", "TI")


def modal_geometry(value, identities, cameras, queries):
    """Record all nine modality pair geometries, never construct new retrieval heads."""
    positive = (identities[queries, None] == identities[None, :]) & (
        cameras[queries, None] != cameras[None, :]
    )
    negative = identities[queries, None] != identities[None, :]
    result = {}
    for left, query_modality in enumerate(MODALITIES):
        for right, gallery_modality in enumerate(MODALITIES):
            similarity = (value[queries, left] @ value[:, right].T).numpy()
            nearest_negative = np.where(negative, similarity, -np.inf).argmax(axis=1)
            rows = {
                "same_instance_cosine": similarity[np.arange(len(queries)), queries].tolist(),
                "mean_cross_camera_positive_cosine": (
                    np.where(positive, similarity, 0).sum(axis=1) / positive.sum(axis=1)
                ).tolist(),
                "nearest_positive_cosine": np.where(positive, similarity, -np.inf).max(axis=1).tolist(),
                "nearest_negative_cosine": similarity[np.arange(len(queries)), nearest_negative].tolist(),
                "nearest_negative_index": nearest_negative.tolist(),
                "nearest_negative_same_camera": (cameras[nearest_negative] == cameras[queries]).tolist(),
            }
            result[f"{query_modality}_to_{gallery_modality}"] = {
                "query_indices": queries.tolist(), "arrays": rows,
                "means": {key: float(np.mean(values)) for key, values in rows.items()
                          if key != "nearest_negative_index"},
            }
    return result


def extract(model, records, config, *, source):
    model.eval()
    before = _model_state_sha256(model)
    features = {name: [] for name in OUTPUTS}
    modal = {name: [] for name in EXPERTS}
    logits = {name: [] for name in ("fused", *EXPERTS, *("residual_" + e for e in EXPERTS))}
    with torch.inference_mode():
        for images, _, _, camera_labels, _, _ in loader_for(records, config):
            output = model(image_batch(images, camera_labels), return_aux=True)
            assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
            features["baseline_only"].append(output.baseline_embedding.float().cpu())
            features["fused"].append(output.fused_embedding.float().cpu())
            for expert in EXPERTS:
                features[expert].append(output.branch_embeddings[expert].float().cpu())
                modal[expert].append(output.modal_residual_embeddings[expert].float().cpu())
            if source:
                logits["fused"].append(output.fused_logits.float().cpu())
                for expert in EXPERTS:
                    logits[expert].append(output.branch_logits[expert].float().cpu())
                    logits["residual_" + expert].append(output.residual_logits[expert].float().cpu())
    assert before == _model_state_sha256(model)
    identities = np.array([r[1] for r in records])
    cameras = np.array([r[2] for r in records])
    counts = Counter(identities.tolist())
    camera_counts = Counter(zip(identities.tolist(), cameras.tolist()))
    queries = np.array([i for i, (identity, camera) in enumerate(zip(identities, cameras, strict=True))
                        if counts[identity] > camera_counts[(identity, camera)]])
    outputs = {}
    for name, batches in features.items():
        value = F.normalize(torch.cat(batches), dim=1)
        distance = torch.cdist(value, value).numpy()
        scores = full_gallery_scores(distance, identities, cameras)
        assert scores["query_indices"] == queries.tolist()
        outputs[name] = {"scores": scores, "queries": match_rows(distance, identities, cameras, scores)}
    result = {
        "gallery": len(records), "identities": len(counts), "eligible_queries": len(queries),
        "cross_camera_identities": len(set(identities[queries].tolist())),
        "excluded_only_from_query": len(records) - len(queries),
        "gallery_manifest": [{"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in records],
        "outputs": outputs,
        "modal_geometry": {expert: modal_geometry(torch.cat(modal[expert]), identities, cameras, queries)
                           for expert in EXPERTS},
        "state_unchanged": True, "model_state_sha256": before,
    }
    if source:
        labels = torch.as_tensor(identities)
        classification = {}
        for name, batches in logits.items():
            value = torch.cat(batches)
            ce = F.cross_entropy(value, labels, label_smoothing=config["LOSS"]["LABEL_SMOOTHING"], reduction="none")
            prediction = value.argmax(dim=1)
            classification[name] = {"predicted_identity": prediction.tolist(), "smoothed_cross_entropy": ce.tolist(),
                                    "accuracy_percent": float((prediction == labels).float().mean() * 100),
                                    "mean_smoothed_cross_entropy": float(ce.mean())}
        result["clean_source_classification"] = classification
    return result


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    assert _sha256(args.q1_summary) == args.q1_sha256
    assert _sha256(args.protocol) == args.protocol_sha256
    q1 = json.loads(args.q1_summary.read_text())
    assert q1["status"] == "Q1_FAIL"
    assert _sha256(args.config) == q1["config_sha256"]
    for path, expected in q1["source_file_sha256"].items():
        assert _sha256(Path(path)) == expected, path
    config, sources = load_contract(args.config)
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    assert (signal_commit, signal_diff) == (q1["signal_commit"], q1["signal_diff_sha256"])
    records = _load_records(config)
    assert len(records) == 3126
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING", "scope": "read_only_train_source_and_reused_oof_postmortem_not_new_validation",
        "q1_summary_sha256": args.q1_sha256, "protocol_sha256": args.protocol_sha256,
        "script_sha256": _sha256(Path(__file__)),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_execution_commit": q1["repository_commit"], "seed": 42,
        "optimizer_steps": 0, "checkpoint_writes": 0, "dev_access_count": 0, "official_test_access_count": 0,
        "folds": [],
    }
    for index, registry in enumerate(sources["fold_receipts"]):
        split = build_complete_path_fold_records(records, heldout_ids=set(registry["heldout_identity_ids"]))
        assert not split["identity_overlap"]
        fold = {"fold": index, "source_identity_mapping": list(split["fit_identity_ids"]), "endpoints": {}}
        for endpoint in ENDPOINTS:
            saved = q1["folds"][index]["endpoints"][endpoint]
            checkpoint = Path(saved["checkpoint"])
            assert _sha256(checkpoint) == saved["checkpoint_sha256"]
            trained = endpoint == ENDPOINTS[1]
            model, binding = build_model(config, signal_cfg, index, split, trained)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["train_private_tail"] == trained
            assert payload["source_binding"] == binding["source"]
            assert payload["plan_sha256"] == q1["plan_sha256"] and payload["config_sha256"] == q1["config_sha256"]
            assert payload["fit_identity_ids"] == list(split["fit_identity_ids"])
            assert payload["heldout_identity_ids"] == list(split["heldout_identity_ids"])
            state = model.state_dict()
            assert set(payload["v19_state_dict"]) == {k for k in state if not k.startswith("baseline.")}
            state.update(payload["v19_state_dict"])
            model.load_state_dict(state, strict=True)
            del payload, state
            assert _model_state_sha256(model) == saved["training"]["final_state_sha256"]
            heldout = extract(model, split["heldout_records"], config, source=False)
            assert heldout["gallery_manifest"] == q1["folds"][index]["gallery_manifest"]
            for name in OUTPUTS:
                assert heldout["outputs"][name]["scores"] == saved["outputs"][name], (index, endpoint, name)
            source = extract(model, split["train_records"], config, source=True)
            assert source["identities"] == 94 and source["cross_camera_identities"] == 14
            assert heldout["identities"] == 47 and heldout["cross_camera_identities"] == 7
            assert source["gallery"] + heldout["gallery"] == 3126
            assert _sha256(checkpoint) == saved["checkpoint_sha256"]
            fold["endpoints"][endpoint] = {
                "checkpoint_sha256": saved["checkpoint_sha256"], "strict_reload": True,
                "all_heldout_metric_arrays_exact": True, "heldout": heldout, "source": source,
            }
            print(json.dumps({"fold": index, "endpoint": endpoint, "all_heldout_metric_arrays_exact": True,
                              "source_fused_mAP": source["outputs"]["fused"]["scores"]["metrics_percent"]["mAP"],
                              "elapsed_seconds": time.time() - started}), flush=True)
            del model
            torch.cuda.empty_cache()
        report["folds"].append(fold)
        (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "ALL_SIX_FINAL_MODELS_REPLAY_AND_SOURCE_DIAGNOSIS_COMPLETE"
    report["elapsed_seconds"] = time.time() - started
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--q1-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
