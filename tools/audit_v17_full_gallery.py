#!/usr/bin/env python3
"""Reload all six V17 endpoints and audit every held-out gallery record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def full_gallery_scores(distances, identities, cameras):
    """Score every eligible query; keep single-camera identities as distractors."""
    from tools.diagnose_v6_oracle_complementarity import per_query_reid_scores

    identities = np.asarray(identities)
    cameras = np.asarray(cameras)
    eligible = np.array([
        np.any((identities == identity) & (cameras != camera))
        for identity, camera in zip(identities, cameras, strict=True)
    ])
    queries = np.flatnonzero(eligible)
    scores = per_query_reid_scores(
        distances[queries], query_ids=identities[queries], gallery_ids=identities,
        query_cameras=cameras[queries], gallery_cameras=cameras,
    )
    first_hits = []
    for query, order in zip(queries, np.argsort(distances[queries], axis=1, kind="stable"), strict=True):
        valid = ~((identities[order] == identities[query]) & (cameras[order] == cameras[query]))
        first_hits.append(int(np.flatnonzero(identities[order][valid] == identities[query])[0]))
    first_hits = np.asarray(first_hits)
    return {
        "query_indices": queries.tolist(),
        "excluded_no_cross_camera_positive": np.flatnonzero(~eligible).tolist(),
        "average_precision": scores.average_precision.tolist(),
        "first_match_rank": (first_hits + 1).tolist(),
        "metrics_percent": {
            "mAP": float(scores.average_precision.mean() * 100),
            **{f"Rank-{k}": float((first_hits < k).mean() * 100) for k in (1, 5, 10)},
        },
    }


def run(args):
    import torch
    import torch.nn.functional as F
    from tools.train_signal_preserving_v17 import (
        _load_contract, _build_fold_model, _model_state_sha256,
        _current_source_hashes, _sha256,
    )
    from tools.build_v12_complete_path_oof_targets import (
        _load_records, _configure_signal, _eval_loader, build_complete_path_fold_records,
    )
    from tools.run_signal_preserving_v5 import _set_seed

    started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = _load_contract(args.config.resolve())
    config = contract["config"]
    original = json.loads(args.q1_summary.read_text())
    if _current_source_hashes(contract) != original["source_file_sha256"]:
        raise ValueError("V17 source differs from the executed Q1")
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    if (signal_commit, signal_diff) != (original["source_commit"], original["signal_source_diff_sha256"]):
        raise ValueError("Signal source differs from Q1")
    records = _load_records(config)
    names = ("baseline_only", "fused", "cnn", "transformer", "mamba")
    folds = []
    all_ap = {endpoint: {name: [] for name in names} for endpoint in ("weight0", "dtred")}
    all_ranks = {endpoint: {name: [] for name in names} for endpoint in all_ap}
    all_ids = []
    for receipt in original["fold_receipts"]:
        fold = receipt["fold"]
        registry = contract["v12_summary"]["fold_receipts"][fold]
        split = build_complete_path_fold_records(records, heldout_ids=set(registry["heldout_identity_ids"]))
        gallery = split["heldout_records"]
        identities = np.array([r[1] for r in gallery])
        cameras = np.array([r[2] for r in gallery])
        template = _eval_loader(gallery, config)
        loader = torch.utils.data.DataLoader(
            ImageDataset(gallery, template.dataset.transform),
            batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]), shuffle=False,
            num_workers=int(config["DATA"]["NUM_WORKERS"]), collate_fn=val_collate_fn,
        )
        result = {"fold": fold, "gallery_records": len(gallery),
                  "gallery_identities": len(set(identities.tolist())),
                  "gallery_manifest": [{"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in gallery],
                  "endpoints": {}}
        for endpoint in ("weight0", "dtred"):
            saved = receipt["endpoints"][endpoint]
            checkpoint = Path(saved["checkpoint"])
            if _sha256(checkpoint) != saved["checkpoint_sha256"]:
                raise ValueError("Saved endpoint checkpoint SHA mismatch")
            _set_seed(42)
            model, _ = _build_fold_model(config, signal_cfg, fold_index=fold, split=split)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = model.state_dict()
            expected = {name for name in state if not name.startswith("base_v8.")}
            if set(payload["v17_state_dict"]) != expected:
                raise ValueError("Incomplete V17 endpoint state")
            if payload["fit_identity_ids"] != list(split["fit_identity_ids"]) or payload["heldout_identity_ids"] != list(split["heldout_identity_ids"]):
                raise ValueError("Checkpoint identity split mismatch")
            state.update(payload["v17_state_dict"])
            model.load_state_dict(state, strict=True)
            del state, payload
            before = _model_state_sha256(model)
            if before != saved["training"]["final_state_sha256"]:
                raise ValueError("Reloaded model differs from final training state")
            model.eval()
            features = {name: [] for name in names}
            exact = True
            with torch.inference_mode():
                for images, _, _, camera_labels, _, _ in loader:
                    output = model({
                        "images": {name: value.cuda() for name, value in images.items()},
                        "camera_ids": camera_labels.cuda(),
                        "modality_mask": torch.ones(len(camera_labels), 3, dtype=torch.bool, device="cuda"),
                    }, return_aux=True)
                    features["baseline_only"].append(output.baseline_embedding.float().cpu())
                    features["fused"].append(output.fused_embedding.float().cpu())
                    for name in names[2:]:
                        features[name].append(output.branch_embeddings[name].float().cpu())
                    exact = exact and bool(output.diagnostics["baseline_exact_prefix"])
            outputs = {}
            for name in names:
                embedding = F.normalize(torch.cat(features[name]), dim=1)
                outputs[name] = full_gallery_scores(torch.cdist(embedding, embedding).numpy(), identities, cameras)
                all_ap[endpoint][name].extend(outputs[name]["average_precision"])
                all_ranks[endpoint][name].extend(outputs[name]["first_match_rank"])
            after = _model_state_sha256(model)
            unchanged = before == after and _sha256(checkpoint) == saved["checkpoint_sha256"]
            if not unchanged or not exact:
                raise RuntimeError("Read-only evaluation changed state or Signal prefix")
            result["endpoints"][endpoint] = {
                "checkpoint_sha256": saved["checkpoint_sha256"], "strict_reload": True,
                "model_state_sha256": before, "state_unchanged": unchanged,
                "exact_signal_prefix": exact, "outputs": outputs,
            }
            print(json.dumps({"fold": fold, "endpoint": endpoint, "gallery": len(gallery),
                              "metrics": {name: value["metrics_percent"] for name, value in outputs.items()}}), flush=True)
            del model
            torch.cuda.empty_cache()
        qidx = outputs["fused"]["query_indices"]
        all_ids.extend(identities[qidx].tolist())
        result["eligible_queries"] = len(qidx)
        result["excluded_queries"] = len(gallery) - len(qidx)
        folds.append(result)
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound
    gains = torch.tensor(all_ap["dtred"]["fused"]) - torch.tensor(all_ap["weight0"]["fused"])
    bootstrap = identity_cluster_bootstrap_lower_bound(gains, torch.tensor(all_ids), seed=42, resamples=10000)
    aggregate = {endpoint: {name: {
        "mAP": float(np.mean(all_ap[endpoint][name]) * 100),
        **{f"Rank-{k}": float(np.mean(np.array(all_ranks[endpoint][name]) <= k) * 100) for k in (1, 5, 10)},
    } for name in names} for endpoint in all_ap}
    result = {
        "schema_version": "v17-all-heldout-gallery-readonly-v1", "status": "PASS",
        "evaluation_type": "real_gt_train_only_full_heldout_gallery",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": _sha256(Path(__file__)), "q1_summary_sha256": _sha256(args.q1_summary),
        "source_file_sha256": _current_source_hashes(contract),
        "folds": folds, "aggregate_metrics_percent": aggregate,
        "matched_mAP_gains": {name: aggregate["dtred"][name]["mAP"] - aggregate["weight0"][name]["mAP"] for name in names},
        "fused_gain_identity_bootstrap": {
            "observed_mean_percent": bootstrap.observed_mean * 100,
            "lower_bound_95_percent": bootstrap.lower_bound * 100,
            "identity_clusters": bootstrap.cluster_count,
            "resamples": bootstrap.resamples,
        },
        "total_gallery_records": sum(f["gallery_records"] for f in folds),
        "total_eligible_queries": sum(f["eligible_queries"] for f in folds),
        "total_excluded_queries": sum(f["excluded_queries"] for f in folds),
        "optimizer_steps": 0, "checkpoint_writes": 0, "dev_access_count": 0,
        "official_test_access_count": 0, "next_phase_authorized": False,
        "original_q1_gate_unchanged": original["scientific_gate"],
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
