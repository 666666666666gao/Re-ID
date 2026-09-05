#!/usr/bin/env python3
"""Read-only complete-gallery V17 match, correction and CNN-part diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def match_rows(distances, identities, cameras, scores):
    rows = []
    for row, query in enumerate(scores["query_indices"]):
        positive = (identities == identities[query]) & (cameras != cameras[query])
        negative = identities != identities[query]
        order = np.argsort(distances[query], kind="stable")
        valid = order[positive[order] | negative[order]]
        pos = order[positive[order]]
        neg = order[negative[order]]
        rows.append({
            "query_index": query,
            "average_precision": scores["average_precision"][row],
            "first_match_rank": scores["first_match_rank"][row],
            "top5_gallery_indices": valid[:5].tolist(),
            "nearest_positive": int(pos[0]),
            "nearest_negative": int(neg[0]),
            "positive_distance": float(distances[query, pos[0]]),
            "negative_distance": float(distances[query, neg[0]]),
            "positive_mean_distance": float(distances[query, positive].mean()),
            "negative_mean_distance": float(distances[query, negative].mean()),
            "nearest_margin": float(distances[query, neg[0]] - distances[query, pos[0]]),
        })
    return rows


def run(args):
    import torch
    import torch.nn.functional as F
    from tools.audit_v17_full_gallery import full_gallery_scores
    from tools.train_signal_preserving_v17 import (
        _load_contract, _build_fold_model, _model_state_sha256,
        _current_source_hashes, _sha256,
    )
    from tools.build_v12_complete_path_oof_targets import (
        _load_records, _configure_signal, _eval_loader, build_complete_path_fold_records,
    )
    from tools.run_signal_preserving_v5 import _set_seed

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = _load_contract(args.config.resolve())
    config = contract["config"]
    original = json.loads(args.q1_summary.read_text())
    prior = json.loads(args.full_gallery_summary.read_text())
    assert prior["q1_summary_sha256"] == _sha256(args.q1_summary)
    assert _current_source_hashes(contract) == original["source_file_sha256"]
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    assert (signal_commit, signal_diff) == (original["source_commit"], original["signal_source_diff_sha256"])
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn

    records = _load_records(config)
    experts = ("cnn", "transformer", "mamba")
    names = ("baseline_only", "fused", *experts)
    folds = []
    for receipt, previous in zip(original["fold_receipts"], prior["folds"], strict=True):
        fold = receipt["fold"]
        assert previous["fold"] == fold
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
        manifest = [{"files": list(r[0]), "identity": r[1], "camera": r[2]} for r in gallery]
        assert [Path(r["files"][0]).name for r in manifest] == [r["file"] for r in previous["gallery_manifest"]]
        result = {"fold": fold, "gallery_manifest": manifest, "endpoints": {}}
        for endpoint in ("weight0", "dtred"):
            saved = receipt["endpoints"][endpoint]
            checkpoint = Path(saved["checkpoint"])
            assert _sha256(checkpoint) == saved["checkpoint_sha256"]
            _set_seed(42)
            model, _ = _build_fold_model(config, signal_cfg, fold_index=fold, split=split)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = model.state_dict()
            assert set(payload["v17_state_dict"]) == {n for n in state if not n.startswith("base_v8.")}
            assert payload["fit_identity_ids"] == list(split["fit_identity_ids"])
            assert payload["heldout_identity_ids"] == list(split["heldout_identity_ids"])
            state.update(payload["v17_state_dict"])
            model.load_state_dict(state, strict=True)
            del state, payload
            before = _model_state_sha256(model)
            assert before == saved["training"]["final_state_sha256"]
            model.eval()
            cache = {"baseline": [], "cnn_parts": []}
            for group in ("teacher", "residual", "delta"):
                cache.update({f"{group}_{expert}": [] for expert in experts})

            def capture_parts(_module, _inputs, value):
                # Fixed four horizontal regions at the actual CNN head input.
                parts = value.reshape(len(value), 3, 4, 4, 8, 768).mean(dim=(3, 4))
                cache["cnn_parts"].append(parts.float().cpu())

            def capture_delta(expert):
                def hook(_module, _inputs, value):
                    cache[f"delta_{expert}"].append(value.float().cpu())
                return hook

            handles = [model.base_v8.encoder.cnn_head.norm.register_forward_hook(capture_parts)]
            handles.extend(model.correction.receiver_projections[e].register_forward_hook(capture_delta(e)) for e in experts)
            features = {name: [] for name in names}
            with torch.inference_mode():
                for images, _, _, camera_labels, _, _ in loader:
                    output = model({
                        "images": {name: value.cuda() for name, value in images.items()},
                        "camera_ids": camera_labels.cuda(),
                        "modality_mask": torch.ones(len(camera_labels), 3, dtype=torch.bool, device="cuda"),
                    }, return_aux=True)
                    assert bool(output.diagnostics["baseline_exact_prefix"])
                    cache["baseline"].append(output.baseline_embedding.float().cpu())
                    features["baseline_only"].append(output.baseline_embedding.float().cpu())
                    features["fused"].append(output.fused_embedding.float().cpu())
                    for expert in experts:
                        features[expert].append(output.branch_embeddings[expert].float().cpu())
                        cache[f"teacher_{expert}"].append(output.teacher_residual_embeddings[expert].float().cpu())
                        cache[f"residual_{expert}"].append(output.residual_embeddings[expert].float().cpu())
            for handle in handles:
                handle.remove()
            tensors = {name: torch.cat(value) for name, value in cache.items()}
            assert all(len(value) == len(gallery) for value in tensors.values())
            cache_path = args.output_dir / f"fold_{fold}_{endpoint}_features.pt"
            torch.save(tensors, cache_path)
            outputs = {}
            for name in names:
                embedding = F.normalize(torch.cat(features[name]), dim=1)
                distances = torch.cdist(embedding, embedding).numpy()
                scores = full_gallery_scores(distances, identities, cameras)
                expected = previous["endpoints"][endpoint]["outputs"][name]
                assert scores["average_precision"] == expected["average_precision"]
                assert scores["first_match_rank"] == expected["first_match_rank"]
                outputs[name] = {"metrics_percent": scores["metrics_percent"],
                                 "queries": match_rows(distances, identities, cameras, scores)}
            geometry = {}
            for expert in experts:
                teacher = tensors[f"teacher_{expert}"]
                residual = tensors[f"residual_{expert}"]
                delta = tensors[f"delta_{expert}"]
                geometry[expert] = {
                    "delta_norm": delta.norm(dim=1).tolist(),
                    "teacher_corrected_cosine": F.cosine_similarity(teacher, residual).tolist(),
                    "modality_energy_fraction": residual.reshape(-1, 3, 512).square().sum(dim=2).tolist(),
                }
            assert before == _model_state_sha256(model)
            assert _sha256(checkpoint) == saved["checkpoint_sha256"]
            result["endpoints"][endpoint] = {
                "checkpoint_sha256": saved["checkpoint_sha256"], "model_state_sha256": before,
                "cache": str(cache_path), "cache_sha256": _sha256(cache_path),
                "state_unchanged": True, "full_gallery_metric_parity": True,
                "outputs": outputs, "geometry": geometry,
            }
            print(json.dumps({"fold": fold, "endpoint": endpoint, "parity": True,
                              "cache": str(cache_path), "gallery": len(gallery)}), flush=True)
            del model, features, cache, tensors
            torch.cuda.empty_cache()
        folds.append(result)
    report = {
        "schema_version": "v17-full-gallery-failure-geometry-v1", "status": "PASS",
        "scope": "read_only_development_diagnosis_not_new_validation",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": _sha256(Path(__file__)),
        "q1_summary_sha256": _sha256(args.q1_summary),
        "full_gallery_summary_sha256": _sha256(args.full_gallery_summary),
        "optimizer_steps": 0, "checkpoint_writes": 0, "dev_access_count": 0,
        "official_test_access_count": 0, "folds": folds,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--full-gallery-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
