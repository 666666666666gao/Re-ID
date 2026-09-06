#!/usr/bin/env python3
"""Complete read-only source-instance/prototype census after the fixed V26 Q1."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.build_v12_complete_path_oof_targets import (
    _configure_signal, _load_records, build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _training_batch
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256,
)
from tools.train_signal_preserving_v18 import loader_for
from tools.train_signal_preserving_v19 import seed_everything
from tools.train_signal_preserving_v22 import build_model
from tools.train_signal_preserving_v26 import load_contract
from trifusion.aligned_data import AlignedTripletImageDataset, SharedGeometryTripletTransform

EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
OUTPUTS = ("baseline_only", "fused", *EXPERTS,
           *(f"{e}_{m}_residual" for e in EXPERTS for m in MODALITIES))
VIEWS = ("clean", "augmented")
PROTOCOLS = ("identity_exclude_record", "cross_camera")


def unit(value):
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    assert np.all(norms > 0)
    return value / norms


def source_rows(query, gallery, identities, cameras, protocol):
    """Enumerate every query; a query without positives remains in the gallery.

    Both views preserve the same record order. The true prototype uses ONLY the
    legal positive records for this query. Negative prototypes use every source
    instance of the corresponding different identity, never a target identity.
    """
    query, gallery = query.astype(np.float64), gallery.astype(np.float64)
    identities, cameras = np.asarray(identities), np.asarray(cameras)
    assert query.shape == gallery.shape and protocol in PROTOCOLS
    classes = np.unique(identities)
    assert np.array_equal(classes, np.arange(len(classes)))
    sums = np.stack([gallery[identities == label].sum(axis=0) for label in classes])
    prototypes = unit(sums)
    similarity = query @ gallery.T
    proto_similarity = query @ prototypes.T
    assert np.isfinite(similarity).all() and np.isfinite(proto_similarity).all()
    for q in range(len(query)):
        same = identities == identities[q]
        negative = ~same
        positive = same.copy()
        if protocol == "identity_exclude_record":
            positive[q] = False
        else:
            positive &= cameras != cameras[q]
        indices = np.flatnonzero(positive)
        row = {"query_index": q, "identity": int(identities[q]), "camera": int(cameras[q]),
               "gallery_records": len(gallery), "positive_records": len(indices),
               "negative_records": int(negative.sum()), "eligible": bool(len(indices))}
        if not len(indices):
            yield row
            continue
        positive_prototype = unit(gallery[indices].sum(axis=0, keepdims=True))[0]
        positive_proto_similarity = float(query[q] @ positive_prototype)
        negative_proto_similarity = float(proto_similarity[q, classes != identities[q]].max())
        positive_scores = similarity[q, positive]
        negative_scores = similarity[q, negative]
        best_positive, worst_positive = float(positive_scores.max()), float(positive_scores.min())
        highest_negative = float(negative_scores.max())
        sorted_negatives = np.sort(negative_scores)
        inversions = int((len(sorted_negatives) - np.searchsorted(
            sorted_negatives, positive_scores, side="left")).sum())
        order = np.argsort(-similarity[q], kind="stable")
        order = order[(positive | negative)[order]]
        positions = np.flatnonzero(identities[order] == identities[q]) + 1
        assert len(positions) == len(indices)
        ap = float(np.mean(np.arange(1, len(positions) + 1) / positions))
        prototype_correct = positive_proto_similarity > negative_proto_similarity
        rank1_correct = identities[order[0]] == identities[q]
        negative_indices = np.flatnonzero(negative)
        nearest_negative = int(negative_indices[np.argmax(negative_scores)])
        row.update({
            "average_precision": ap, "first_match_rank": int(positions[0]),
            "first_gallery_index": int(order[0]), "nearest_negative_gallery_index": nearest_negative,
            "nearest_negative_identity": int(identities[nearest_negative]),
            "nearest_negative_camera": int(cameras[nearest_negative]),
            "positive_prototype_cosine": positive_proto_similarity,
            "nearest_negative_prototype_cosine": negative_proto_similarity,
            "prototype_margin": positive_proto_similarity - negative_proto_similarity,
            "best_positive_minus_nearest_negative": best_positive - highest_negative,
            "worst_positive_minus_nearest_negative": worst_positive - highest_negative,
            "nonpositive_ordered_positive_negative_pairs": inversions,
            "all_ordered_positive_negative_pairs": len(indices) * len(negative_scores),
            "prototype_correct": bool(prototype_correct),
            "prototype_correct_but_instance_rank1_wrong": bool(prototype_correct and not rank1_correct),
            "prototype_correct_but_instance_ap_below_one": bool(prototype_correct and ap < 1 - 1e-12),
            "negative_prototype_misses_rank1_competitor": bool(
                highest_negative >= best_positive and negative_proto_similarity < best_positive),
        })
        yield row


def mathematical_check():
    angles = np.deg2rad([0, 60, 5, 180])
    features = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    identities, cameras = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0])
    a = list(source_rows(features, features, identities, cameras, "identity_exclude_record"))
    b = list(source_rows(features, features, identities, cameras, "cross_camera"))
    assert len(a) == len(b) == 4
    for row in (a[0], b[0]):
        assert row["positive_records"] == 1 and row["negative_records"] == 2
        assert row["average_precision"] == 0.5 and row["first_match_rank"] == 2
        assert row["prototype_correct_but_instance_rank1_wrong"]
        assert row["negative_prototype_misses_rank1_competitor"]
        assert row["nonpositive_ordered_positive_negative_pairs"] == 1
        assert row["nearest_negative_gallery_index"] == 2
    assert not b[2]["eligible"] and not b[3]["eligible"]
    assert b[0]["gallery_records"] == 4  # Those two records still compete as negatives.
    return {"status": "PASS_KNOWN_HIDDEN_INSTANCE_SELF_EXCLUSION_CLASS_ZERO_FULL_GALLERY",
            "model_forwards": 0, "image_reads": 0, "optimizer_updates": 0}


def extract(model, records, config, view):
    from data.datasets.make_dataloader import train_collate_fn
    seed_everything()
    model.eval()
    before = _model_state_sha256(model)
    if view == "clean":
        dataset = loader_for(records, config).dataset
    else:
        assert view == "augmented"
        dataset = AlignedTripletImageDataset(records, transform=SharedGeometryTripletTransform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False,
                                       num_workers=4, collate_fn=train_collate_fn)
    index = _record_index_by_path(records)
    seen, receipts = [], []
    values = {name: [] for name in OUTPUTS}
    with torch.inference_mode():
        for raw in loader:
            receipt = _raw_batch_receipt(raw, record_index_by_path=index)
            indices = receipt["sampler_indices"]
            assert indices == list(range(len(seen), len(seen) + len(indices)))
            assert raw[1].tolist() == [records[i][1] for i in indices]
            assert raw[2].tolist() == [records[i][2] for i in indices]
            seen.extend(indices)
            receipts.append(receipt)
            batch, _ = _training_batch(raw)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(batch, return_aux=True)
            assert output.diagnostics["baseline_exact_prefix"] and output.diagnostics["all_finite"]
            features = {"baseline_only": output.baseline_embedding, "fused": output.fused_embedding,
                        **dict(output.branch_embeddings),
                        **{f"{e}_{m}_residual": output.modal_residual_embeddings[e][:, i]
                           for e in EXPERTS for i, m in enumerate(MODALITIES)}}
            for name, feature in features.items():
                values[name].append(F.normalize(feature.float().cpu(), dim=1))
    assert seen == list(range(len(records))) and before == _model_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    return {name: torch.cat(rows) for name, rows in values.items()}, {
        "view": view, "records": len(seen), "batches": len(receipts), "batch_receipts": receipts,
        "model_state_sha256": before, "model_state_unchanged": True, "gradients_absent": True,
        "model_mode": "eval", "forward_amp_dtype": "float16", "sampler": "sequential_all_records_once",
    }


def summarize(rows):
    valid = [r for r in rows if r["eligible"]]
    assert valid
    return {"all_query_records": len(rows), "eligible_queries": len(valid),
            "excluded_query_records_kept_in_gallery": len(rows) - len(valid),
            "source_probe_mAP_percent": float(np.mean([r["average_precision"] for r in valid]) * 100),
            "source_probe_Rank1_percent": float(np.mean([r["first_match_rank"] == 1 for r in valid]) * 100),
            **{key: sum(r[key] for r in valid) for key in (
                "prototype_correct", "prototype_correct_but_instance_rank1_wrong",
                "prototype_correct_but_instance_ap_below_one", "negative_prototype_misses_rank1_competitor",
                "nonpositive_ordered_positive_negative_pairs", "all_ordered_positive_negative_pairs")}}


def run(args):
    started = time.time()
    assert _sha256(args.contract) == args.contract_sha256
    contract = json.loads(args.contract.read_bytes())
    assert contract["views"] == list(VIEWS) and contract["protocols"] == list(PROTOCOLS)
    assert contract["output_names"] == list(OUTPUTS)
    assert contract["batch_size"] == 64 and contract["workers"] == 4 and contract["seed"] == 42
    assert shutil.disk_usage(args.output_dir.parent).free >= contract["minimum_free_bytes"]
    for name, expected in contract["source_file_sha256"].items():
        assert _sha256(Path(name)) == expected, name
    prior = Path(contract["required_v26_run"])
    summary_raw = (prior / "run_summary.json").read_bytes()
    terminal = json.loads(summary_raw)
    verification_raw = (prior / "terminal_verification.json").read_bytes()
    verification = json.loads(verification_raw)
    assert terminal["status"] in ("Q1_PASS", "Q1_FAIL")
    assert len(terminal["folds"]) == 3 and int((prior / "terminal_verification.exit").read_text()) == 0
    assert verification["status"] == "PASS_COMPLETE_V26_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES"
    assert verification["run_summary_sha256"] == hashlib.sha256(summary_raw).hexdigest()
    assert verification["verifier_sha256"] == contract["required_v26_verifier_sha256"]
    assert _sha256(Path(contract["v26_config"])) == contract["v26_config_sha256"]
    config, sources = load_contract(Path(contract["v26_config"]))
    m0 = json.loads(Path(contract["m0_receipt"]).read_bytes())
    torch.set_num_threads(4)
    math_check = mathematical_check()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING_SOURCE_ONLY_CENSUS", "math_check": math_check,
              "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": _sha256(Path(__file__)), "contract_sha256": args.contract_sha256,
              "prior_v26_summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
              "prior_v26_verification_sha256": hashlib.sha256(verification_raw).hexdigest(),
              "model_checkpoint_scope": "original_V12_source_initialization_not_V26_final",
              "optimizer_updates": 0, "heldout_image_reads": 0, "dev_access_count": 0,
              "official_test_access_count": 0, "output_names": list(OUTPUTS), "folds": []}
    report["disk_before"] = shutil.disk_usage(args.output_dir)._asdict()

    def save():
        report["elapsed_seconds"] = time.time() - started
        (args.output_dir / "source_census.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())

    save()
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    report.update(signal_commit=signal_commit, signal_diff_sha256=signal_diff)
    records = _load_records(config)
    replay = json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    for fold, source in enumerate(sources["fold_receipts"]):
        split = build_complete_path_fold_records(records, heldout_ids=set(source["heldout_identity_ids"]))
        train = split["train_records"]
        manifest = [{"file": Path(r[0][0]).name, "identity": r[1], "camera": r[2]} for r in train]
        assert manifest == replay["folds"][fold]["source_manifest"] and not split["identity_overlap"]
        assert len(train) == (2126, 2075, 2051)[fold]
        model, binding = build_model(config, signal_cfg, fold, split)
        assert _model_state_sha256(model) == m0["preflight"][fold]["endpoints"][0]["initial_state_sha256"]
        views, extraction = {}, {}
        for view in VIEWS:
            views[view], extraction[view] = extract(model, train, config, view)
        del model
        torch.cuda.empty_cache()
        identities = np.asarray([r[1] for r in train])
        cameras = np.asarray([r[2] for r in train])
        assert len(np.unique(identities)) == 94
        feature_path = args.output_dir / f"fold_{fold}_all_source_views.pt"
        torch.save({"views": views, "manifest": manifest, "binding": binding,
                    "extraction": extraction}, feature_path)
        result = {"fold": fold, "binding": binding, "manifest": manifest, "extraction": extraction,
                  "features": {"path": str(feature_path), "bytes": feature_path.stat().st_size,
                               "sha256": _sha256(feature_path)}, "comparisons": []}
        for view in VIEWS:
            for protocol in PROTOCOLS:
                path = args.output_dir / f"fold_{fold}_{view}_{protocol}_all_rows.jsonl.gz"
                aggregates = {}
                with gzip.open(path, "xt", encoding="utf-8") as handle:
                    for name in OUTPUTS:
                        rows = list(source_rows(views[view][name].numpy(), views["clean"][name].numpy(),
                                                identities, cameras, protocol))
                        assert len(rows) == len(train)
                        for row in rows:
                            handle.write(json.dumps({"fold": fold, "view": view, "protocol": protocol,
                                                     "output": name, **row}) + "\n")
                        aggregates[name] = summarize(rows)
                result["comparisons"].append({"view": view, "protocol": protocol, "aggregates": aggregates,
                                               "all_rows": {"path": str(path), "bytes": path.stat().st_size,
                                                            "sha256": _sha256(path), "rows": len(train) * 14}})
        report["folds"].append(result)
        save()
        print(json.dumps({"stage": "complete_source_fold", "fold": fold, "source_records": len(train),
                          "view_count": 2, "output_count": 14, "protocol_count": 2}), flush=True)
        del views
    assert sum(len(f["manifest"]) for f in report["folds"]) == 6252
    assert sum(e["batches"] for f in report["folds"] for e in f["extraction"].values()) == 200
    assert sum(c["all_rows"]["rows"] for f in report["folds"] for c in f["comparisons"]) == 350112
    report.update(status="COMPLETE_SOURCE_ONLY_INSTANCE_PROTOTYPE_CENSUS",
                  total_source_record_model_pairs=6252, total_triplet_image_forward_exposures=12504,
                  total_model_forward_batches=200, total_all_query_output_protocol_rows=350112,
                  limitations=["one_fixed_augmented_view_per_record", "source_models_seen_these_training_identities",
                               "repeated_source_records_across_folds_not_independent", "no_training_time_cache_drift_measurement",
                               "source_diagnostic_not_retrieval_qualification_or_SOTA"])
    report["disk_after"] = shutil.disk_usage(args.output_dir)._asdict()
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("folds", "output_names")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
