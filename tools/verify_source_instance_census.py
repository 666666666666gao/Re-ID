#!/usr/bin/env python3
"""Recompute every source-census row from all saved features, without a model."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba",
           *(f"{e}_{m}_residual" for e in ("cnn", "transformer", "mamba")
             for m in ("RGB", "NI", "TI")))
PREDICATES = ("prototype_correct", "prototype_correct_but_instance_rank1_wrong",
              "prototype_correct_but_instance_ap_below_one", "negative_prototype_misses_rank1_competitor",
              "nonpositive_ordered_positive_negative_pairs", "all_ordered_positive_negative_pairs")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(x):
    norms = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    assert np.all(norms > 0)
    return x / norms


def verify(args):
    started = time.time()
    assert not args.output.exists()
    summary = json.loads(args.summary.read_bytes())
    assert summary["status"] == "COMPLETE_SOURCE_ONLY_INSTANCE_PROTOTYPE_CENSUS"
    assert summary["output_names"] == list(OUTPUTS) and len(summary["folds"]) == 3
    assert sha(args.contract) == summary["contract_sha256"]
    contract = json.loads(args.contract.read_bytes())
    for name, expected in contract["source_file_sha256"].items():
        assert sha(name) == expected, name
    m0 = json.loads(Path(contract["m0_receipt"]).read_bytes())
    replay = json.loads(Path(contract["sampler_metadata"]).read_bytes())
    torch.set_num_threads(4)
    count, batches, records, max_error = 0, 0, 0, 0.0
    all_aggregates = []
    for fold in summary["folds"]:
        k = fold["fold"]
        n = (2126, 2075, 2051)[k]
        assert fold["manifest"] == replay["folds"][k]["source_manifest"]
        records += n
        assert len(fold["manifest"]) == n
        descriptor = fold["features"]
        assert sha(descriptor["path"]) == descriptor["sha256"]
        assert Path(descriptor["path"]).stat().st_size == descriptor["bytes"]
        saved = torch.load(descriptor["path"], map_location="cpu", weights_only=True)
        assert saved["manifest"] == fold["manifest"] and saved["binding"] == fold["binding"]
        assert saved["extraction"] == fold["extraction"]
        ids = np.asarray([r["identity"] for r in saved["manifest"]])
        cams = np.asarray([r["camera"] for r in saved["manifest"]])
        assert np.array_equal(np.unique(ids), np.arange(94))
        for view in ("clean", "augmented"):
            extraction = saved["extraction"][view]
            assert extraction["model_state_unchanged"] and extraction["gradients_absent"]
            assert extraction["model_state_sha256"] == m0["preflight"][k]["endpoints"][0]["initial_state_sha256"]
            indices = [i for receipt in extraction["batch_receipts"] for i in receipt["sampler_indices"]]
            assert indices == list(range(n))
            assert extraction["batches"] == (n + 63) // 64
            batches += extraction["batches"]
            assert set(saved["views"][view]) == set(OUTPUTS)
            for name, tensor in saved["views"][view].items():
                width = 3072 if name == "baseline_only" else 7680 if name == "fused" else 4608 if name in OUTPUTS[2:5] else 512
                assert tensor.shape == (n, width) and tensor.dtype == torch.float32
                assert bool(torch.isfinite(tensor).all())
                assert float((tensor.norm(dim=1) - 1).abs().max()) < 1e-5
        assert len(fold["comparisons"]) == 4
        for comparison in fold["comparisons"]:
            view, protocol = comparison["view"], comparison["protocol"]
            descriptor = comparison["all_rows"]
            assert sha(descriptor["path"]) == descriptor["sha256"]
            assert Path(descriptor["path"]).stat().st_size == descriptor["bytes"]
            with gzip.open(descriptor["path"], "rt", encoding="utf-8") as handle:
                for name in OUTPUTS:
                    query = saved["views"][view][name].numpy().astype(np.float64)
                    gallery = saved["views"]["clean"][name].numpy().astype(np.float64)
                    scores = query @ gallery.T
                    prototypes = normalize(np.stack([gallery[ids == label].sum(axis=0) for label in range(94)]))
                    proto_scores = query @ prototypes.T
                    totals = dict.fromkeys(PREDICATES, 0)
                    aps, hits = [], []
                    for q in range(n):
                        actual = json.loads(next(handle))
                        assert (actual["fold"], actual["view"], actual["protocol"], actual["output"], actual["query_index"]) == (k, view, protocol, name, q)
                        positive = np.flatnonzero((ids == ids[q]) & ((np.arange(n) != q) if protocol == "identity_exclude_record" else (cams != cams[q])))
                        negative = np.flatnonzero(ids != ids[q])
                        assert actual["positive_records"] == len(positive) and actual["negative_records"] == len(negative)
                        assert actual["gallery_records"] == n and actual["identity"] == int(ids[q]) and actual["camera"] == int(cams[q])
                        assert actual["eligible"] == bool(len(positive))
                        count += 1
                        if not len(positive):
                            continue
                        legal = np.sort(np.concatenate((positive, negative)))
                        order = legal[np.argsort(-scores[q, legal], kind="stable")]
                        positions = np.flatnonzero(ids[order] == ids[q]) + 1
                        ap = float(np.mean(np.arange(1, len(positions) + 1) / positions))
                        best, worst = float(scores[q, positive].max()), float(scores[q, positive].min())
                        ni = int(negative[np.argmax(scores[q, negative])])
                        highest = float(scores[q, ni])
                        pp = float(query[q] @ normalize(gallery[positive].sum(axis=0)))
                        npmax = float(proto_scores[q, np.arange(94) != ids[q]].max())
                        correct = pp > npmax
                        # Direct pair comparisons independently check searchsorted counts.
                        inversions = int(np.sum(scores[q, negative][None, :] >= scores[q, positive][:, None]))
                        expected = {"average_precision": ap, "first_match_rank": int(positions[0]),
                                    "first_gallery_index": int(order[0]), "nearest_negative_gallery_index": ni,
                                    "nearest_negative_identity": int(ids[ni]), "nearest_negative_camera": int(cams[ni]),
                                    "positive_prototype_cosine": pp, "nearest_negative_prototype_cosine": npmax,
                                    "prototype_margin": pp - npmax, "best_positive_minus_nearest_negative": best - highest,
                                    "worst_positive_minus_nearest_negative": worst - highest,
                                    "nonpositive_ordered_positive_negative_pairs": inversions,
                                    "all_ordered_positive_negative_pairs": len(positive) * len(negative),
                                    "prototype_correct": bool(correct),
                                    "prototype_correct_but_instance_rank1_wrong": bool(correct and positions[0] != 1),
                                    "prototype_correct_but_instance_ap_below_one": bool(correct and ap < 1 - 1e-12),
                                    "negative_prototype_misses_rank1_competitor": bool(highest >= best and npmax < best)}
                        for key, value in expected.items():
                            if isinstance(value, float):
                                error = abs(actual[key] - value)
                                max_error = max(max_error, error)
                                assert error < 1e-10, (k, view, protocol, name, q, key, error)
                            else:
                                assert actual[key] == value, (k, view, protocol, name, q, key)
                        aps.append(ap)
                        hits.append(positions[0] == 1)
                        for key in PREDICATES:
                            totals[key] += expected[key]
                    aggregate = {"all_query_records": n, "eligible_queries": len(aps),
                                 "excluded_query_records_kept_in_gallery": n - len(aps),
                                 "source_probe_mAP_percent": float(np.mean(aps) * 100),
                                 "source_probe_Rank1_percent": float(np.mean(hits) * 100), **totals}
                    original = comparison["aggregates"][name]
                    for key, value in aggregate.items():
                        assert abs(original[key] - value) < 1e-10, (k, view, protocol, name, key)
                    all_aggregates.append({"fold": k, "view": view, "protocol": protocol, "output": name, **aggregate})
                assert not handle.read()
        del saved
    assert (count, batches, records, len(all_aggregates)) == (350112, 200, 6252, 168)
    assert summary["optimizer_updates"] == summary["heldout_image_reads"] == summary["dev_access_count"] == summary["official_test_access_count"] == 0
    proof = {"status": "PASS_ALL_SOURCE_INSTANCE_CENSUS_ROWS_AND_AGGREGATES", "rows_checked": count,
             "source_record_model_pairs": records, "forward_batch_receipts_checked": batches,
             "summary_sha256": sha(args.summary), "contract_sha256": sha(args.contract),
             "verifier_sha256": sha(__file__), "all_aggregates": all_aggregates, "maximum_numeric_error": max_error,
             "saved_feature_files_loaded": 3, "model_forwards": 0, "image_reads": 0, "optimizer_updates": 0,
             "independent_audit": False, "elapsed_seconds": time.time() - started}
    args.output.write_bytes((json.dumps(proof, indent=2) + "\n").encode())
    print(json.dumps({k: v for k, v in proof.items() if k != "all_aggregates"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
