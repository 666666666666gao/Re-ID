#!/usr/bin/env python3
"""Describe paired Signal/fused retrieval changes from a fixed official receipt."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path,
                        help="Frozen SIGNAL_V8 receipt supplying the original author Signal reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE"
    checkpoint_policy = receipt.get("checkpoint_policy", "fixed_final_epoch")
    if checkpoint_policy == "best_official_map":
        assert receipt["fixed_epoch"] is None
        training_epochs = receipt.get("training_epochs", 20)
        assert training_epochs in (20, 50)
        assert 1 <= receipt["selected_epoch"] <= training_epochs
    else:
        assert checkpoint_policy == "fixed_final_epoch" and receipt["fixed_epoch"] == 20
    distance_path = Path(receipt["distance_arrays"])
    assert sha256(distance_path) == receipt["distance_arrays_sha256"]
    saved = torch.load(distance_path, map_location="cpu", weights_only=False)
    assert saved["protocol_sha256"] == receipt["protocol_sha256"]
    qids, gids = saved["query_ids"], saved["gallery_ids"]
    environment = "scenes" if receipt["dataset"] == "MSVR310" else "cameras"
    qenv, genv = saved["query_" + environment], saved["gallery_" + environment]
    distances = {name: saved["distances"][name].numpy() for name in OUTPUTS}
    assert len(qids) == receipt["query_count"] and len(gids) == receipt["gallery_count"]
    ap = {name: np.asarray(receipt["outputs"][name]["average_precision"]) for name in OUTPUTS}
    first = {name: np.asarray(receipt["outputs"][name]["first_match_rank"]) for name in OUTPUTS}
    baseline_receipt = receipt
    if args.reference_receipt is not None:
        baseline_receipt = json.loads(args.reference_receipt.read_text(encoding="utf-8"))
        assert baseline_receipt["status"] == "COMPLETE" and baseline_receipt["method"] == "SIGNAL_V8"
        for key in ("dataset", "protocol_sha256", "author_checkpoint_sha256", "filter", "reranking"):
            assert baseline_receipt[key] == receipt[key], key
        reference_path = Path(baseline_receipt["distance_arrays"])
        assert sha256(reference_path) == baseline_receipt["distance_arrays_sha256"]
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        assert reference["protocol_sha256"] == saved["protocol_sha256"]
        for key in ("query_ids", "gallery_ids", "query_" + environment, "gallery_" + environment):
            assert np.array_equal(reference[key], saved[key]), key
        distances["baseline_only"] = reference["distances"]["baseline_only"].numpy()
        ap["baseline_only"] = np.asarray(baseline_receipt["outputs"]["baseline_only"]["average_precision"])
        first["baseline_only"] = np.asarray(baseline_receipt["outputs"]["baseline_only"]["first_match_rank"])
    for name in OUTPUTS:
        metrics = (baseline_receipt if name == "baseline_only" else receipt)["outputs"][name]["metrics"]
        assert abs(ap[name].mean() * 100 - metrics["mAP"]) < 1e-8
        assert abs((first[name] == 1).mean() * 100 - metrics["Rank-1"]) < 1e-8
    delta = ap["fused"] - ap["baseline_only"]
    repaired = np.flatnonzero((first["baseline_only"] > 1) & (first["fused"] == 1))
    broken = np.flatnonzero((first["baseline_only"] == 1) & (first["fused"] > 1))
    by_id = defaultdict(list)
    for i, identity in enumerate(qids):
        by_id[str(identity)].append(float(delta[i]))
    identity_deltas = {identity: float(np.mean(values) * 100) for identity, values in by_id.items()}

    pair_counts = dict(both_correct=0, repaired=0, broken=0, both_wrong=0, tied=0)
    query_pair_deltas = []
    for q in range(len(qids)):
        positive = (gids == qids[q]) & (genv != qenv[q])
        negative = gids != qids[q]
        base = distances["baseline_only"][q]
        fused = distances["fused"][q]
        base_margin = base[negative][None, :] - base[positive][:, None]
        fused_margin = fused[negative][None, :] - fused[positive][:, None]
        comparable = (base_margin != 0) & (fused_margin != 0)
        base_correct = base_margin > 0
        fused_correct = fused_margin > 0
        counts = {
            "both_correct": int((comparable & base_correct & fused_correct).sum()),
            "repaired": int((comparable & ~base_correct & fused_correct).sum()),
            "broken": int((comparable & base_correct & ~fused_correct).sum()),
            "both_wrong": int((comparable & ~base_correct & ~fused_correct).sum()),
            "tied": int((~comparable).sum()),
        }
        assert sum(counts.values()) == int(positive.sum() * negative.sum())
        for key, value in counts.items():
            pair_counts[key] += value
        query_pair_deltas.append((counts["repaired"] - counts["broken"]) /
                                 (sum(counts.values()) - counts["tied"]))

    def ordered(q, name):
        valid = ~((gids == qids[q]) & (genv == qenv[q]))
        indices = np.argsort(distances[name][q])
        return indices[valid[indices]]

    def describe(q):
        base_order, fused_order = ordered(q, "baseline_only"), ordered(q, "fused")
        positive = int(next(j for j in base_order if gids[j] == qids[q]))
        negative = int(next(j for j in fused_order if gids[j] != qids[q]))
        assert gids[positive] == qids[q] and gids[negative] != qids[q]
        assert base_order[0] == positive and fused_order[0] == negative
        margins = {name: float(distances[name][q, negative] - distances[name][q, positive])
                   for name in OUTPUTS}
        mean_roles = np.mean([margins[name] for name in ("cnn", "transformer", "mamba")])
        assert abs(mean_roles - margins["fused"]) < 1e-4
        return {
            "query_index": int(q), "query_identity": str(qids[q]),
            "query_environment": str(qenv[q]),
            "signal_first_positive_gallery_index": positive,
            "signal_first_positive_environment": str(genv[positive]),
            "fused_first_negative_gallery_index": negative,
            "fused_first_negative_identity": str(gids[negative]),
            "fused_first_negative_environment": str(genv[negative]),
            "same_environment_negative": bool(genv[negative] == qenv[q]),
            "signal_ap": float(ap["baseline_only"][q]),
            "fused_ap": float(ap["fused"][q]),
            "ap_change_pp": float(delta[q] * 100),
            "signal_first_positive_rank": int(first["baseline_only"][q]),
            "fused_first_positive_rank": int(first["fused"][q]),
            "top20_signal": "".join("P" if gids[j] == qids[q] else "." for j in base_order[:20]),
            "top20_fused": "".join("P" if gids[j] == qids[q] else "." for j in fused_order[:20]),
            "pair_distance_margin_negative_minus_positive": margins,
        }

    examples = [describe(int(q)) for q in sorted(broken, key=lambda i: delta[i])[:3]]
    new_errors_same_environment = sum(item["same_environment_negative"] for item in
                                      (describe(int(q)) for q in broken))
    report = {
        "status": "COMPLETE_OFFICIAL_POSTHOC_DIAGNOSIS",
        "scope": "Official test receipt, read-only posthoc diagnosis",
        "checkpoint_policy": checkpoint_policy,
        "selected_epoch": receipt["selected_epoch"] if checkpoint_policy == "best_official_map" else None,
        "dataset": receipt["dataset"], "method": receipt["method"], "seed": receipt["seed"],
        "receipt_sha256": sha256(args.receipt),
        "distance_arrays_sha256": receipt["distance_arrays_sha256"],
        "query_count": int(len(qids)), "gallery_count": int(len(gids)),
        "signal_metrics": baseline_receipt["outputs"]["baseline_only"]["metrics"],
        "fused_metrics": receipt["outputs"]["fused"]["metrics"],
        "ap_improved_queries": int((delta > 1e-12).sum()),
        "ap_declined_queries": int((delta < -1e-12).sum()),
        "ap_unchanged_queries": int((abs(delta) <= 1e-12).sum()),
        "rank1_repaired_queries": int(len(repaired)),
        "rank1_new_errors": int(len(broken)),
        "new_errors_same_environment": int(new_errors_same_environment),
        "identities_improved": sum(value > 1e-12 for value in identity_deltas.values()),
        "identities_declined": sum(value < -1e-12 for value in identity_deltas.values()),
        "identities_unchanged": sum(abs(value) <= 1e-12 for value in identity_deltas.values()),
        "positive_negative_pair_transitions": pair_counts,
        "mean_query_pair_accuracy_change_pp": float(np.mean(query_pair_deltas) * 100),
        "queries_pair_accuracy_improved": int((np.asarray(query_pair_deltas) > 0).sum()),
        "queries_pair_accuracy_declined": int((np.asarray(query_pair_deltas) < 0).sum()),
        "worst_identity_delta_pp": sorted(identity_deltas.items(), key=lambda pair: pair[1])[:3],
        "best_identity_delta_pp": sorted(identity_deltas.items(), key=lambda pair: pair[1], reverse=True)[:3],
        "illustrative_new_rank1_errors": examples,
    }
    if args.reference_receipt is not None:
        report["reference"] = {
            "kind": "original_frozen_author_signal",
            "receipt": str(args.reference_receipt),
            "receipt_sha256": sha256(args.reference_receipt),
            "distance_arrays_sha256": baseline_receipt["distance_arrays_sha256"],
        }
        report["student_baseline_metrics"] = receipt["outputs"]["baseline_only"]["metrics"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "illustrative_new_rank1_errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
