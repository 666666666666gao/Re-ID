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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE" and receipt["fixed_epoch"] == 20
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
    for name in OUTPUTS:
        assert abs(ap[name].mean() * 100 - receipt["outputs"][name]["metrics"]["mAP"]) < 1e-8
        assert abs((first[name] == 1).mean() * 100 - receipt["outputs"][name]["metrics"]["Rank-1"]) < 1e-8
    delta = ap["fused"] - ap["baseline_only"]
    repaired = np.flatnonzero((first["baseline_only"] > 1) & (first["fused"] == 1))
    broken = np.flatnonzero((first["baseline_only"] == 1) & (first["fused"] > 1))
    by_id = defaultdict(list)
    for i, identity in enumerate(qids):
        by_id[str(identity)].append(float(delta[i]))
    identity_deltas = {identity: float(np.mean(values) * 100) for identity, values in by_id.items()}

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
        "scope": "Fixed official test receipt, no training, no model selection or parameter update",
        "dataset": receipt["dataset"], "method": receipt["method"], "seed": receipt["seed"],
        "receipt_sha256": sha256(args.receipt),
        "distance_arrays_sha256": receipt["distance_arrays_sha256"],
        "query_count": int(len(qids)), "gallery_count": int(len(gids)),
        "signal_metrics": receipt["outputs"]["baseline_only"]["metrics"],
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
        "worst_identity_delta_pp": sorted(identity_deltas.items(), key=lambda pair: pair[1])[:3],
        "best_identity_delta_pp": sorted(identity_deltas.items(), key=lambda pair: pair[1], reverse=True)[:3],
        "illustrative_new_rank1_errors": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "illustrative_new_rank1_errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
