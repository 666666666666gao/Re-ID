#!/usr/bin/env python3
"""Decompose every source relation using previously verified, fixed features."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
SLOTS = tuple(f"{e}_{m}_residual" for e in EXPERTS for m in MODALITIES)
OLD_OUTPUTS = ("baseline_only", "fused", *EXPERTS, *SLOTS)
NEW_OUTPUTS = ("pure_bank", *(f"pure_{e}" for e in EXPERTS))
OUTPUTS = (*OLD_OUTPUTS, *NEW_OUTPUTS)
VIEWS = ("clean", "augmented")
PROTOCOLS = ("identity_exclude_record", "cross_camera")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    assert np.all(norms > 0)
    return array / norms


def query_metrics(scores, positive, negative):
    legal = np.sort(np.concatenate((positive, negative)))
    order = legal[np.argsort(-scores[legal], kind="stable")]
    positions = np.flatnonzero(np.isin(order, positive)) + 1
    ap = float(np.mean(np.arange(1, len(positions) + 1) / positions))
    margins = scores[positive, None] - scores[None, negative]
    return {"average_precision": ap, "first_match_rank": int(positions[0]),
            "nonpositive_pairs": int((margins <= 0).sum())}, margins


def relation_patterns(slot_margin, role_margin, bank_margin, signal_margin, fused_margin):
    # The four bits are overlapping factual signs, not a mutually exclusive story.
    bits = ((signal_margin <= 0).astype(np.int8) * 8
            + (bank_margin <= 0).astype(np.int8) * 4
            + (role_margin <= 0).astype(np.int8) * 2
            + (fused_margin <= 0).astype(np.int8))
    wrong = slot_margin <= 0
    return np.bincount(bits[wrong], minlength=16).tolist()


def mathematical_check():
    # slot wrong throughout: role rescue, cross-role rescue, Signal rescue,
    # unresolved, and role-correct but bank-harmful/Signal-rescued respectively.
    slot = np.asarray([[-.2, -.2, -.2, -.2, -.2]])
    role = np.asarray([[.1, -.1, -.1, -.1, .1]])
    bank = np.asarray([[.1, .1, -.1, -.1, -.1]])
    signal = np.asarray([[.3, .3, .3, -.3, .3]])
    fused = .5 * (bank + signal)
    actual = relation_patterns(slot, role, bank, signal, fused)
    expected = [0] * 16
    for code in (0, 2, 6, 15, 4):
        expected[code] += 1
    assert actual == expected
    assert sum(actual) == 5
    return {"status": "PASS_FIVE_KNOWN_SUPPORT_PATTERNS", "relation_count": 5}


def run(args):
    started = time.time()
    assert sha(args.contract) == args.contract_sha256
    contract = json.loads(args.contract.read_bytes())
    assert sha(__file__) == contract["runner_sha256"]
    source = Path(contract["source_run"])
    assert sha(source / "source_census.json") == contract["source_summary_sha256"]
    assert sha(source / "verification.json") == contract["source_verification_sha256"]
    source_summary = json.loads((source / "source_census.json").read_bytes())
    verification = json.loads((source / "verification.json").read_bytes())
    assert verification["status"] == "PASS_ALL_SOURCE_INSTANCE_CENSUS_ROWS_AND_AGGREGATES"
    assert verification["rows_checked"] == 350112
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING_ALL_SOURCE_SUPPORT_DECOMPOSITION", "math_check": mathematical_check(),
              "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": sha(__file__), "contract_sha256": args.contract_sha256,
              "source_summary_sha256": contract["source_summary_sha256"],
              "source_verification_sha256": contract["source_verification_sha256"],
              "pattern_bits": {"8": "signal_nonpositive", "4": "pure_bank_nonpositive",
                               "2": "same_role_residual_nonpositive", "1": "fused_nonpositive"},
              "model_forwards": 0, "image_reads": 0, "optimizer_updates": 0,
              "new_model_weights": 0, "folds": []}
    output = args.output_dir / "support_decomposition.json"

    def save():
        report["elapsed_seconds"] = time.time() - started
        output.write_bytes((json.dumps(report, indent=2) + "\n").encode())

    save()
    total_rows = total_relations = 0
    for fold in source_summary["folds"]:
        k = fold["fold"]
        descriptor = fold["features"]
        assert sha(descriptor["path"]) == descriptor["sha256"]
        features = torch.load(descriptor["path"], map_location="cpu", weights_only=True)
        assert features["manifest"] == fold["manifest"]
        ids = np.asarray([r["identity"] for r in fold["manifest"]])
        cams = np.asarray([r["camera"] for r in fold["manifest"]])
        n = len(ids)
        result = {"fold": k, "source_records": n, "input_feature_sha256": descriptor["sha256"],
                  "comparisons": [], "numerical_checks": []}
        arrays = {v: {name: t.numpy().astype(np.float64) for name, t in features["views"][v].items()} for v in VIEWS}
        for view in VIEWS:
            arrays[view]["pure_bank"] = unit(arrays[view]["fused"][:, 3072:])
            for e in EXPERTS:
                arrays[view][f"pure_{e}"] = unit(arrays[view][e][:, 3072:])
        for view in VIEWS:
            query, gallery = arrays[view], arrays["clean"]
            scores = {name: query[name] @ gallery[name].T for name in OUTPUTS}
            prefix = query["fused"][:, :3072] @ gallery["fused"][:, :3072].T
            residual = query["fused"][:, 3072:] @ gallery["fused"][:, 3072:].T
            exact_error = float(np.max(np.abs(scores["fused"] - prefix - residual)))
            mix_error = float(np.max(np.abs(scores["fused"] - .5 * (scores["baseline_only"] + scores["pure_bank"])) ))
            bank_error = float(np.max(np.abs(scores["pure_bank"] - sum(scores[s] for s in SLOTS) / 9)))
            role_error = max(float(np.max(np.abs(scores[f"pure_{e}"] - sum(scores[f"{e}_{m}_residual"] for m in MODALITIES) / 3))) for e in EXPERTS)
            assert exact_error < 1e-12 and max(mix_error, bank_error, role_error) < 1e-5
            result["numerical_checks"].append({"view": view, "exact_prefix_plus_residual_error": exact_error,
                                               "equal_energy_mix_error": mix_error,
                                               "bank_slot_mean_error": bank_error, "role_slot_mean_error": role_error})
            del prefix, residual
            for protocol in PROTOCOLS:
                previous = next(c for c in fold["comparisons"] if (c["view"], c["protocol"]) == (view, protocol))
                assert sha(previous["all_rows"]["path"]) == previous["all_rows"]["sha256"]
                expected = {}
                with gzip.open(previous["all_rows"]["path"], "rt", encoding="utf-8") as handle:
                    for line in handle:
                        r = json.loads(line)
                        expected[(r["output"], r["query_index"])] = r
                assert len(expected) == n * 14
                stats = {name: {"ap_sum": 0., "rank1_hits": 0, "nonpositive_pairs": 0, "queries_ap_below_one": 0} for name in OUTPUTS}
                patterns = {s: [0] * 16 for s in SLOTS}
                eligible = relations = 0
                path = args.output_dir / f"fold_{k}_{view}_{protocol}_all_query_support.jsonl.gz"
                with gzip.open(path, "xt", encoding="utf-8") as handle:
                    for q in range(n):
                        positive = np.flatnonzero((ids == ids[q]) & ((np.arange(n) != q) if protocol == "identity_exclude_record" else (cams != cams[q])))
                        negative = np.flatnonzero(ids != ids[q])
                        row = {"fold": k, "view": view, "protocol": protocol, "query_index": q,
                               "file": fold["manifest"][q]["file"], "identity": int(ids[q]), "camera": int(cams[q]),
                               "eligible": bool(len(positive)), "gallery_records": n,
                               "positive_records": len(positive), "negative_records": len(negative)}
                        if len(positive):
                            eligible += 1
                            relation_count = len(positive) * len(negative)
                            relations += relation_count
                            total_relations += relation_count
                            rows, margins = {}, {}
                            for name in OUTPUTS:
                                rows[name], margins[name] = query_metrics(scores[name][q], positive, negative)
                                if name in OLD_OUTPUTS:
                                    old = expected[(name, q)]
                                    assert old["eligible"]
                                    assert abs(rows[name]["average_precision"] - old["average_precision"]) < 1e-12
                                    assert rows[name]["first_match_rank"] == old["first_match_rank"]
                                    assert rows[name]["nonpositive_pairs"] == old["nonpositive_ordered_positive_negative_pairs"]
                                stats[name]["ap_sum"] += rows[name]["average_precision"]
                                stats[name]["rank1_hits"] += rows[name]["first_match_rank"] == 1
                                stats[name]["nonpositive_pairs"] += rows[name]["nonpositive_pairs"]
                                stats[name]["queries_ap_below_one"] += rows[name]["average_precision"] < 1 - 1e-12
                            per_slot = {}
                            for s in SLOTS:
                                e = s.split("_", 1)[0]
                                counts = relation_patterns(margins[s], margins[f"pure_{e}"], margins["pure_bank"], margins["baseline_only"], margins["fused"])
                                assert sum(counts) == rows[s]["nonpositive_pairs"]
                                patterns[s] = [a + b for a, b in zip(patterns[s], counts)]
                                per_slot[s] = counts
                            row.update(outputs=rows, slot_error_joint_sign_counts=per_slot)
                        else:
                            assert all(not expected[(name, q)]["eligible"] for name in OLD_OUTPUTS)
                        handle.write(json.dumps(row) + "\n")
                        total_rows += 1
                for name in OUTPUTS:
                    stats[name]["source_mAP_percent"] = stats[name]["ap_sum"] / eligible * 100
                    stats[name]["source_Rank1_percent"] = stats[name]["rank1_hits"] / eligible * 100
                for name in OLD_OUTPUTS:
                    assert abs(stats[name]["source_mAP_percent"] - previous["aggregates"][name]["source_probe_mAP_percent"]) < 1e-9
                    assert stats[name]["nonpositive_pairs"] == previous["aggregates"][name]["nonpositive_ordered_positive_negative_pairs"]
                result["comparisons"].append({"view": view, "protocol": protocol, "all_queries": n,
                                               "eligible_queries": eligible, "all_positive_negative_relations": relations,
                                               "outputs": stats, "slot_error_joint_sign_counts": patterns,
                                               "all_query_rows": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "rows": n}})
        report["folds"].append(result)
        save()
        print(json.dumps({"stage": "complete_support_fold", "fold": k, "source_records": n}), flush=True)
        del features, arrays, scores
    assert total_rows == 25008 and len(report["folds"]) == 3
    assert total_relations == 2 * (286626942 + 30550172)
    report.update(status="COMPLETE_ALL_SOURCE_SUPPORT_DECOMPOSITION", all_query_rows=total_rows,
                  all_positive_negative_relations=total_relations, derived_output_metric_rows=3 * 2 * 2 * 4,
                  original_output_metric_rows_revalidated=168, feature_files_loaded=3,
                  limitations=["source_only_fixed_features", "no_model_intervention", "one_fixed_augmented_query_view",
                               "clean_gallery_only", "sign_partitions_not_unique_causal_attribution",
                               "not_unseen_identity_performance_or_ablation_qualification"])
    save()
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
