#!/usr/bin/env python3
"""Summarize all fixed V17 queries and render the preregistered image cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def run(args):
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    started = time.time()
    torch.set_num_threads(4)
    report = json.loads(args.diagnostic.read_text())
    output = args.diagnostic.parent
    assert not (output / "analysis.json").exists()
    names = ("fused", "cnn", "transformer", "mamba")
    experts = names[1:]
    all_rows = {name: [] for name in names}
    part_rows = []
    cases = []
    for fold in report["folds"]:
        endpoints = fold["endpoints"]
        caches = {}
        for name, endpoint in endpoints.items():
            path = Path(endpoint["cache"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == endpoint["cache_sha256"]
            caches[name] = torch.load(path, map_location="cpu", weights_only=True)
        left, right = caches["weight0"], caches["dtred"]
        assert torch.equal(left["baseline"], right["baseline"])
        assert torch.equal(left["cnn_parts"], right["cnn_parts"])
        for expert in experts:
            assert torch.equal(left[f"teacher_{expert}"], right[f"teacher_{expert}"])
        baseline = F.normalize(left["baseline"], dim=1)
        distances = {}
        for endpoint, tensors in caches.items():
            residual = {e: F.normalize(tensors[f"residual_{e}"], dim=1) for e in experts}
            residual["fused"] = F.normalize(torch.cat([residual[e] for e in experts], dim=1), dim=1)
            distances[endpoint] = {
                name: (2.0 - (baseline @ baseline.T + value @ value.T)).clamp_min(0).sqrt().numpy()
                for name, value in residual.items()
            }
        for name in names:
            queries0 = endpoints["weight0"]["outputs"][name]["queries"]
            queries1 = endpoints["dtred"]["outputs"][name]["queries"]
            for a, b in zip(queries0, queries1, strict=True):
                query = a["query_index"]
                assert query == b["query_index"]
                pos, neg = a["nearest_positive"], a["nearest_negative"]
                d0, d1 = distances["weight0"][name], distances["dtred"][name]
                # Fixed weight0-selected pairs separate shifts from changing neighbors.
                row = {
                    "fold": fold["fold"], "query_index": query,
                    "file": Path(fold["gallery_manifest"][query]["files"][0]).name,
                    "ap_delta_pp": (b["average_precision"] - a["average_precision"]) * 100,
                    "rank1_break": a["first_match_rank"] == 1 and b["first_match_rank"] != 1,
                    "rank1_fix": a["first_match_rank"] != 1 and b["first_match_rank"] == 1,
                    "nearest_positive_distance_delta": b["positive_distance"] - a["positive_distance"],
                    "nearest_negative_distance_delta": b["negative_distance"] - a["negative_distance"],
                    "nearest_margin_delta": b["nearest_margin"] - a["nearest_margin"],
                    "fixed_pair_positive_distance_delta": float(d1[query, pos] - d0[query, pos]),
                    "fixed_pair_negative_distance_delta": float(d1[query, neg] - d0[query, neg]),
                }
                if name in experts:
                    for endpoint in endpoints:
                        geom = endpoints[endpoint]["geometry"][name]
                        row[endpoint + "_correction_norm"] = geom["delta_norm"][query]
                        row[endpoint + "_teacher_cosine"] = geom["teacher_corrected_cosine"][query]
                        row[endpoint + "_modality_energy"] = geom["modality_energy_fraction"][query]
                all_rows[name].append(row)

        parts = F.normalize(left["cnn_parts"], dim=-1)
        cnn_queries = endpoints["weight0"]["outputs"]["cnn"]["queries"]
        qi = torch.tensor([row["query_index"] for row in cnn_queries])
        pi = torch.tensor([row["nearest_positive"] for row in cnn_queries])
        ni = torch.tensor([row["nearest_negative"] for row in cnn_queries])
        sim_pos = torch.einsum("nmid,nmjd->nmij", parts[qi], parts[pi])
        sim_neg = torch.einsum("nmid,nmjd->nmij", parts[qi], parts[ni])
        for row, query in enumerate(qi.tolist()):
            diagonal_pos = sim_pos[row].diagonal(dim1=-2, dim2=-1)
            diagonal_neg = sim_neg[row].diagonal(dim1=-2, dim2=-1)
            part_rows.append({
                "fold": fold["fold"], "query_index": query,
                "positive_similarity": sim_pos[row].tolist(),
                "negative_similarity": sim_neg[row].tolist(),
                "diagonal_margin_by_modality_part": (diagonal_pos - diagonal_neg).tolist(),
                "positive_off_diagonal_best_fraction": float((sim_pos[row].argmax(dim=-1) != torch.arange(4)).float().mean()),
                "negative_off_diagonal_best_fraction": float((sim_neg[row].argmax(dim=-1) != torch.arange(4)).float().mean()),
            })

        selections = (("cnn_worst_ap", "cnn", False), ("cnn_best_ap", "cnn", True),
                      ("fused_worst_ap", "fused", False))
        for label, name, maximize in selections:
            q0 = endpoints["weight0"]["outputs"][name]["queries"]
            q1 = endpoints["dtred"]["outputs"][name]["queries"]
            deltas = np.array([b["average_precision"] - a["average_precision"] for a, b in zip(q0, q1, strict=True)])
            index = int(deltas.argmax() if maximize else deltas.argmin())
            a, b = q0[index], q1[index]
            selected = [a["query_index"], a["nearest_positive"], a["nearest_negative"], b["nearest_positive"], b["nearest_negative"]]
            roles = ["Query", "weight0 positive", "weight0 negative", "DTRED positive", "DTRED negative"]
            fig, axes = plt.subplots(3, 5, figsize=(12, 8))
            for col, (gallery_index, role) in enumerate(zip(selected, roles, strict=True)):
                record = fold["gallery_manifest"][gallery_index]
                for modal in range(3):
                    with Image.open(record["files"][modal]) as source:
                        axes[modal, col].imshow(source.convert("RGB"))
                    axes[modal, col].set_xticks([])
                    axes[modal, col].set_yticks([])
                    if col == 0:
                        axes[modal, col].set_ylabel(("RGB", "NI", "TI")[modal])
                    if modal == 0:
                        axes[modal, col].set_title(role + "\n" + Path(record["files"][0]).name, fontsize=7)
            title = f"fold {fold['fold']} {label}: AP {a['average_precision']*100:.2f} -> {b['average_precision']*100:.2f}; rank {a['first_match_rank']} -> {b['first_match_rank']}"
            fig.suptitle(title, fontsize=11)
            fig.tight_layout()
            path = output / f"fold_{fold['fold']}_{label}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            cases.append({"fold": fold["fold"], "selection": label, "output": name,
                          "query_index": a["query_index"], "ap_delta_pp": float(deltas[index] * 100),
                          "gallery_indices": selected, "image": str(path),
                          "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        del caches, left, right, distances

    summary = {}
    for name, rows in all_rows.items():
        groups = {
            "all": rows,
            "ap_harmed": [r for r in rows if r["ap_delta_pp"] < 0],
            "ap_improved": [r for r in rows if r["ap_delta_pp"] > 0],
            "rank1_broken": [r for r in rows if r["rank1_break"]],
            "rank1_fixed": [r for r in rows if r["rank1_fix"]],
        }
        summary[name] = {}
        keys = [k for k in rows[0] if k.endswith("delta") or k.endswith("norm") or k.endswith("cosine") or k == "ap_delta_pp"]
        for group, group_rows in groups.items():
            summary[name][group] = {"count": len(group_rows), **{
                k: float(np.mean([r[k] for r in group_rows])) for k in keys
            }}
        if name in experts:
            summary[name]["modality_energy_mean"] = {
                endpoint: np.mean([r[endpoint + "_modality_energy"] for r in rows], axis=0).tolist()
                for endpoint in ("weight0", "dtred")
            }
    part_summary = {
        "count": len(part_rows),
        "diagonal_margin_by_modality_part": np.mean([r["diagonal_margin_by_modality_part"] for r in part_rows], axis=0).tolist(),
        "positive_off_diagonal_best_fraction": float(np.mean([r["positive_off_diagonal_best_fraction"] for r in part_rows])),
        "negative_off_diagonal_best_fraction": float(np.mean([r["negative_off_diagonal_best_fraction"] for r in part_rows])),
    }
    result = {
        "schema_version": "v17-fixed-failure-analysis-v1", "status": "PASS",
        "diagnostic_sha256": hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "all_571_consumed_fit_queries_not_independent_validation",
        "summary": summary, "cnn_part_summary": part_summary, "cases": cases,
        "all_query_geometry": all_rows, "all_query_part_similarity": part_rows,
        "optimizer_steps": 0, "checkpoint_writes": 0, "dev_access_count": 0,
        "official_test_access_count": 0, "elapsed_seconds": time.time() - started,
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "cnn_part_summary": part_summary, "cases": cases}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    run(parser.parse_args())
