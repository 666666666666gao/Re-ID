#!/usr/bin/env python3
"""Read-only legal first-match margins from a completed MSVR310 Signal OOF run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert summary["evaluation_type"] == "train_internal_identity_oof_baseline"
    assert len(summary["folds"]) == 3

    rows = []
    gallery_count = 0
    for fold in summary["folds"]:
        retrieval = fold["retrieval"]
        array_path = Path(fold["checkpoint"]).parent / "retrieval_arrays.pt"
        assert hashlib.sha256(array_path.read_bytes()).hexdigest() == retrieval["retrieval_arrays_sha256"]
        arrays = torch.load(array_path, map_location="cpu", weights_only=True)
        distances = arrays["distances"].numpy()
        manifest = retrieval["gallery_manifest"]
        positions = arrays["query_gallery_positions"]
        assert distances.shape == (len(positions), len(manifest))
        assert len(positions) == len(retrieval["query_rows"])
        gallery_count += len(manifest)
        identities = np.asarray([item["identity"] for item in manifest])
        scenes = np.asarray([item["scene"] for item in manifest])

        for index, (query, position) in enumerate(zip(retrieval["query_rows"], positions, strict=True)):
            assert manifest[position]["index"] == query["record_index"]
            identity, scene = identities[position], scenes[position]
            positive = (identities == identity) & (scenes != scene)
            negative = identities != identity
            assert positive.any() and negative.any()
            order = np.argsort(distances[index])
            legal = ~((identities[order] == identity) & (scenes[order] == scene))
            first_rank = int(np.flatnonzero(identities[order][legal] == identity)[0] + 1)
            assert first_rank == retrieval["first_match_rank"][index]
            nearest_positive = float(distances[index, positive].min())
            nearest_negative = float(distances[index, negative].min())
            rows.append({
                "fold": fold["fold"],
                "query_record_index": query["record_index"],
                "identity": int(identity),
                "first_match_rank": first_rank,
                "legal_positive_count": int(positive.sum()),
                "legal_positive_scene_count": int(np.unique(scenes[positive]).size),
                "nearest_positive_squared_distance": nearest_positive,
                "nearest_negative_squared_distance": nearest_negative,
                "first_match_margin_cosine": (nearest_negative - nearest_positive) / 2,
            })

    assert len(rows) == 600 and gallery_count == 1032
    rank1_correct = sum(row["first_match_rank"] == 1 for row in rows)
    assert rank1_correct / len(rows) * 100 == summary["aggregate"]["Rank-1"]
    margins = np.asarray([row["first_match_margin_cosine"] for row in rows])
    quantiles = {str(q): float(np.quantile(margins, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    report = {
        "schema": "msvr310-signal-oof-first-match-margin-v1",
        "status": "COMPLETE_DIAGNOSTIC_NOT_MODEL_RESULT",
        "baseline_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "baseline_source_oof_metrics": summary["aggregate"],
        "legal_query_count": len(rows),
        "full_gallery_record_count": gallery_count,
        "rank1_correct_count": rank1_correct,
        "rank1_incorrect_count": len(rows) - rank1_correct,
        "first_match_margin_cosine_definition": "(nearest different-identity squared distance - nearest legal same-identity different-scene squared distance) / 2",
        "first_match_margin_cosine_quantiles": quantiles,
        "legal_positive_count_quantiles": {
            str(q): float(np.quantile([row["legal_positive_count"] for row in rows], q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "query_rows": rows,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.summary, args.output)
