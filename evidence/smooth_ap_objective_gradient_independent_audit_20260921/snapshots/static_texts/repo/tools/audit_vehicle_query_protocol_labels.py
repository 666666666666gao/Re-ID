#!/usr/bin/env python3
"""Count vehicle query/gallery label masks without images, features or retrieval."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def msvr_records(directory):
    return [
        {"path": str(path.relative_to(directory)), "identity": int(path.name[:4]),
         "camera": int(path.name[11]), "scene": int(path.name[6:9])}
        for path in sorted(directory.glob("*/vis/*.jpg"))
    ]


def rgbnt_records(directory):
    pattern = re.compile(r"([-\d]+)_c([-\d]+)")
    records = []
    for path in sorted(directory.glob("*.jpg")):
        identity, camera = map(int, pattern.search(path.name).groups())
        assert 1 <= identity <= 600 and 1 <= camera <= 8
        records.append({"path": path.name, "identity": identity,
                        "camera": camera - 1})
    return records


def audit(name, root, query_name, reader, exclusion_key):
    records = {split: reader(root / split) for split in
               ("bounding_box_train", query_name, "bounding_box_test")}
    train, query, gallery = records.values()
    assert train and query and gallery
    train_ids = {row["identity"] for row in train}
    gallery_ids = {row["identity"] for row in gallery}
    assert train_ids.isdisjoint(gallery_ids)
    assert {row["path"] for row in query} <= {row["path"] for row in gallery}
    id_counts = Counter(row["identity"] for row in gallery)
    mask_counts = Counter((row["identity"], row[exclusion_key]) for row in gallery)
    camera_counts = Counter((row["identity"], row["camera"]) for row in gallery)
    rows = []
    for row in query:
        total_positives = id_counts[row["identity"]]
        removed = mask_counts[(row["identity"], row[exclusion_key])]
        camera_removed = camera_counts[(row["identity"], row["camera"])]
        rows.append({**row, "removed_same_identity_and_protocol_key": removed,
                     "retained_gallery": len(gallery) - removed,
                     "valid_positives": total_positives - removed,
                     "negative_identity_distractors": len(gallery) - total_positives,
                     "camera_only_valid_positives": total_positives - camera_removed})
    raw = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "dataset": name, "root": str(root), "exclusion_key": exclusion_key,
        "exclusion_rule": "gallery identity equals query identity AND protocol key equals query key",
        "train_test_identity_disjoint": True, "query_paths_subset_of_gallery": True,
        "split_counts": {split: {"records": len(items),
                                 "identities": len({r["identity"] for r in items})}
                         for split, items in records.items()},
        "record_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "eligible_queries": sum(row["valid_positives"] > 0 for row in rows),
        "invalid_queries": sum(row["valid_positives"] == 0 for row in rows),
        "positive_count_range": [min(row["valid_positives"] for row in rows),
                                 max(row["valid_positives"] for row in rows)],
        "queries_with_different_camera_only_positive_count": sum(
            row["valid_positives"] != row["camera_only_valid_positives"] for row in rows),
        "query_rows": rows, "record_manifest": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "vehicle-query-protocol-label-audit-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "dataset label inventory only; no image or checkpoint loading",
        "optimizer_steps": 0, "retrieval_evaluation_runs": 0,
        "datasets": [
            audit("MSVR310", args.data_root / "MSVR310", "query3", msvr_records, "scene"),
            audit("RGBNT100", args.data_root / "RGBNT100/rgbir", "query", rgbnt_records, "camera"),
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in item.items()
                       if k not in ("query_rows", "record_manifest")}
                      for item in report["datasets"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
