#!/usr/bin/env python3
"""Freeze all RGBNT100 official files, labels and complete query masks remotely."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    assert not args.output.exists()
    inventory = json.loads(args.train_inventory.read_text(encoding="utf-8"))
    original_train = {row["path"]: row for row in inventory["rows"]}
    assert len(original_train) == 8675
    expected_counts = {"train": 8675, "query": 1715, "gallery": 8575}
    directories = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}
    rows = {name: [] for name in directories}
    pattern = re.compile(r"([-\d]+)_c([-\d]+)")
    for split, directory in directories.items():
        paths = sorted((args.dataset_root / "rgbir" / directory).glob("*.jpg"))
        assert len(paths) == expected_counts[split]
        for index, path in enumerate(paths):
            identity, camera = map(int, pattern.search(path.name).groups())
            assert 501 <= identity <= 600 and 1 <= camera <= 8
            relative = path.relative_to(args.dataset_root).as_posix()
            with Image.open(path) as image:
                assert image.size == (768, 128) and image.mode == "RGB"
                image.verify()
            row = {"index": index, "path": relative, "identity": identity,
                   "camera": camera - 1, "view": -1, "bytes": path.stat().st_size,
                   "sha256": sha(path)}
            if split == "train":
                old = original_train[relative]
                assert all(row[key] == old[key] for key in
                           ("identity", "camera", "view", "bytes", "sha256"))
            rows[split].append(row)
    ids = {name: sorted({r["identity"] for r in values}) for name, values in rows.items()}
    assert ids["train"] == list(range(501, 600, 2))
    assert ids["query"] == ids["gallery"] == list(range(502, 601, 2))
    cameras = {name: sorted({r["camera"] for r in values}) for name, values in rows.items()}
    assert all(values == list(range(8)) for values in cameras.values())
    gallery_members = defaultdict(list)
    for row in rows["gallery"]:
        gallery_members[row["identity"]].append(row)
    queries = []
    for row in rows["query"]:
        members = gallery_members[row["identity"]]
        positives = [r["index"] for r in members if r["camera"] != row["camera"]]
        excluded = [r["index"] for r in members if r["camera"] == row["camera"]]
        assert positives and len(positives) + len(excluded) == len(members)
        queries.append({"query_index": row["index"], "identity": row["identity"],
                        "positive_gallery_positions": positives,
                        "excluded_gallery_positions": excluded,
                        "valid_positive_count": len(positives),
                        "negative_count": len(rows["gallery"]) - len(members),
                        "retained_gallery_count": len(rows["gallery"]) - len(excluded)})
    result = {
        "schema": "rgbnt100-official-fixed-main-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_root": str(args.dataset_root), "counts": expected_counts,
        "identities": ids, "camera_values": cameras,
        "train_label_map": {str(pid): i for i, pid in enumerate(ids["train"])},
        "records": rows, "query_rows": queries,
        "evaluation": {"query": "All1715 official query files",
                       "gallery": "All8575 official gallery files",
                       "remove": "same identity AND same camera",
                       "retain": "Every different identity, including same-camera negatives",
                       "order": "Sorted relative file path within each official split",
                       "checkpoint_selection": "Fixed endpoints before any official model evaluation"},
        "identity_record_counts": {name: dict(sorted(Counter(r["identity"] for r in values).items()))
                                   for name, values in rows.items()},
        "train_inventory_sha256": sha(args.train_inventory), "builder_sha256": sha(Path(__file__)),
        "all_file_hashes_and_rgb_montage_headers_verified": True,
        "official_test_file_metadata_records": 10290,
        "model_forwards": 0, "training_updates": 0,
        "status": "FROZEN_OFFICIAL_PROTOCOL_NO_MODEL_EVALUATION",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "counts": expected_counts, "identities": {k: len(v) for k, v in ids.items()},
                      "status": result["status"], "model_forwards": 0, "training_updates": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("dataset-root", "train-inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
