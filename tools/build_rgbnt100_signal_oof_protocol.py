#!/usr/bin/env python3
"""Freeze identity folds from the complete RGBNT100 official-training inventory."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    inventory_path = ROOT / "evidence/rgbnt100_signal_train_file_inventory_20260906.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = inventory["rows"]
    assert len(records) == 8675
    by_id = defaultdict(list)
    for i, row in enumerate(records):
        assert row["index"] == i
        assert row["path"].startswith("rgbir/bounding_box_train/")
        assert row["view"] == -1 and row["camera"] in range(8)
        by_id[row["identity"]].append(row)
    assert sorted(by_id) == list(range(501, 600, 2))
    eligible = sorted(pid for pid, rows in by_id.items() if len({r["camera"] for r in rows}) >= 2)
    single_camera = sorted(set(by_id) - set(eligible))
    heldout_groups = [[] for _ in range(3)]
    for group in (eligible, single_camera):
        for i, pid in enumerate(group):
            heldout_groups[i % 3].append(pid)
    folds = []
    for fold_id, heldout in enumerate(heldout_groups):
        heldout = sorted(heldout)
        source_ids = sorted(set(by_id) - set(heldout))
        source = [r["index"] for r in records if r["identity"] in source_ids]
        gallery = [r["index"] for r in records if r["identity"] in heldout]
        query_rows = []
        for position, index in enumerate(gallery):
            row = records[index]
            positives = [x["index"] for x in by_id[row["identity"]] if x["camera"] != row["camera"]]
            if positives:
                query_rows.append({"record_index": index, "gallery_position": position,
                                   "identity": row["identity"], "valid_positive_count": len(positives)})
        source_cameras = sorted({records[i]["camera"] for i in source})
        assert source_cameras == list(range(8))
        assert set(source).isdisjoint(gallery) and len(source) + len(gallery) == 8675
        folds.append({"fold": fold_id, "source_ids": source_ids, "heldout_ids": heldout,
                      "source_label_map": {str(pid): i for i, pid in enumerate(source_ids)},
                      "source_camera_values": source_cameras,
                      "source_record_indices": source, "gallery_record_indices": gallery,
                      "query_rows": query_rows,
                      "counts": {"source_identities": len(source_ids), "source_records": len(source),
                                 "heldout_identities": len(heldout), "gallery_records": len(gallery),
                                 "query_records": len(query_rows),
                                 "query_identities": len({r["identity"] for r in query_rows})}})
    assert sorted(i for f in folds for i in f["gallery_record_indices"]) == list(range(8675))
    report = {"schema": "rgbnt100-train-identity-oof-v1",
              "status": "FROZEN_DATA_PROTOCOL_TRAINING_CONTRACT_PENDING",
              "dataset": "RGBNT100", "source_split": "rgbir/bounding_box_train",
              "inventory_sha256": sha256(inventory_path), "builder_sha256": sha256(__file__),
              "split_rule": "Within each sorted cross-camera/single-camera identity group, round-robin identities to folds0/1/2; labels only, no split scan.",
              "evaluation": {"type": "train_internal_identity_oof_not_official",
                             "gallery": "All records of every heldout identity in its own fold",
                             "query": "Every heldout record with at least one same-ID different-camera positive",
                             "remove": "same identity AND same camera",
                             "retain": "Every different-identity record, including same-camera negatives",
                             "aggregation": "Pool per-query results after fold-local distances; no cross-fold feature comparison",
                             "official_query_gallery_access": 0},
              "counts": {"identities": len(by_id), "cross_camera_identities": len(eligible),
                         "single_camera_identities": len(single_camera), "gallery_records": len(records),
                         "query_records": sum(f["counts"]["query_records"] for f in folds)},
              "identity_camera_counts": {str(pid): dict(sorted(Counter(r["camera"] for r in rows).items()))
                                         for pid, rows in sorted(by_id.items())},
              "records": records, "folds": folds}
    output = ROOT / "protocols/rgbnt100_train_oof_v1.json"
    assert not output.exists()
    output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"protocol": str(output), "sha256": sha256(output),
                      "counts": report["counts"], "folds": [f["counts"] for f in folds]}))


if __name__ == "__main__":
    build()

