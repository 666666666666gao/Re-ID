#!/usr/bin/env python3
"""Prepare the MSVR310 train-only identity protocol from existing label evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def build_protocol(label_file: Path) -> dict:
    label_bytes = label_file.read_bytes()
    assert hashlib.sha256(label_bytes).hexdigest() == (
        "c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8"
    )
    dataset = next(
        row for row in json.loads(label_bytes)["datasets"]
        if row["dataset"] == "MSVR310"
    )
    train = dataset["record_manifest"]["bounding_box_train"]
    assert len(train) == 1032
    assert train == sorted(train, key=lambda row: row["path"])
    identities = sorted({row["identity"] for row in train})
    assert len(identities) == 155
    scenes = {
        identity: sorted({row["scene"] for row in train if row["identity"] == identity})
        for identity in identities
    }
    eligible = [identity for identity in identities if len(scenes[identity]) > 1]
    ineligible = [identity for identity in identities if len(scenes[identity]) == 1]
    assert (len(eligible), len(ineligible)) == (60, 95)

    heldout_sets = [set() for _ in range(3)]
    for group in (eligible, ineligible):
        for index, identity in enumerate(group):
            heldout_sets[index % 3].add(identity)

    records = []
    for index, row in enumerate(train):
        path = Path(row["path"])
        assert path.parts[1] == "vis"
        name = path.name
        assert (int(name[:4]), int(name[11]), int(name[6:9])) == (
            row["identity"], row["camera"], row["scene"]
        )
        prefix = Path("bounding_box_train") / path.parts[0]
        records.append({
            "index": index,
            "identity": row["identity"],
            "camera": row["camera"],
            "scene": row["scene"],
            "paths": [(prefix / modality / name).as_posix()
                      for modality in ("vis", "ni", "th")],
        })

    folds = []
    for fold_index, heldout_ids in enumerate(heldout_sets):
        source_ids = sorted(set(identities) - heldout_ids)
        source_indices = [row["index"] for row in records
                          if row["identity"] in source_ids]
        gallery_indices = [row["index"] for row in records
                           if row["identity"] in heldout_ids]
        gallery = [records[index] for index in gallery_indices]
        id_counts = Counter(row["identity"] for row in gallery)
        scene_counts = Counter((row["identity"], row["scene"]) for row in gallery)
        query_rows = []
        excluded_query_indices = []
        for position, row in enumerate(gallery):
            removed = scene_counts[(row["identity"], row["scene"])]
            positives = id_counts[row["identity"]] - removed
            if positives == 0:
                excluded_query_indices.append(row["index"])
                continue
            query_rows.append({
                "record_index": row["index"],
                "gallery_position": position,
                "identity": row["identity"],
                "scene": row["scene"],
                "valid_positives": positives,
                "removed_same_identity_same_scene": removed,
                "retained_gallery": len(gallery) - removed,
                "negative_identity_distractors": len(gallery) - id_counts[row["identity"]],
            })
        source = [records[index] for index in source_indices]
        assert set(source_indices).isdisjoint(gallery_indices)
        assert sorted(source_indices + gallery_indices) == list(range(1032))
        assert len({row["identity"] for row in query_rows}) == 20
        folds.append({
            "fold": fold_index,
            "source_ids": source_ids,
            "source_label_map": {str(identity): local_label
                                 for local_label, identity in enumerate(source_ids)},
            "heldout_ids": sorted(heldout_ids),
            "source_record_indices": source_indices,
            "gallery_record_indices": gallery_indices,
            "query_rows": query_rows,
            "excluded_query_record_indices": excluded_query_indices,
            "counts": {
                "source_identities": len(source_ids),
                "source_records": len(source),
                "source_cross_scene_identities": len(set(source_ids) & set(eligible)),
                "source_identity_camera_pairs": len({(r["identity"], r["camera"]) for r in source}),
                "source_identity_scene_pairs": len({(r["identity"], r["scene"]) for r in source}),
                "heldout_identities": len(heldout_ids),
                "gallery_records": len(gallery),
                "query_identities": len({row["identity"] for row in query_rows}),
                "valid_queries": len(query_rows),
                "excluded_queries_without_cross_scene_positive": len(excluded_query_indices),
                "gallery_only_distractor_identities": len(set(heldout_ids) & set(ineligible)),
                "gallery_only_distractor_records": len(excluded_query_indices),
            },
            "source_camera_values": sorted({row["camera"] for row in source}),
            "source_scene_values": sorted({row["scene"] for row in source}),
            "heldout_scene_values": sorted({row["scene"] for row in gallery}),
        })

    heldout_union = [identity for fold in folds for identity in fold["heldout_ids"]]
    gallery_union = [index for fold in folds for index in fold["gallery_record_indices"]]
    query_union = [row["record_index"] for fold in folds for row in fold["query_rows"]]
    assert sorted(heldout_union) == identities
    assert sorted(gallery_union) == list(range(1032))
    assert len(query_union) == len(set(query_union)) == 600
    return {
        "schema": "msvr310-train-identity-oof-v1",
        "status": "DATA_PROTOCOL_FROZEN_TRAINING_CONTRACT_PENDING",
        "scope": "Existing official-training labels only; no image, feature, model, tensor or retrieval execution",
        "dataset": "MSVR310",
        "source_split": "bounding_box_train",
        "dataset_root_at_inventory": dataset["root"],
        "label_evidence_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_rule": (
            "Partition sorted identities by whether they have >=2 distinct scene labels; "
            "round-robin each sorted group into folds 0/1/2. Label-only deterministic rule; "
            "no random draw, feature ranking, hard-identity selection or split scan."
        ),
        "split_rule_source": (
            "tools/build_v8_oof_router_targets.py:19-39 supplies the existing stratified "
            "round-robin pattern; this protocol uses MSVR310 scene eligibility."
        ),
        "training_seed_when_registered": 42,
        "evaluation": {
            "type": "train_internal_identity_oof_not_official_test",
            "gallery": "Every record of every heldout identity in its own fold",
            "query": "Heldout records with at least one same-identity different-scene gallery record",
            "remove": "same identity AND same scene",
            "retain": "All different-identity gallery records, including same-scene negatives and single-scene identities",
            "aggregation": "Pool per-query AP/Rank results after fold-local distances; never compare features across folds",
            "separate_development_split": "None registered in this data protocol",
            "checkpoint_selection": "Future training contract must use fixed source-only endpoints",
            "official_query_and_gallery": "Not used to construct folds, source labels, query masks or scores",
        },
        "records": records,
        "identity_scene_membership": {str(identity): scenes[identity] for identity in identities},
        "folds": folds,
        "aggregate_counts": {
            "unique_training_identities": len(identities),
            "unique_gallery_records": len(gallery_union),
            "unique_query_identities": 60,
            "valid_queries": len(query_union),
            "excluded_query_records_retained_in_gallery": 1032 - len(query_union),
            "gallery_only_distractor_identities": len(ineligible),
        },
        "new_optimizer_steps": 0,
        "new_retrieval_evaluations": 0,
        "new_image_reads": 0,
        "new_checkpoint_loads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_protocol(args.label_evidence)
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({
        "status": report["status"],
        "folds": [{"fold": row["fold"], **row["counts"]} for row in report["folds"]],
        "aggregate_counts": report["aggregate_counts"],
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "new_optimizer_steps": 0,
        "new_retrieval_evaluations": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
