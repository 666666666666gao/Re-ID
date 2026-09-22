"""Bind the audited training and official retrieval records for all three datasets."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SPLITS = {
    "RGBNT201": ("train_171", "test", "test", "camera", (3951, 836, 836)),
    "RGBNT100": ("bounding_box_train", "query", "bounding_box_test", "camera", (8675, 1715, 8575)),
    "MSVR310": ("bounding_box_train", "query3", "bounding_box_test", "scene", (1032, 591, 1055)),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_paths(dataset, split, row):
    if dataset == "RGBNT201":
        return [f"{split}/{modality}/{row['path']}" for modality in ("RGB", "NI", "TI")]
    if dataset == "MSVR310":
        identity, modality, filename = Path(row["path"]).parts
        assert modality == "vis"
        return [f"{split}/{identity}/{name}/{filename}" for name in ("vis", "ni", "th")]
    assert dataset == "RGBNT100"
    return [f"rgbir/{split}/{row['path']}"]


def build(dataset, data_root, inventory_sha):
    name = dataset["dataset"]
    train_split, query_split, gallery_split, environment, expected = SPLITS[name]
    source = dataset["record_manifest"]
    assert dataset["train_test_identity_disjoint"]
    assert tuple(len(source[split]) for split in (train_split, query_split, gallery_split)) == expected
    train_ids = sorted({row["identity"] for row in source[train_split]})
    test_ids = {row["identity"] for row in source[gallery_split]}
    assert set(train_ids).isdisjoint(test_ids)
    label_map = {str(identity): index for index, identity in enumerate(train_ids)}
    root = data_root / name
    records = {}
    for label, split in (("train", train_split), ("query", query_split), ("gallery", gallery_split)):
        rows = []
        for index, source_row in enumerate(source[split]):
            paths = record_paths(name, split, source_row)
            assert all((root / path).is_file() for path in paths), (name, split, index)
            camera = source_row["camera"] - 1 if name == "RGBNT201" else source_row["camera"]
            scene = source_row["scene"] if name == "MSVR310" else camera
            rows.append(dict(index=index, paths=paths, identity=source_row["identity"],
                             camera=camera, scene=scene,
                             view=scene if name == "MSVR310" else -1,
                             label=label_map[str(source_row["identity"])] if label == "train" else None))
        records[label] = rows
    gallery_counts = Counter(row["identity"] for row in records["gallery"])
    excluded_counts = Counter((row["identity"], row[environment]) for row in records["gallery"])
    query_rows = []
    for row in records["query"]:
        positives = gallery_counts[row["identity"]] - excluded_counts[(row["identity"], row[environment])]
        assert positives > 0
        query_rows.append(dict(index=row["index"], identity=row["identity"],
                               valid_positive_count=positives,
                               excluded_same_identity_same_environment=excluded_counts[(row["identity"], row[environment])]))
    assert len(query_rows) == dataset["eligible_queries"] and dataset["invalid_queries"] == 0
    assert [min(row["valid_positive_count"] for row in query_rows),
            max(row["valid_positive_count"] for row in query_rows)] == dataset["positive_count_range"]
    if name == "RGBNT201":
        assert records["query"] == records["gallery"]
    else:
        gallery_names = {Path(row["paths"][0]).name for row in records["gallery"]}
        assert len(gallery_names) == len(records["gallery"])
        assert all(Path(row["paths"][0]).name in gallery_names for row in records["query"])
    return dict(schema="trifusion-official-three-dataset-protocol-v1", dataset=name,
                inventory_sha256=inventory_sha, dataset_root=str(root), seed=42,
                environment_key=environment, train_split=train_split, query_split=query_split,
                gallery_split=gallery_split, train_label_map=label_map, records=records,
                query_rows=query_rows, counts={key: len(value) for key, value in records.items()},
                filter="exclude same identity and same environment; retain all different identities",
                model_forwards=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert sha256(args.inventory) == args.inventory_sha256
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    datasets = {dataset["dataset"]: dataset for dataset in inventory["datasets"]}
    assert set(datasets) == set(SPLITS)
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    for name in SPLITS:
        protocol = build(datasets[name], args.data_root, args.inventory_sha256)
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(dataset=name, path=str(path), sha256=sha256(path),
                              counts=protocol["counts"], environment=protocol["environment_key"])))


if __name__ == "__main__":
    main()
