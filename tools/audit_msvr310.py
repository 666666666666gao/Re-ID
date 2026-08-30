#!/usr/bin/env python3
"""Audit MSVR310 identity splits and RGB/NIR/TIR triplet alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("bounding_box_train", "query3", "bounding_box_test")
MODALITIES = ("vis", "ni", "th")


def jpeg_has_valid_markers(path: Path) -> bool:
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, 2)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def audit_split(root: Path, split: str) -> tuple[dict[str, object], set[int]]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing split: {split_root}")

    identities: set[int] = set()
    triplets = 0
    cameras: set[int] = set()
    scenes: set[int] = set()
    pairing_mismatches: dict[str, object] = {}
    invalid_jpegs: list[str] = []

    for identity_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
        identity = int(identity_root.name)
        identities.add(identity)
        names_by_modality: dict[str, set[str]] = {}
        for modality in MODALITIES:
            modality_root = identity_root / modality
            if not modality_root.is_dir():
                raise FileNotFoundError(f"missing modality: {modality_root}")
            images = sorted(modality_root.glob("*.jpg"))
            names_by_modality[modality] = {path.name for path in images}
            invalid_jpegs.extend(
                str(path.relative_to(root)) for path in images if not jpeg_has_valid_markers(path)
            )

        reference = names_by_modality["vis"]
        triplets += len(reference)
        for filename in reference:
            if int(filename[:4]) != identity:
                raise ValueError(f"identity directory/name mismatch: {identity_root / filename}")
            cameras.add(int(filename[11]))
            scenes.add(int(filename[6:9]))
        mismatched = {
            modality: {
                "missing_vs_vis": sorted(reference - names),
                "extra_vs_vis": sorted(names - reference),
            }
            for modality, names in names_by_modality.items()
            if names != reference
        }
        if mismatched:
            pairing_mismatches[identity_root.name] = mismatched

    return (
        {
            "identities": len(identities),
            "triplets": triplets,
            "images": triplets * len(MODALITIES),
            "cameras": sorted(cameras),
            "scenes": len(scenes),
            "pairing_mismatches": pairing_mismatches,
            "invalid_jpegs": invalid_jpegs,
        },
        identities,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("/root/mmreid-trifusion/data/MSVR310"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    reports: dict[str, object] = {}
    identities: dict[str, set[int]] = {}
    for split in SPLITS:
        reports[split], identities[split] = audit_split(root, split)

    train_ids = identities["bounding_box_train"]
    query_ids = identities["query3"]
    gallery_ids = identities["bounding_box_test"]
    protocol_checks = {
        "train_test_disjoint": train_ids.isdisjoint(gallery_ids),
        "query_ids_subset_of_gallery": query_ids <= gallery_ids,
        "total_unique_identities_is_310": len(train_ids | gallery_ids) == 310,
    }
    valid = (
        all(not report["pairing_mismatches"] and not report["invalid_jpegs"] for report in reports.values())
        and all(protocol_checks.values())
    )
    output = {
        "dataset": "MSVR310",
        "root": str(root),
        "splits": reports,
        "protocol_checks": protocol_checks,
        "valid": valid,
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
