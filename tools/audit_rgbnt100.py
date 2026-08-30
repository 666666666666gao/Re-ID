#!/usr/bin/env python3
"""Audit RGBNT100 raw triplets and the official composite ReID protocol."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


MODALITIES = ("R", "N", "T")
SPLITS = ("bounding_box_train", "query", "bounding_box_test")
EXPECTED_SPLIT_TRIPLETS = {
    "bounding_box_train": 8_675,
    "query": 1_715,
    "bounding_box_test": 8_575,
}
NAME_PATTERN = re.compile(r"^(?P<pid>\d{4})_c(?P<camera>\d{4})_(?P<frame>\d{3})\.jpg$")


def parse_name(path: Path) -> tuple[int, int, int]:
    match = NAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected RGBNT100 filename: {path.name}")
    return tuple(int(match.group(key)) for key in ("pid", "camera", "frame"))


def inspect_jpeg(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with path.open("rb") as handle:
            start = handle.read(2)
            handle.seek(-2, 2)
            end = handle.read(2)
        if start != b"\xff\xd8" or end != b"\xff\xd9":
            return False, None
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return True, size
    except (OSError, ValueError):
        return False, None


def audit_images(root: Path, images: list[Path]) -> dict[str, object]:
    invalid: list[str] = []
    dimensions: Counter[str] = Counter()
    for path in images:
        valid, size = inspect_jpeg(path)
        if not valid or size is None:
            invalid.append(str(path.relative_to(root)))
        else:
            dimensions[f"{size[0]}x{size[1]}"] += 1
    return {
        "invalid_jpegs": invalid,
        "dimensions": dict(sorted(dimensions.items())),
        "bytes": sum(path.stat().st_size for path in images),
    }


def audit_raw(root: Path) -> tuple[dict[str, object], dict[str, set[str]]]:
    names_by_modality: dict[str, set[str]] = {}
    reports: dict[str, object] = {}
    for modality in MODALITIES:
        modality_root = root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"missing raw modality: {modality_root}")
        images = sorted(modality_root.glob("*/*.jpg"))
        names = {path.name for path in images}
        names_by_modality[modality] = names
        identities = {int(path.parent.name) for path in images}
        cameras = {parse_name(path)[1] for path in images}
        image_audit = audit_images(root, images)
        reports[modality] = {
            "triplets": len(images),
            "identities": len(identities),
            "identity_min": min(identities) if identities else None,
            "identity_max": max(identities) if identities else None,
            "cameras": sorted(cameras),
            **image_audit,
        }

    reference = names_by_modality["R"]
    pairing_mismatches = {
        modality: {
            "missing_vs_R": sorted(reference - names),
            "extra_vs_R": sorted(names - reference),
        }
        for modality, names in names_by_modality.items()
        if names != reference
    }
    return {
        "modalities": reports,
        "aligned_triplets": len(reference),
        "images": len(reference) * len(MODALITIES),
        "pairing_mismatches": pairing_mismatches,
    }, names_by_modality


def audit_split(
    root: Path, split: str, raw_names: set[str]
) -> tuple[dict[str, object], set[int], set[str]]:
    split_root = root / "rgbir" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing protocol split: {split_root}")
    images = sorted(split_root.glob("*.jpg"))
    names = {path.name for path in images}
    identities: set[int] = set()
    cameras: set[int] = set()
    for path in images:
        pid, camera, _ = parse_name(path)
        identities.add(pid)
        cameras.add(camera)
    image_audit = audit_images(root, images)
    return (
        {
            "triplets": len(images),
            "composite_images": len(images),
            "spectral_images_represented": len(images) * len(MODALITIES),
            "identities": len(identities),
            "identity_min": min(identities) if identities else None,
            "identity_max": max(identities) if identities else None,
            "cameras": sorted(cameras),
            "missing_raw_counterparts": sorted(names - raw_names),
            **image_audit,
        },
        identities,
        names,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("/root/mmreid-trifusion/data/RGBNT100"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    raw_report, raw_names_by_modality = audit_raw(root)
    raw_names = raw_names_by_modality["R"]
    split_reports: dict[str, object] = {}
    split_ids: dict[str, set[int]] = {}
    split_names: dict[str, set[str]] = {}
    for split in SPLITS:
        split_reports[split], split_ids[split], split_names[split] = audit_split(
            root, split, raw_names
        )

    train_ids = split_ids["bounding_box_train"]
    query_ids = split_ids["query"]
    gallery_ids = split_ids["bounding_box_test"]
    expected_dimensions = {
        "raw_all_dimensions_readable": all(
            sum(report["dimensions"].values()) == 17_250
            for report in raw_report["modalities"].values()
        ),
        "composites_all_768x128": all(
            report["dimensions"] == {
                "768x128": EXPECTED_SPLIT_TRIPLETS[split]
            }
            for split, report in split_reports.items()
        ),
    }
    protocol_checks = {
        "raw_has_17250_aligned_triplets": raw_report["aligned_triplets"] == 17_250,
        "raw_has_100_identities_per_modality": all(
            report["identities"] == 100
            for report in raw_report["modalities"].values()
        ),
        "split_counts_match_readme": all(
            split_reports[split]["triplets"] == expected
            for split, expected in EXPECTED_SPLIT_TRIPLETS.items()
        ),
        "train_has_50_identities": len(train_ids) == 50,
        "gallery_has_50_identities": len(gallery_ids) == 50,
        "query_has_50_identities": len(query_ids) == 50,
        "train_test_identities_disjoint": train_ids.isdisjoint(gallery_ids),
        "query_identities_equal_gallery": query_ids == gallery_ids,
        "query_files_are_gallery_subset": split_names["query"] <= split_names["bounding_box_test"],
        "all_processed_files_have_raw_counterparts": all(
            not report["missing_raw_counterparts"] for report in split_reports.values()
        ),
        **expected_dimensions,
    }
    no_invalid_jpegs = (
        all(
            not report["invalid_jpegs"]
            for report in raw_report["modalities"].values()
        )
        and all(not report["invalid_jpegs"] for report in split_reports.values())
    )
    valid = (
        not raw_report["pairing_mismatches"]
        and no_invalid_jpegs
        and all(protocol_checks.values())
    )
    output = {
        "dataset": "RGBNT100",
        "root": str(root),
        "raw": raw_report,
        "splits": split_reports,
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
