#!/usr/bin/env python3
"""Audit the extracted RGBNT201 dataset before any experiment consumes it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODALITIES = ("RGB", "NI", "TI")
SPLITS = ("train_171", "train_141", "test")
EXPECTED_IDENTITIES = {"train_171": 171, "train_141": 141, "test": 30}
EXPECTED_TRIPLETS = {"train_171": 3951, "train_141": 3280, "test": 836}
EXPECTED_CAMERAS = {
    "train_171": [1, 2, 3, 4],
    "train_141": [1, 2, 3, 4],
    "test": [1, 2],
}


def parse_name(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 2 or not parts[0][:6].isdigit() or not parts[1].startswith("cam"):
        raise ValueError(f"unexpected RGBNT201 filename: {path.name}")
    return int(parts[0][:6]), int(parts[1][3:])


def jpeg_has_valid_markers(path: Path) -> bool:
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, 2)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def audit_split(root: Path, split: str) -> dict[str, object]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing split: {split_root}")

    names_by_modality: dict[str, set[str]] = {}
    sizes_by_modality: dict[str, int] = {}
    invalid_jpegs: list[str] = []
    for modality in MODALITIES:
        modality_root = split_root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"missing modality: {modality_root}")
        images = sorted(modality_root.glob("*.jpg"))
        names_by_modality[modality] = {path.name for path in images}
        sizes_by_modality[modality] = sum(path.stat().st_size for path in images)
        invalid_jpegs.extend(
            str(path.relative_to(root)) for path in images if not jpeg_has_valid_markers(path)
        )

    reference = names_by_modality["RGB"]
    mismatches = {
        modality: {
            "missing_vs_rgb": sorted(reference - names),
            "extra_vs_rgb": sorted(names - reference),
        }
        for modality, names in names_by_modality.items()
        if names != reference
    }
    identities: set[int] = set()
    cameras: set[int] = set()
    for name in reference:
        pid, camera = parse_name(Path(name))
        identities.add(pid)
        cameras.add(camera)

    return {
        "triplets": len(reference),
        "images": len(reference) * len(MODALITIES),
        "identities": len(identities),
        "identity_min": min(identities) if identities else None,
        "identity_max": max(identities) if identities else None,
        "cameras": sorted(cameras),
        "bytes_by_modality": sizes_by_modality,
        "pairing_mismatches": mismatches,
        "invalid_jpegs": invalid_jpegs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("/root/mmreid-trifusion/data/RGBNT201"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    report = {
        "dataset": "RGBNT201",
        "root": str(root),
        "splits": {split: audit_split(root, split) for split in SPLITS},
    }
    report["expected_identity_counts_match"] = {
        split: report["splits"][split]["identities"] == expected
        for split, expected in EXPECTED_IDENTITIES.items()
    }
    report["expected_triplet_counts_match"] = {
        split: report["splits"][split]["triplets"] == expected
        for split, expected in EXPECTED_TRIPLETS.items()
    }
    report["expected_camera_sets_match"] = {
        split: report["splits"][split]["cameras"] == expected
        for split, expected in EXPECTED_CAMERAS.items()
    }
    report["valid"] = all(
        not split["pairing_mismatches"] and not split["invalid_jpegs"]
        for split in report["splits"].values()
    ) and all(
        all(report[key].values())
        for key in (
            "expected_identity_counts_match",
            "expected_triplet_counts_match",
            "expected_camera_sets_match",
        )
    )

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
