#!/usr/bin/env python3
"""Build a deterministic, train-only cross-camera RGBNT201 dev protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


NAME_PATTERN = re.compile(r"^(?P<pid>\d{6})_cam(?P<camera>\d+)_.*\.jpg$")
SALT = "TriFusion-RGBNT201-dev-v1"


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def file_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.jpg")}


def identity_records(directory: Path) -> dict[str, list[tuple[str, int]]]:
    records: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in sorted(directory.glob("*.jpg")):
        match = NAME_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Unexpected RGBNT201 filename: {path.name}")
        records[match.group("pid")].append(
            (path.name, int(match.group("camera")))
        )
    return dict(records)


def hash_rank(identity: str) -> str:
    return hashlib.sha256(f"{SALT}:{identity}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/data/RGBNT201"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("/root/mmreid-trifusion/TriFusion-ReID/protocols"),
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path(
            "/root/mmreid-trifusion/artifacts/rgbnt201_dev_protocol_v1_audit.json"
        ),
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    split171 = root / "train_171"
    split141 = root / "train_141"
    test = root / "test"
    for required in (split171, split141, test):
        if not required.is_dir():
            raise FileNotFoundError(required)
    modality_pairing = {}
    for split_name, split in (
        ("train_171", split171),
        ("train_141", split141),
        ("test", test),
    ):
        names = {modality: file_names(split / modality) for modality in ("RGB", "NI", "TI")}
        modality_pairing[split_name] = {
            "counts": {modality: len(values) for modality, values in names.items()},
            "paired": names["RGB"] == names["NI"] == names["TI"],
        }

    records171 = identity_records(split171 / "RGB")
    records141 = identity_records(split141 / "RGB")
    test_records = identity_records(test / "RGB")
    identities171 = set(records171)
    eligible = sorted(
        identity
        for identity, records in records171.items()
        if len({camera for _name, camera in records}) >= 2
    )
    ranked = sorted(eligible, key=lambda identity: (hash_rank(identity), identity))
    dev_ids = sorted(ranked[:30])
    train_ids = sorted(identities171 - set(dev_ids))
    provided_dev_ids = sorted(identities171 - set(records141))

    dev_records = [record for identity in dev_ids for record in records171[identity]]
    queries_without_positive = [
        name
        for identity in dev_ids
        for name, camera in records171[identity]
        if not any(
            other_camera != camera
            for _other_name, other_camera in records171[identity]
        )
    ]
    test_overlap = sorted(set(test_records) & identities171)
    checks = {
        "train171_modalities_paired": modality_pairing["train_171"]["paired"],
        "train141_modalities_paired": modality_pairing["train_141"]["paired"],
        "test_modalities_paired": modality_pairing["test"]["paired"],
        "source_has_171_identities": len(identities171) == 171,
        "eligible_cross_camera_identities_is_51": len(eligible) == 51,
        "train_has_141_identities": len(train_ids) == 141,
        "dev_has_30_identities": len(dev_ids) == 30,
        "train_dev_disjoint": not (set(train_ids) & set(dev_ids)),
        "train_dev_cover_train171": set(train_ids) | set(dev_ids) == identities171,
        "test_identity_disjoint": not test_overlap,
        "every_dev_query_has_cross_camera_positive": not queries_without_positive,
    }
    protocol = {
        "name": "RGBNT201 train-only dev protocol v1",
        "valid": all(checks.values()),
        "checks": checks,
        "selection": {
            "salt": SALT,
            "rule": (
                "Among train_171 identities with >=2 cameras, sort by "
                "SHA256(salt:identity), hold out the first 30."
            ),
            "uses_test_labels": False,
            "eligible_identity_count": len(eligible),
        },
        "evaluation": {
            "query": "all held-out identity triplets",
            "gallery": "the same held-out triplet list",
            "filter": "exclude same identity and same camera, matching official RGBNT201 evaluator",
            "query_triplets": len(dev_records),
            "gallery_triplets": len(dev_records),
            "queries_without_valid_positive": len(queries_without_positive),
        },
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "test_ids": sorted(test_records),
        "counts": {
            "train_triplets": sum(len(records171[identity]) for identity in train_ids),
            "dev_triplets": len(dev_records),
            "train171_triplets": sum(map(len, records171.values())),
        },
        "provided_train141_diagnostic": {
            "provided_dev_ids": provided_dev_ids,
            "provided_dev_identity_count": len(provided_dev_ids),
            "provided_dev_cross_camera_identity_count": sum(
                len({camera for _name, camera in records171[identity]}) >= 2
                for identity in provided_dev_ids
            ),
            "provided_dev_single_camera_identity_count": sum(
                len({camera for _name, camera in records171[identity]}) == 1
                for identity in provided_dev_ids
            ),
            "overlap_with_v1_dev": len(set(provided_dev_ids) & set(dev_ids)),
            "reason_not_used_for_retrieval_selection": (
                "Single-camera identities have no positive after the official "
                "same-pid/same-camera exclusion."
            ),
        },
        "modality_pairing": modality_pairing,
        "test_identity_overlap": test_overlap,
    }
    encoded = json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.protocol_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(args.protocol_dir / "rgbnt201_dev_v1.json", encoded)
    atomic_text(
        args.protocol_dir / "rgbnt201_train_ids_v1.txt",
        "\n".join(train_ids) + "\n",
    )
    atomic_text(
        args.protocol_dir / "rgbnt201_dev_ids_v1.txt",
        "\n".join(dev_ids) + "\n",
    )
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.audit_out, encoded)
    print(encoded, end="")
    return 0 if protocol["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
