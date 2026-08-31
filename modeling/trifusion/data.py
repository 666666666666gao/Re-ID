"""Train-only RGBNT201 loaders for preregistered TriFusion development."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.datasets.bases import ImageDataset
from data.datasets.make_dataloader import (
    RandomErasing,
    train_collate_fn,
    val_collate_fn,
)
from data.datasets.sampler import RandomIdentitySampler

from .intervention_targets import assign_identity_fold


Record = tuple[list[str], int, int, int]


@dataclass(frozen=True, eq=False)
class TriFusionDataLoaders:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_query: int
    num_classes: int
    train_records: tuple[Record, ...]
    query_records: tuple[Record, ...]
    gallery_records: tuple[Record, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_registry_sha256(identities: set[str]) -> str:
    payload = json.dumps(
        sorted(int(identity) for identity in identities),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_record(rgb_path: Path, *, label: int) -> Record:
    filename = rgb_path.name
    fields = filename.split("_")
    if len(fields) < 2 or len(fields[0]) < 6 or len(fields[1]) < 4:
        raise ValueError(f"unexpected RGBNT201 filename: {filename}")
    camera_id = int(fields[1][3]) - 1
    paths = [
        rgb_path,
        rgb_path.parents[1] / "NI" / filename,
        rgb_path.parents[1] / "TI" / filename,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"unpaired RGBNT201 modalities: {missing}")
    return ([str(path) for path in paths], label, camera_id, -1)


def _records_for_ids(
    train_root: Path,
    identities: set[str],
    *,
    relabel: bool,
) -> tuple[Record, ...]:
    label_map = {
        identity: index for index, identity in enumerate(sorted(identities))
    }
    records = []
    for rgb_path in sorted((train_root / "RGB").glob("*.jpg")):
        identity = rgb_path.name[:6]
        if identity not in identities:
            continue
        label = label_map[identity] if relabel else int(identity)
        records.append(_parse_record(rgb_path, label=label))
    return tuple(records)


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([256, 128], interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop([256, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            RandomErasing(
                probability=0.5,
                mode="pixel",
                max_count=1,
                device="cpu",
            ),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([256, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _make_loaders(
    *,
    train_records: tuple[Record, ...],
    eval_records: tuple[Record, ...],
    train_batch_size: int,
    num_instances: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = ImageDataset(train_records, _train_transform())
    eval_dataset = ImageDataset(eval_records + eval_records, _eval_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=RandomIdentitySampler(
            train_records,
            batch_size=train_batch_size,
            num_instances=num_instances,
        ),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader


def build_rgbnt201_oof_loaders(
    *,
    dataset_root: Path | str,
    protocol_path: Path | str,
    circ_protocol_path: Path | str,
    target_fold: int,
    train_batch_size: int,
    num_instances: int,
    eval_batch_size: int,
    num_workers: int,
) -> TriFusionDataLoaders:
    """Build one development OOF target-generator complement and target fold."""

    if train_batch_size <= 0 or num_instances <= 0:
        raise ValueError("train batch size and K must be positive")
    if train_batch_size % num_instances:
        raise ValueError("train batch size must be divisible by num_instances")
    if eval_batch_size <= 0 or num_workers < 0:
        raise ValueError("eval batch size must be positive and workers nonnegative")

    dataset_root = Path(dataset_root).expanduser().resolve()
    protocol_path = Path(protocol_path).expanduser().resolve()
    circ_protocol_path = Path(circ_protocol_path).expanduser().resolve()
    train_root = dataset_root / "train_171"
    if (
        not train_root.is_dir()
        or not protocol_path.is_file()
        or not circ_protocol_path.is_file()
    ):
        raise FileNotFoundError(
            "RGBNT201 train_171, dev protocol, or frozen CIRC protocol is missing"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("selection", {}).get("uses_test_labels") is not False:
        raise ValueError("OOF development protocol must be test-label blind")
    eligible_ids = {str(value) for value in protocol["train_ids"]}
    forbidden_dev_ids = {str(value) for value in protocol["dev_ids"]}
    if (
        len(eligible_ids) != 141
        or len(forbidden_dev_ids) != 30
        or eligible_ids & forbidden_dev_ids
    ):
        raise ValueError("frozen 141-fit/30-dev identity registry is invalid")

    circ_protocol = json.loads(circ_protocol_path.read_text(encoding="utf-8"))
    if circ_protocol.get("schema_version") != "circ-protocol-v1":
        raise ValueError("unsupported frozen CIRC protocol schema")
    if int(circ_protocol.get("official_test_access_count", -1)) != 0:
        raise ValueError("OOF development must have zero official-test access")
    if circ_protocol.get("dev_protocol_sha256") != _sha256(protocol_path):
        raise ValueError("CIRC protocol does not bind the frozen dev protocol")
    folds = dict(circ_protocol.get("folds", {}))
    fold_count = int(folds.get("count", 0))
    fold_salt = str(folds.get("salt", ""))
    if fold_count != 3 or not fold_salt:
        raise ValueError("CIRC development requires exactly three registered folds")
    if folds.get("identity_canonicalization") != "unsigned-decimal":
        raise ValueError("CIRC fold identity canonicalization is not registered")
    registry_sha256 = _identity_registry_sha256(eligible_ids)
    if folds.get("development_identity_sha256") != registry_sha256:
        raise ValueError("CIRC protocol development identity registry mismatch")
    if target_fold < 0 or target_fold >= fold_count:
        raise ValueError("target_fold must identify one of the three registered folds")

    target_ids = {
        identity
        for identity in eligible_ids
        if assign_identity_fold(
            identity,
            fold_salt=fold_salt,
            fold_count=fold_count,
        )
        == target_fold
    }
    generator_ids = eligible_ids - target_ids
    overlap = generator_ids & target_ids
    if overlap or not target_ids or not generator_ids:
        raise ValueError("OOF identity partition is empty or overlapping")

    train_records = _records_for_ids(train_root, generator_ids, relabel=True)
    target_records = _records_for_ids(train_root, target_ids, relabel=False)
    target_cameras: dict[int, set[int]] = defaultdict(set)
    target_frequency: dict[int, int] = defaultdict(int)
    for _paths, identity, camera_id, _view in target_records:
        target_cameras[identity].add(camera_id)
        target_frequency[identity] += 1
    supported_ids = {
        identity for identity, cameras in target_cameras.items() if len(cameras) >= 2
    }
    supported_records = sum(target_frequency[identity] for identity in supported_ids)
    train_loader, eval_loader = _make_loaders(
        train_records=train_records,
        eval_records=target_records,
        train_batch_size=train_batch_size,
        num_instances=num_instances,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    provenance = {
        "dataset_root": str(dataset_root),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "circ_protocol_path": str(circ_protocol_path),
        "circ_protocol_sha256": _sha256(circ_protocol_path),
        "fold_salt": fold_salt,
        "fold_count": fold_count,
        "target_fold": target_fold,
        "generator_training_identities": len(generator_ids),
        "target_identities": len(target_ids),
        "generator_training_identity_values": sorted(int(value) for value in generator_ids),
        "target_identity_values": sorted(int(value) for value in target_ids),
        "generator_training_records": len(train_records),
        "target_records": len(target_records),
        "target_cross_camera_supported_identities": len(supported_ids),
        "target_cross_camera_unsupported_identities": len(target_ids) - len(supported_ids),
        "target_cross_camera_supported_records": supported_records,
        "target_cross_camera_unsupported_records": len(target_records) - supported_records,
        "generator_target_identity_overlap": len(overlap),
        "forbidden_dev_identities": len(forbidden_dev_ids),
        "target_forbidden_dev_identity_overlap": len(target_ids & forbidden_dev_ids),
        "protocol_uses_test_labels": False,
        "official_test_records": 0,
        "query_equals_gallery_record_list": True,
    }
    return TriFusionDataLoaders(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_query=len(target_records),
        num_classes=len(generator_ids),
        train_records=train_records,
        query_records=target_records,
        gallery_records=target_records,
        provenance=provenance,
    )


def build_rgbnt201_dev_loaders(
    *,
    dataset_root: Path | str,
    protocol_path: Path | str,
    train_batch_size: int,
    num_instances: int,
    eval_batch_size: int,
    num_workers: int,
) -> TriFusionDataLoaders:
    """Build the 141-fit/30-dev protocol without touching official test files."""

    if train_batch_size <= 0 or num_instances <= 0:
        raise ValueError("train batch size and K must be positive")
    if train_batch_size % num_instances:
        raise ValueError("train batch size must be divisible by num_instances")
    if eval_batch_size <= 0 or num_workers < 0:
        raise ValueError("eval batch size must be positive and workers nonnegative")
    dataset_root = Path(dataset_root).expanduser().resolve()
    protocol_path = Path(protocol_path).expanduser().resolve()
    train_root = dataset_root / "train_171"
    if not train_root.is_dir() or not protocol_path.is_file():
        raise FileNotFoundError("RGBNT201 train_171 or frozen dev protocol is missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("selection", {}).get("uses_test_labels") is not False:
        raise ValueError("development protocol must be test-label blind")
    train_ids = set(protocol["train_ids"])
    dev_ids = set(protocol["dev_ids"])
    overlap = train_ids & dev_ids
    if overlap or len(train_ids) != 141 or len(dev_ids) != 30:
        raise ValueError("frozen protocol identity partition is invalid")

    train_records = _records_for_ids(train_root, train_ids, relabel=True)
    dev_records = _records_for_ids(train_root, dev_ids, relabel=False)
    if len(train_records) != int(protocol["counts"]["train_triplets"]):
        raise ValueError("fit record count differs from the frozen protocol")
    if len(dev_records) != int(protocol["counts"]["dev_triplets"]):
        raise ValueError("dev record count differs from the frozen protocol")
    dev_cameras: dict[int, set[int]] = defaultdict(set)
    for _paths, identity, camera_id, _view in dev_records:
        dev_cameras[identity].add(camera_id)
    if any(len(cameras) < 2 for cameras in dev_cameras.values()):
        raise ValueError("a dev identity lacks cross-camera positive support")

    query_records = dev_records
    gallery_records = dev_records
    train_loader, eval_loader = _make_loaders(
        train_records=train_records,
        eval_records=dev_records,
        train_batch_size=train_batch_size,
        num_instances=num_instances,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    provenance = {
        "dataset_root": str(dataset_root),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_identities": len(train_ids),
        "dev_identities": len(dev_ids),
        "train_dev_identity_overlap": len(overlap),
        "protocol_uses_test_labels": False,
        "official_test_records": 0,
        "query_equals_gallery_record_list": True,
        "all_dev_identities_have_cross_camera_support": True,
    }
    return TriFusionDataLoaders(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_query=len(query_records),
        num_classes=len(train_ids),
        train_records=train_records,
        query_records=query_records,
        gallery_records=gallery_records,
        provenance=provenance,
    )


__all__ = [
    "TriFusionDataLoaders",
    "build_rgbnt201_dev_loaders",
    "build_rgbnt201_oof_loaders",
]
