"""Geometry-aligned RGB/NIR/TIR training input."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torchvision.transforms import InterpolationMode, RandomCrop, RandomErasing
from torchvision.transforms import functional as transform_functional


def _read_triplet(paths: Sequence[str]) -> tuple[Image.Image, Image.Image, Image.Image]:
    if len(paths) != 3:
        raise ValueError("RGBNT201 records must contain RGB, NIR and TIR paths")
    images = tuple(Image.open(path).convert("RGB") for path in paths)
    return images


class SharedGeometryTripletTransform:
    """Sample flip/crop once, then apply independent tensor-space erasing."""

    def __init__(
        self,
        *,
        size: tuple[int, int] = (256, 128),
        padding: int = 10,
        flip_probability: float = 0.5,
        erase_probability: float = 0.5,
    ) -> None:
        self.size = tuple(int(value) for value in size)
        self.padding = int(padding)
        self.flip_probability = float(flip_probability)
        self.erase = RandomErasing(p=float(erase_probability), value="random")

    def __call__(self, images: Sequence[Image.Image]) -> list[torch.Tensor]:
        if len(images) != 3:
            raise ValueError("triplet transform requires exactly three modalities")
        transformed = [
            transform_functional.resize(
                image,
                self.size,
                interpolation=InterpolationMode.BICUBIC,
            )
            for image in images
        ]
        if random.random() < self.flip_probability:
            transformed = [transform_functional.hflip(image) for image in transformed]
        transformed = [
            transform_functional.pad(image, self.padding) for image in transformed
        ]
        top, left, height, width = RandomCrop.get_params(
            transformed[0], output_size=self.size
        )
        transformed = [
            transform_functional.crop(image, top, left, height, width)
            for image in transformed
        ]
        tensors = [
            transform_functional.normalize(
                transform_functional.to_tensor(image),
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
            for image in transformed
        ]
        return [self.erase(tensor) for tensor in tensors]


class AlignedTripletImageDataset(Dataset):
    """Apply one triplet-aware transform to one RGBNT201 record."""

    def __init__(
        self,
        records: Sequence[tuple[Sequence[str], int, int, int]],
        *,
        transform: Callable[[Sequence[Image.Image]], list[torch.Tensor]],
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        paths, identity, camera, view = self.records[index]
        images = self.transform(_read_triplet(paths))
        return images, identity, camera, view, Path(paths[0]).name


class CrossCameraIdentitySampler(Sampler):
    """Emit deterministic identity batches with one cross-camera group each."""

    def __init__(
        self,
        records: Sequence[tuple[Sequence[str], int, int, int]],
        *,
        batch_size: int,
        num_instances: int,
        seed: int,
    ) -> None:
        self.records = records
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.identities_per_batch = self.batch_size // self.num_instances
        self.seed = int(seed)
        self.iteration = 0
        self.indices_by_identity: dict[int, list[int]] = defaultdict(list)
        self.camera_by_index: dict[int, int] = {}
        for index, (_paths, identity, camera, _view) in enumerate(records):
            self.indices_by_identity[int(identity)].append(index)
            self.camera_by_index[index] = int(camera)
        total_groups = sum(
            max(len(indices), self.num_instances) // self.num_instances
            for indices in self.indices_by_identity.values()
        )
        self.length = (
            total_groups // self.identities_per_batch * self.batch_size
        )

    def _identity_groups(
        self,
        identity: int,
        rng: random.Random,
    ) -> tuple[list[list[int]], bool]:
        indices = list(self.indices_by_identity[identity])
        if len(indices) < self.num_instances:
            indices.extend(
                rng.choices(indices, k=self.num_instances - len(indices))
            )
        group_count = len(indices) // self.num_instances
        cameras: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            cameras[self.camera_by_index[index]].append(index)
        if len(cameras) == 1:
            rng.shuffle(indices)
            return (
                [
                    indices[offset : offset + self.num_instances]
                    for offset in range(
                        0,
                        group_count * self.num_instances,
                        self.num_instances,
                    )
                ],
                False,
            )

        ordered_cameras = sorted(
            cameras.values(),
            key=len,
            reverse=True,
        )
        primary = list(ordered_cameras[0])
        secondary = [
            index for camera_indices in ordered_cameras[1:] for index in camera_indices
        ]
        if len(primary) < group_count or len(secondary) < group_count:
            raise ValueError(
                "cross-camera identities must supply one sample per camera group"
            )
        rng.shuffle(primary)
        rng.shuffle(secondary)
        anchors = list(zip(primary[:group_count], secondary[:group_count], strict=True))
        remainder = list(indices)
        for pair in anchors:
            for index in pair:
                remainder.remove(index)
        rng.shuffle(remainder)
        groups = []
        fill = self.num_instances - 2
        for group_index, pair in enumerate(anchors):
            start = group_index * fill
            groups.append([*pair, *remainder[start : start + fill]])
        return groups, True

    def __iter__(self):
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        queues: dict[int, list[list[int]]] = {}
        cross_camera: set[int] = set()
        for identity in self.indices_by_identity:
            groups, is_cross_camera = self._identity_groups(identity, rng)
            rng.shuffle(groups)
            queues[identity] = groups
            if is_cross_camera:
                cross_camera.add(identity)

        final_indices = []
        batch_count = self.length // self.batch_size
        for _batch in range(batch_count):
            anchors = [identity for identity in cross_camera if queues[identity]]
            if not anchors:
                raise ValueError("cross-camera groups cannot cover every formal batch")
            rng.shuffle(anchors)
            anchors.sort(key=lambda identity: len(queues[identity]), reverse=True)
            selected = [anchors[0]]
            available = [
                identity
                for identity, groups in queues.items()
                if groups and identity not in selected
            ]
            rng.shuffle(available)
            available.sort(
                key=lambda identity: (
                    identity in cross_camera,
                    -len(queues[identity]),
                )
            )
            selected.extend(available[: self.identities_per_batch - 1])
            if len(selected) != self.identities_per_batch:
                raise ValueError("identity groups cannot fill a formal batch")
            for identity in selected:
                final_indices.extend(queues[identity].pop())
        return iter(final_indices)

    def __len__(self) -> int:
        return self.length


def build_aligned_train_loader(
    records: Sequence[tuple[Sequence[str], int, int, int]],
    *,
    batch_size: int,
    num_instances: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Build the V7 B64/K8 loader with one shared triplet transform."""

    from data.datasets.make_dataloader import train_collate_fn
    from data.datasets.sampler import RandomIdentitySampler

    dataset = AlignedTripletImageDataset(
        records,
        transform=SharedGeometryTripletTransform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(
            records,
            batch_size,
            num_instances,
            seed,
        ),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def build_cross_camera_train_loader(
    records: Sequence[tuple[Sequence[str], int, int, int]],
    *,
    batch_size: int,
    num_instances: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Build the V17 aligned loader with a cross-camera relation per batch."""

    from data.datasets.make_dataloader import train_collate_fn

    dataset = AlignedTripletImageDataset(
        records,
        transform=SharedGeometryTripletTransform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=CrossCameraIdentitySampler(
            records,
            batch_size=batch_size,
            num_instances=num_instances,
            seed=seed,
        ),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = [
    "AlignedTripletImageDataset",
    "CrossCameraIdentitySampler",
    "SharedGeometryTripletTransform",
    "build_aligned_train_loader",
    "build_cross_camera_train_loader",
]
