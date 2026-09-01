"""Geometry-aligned RGB/NIR/TIR training input."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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


__all__ = [
    "AlignedTripletImageDataset",
    "SharedGeometryTripletTransform",
    "build_aligned_train_loader",
]
