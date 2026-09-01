from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


def _coordinate_image() -> Image.Image:
    rows = np.arange(32, dtype=np.uint8)[:, None]
    columns = np.arange(16, dtype=np.uint8)[None, :]
    image = np.stack(
        (
            np.broadcast_to(rows, (32, 16)),
            np.broadcast_to(columns, (32, 16)),
            np.broadcast_to(rows + columns, (32, 16)),
        ),
        axis=-1,
    )
    return Image.fromarray(image, mode="RGB")


def test_triplet_transform_shares_flip_and_crop_geometry() -> None:
    from modeling.trifusion.aligned_data import SharedGeometryTripletTransform

    random.seed(17)
    transform = SharedGeometryTripletTransform(
        size=(24, 12),
        padding=4,
        flip_probability=0.5,
        erase_probability=0.0,
    )
    image = _coordinate_image()

    transformed = transform((image.copy(), image.copy(), image.copy()))

    assert len(transformed) == 3
    assert torch.equal(transformed[0], transformed[1])
    assert torch.equal(transformed[1], transformed[2])


def test_triplet_dataset_calls_one_triplet_transform(monkeypatch) -> None:
    from modeling.trifusion.aligned_data import AlignedTripletImageDataset

    calls = []

    def transform(images):
        calls.append(tuple(images))
        return [torch.tensor([float(index)]) for index in range(3)]

    monkeypatch.setattr(
        "modeling.trifusion.aligned_data._read_triplet",
        lambda _paths: (object(), object(), object()),
    )
    dataset = AlignedTripletImageDataset(
        [(["rgb", "nir", "tir"], 3, 2, -1)],
        transform=transform,
    )

    images, identity, camera, view, filename = dataset[0]

    assert len(calls) == 1
    assert [float(image.item()) for image in images] == [0.0, 1.0, 2.0]
    assert (identity, camera, view, filename) == (3, 2, -1, "rgb")
