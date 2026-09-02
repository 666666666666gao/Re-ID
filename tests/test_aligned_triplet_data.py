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


def test_cross_camera_sampler_keeps_b64k8_semantics_and_pairs_every_batch() -> None:
    from modeling.trifusion.aligned_data import CrossCameraIdentitySampler

    records = []
    for identity in range(8):
        for sample in range(4):
            camera = sample % 2 if identity < 2 else 0
            records.append(([f"{identity}_{sample}"] * 3, identity, camera, -1))

    first = list(
        CrossCameraIdentitySampler(
            records,
            batch_size=8,
            num_instances=2,
            seed=42,
        )
    )
    second = list(
        CrossCameraIdentitySampler(
            records,
            batch_size=8,
            num_instances=2,
            seed=42,
        )
    )

    assert first == second
    assert len(first) == 32
    for offset in range(0, len(first), 8):
        batch = [records[index] for index in first[offset : offset + 8]]
        identities = torch.tensor([record[1] for record in batch])
        cameras = torch.tensor([record[2] for record in batch])
        assert identities.unique().numel() == 4
        assert all(int((identities == identity).sum()) == 2 for identity in identities.unique())
        same_identity = identities[:, None] == identities[None, :]
        different_camera = cameras[:, None] != cameras[None, :]
        assert bool((same_identity & different_camera).any())
