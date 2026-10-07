"""V24 ordinary/strong views with one geometry across both views and modalities."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomCrop, RandomErasing
from torchvision.transforms import functional as TF

from .aligned_data import CrossCameraIdentitySampler, _read_triplet


class SharedGeometryDualViewTransform:
    def __init__(self, *, size=(256, 128), padding=10, flip_probability=0.5,
                 weak_erase=0.5, strong_erase=0.6, brightness=0.2):
        self.size = tuple(size)
        self.padding = int(padding)
        self.flip_probability = float(flip_probability)
        self.brightness = float(brightness)
        self.weak_erase = RandomErasing(p=weak_erase, value="random")
        self.strong_erase = RandomErasing(p=strong_erase, value="random")

    def __call__(self, images):
        assert len(images) == 3
        images = [TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC) for image in images]
        if random.random() < self.flip_probability:
            images = [TF.hflip(image) for image in images]
        images = [TF.pad(image, self.padding) for image in images]
        crop = RandomCrop.get_params(images[0], output_size=self.size)
        images = [TF.crop(image, *crop) for image in images]
        weak, strong = [], []
        for image in images:
            value = TF.to_tensor(image)
            weak.append(self.weak_erase(TF.normalize(value, (0.5,) * 3, (0.5,) * 3)))
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            perturbed = TF.adjust_brightness(value, factor)
            strong.append(self.strong_erase(TF.normalize(perturbed, (0.5,) * 3, (0.5,) * 3)))
        return weak, strong


class DualViewTripletDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = SharedGeometryDualViewTransform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        paths, identity, camera, view = self.records[index]
        weak, strong = self.transform(_read_triplet(paths))
        return weak, strong, identity, camera, view, Path(paths[0]).name


def collate_dual_view(items):
    from data.datasets.make_dataloader import train_collate_fn
    weak = train_collate_fn([(w, identity, camera, view, path) for w, _, identity, camera, view, path in items])
    strong = train_collate_fn([(s, identity, camera, view, path) for _, s, identity, camera, view, path in items])
    return weak, strong


def build_dual_view_loader(records, config):
    return DataLoader(
        DualViewTripletDataset(records),
        batch_size=config["DATA"]["TRAIN_BATCH_SIZE"],
        sampler=CrossCameraIdentitySampler(records, batch_size=64, num_instances=8, seed=42),
        num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=collate_dual_view,
        pin_memory=torch.cuda.is_available(),
    )
