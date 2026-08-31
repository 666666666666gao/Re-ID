from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def test_rgbnt201_dev_loaders_use_only_train171_and_preserve_pk_batches() -> None:
    from modeling.trifusion.data import build_rgbnt201_dev_loaders

    random.seed(53)
    np.random.seed(53)
    loaders = build_rgbnt201_dev_loaders(
        dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
        protocol_path=PROJECT / "protocols/rgbnt201_dev_v1.json",
        train_batch_size=16,
        num_instances=4,
        eval_batch_size=64,
        num_workers=0,
    )

    assert loaders.num_classes == 141
    assert loaders.num_query == 825
    assert len(loaders.train_loader.dataset) == 3126
    assert len(loaders.eval_loader.dataset) == 1650
    assert loaders.provenance["official_test_records"] == 0
    assert loaders.provenance["train_dev_identity_overlap"] == 0
    assert loaders.provenance["protocol_uses_test_labels"] is False
    all_paths = [
        path
        for record in loaders.train_records + loaders.query_records + loaders.gallery_records
        for path in record[0]
    ]
    assert all("/test/" not in path for path in all_paths)

    images, labels, _camera_ids, _view_ids, sample_keys = next(
        iter(loaders.train_loader)
    )
    assert tuple(images) == ("RGB", "NI", "TI")
    assert all(tuple(tensor.shape) == (16, 3, 256, 128) for tensor in images.values())
    assert sorted(Counter(labels.tolist()).values()) == [4, 4, 4, 4]
    assert len(sample_keys) == 16
