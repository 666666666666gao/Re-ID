from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def test_rgbnt201_oof_loader_keeps_target_identities_out_of_generator_training() -> None:
    from modeling.trifusion.data import build_rgbnt201_oof_loaders

    loaders = build_rgbnt201_oof_loaders(
        dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
        protocol_path=PROJECT / "protocols/rgbnt201_dev_v1.json",
        fold_salt="TriFusion-CIRC-fold-v1",
        fold_count=3,
        target_fold=0,
        train_batch_size=8,
        num_instances=4,
        eval_batch_size=32,
        num_workers=0,
    )

    assert loaders.num_classes == 97
    assert loaders.num_query == 1008
    assert len(loaders.train_loader.dataset) == 2118
    assert len(loaders.eval_loader.dataset) == 2016
    assert loaders.provenance["target_fold"] == 0
    assert loaders.provenance["fold_count"] == 3
    assert loaders.provenance["generator_training_identities"] == 97
    assert loaders.provenance["target_identities"] == 44
    assert loaders.provenance["generator_target_identity_overlap"] == 0
    assert loaders.provenance["official_test_records"] == 0
    generator_paths = {
        record[0][0] for record in loaders.train_records
    }
    target_paths = {
        record[0][0] for record in loaders.query_records
    }
    assert generator_paths.isdisjoint(target_paths)
    assert all("/train_171/" in path for path in generator_paths | target_paths)


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
