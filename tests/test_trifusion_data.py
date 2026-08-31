from __future__ import annotations

from collections import Counter
import json
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
DEV_PROTOCOL_SHA256 = (
    "d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946"
)
DEVELOPMENT_IDENTITY_SHA256 = (
    "84e071d8d26cf6b038bcbd9951b22ce61f07e5e4ff7ea40b65d82a3ebd68dde2"
)


def _write_circ_protocol(
    tmp_path: Path,
    *,
    development_identity_sha256: str = DEVELOPMENT_IDENTITY_SHA256,
    dev_protocol_sha256: str = DEV_PROTOCOL_SHA256,
) -> Path:
    circ_protocol_path = tmp_path / "circ-target-v1.json"
    circ_protocol_path.write_text(
        json.dumps(
            {
                "schema_version": "circ-protocol-v1",
                "dev_protocol_sha256": dev_protocol_sha256,
                "official_test_access_count": 0,
                "folds": {
                    "count": 3,
                    "salt": "TriFusion-CIRC-fold-v1",
                    "identity_canonicalization": "unsigned-decimal",
                    "development_identity_sha256": development_identity_sha256,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return circ_protocol_path


def test_rgbnt201_oof_loader_keeps_target_identities_out_of_generator_training(
    tmp_path: Path,
) -> None:
    from modeling.trifusion.data import build_rgbnt201_oof_loaders

    circ_protocol_path = _write_circ_protocol(tmp_path)
    loaders = build_rgbnt201_oof_loaders(
        dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
        protocol_path=PROJECT / "protocols/rgbnt201_dev_v1.json",
        circ_protocol_path=circ_protocol_path,
        target_fold=0,
        train_batch_size=8,
        num_instances=4,
        eval_batch_size=32,
        num_workers=0,
    )

    assert loaders.num_classes == 93
    assert loaders.num_query == 258
    assert len(loaders.train_loader.dataset) == 2045
    assert len(loaders.eval_loader.dataset) == 1339
    assert loaders.provenance["target_fold"] == 0
    assert loaders.provenance["fold_count"] == 3
    assert loaders.provenance["generator_training_identities"] == 93
    assert loaders.provenance["target_identities"] == 48
    assert loaders.provenance["target_cross_camera_supported_identities"] == 10
    assert loaders.provenance["target_cross_camera_unsupported_identities"] == 38
    assert loaders.provenance["target_cross_camera_supported_records"] == 258
    assert loaders.provenance["target_cross_camera_unsupported_records"] == 823
    assert loaders.provenance["generator_target_identity_overlap"] == 0
    assert loaders.provenance["target_forbidden_dev_identity_overlap"] == 0
    assert loaders.provenance["official_test_records"] == 0
    assert loaders.provenance["query_equals_gallery_record_list"] is False
    assert loaders.provenance["circ_protocol_path"] == str(
        circ_protocol_path.resolve()
    )
    generator_paths = {
        record[0][0] for record in loaders.train_records
    }
    target_paths = {
        record[0][0] for record in loaders.query_records
    }
    assert generator_paths.isdisjoint(target_paths)
    assert all("/train_171/" in path for path in generator_paths | target_paths)


def test_rgbnt201_oof_folds_are_disjoint_and_cover_every_fit_identity(
    tmp_path: Path,
) -> None:
    from modeling.trifusion.data import build_rgbnt201_oof_loaders

    circ_protocol_path = _write_circ_protocol(tmp_path)
    target_sets = []
    for target_fold in range(3):
        loaders = build_rgbnt201_oof_loaders(
            dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
            protocol_path=PROJECT / "protocols/rgbnt201_dev_v1.json",
            circ_protocol_path=circ_protocol_path,
            target_fold=target_fold,
            train_batch_size=8,
            num_instances=4,
            eval_batch_size=32,
            num_workers=0,
        )
        target_sets.append(set(loaders.provenance["target_identity_values"]))
        assert loaders.provenance["generator_target_identity_overlap"] == 0
        assert loaders.provenance["target_forbidden_dev_identity_overlap"] == 0

    assert all(
        target_sets[left].isdisjoint(target_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    protocol = json.loads(
        (PROJECT / "protocols/rgbnt201_dev_v1.json").read_text(encoding="utf-8")
    )
    assert set.union(*target_sets) == {int(identity) for identity in protocol["train_ids"]}


def test_rgbnt201_oof_loader_rejects_semantic_train_dev_identity_overlap(
    tmp_path: Path,
) -> None:
    from modeling.trifusion.data import build_rgbnt201_oof_loaders

    source_protocol = PROJECT / "protocols/rgbnt201_dev_v1.json"
    protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
    protocol["dev_ids"][0] = str(int(protocol["train_ids"][0]))
    protocol_path = tmp_path / "overlap-dev-protocol.json"
    encoded = json.dumps(protocol, sort_keys=True).encode("utf-8")
    protocol_path.write_bytes(encoded)
    circ_protocol_path = _write_circ_protocol(
        tmp_path,
        dev_protocol_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    with pytest.raises(ValueError, match="141-fit/30-dev identity registry is invalid"):
        build_rgbnt201_oof_loaders(
            dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
            protocol_path=protocol_path,
            circ_protocol_path=circ_protocol_path,
            target_fold=0,
            train_batch_size=8,
            num_instances=4,
            eval_batch_size=32,
            num_workers=0,
        )


def test_rgbnt201_oof_loader_rejects_unbound_identity_registry(
    tmp_path: Path,
) -> None:
    from modeling.trifusion.data import build_rgbnt201_oof_loaders

    circ_protocol_path = _write_circ_protocol(
        tmp_path,
        development_identity_sha256="0" * 64,
    )
    with pytest.raises(
        ValueError,
        match="development identity registry mismatch",
    ):
        build_rgbnt201_oof_loaders(
            dataset_root=Path("/root/mmreid-trifusion/data/RGBNT201"),
            protocol_path=PROJECT / "protocols/rgbnt201_dev_v1.json",
            circ_protocol_path=circ_protocol_path,
            target_fold=0,
            train_batch_size=8,
            num_instances=4,
            eval_batch_size=32,
            num_workers=0,
        )


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
