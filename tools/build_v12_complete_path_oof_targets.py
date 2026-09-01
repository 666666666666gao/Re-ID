#!/usr/bin/env python3
"""Build complete-path identity-OOF utility targets for the V12 Router."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_complete_path_fold_records(
    records: Sequence[tuple[Any, int, int, int]],
    *,
    heldout_ids: set[int],
) -> dict[str, Any]:
    """Split one identity fold and relabel only its fit records."""

    heldout = {int(identity) for identity in heldout_ids}
    fit_ids = sorted({int(record[1]) for record in records} - heldout)
    label_map = {identity: index for index, identity in enumerate(fit_ids)}
    train_records = [
        (paths, label_map[int(identity)], camera, view)
        for paths, identity, camera, view in records
        if int(identity) not in heldout
    ]
    heldout_records = [
        record for record in records if int(record[1]) in heldout
    ]
    heldout_present = sorted({int(record[1]) for record in heldout_records})
    overlap = sorted(set(fit_ids) & set(heldout_present))
    return {
        "train_records": train_records,
        "heldout_records": heldout_records,
        "label_map": label_map,
        "fit_identity_ids": tuple(fit_ids),
        "heldout_identity_ids": tuple(heldout_present),
        "identity_overlap": tuple(overlap),
    }


__all__ = ["build_complete_path_fold_records"]
