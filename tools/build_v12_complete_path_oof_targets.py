#!/usr/bin/env python3
"""Build complete-path identity-OOF utility targets for the V12 Router."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


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


def evaluate_complete_path_oof_gate(
    *,
    fold_receipts: Sequence[dict[str, Any]],
    query_count: int,
    fixed_map: dict[str, float],
    expert_winner_counts: dict[str, int],
    modality_winner_counts: dict[str, int],
    residual_oracle_gain_map: float,
    slot_oracle_margin_gain: float,
) -> dict[str, Any]:
    """Reject any utility target whose Signal or expert path saw held-out IDs."""

    identity_isolation = all(
        not (
            set(receipt["signal_fit_identity_ids"])
            & set(receipt["heldout_identity_ids"])
        )
        and not (
            set(receipt["expert_fit_identity_ids"])
            & set(receipt["heldout_identity_ids"])
        )
        for receipt in fold_receipts
    )
    return {
        "passed": identity_isolation,
        "complete_path_identity_isolation_passed": identity_isolation,
    }


__all__ = [
    "build_complete_path_fold_records",
    "evaluate_complete_path_oof_gate",
]
