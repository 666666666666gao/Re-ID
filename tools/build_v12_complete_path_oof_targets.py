#!/usr/bin/env python3
"""Build complete-path identity-OOF utility targets for the V12 Router."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
EXPECTED_FOLDS = 3
EXPECTED_QUERIES = 571
SIGNAL_EPOCHS = 50
EXPERT_EPOCHS = 20
SATURATION_MAP = 99.0
MIN_RESIDUAL_ORACLE_GAIN_MAP = 1.0


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
    """Apply the preregistered V12 complete-path qualification decision."""

    protocol = len(fold_receipts) == EXPECTED_FOLDS and int(query_count) == EXPECTED_QUERIES
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
    training_schedule = all(
        int(receipt["signal_epochs"]) == SIGNAL_EPOCHS
        and int(receipt["expert_epochs"]) == EXPERT_EPOCHS
        and receipt["signal_checkpoint_selection"] == "final_epoch_only"
        for receipt in fold_receipts
    )
    runtime_integrity = all(
        int(receipt["overflow_events"]) == 0
        and int(receipt["dev_access_count"]) == 0
        and int(receipt["official_test_access_count"]) == 0
        for receipt in fold_receipts
    )
    best_fixed_map = max(float(value) for value in fixed_map.values())
    non_saturation = best_fixed_map < SATURATION_MAP
    expert_diversity = all(
        int(expert_winner_counts[expert]) > 0 for expert in EXPERTS
    )
    modality_diversity = all(
        int(modality_winner_counts[modality]) > 0 for modality in MODALITIES
    )
    residual_oracle_gain = (
        float(residual_oracle_gain_map) >= MIN_RESIDUAL_ORACLE_GAIN_MAP
    )
    slot_oracle_margin = float(slot_oracle_margin_gain) > 0.0
    passed = all(
        (
            protocol,
            identity_isolation,
            training_schedule,
            runtime_integrity,
            non_saturation,
            expert_diversity,
            modality_diversity,
            residual_oracle_gain,
            slot_oracle_margin,
        )
    )
    return {
        "passed": passed,
        "protocol_passed": protocol,
        "complete_path_identity_isolation_passed": identity_isolation,
        "training_schedule_passed": training_schedule,
        "runtime_integrity_passed": runtime_integrity,
        "non_saturation_passed": non_saturation,
        "expert_diversity_passed": expert_diversity,
        "modality_diversity_passed": modality_diversity,
        "residual_oracle_gain_passed": residual_oracle_gain,
        "slot_oracle_margin_gain_passed": slot_oracle_margin,
        "fold_count": len(fold_receipts),
        "query_count": int(query_count),
        "best_fixed_mAP": best_fixed_map,
        "saturation_mAP": SATURATION_MAP,
        "residual_oracle_gain_mAP": float(residual_oracle_gain_map),
        "minimum_residual_oracle_gain_mAP": MIN_RESIDUAL_ORACLE_GAIN_MAP,
        "slot_oracle_margin_gain": float(slot_oracle_margin_gain),
        "expert_unique_winner_counts": {
            expert: int(expert_winner_counts[expert]) for expert in EXPERTS
        },
        "modality_unique_winner_counts": {
            modality: int(modality_winner_counts[modality])
            for modality in MODALITIES
        },
    }


__all__ = [
    "build_complete_path_fold_records",
    "evaluate_complete_path_oof_gate",
]
