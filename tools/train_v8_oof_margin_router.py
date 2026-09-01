#!/usr/bin/env python3
"""Train the frozen-expert V8 Router from identity-OOF margin targets."""

from __future__ import annotations

from typing import Any


def evaluate_router_phase_gate(
    *,
    learned_oof_margin: float,
    fixed_oof_margin: float,
    learned_top1_accuracy: float,
    majority_top1_accuracy: float,
    corrupted_mass_decreases: dict[str, bool],
    missing_modality_max_mass: float,
) -> dict[str, Any]:
    margin_passed = float(learned_oof_margin) > float(fixed_oof_margin)
    alignment_passed = float(learned_top1_accuracy) > float(majority_top1_accuracy)
    quality_passed = all(
        bool(corrupted_mass_decreases[name]) for name in ("RGB", "NI", "TI")
    )
    missing_passed = float(missing_modality_max_mass) == 0.0
    return {
        "passed": margin_passed and alignment_passed and quality_passed and missing_passed,
        "oof_margin_gain_passed": margin_passed,
        "top1_alignment_passed": alignment_passed,
        "quality_response_passed": quality_passed,
        "missing_modality_zero_mass_passed": missing_passed,
        "learned_oof_margin": float(learned_oof_margin),
        "fixed_oof_margin": float(fixed_oof_margin),
        "learned_top1_accuracy": float(learned_top1_accuracy),
        "majority_top1_accuracy": float(majority_top1_accuracy),
        "corrupted_mass_decreases": {
            name: bool(corrupted_mass_decreases[name])
            for name in ("RGB", "NI", "TI")
        },
        "missing_modality_max_mass": float(missing_modality_max_mass),
    }


__all__ = ["evaluate_router_phase_gate"]
