from __future__ import annotations


def test_v8_router_phase_gate_requires_oof_gain_alignment_and_quality() -> None:
    from tools.train_v8_oof_margin_router import evaluate_router_phase_gate

    passing = evaluate_router_phase_gate(
        learned_oof_margin=0.20,
        fixed_oof_margin=0.15,
        learned_top1_accuracy=0.70,
        majority_top1_accuracy=0.60,
        corrupted_mass_decreases={"RGB": True, "NI": True, "TI": True},
        missing_modality_max_mass=0.0,
    )
    collapsed_quality = evaluate_router_phase_gate(
        learned_oof_margin=0.20,
        fixed_oof_margin=0.15,
        learned_top1_accuracy=0.70,
        majority_top1_accuracy=0.60,
        corrupted_mass_decreases={"RGB": True, "NI": False, "TI": True},
        missing_modality_max_mass=0.0,
    )

    assert passing["passed"] is True
    assert collapsed_quality["passed"] is False
    assert collapsed_quality["quality_response_passed"] is False
