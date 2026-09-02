from __future__ import annotations

import torch


def test_v13_q1_gate_requires_fold_replay_bootstrap_and_quality_evidence() -> None:
    from tools.train_v13_deployment_aligned_router import evaluate_v13_q1_gate

    fold_noninferiority = tuple(
        {
            "expected_utility": True,
            "top1": True,
            "replay_average_precision": True,
            "replay_margin": True,
        }
        for _ in range(3)
    )
    bootstrap_lower_bounds = {
        "expected_utility": 0.01,
        "top1": 0.01,
        "replay_average_precision": 0.01,
        "replay_margin": 0.01,
    }
    quality_response = {"RGB": True, "NI": True, "TI": True}

    passing = evaluate_v13_q1_gate(
        fold_noninferiority=fold_noninferiority,
        bootstrap_lower_bounds=bootstrap_lower_bounds,
        corrupted_mass_decreases=quality_response,
        missing_modality_max_mass=0.0,
        frozen_phase_a_unchanged=True,
        dev_access_count=0,
        official_test_access_count=0,
    )
    failed_replay = evaluate_v13_q1_gate(
        fold_noninferiority=(
            fold_noninferiority[0],
            {
                **fold_noninferiority[1],
                "replay_average_precision": False,
            },
            fold_noninferiority[2],
        ),
        bootstrap_lower_bounds=bootstrap_lower_bounds,
        corrupted_mass_decreases=quality_response,
        missing_modality_max_mass=0.0,
        frozen_phase_a_unchanged=True,
        dev_access_count=0,
        official_test_access_count=0,
    )

    assert passing["passed"] is True
    assert passing["aggregate_bootstrap_passed"] is True
    assert failed_replay["passed"] is False
    assert failed_replay["per_fold_noninferiority_passed"] is False


def test_v13_target_diagnostic_reports_scale_stability_and_observability() -> None:
    from tools.diagnose_v13_target_learnability import (
        analyze_v13_target_learnability,
    )

    utility = torch.arange(36, dtype=torch.float32).reshape(4, 3, 3) / 100.0
    cache = {
        "teacher_identity_utility": utility,
        "identities": torch.tensor([0, 0, 1, 1]),
        "fold_indices": torch.tensor([0, 1, 2, 0]),
        "student_direct_modal": torch.randn(4, 3, 2),
        "student_modal_residual": torch.randn(4, 3, 3, 2),
    }

    result = analyze_v13_target_learnability(cache, temperature=0.05)

    assert result["query_count"] == 4
    assert result["identity_count"] == 2
    assert len(result["cross_fold_slot_semantics"]["fixed_slots"]) == 3
    assert 0.0 <= result["distillation_target"]["normalized_entropy_mean"] <= 1.0
    assert result["training_executed"] is False
    assert result["dev_access_count"] == 0
