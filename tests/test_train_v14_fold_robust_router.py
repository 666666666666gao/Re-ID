from __future__ import annotations


def test_v14_q1_gate_uses_retrieval_metrics_not_v13_action_labels() -> None:
    from tools.train_v14_fold_robust_router import evaluate_v14_q1_gate

    passing = evaluate_v14_q1_gate(
        fold_gains=tuple(
            {"retrieval_risk": 0.01, "replay_average_precision": 0.0, "replay_margin": 0.0}
            for _ in range(3)
        ),
        bootstrap_lower_bounds={
            "retrieval_risk": 0.001,
            "replay_average_precision": 0.001,
            "replay_margin": 0.001,
        },
        corrupted_mass_decreases={"RGB": True, "NI": True, "TI": True},
        missing_modality_max_mass=0.0,
        frozen_phase_a_unchanged=True,
        dev_access_count=0,
        official_test_access_count=0,
    )
    failed_zero_risk = evaluate_v14_q1_gate(
        fold_gains=(
            {"retrieval_risk": 0.01, "replay_average_precision": 0.0, "replay_margin": 0.0},
            {"retrieval_risk": 0.0, "replay_average_precision": 0.0, "replay_margin": 0.0},
            {"retrieval_risk": 0.01, "replay_average_precision": 0.0, "replay_margin": 0.0},
        ),
        bootstrap_lower_bounds={
            "retrieval_risk": 0.001,
            "replay_average_precision": 0.001,
            "replay_margin": 0.001,
        },
        corrupted_mass_decreases={"RGB": True, "NI": True, "TI": True},
        missing_modality_max_mass=0.0,
        frozen_phase_a_unchanged=True,
        dev_access_count=0,
        official_test_access_count=0,
    )

    assert passing["passed"] is True
    assert failed_zero_risk["passed"] is False
    assert failed_zero_risk["per_fold_retrieval_passed"] is False

