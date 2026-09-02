from __future__ import annotations


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
