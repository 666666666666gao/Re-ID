from __future__ import annotations


def test_v16_q1_gate_requires_matched_gain_and_initial_activity_only() -> None:
    from tools.train_signal_preserving_v16 import evaluate_v16_q1_gate

    fold_gains = (
        {"fused": 0.2, "cnn": 0.1, "transformer": 0.1, "mamba": -0.1},
        {"fused": 0.4, "cnn": 0.1, "transformer": -0.1, "mamba": 0.2},
        {"fused": 2.4, "cnn": -0.1, "transformer": 0.2, "mamba": 0.1},
    )
    aggregate_branch_gains = {
        "cnn": 0.1,
        "transformer": 0.1,
        "mamba": 0.1,
    }
    aggregate_satr_map = {
        "fused": 89.0,
        "cnn": 88.0,
        "transformer": 87.0,
        "mamba": 86.0,
    }
    initial_coverages = (
        {"cnn": 0.015, "transformer": 0.031, "mamba": 0.047},
        {"cnn": 0.031, "transformer": 0.016, "mamba": 0.141},
        {"cnn": 0.042, "transformer": 0.014, "mamba": 0.056},
    )
    integrity = {
        "fold_isolation": True,
        "paired_initial_state": True,
        "paired_trainable_names": True,
        "paired_sample_order": True,
        "paired_transformed_batches": True,
        "frozen_state_unchanged": True,
        "pre_bn_evaluation": True,
        "access_boundary": True,
    }

    passing = evaluate_v16_q1_gate(
        fold_map_gains=fold_gains,
        weighted_fused_gain_map=1.0,
        fused_gain_bootstrap_lower_bound=0.01,
        aggregate_branch_gains=aggregate_branch_gains,
        aggregate_satr_map=aggregate_satr_map,
        initial_coverages=initial_coverages,
        integrity=integrity,
    )
    bad_initial_activity = evaluate_v16_q1_gate(
        fold_map_gains=fold_gains,
        weighted_fused_gain_map=1.0,
        fused_gain_bootstrap_lower_bound=0.01,
        aggregate_branch_gains=aggregate_branch_gains,
        aggregate_satr_map=aggregate_satr_map,
        initial_coverages=(
            initial_coverages[0],
            {**initial_coverages[1], "transformer": 0.0},
            initial_coverages[2],
        ),
        integrity=integrity,
    )

    assert passing["passed"] is True
    assert bad_initial_activity["passed"] is False
    assert bad_initial_activity["fixed_initial_activity_passed"] is False
    assert "training_coverage" not in passing

    wrong_fold_count = evaluate_v16_q1_gate(
        fold_map_gains=fold_gains[:2],
        weighted_fused_gain_map=1.0,
        fused_gain_bootstrap_lower_bound=0.01,
        aggregate_branch_gains=aggregate_branch_gains,
        aggregate_satr_map=aggregate_satr_map,
        initial_coverages=initial_coverages[:2],
        integrity=integrity,
    )
    assert wrong_fold_count["passed"] is False


def test_v16_m0_gate_requires_exact_capacity_and_overfit_receipts() -> None:
    from tools.train_signal_preserving_v16 import evaluate_v16_m0_gate

    passing = evaluate_v16_m0_gate(
        exact_prefix=True,
        paired_receipts_equal=True,
        initial_activity=True,
        capacity_overflow_events=0,
        capacity_all_trainable_tensors_reached=True,
        capacity_frozen_state_unchanged=True,
        capacity_peak_reserved_mib=23000.0,
        overfit_overflow_events=0,
        overfit_frozen_state_unchanged=True,
        overfit_excess_loss_ratio=0.09,
        overfit_max_loss_ratio=0.1,
    )
    too_large = evaluate_v16_m0_gate(
        exact_prefix=True,
        paired_receipts_equal=True,
        initial_activity=True,
        capacity_overflow_events=0,
        capacity_all_trainable_tensors_reached=True,
        capacity_frozen_state_unchanged=True,
        capacity_peak_reserved_mib=24576.0,
        overfit_overflow_events=0,
        overfit_frozen_state_unchanged=True,
        overfit_excess_loss_ratio=0.09,
        overfit_max_loss_ratio=0.1,
    )

    assert passing["passed"] is True
    assert too_large["passed"] is False
    assert too_large["capacity_passed"] is False


def test_v16_d1_gate_requires_65_map_and_strict_single_checkpoint_wins() -> None:
    from tools.train_signal_preserving_v16 import evaluate_v16_d1_gate

    metrics = {
        "baseline_only": {"mAP": 58.0, "Rank-1": 60.0},
        "fused": {"mAP": 65.1, "Rank-1": 68.0},
        "cnn": {"mAP": 64.0, "Rank-1": 66.0},
        "transformer": {"mAP": 63.0, "Rank-1": 65.0},
        "mamba": {"mAP": 62.0, "Rank-1": 64.0},
    }
    passing = evaluate_v16_d1_gate(
        metrics_percent=metrics,
        minimum_fused_map=65.0,
        v8_phase_b_map=58.405,
        strict_reload=True,
        frozen_state_unchanged=True,
    )
    tied_branch = evaluate_v16_d1_gate(
        metrics_percent={
            **metrics,
            "cnn": {"mAP": 65.1, "Rank-1": 66.0},
        },
        minimum_fused_map=65.0,
        v8_phase_b_map=58.405,
        strict_reload=True,
        frozen_state_unchanged=True,
    )

    assert passing["passed"] is True
    assert tied_branch["passed"] is False
    assert tied_branch["strict_mAP_wins_passed"] is False
