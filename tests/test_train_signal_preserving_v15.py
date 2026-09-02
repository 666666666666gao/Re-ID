from __future__ import annotations


def test_v15_q1_gate_requires_fold_stability_and_aggregate_collaboration() -> None:
    from tools.train_signal_preserving_v15 import evaluate_v15_q1_gate

    fold_gains = (
        {"fused": 0.5, "cnn": 0.2, "transformer": 0.1, "mamba": -0.1},
        {"fused": 1.0, "cnn": 0.1, "transformer": -0.1, "mamba": 0.2},
        {"fused": 1.5, "cnn": -0.1, "transformer": 0.2, "mamba": 0.1},
    )
    aggregate_branch_gains = {
        "cnn": 0.1,
        "transformer": 0.1,
        "mamba": 0.1,
    }
    aggregate_on_map = {
        "fused": 88.0,
        "cnn": 87.0,
        "transformer": 86.0,
        "mamba": 85.0,
    }
    integrity = {
        "fold_isolation": True,
        "same_tensor_pairing": True,
        "frozen_state_unchanged": True,
        "pre_bn_evaluation": True,
        "regret_weight_exact": True,
        "access_boundary": True,
    }

    passing = evaluate_v15_q1_gate(
        fold_map_gains=fold_gains,
        weighted_fused_gain_map=1.0,
        fused_gain_bootstrap_lower_bound=0.01,
        aggregate_branch_gains=aggregate_branch_gains,
        aggregate_on_map=aggregate_on_map,
        integrity=integrity,
    )
    one_receiver_only = evaluate_v15_q1_gate(
        fold_map_gains=(
            fold_gains[0],
            fold_gains[1],
            {"fused": 1.5, "cnn": -0.1, "transformer": -0.1, "mamba": 0.1},
        ),
        weighted_fused_gain_map=1.0,
        fused_gain_bootstrap_lower_bound=0.01,
        aggregate_branch_gains=aggregate_branch_gains,
        aggregate_on_map=aggregate_on_map,
        integrity=integrity,
    )

    assert passing["passed"] is True
    assert one_receiver_only["passed"] is False
    assert one_receiver_only["per_fold_two_receivers_passed"] is False


def test_v15_m0_gate_requires_two_live_exchanges_not_every_tensor_nonzero() -> None:
    from tools.train_signal_preserving_v15 import evaluate_v15_m0_gate

    gate = evaluate_v15_m0_gate(
        step0_passed=True,
        capacity_overflow_events=0,
        capacity_frozen_state_unchanged=True,
        live_exchange_stages=(0, 1),
        overfit_overflow_events=0,
        overfit_frozen_state_unchanged=True,
        overfit_all_trainable_tensors_reached=True,
        overfit_excess_loss_ratio=0.051,
        overfit_max_loss_ratio=0.1,
    )

    assert gate["passed"] is True
    assert gate["capacity_two_exchange_live_passed"] is True
