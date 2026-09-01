from __future__ import annotations


def test_fit_diversity_gate_requires_every_expert_and_oracle_headroom() -> None:
    from tools.probe_v8_phase_a_fit_diversity import evaluate_fit_diversity_gate

    passing = evaluate_fit_diversity_gate(
        {
            "oracle_minus_best_fixed_percent": {"mAP": 3.0},
            "unique_ap_wins": {"cnn": 20, "transformer": 5, "mamba": 8},
            "leave_one_expert_out": {
                "cnn": {"marginal_mAP": 1.0},
                "transformer": {"marginal_mAP": 0.5},
                "mamba": {"marginal_mAP": 0.7},
            },
        },
        min_oracle_gain_map=1.0,
    )
    collapsed = evaluate_fit_diversity_gate(
        {
            "oracle_minus_best_fixed_percent": {"mAP": 3.0},
            "unique_ap_wins": {"cnn": 20, "transformer": 0, "mamba": 8},
            "leave_one_expert_out": {
                "cnn": {"marginal_mAP": 1.0},
                "transformer": {"marginal_mAP": 0.0},
                "mamba": {"marginal_mAP": 0.7},
            },
        },
        min_oracle_gain_map=1.0,
    )

    assert passing["passed"] is True
    assert collapsed["passed"] is False
    assert collapsed["experts"]["transformer"]["passed"] is False
