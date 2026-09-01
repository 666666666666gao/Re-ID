from __future__ import annotations

import numpy as np
import torch


def test_residual_bank_normalizes_each_expert_before_concatenation() -> None:
    from tools.probe_v11_dinov2_oof_residual_complement import compose_residual_bank

    bank = compose_residual_bank(
        {
            "cnn": torch.tensor([[3.0, 4.0]]),
            "transformer": torch.tensor([[0.0, 2.0]]),
            "mamba": torch.tensor([[-1.0, 0.0]]),
        }
    )

    assert torch.allclose(
        bank,
        torch.tensor([[0.6, 0.8, 0.0, 1.0, -1.0, 0.0]]),
    )


def test_fold_score_aggregation_only_concatenates_query_results() -> None:
    from tools.diagnose_v6_oracle_complementarity import QueryRetrievalScores
    from tools.probe_v11_dinov2_oof_residual_complement import aggregate_fold_scores

    combined = aggregate_fold_scores(
        [
            {
                "residual_bank": QueryRetrievalScores(
                    average_precision=np.asarray([0.25, 0.75]),
                    rank1_correct=np.asarray([False, True]),
                ),
                "dinov2": QueryRetrievalScores(
                    average_precision=np.asarray([0.50, 0.25]),
                    rank1_correct=np.asarray([True, False]),
                ),
            },
            {
                "residual_bank": QueryRetrievalScores(
                    average_precision=np.asarray([1.00]),
                    rank1_correct=np.asarray([True]),
                ),
                "dinov2": QueryRetrievalScores(
                    average_precision=np.asarray([0.10]),
                    rank1_correct=np.asarray([False]),
                ),
            },
        ]
    )

    assert combined["residual_bank"].average_precision.tolist() == [0.25, 0.75, 1.0]
    assert combined["residual_bank"].rank1_correct.tolist() == [False, True, True]
    assert combined["dinov2"].average_precision.tolist() == [0.5, 0.25, 0.1]


def test_v11_gate_requires_fixed_gain_oracle_gain_and_two_source_wins() -> None:
    from tools.probe_v11_dinov2_oof_residual_complement import (
        evaluate_qualification_gate,
    )

    passing = evaluate_qualification_gate(
        fixed_map={"residual_bank": 50.0, "dinov2": 40.0, "concat": 52.0},
        oracle_map=53.0,
        unique_ap_wins={"residual_bank": 7, "dinov2": 3},
        fold_count=3,
        query_count=571,
    )
    weak_concat = evaluate_qualification_gate(
        fixed_map={"residual_bank": 50.0, "dinov2": 40.0, "concat": 50.5},
        oracle_map=53.0,
        unique_ap_wins={"residual_bank": 7, "dinov2": 3},
        fold_count=3,
        query_count=571,
    )
    saturated = evaluate_qualification_gate(
        fixed_map={"residual_bank": 99.0, "dinov2": 40.0, "concat": 99.5},
        oracle_map=100.0,
        unique_ap_wins={"residual_bank": 7, "dinov2": 3},
        fold_count=3,
        query_count=571,
    )
    collapsed = evaluate_qualification_gate(
        fixed_map={"residual_bank": 50.0, "dinov2": 40.0, "concat": 52.0},
        oracle_map=53.0,
        unique_ap_wins={"residual_bank": 7, "dinov2": 0},
        fold_count=3,
        query_count=571,
    )

    assert passing["passed"] is True
    assert passing["concat_gain_mAP"] == 2.0
    assert passing["oracle_gain_mAP"] == 3.0
    assert weak_concat["concat_gain_passed"] is False
    assert saturated["non_saturation_passed"] is False
    assert collapsed["two_source_unique_wins_passed"] is False

