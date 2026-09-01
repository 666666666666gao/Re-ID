from __future__ import annotations

import math

import numpy as np
import torch


def test_v7_diagnostic_summarizes_joint_routing_and_bounded_energy() -> None:
    from tools.diagnose_signal_preserving_v7 import summarize_v7_routing

    baseline = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    suffix = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    fused = torch.cat((baseline, suffix), dim=1)
    branches = {
        "cnn": torch.cat((baseline, suffix), dim=1),
        "transformer": torch.cat((baseline, -suffix), dim=1),
        "mamba": torch.cat((baseline, suffix), dim=1),
    }
    weights = torch.full((2, 3, 3), 1.0 / 9.0)
    modal = torch.full((2, 3), 1.0 / 3.0)
    expert = torch.full((2, 3, 3), 1.0 / 3.0)

    result = summarize_v7_routing(
        baseline,
        fused,
        branches,
        weights,
        modal,
        expert,
        torch.full((2, 1), 0.2),
    )

    assert math.isclose(
        result["fused_suffix_to_baseline_norm_mean"], 0.2, rel_tol=1e-6
    )
    assert math.isclose(
        result["joint_router_normalized_entropy_mean"], 1.0, rel_tol=1e-6
    )
    assert math.isclose(
        result["modal_normalized_entropy_mean"], 1.0, rel_tol=1e-6
    )
    assert math.isclose(
        result["conditional_expert_entropy_mean"], 1.0, rel_tol=1e-6
    )
    assert math.isclose(result["alpha"]["mean"], 0.2, rel_tol=1e-6)


def test_v7_diagnostic_reports_fused_vs_mamba_query_repairs_and_breaks() -> None:
    from tools.diagnose_signal_preserving_v7 import summarize_pairwise_query_outcomes
    from tools.diagnose_v6_oracle_complementarity import QueryRetrievalScores

    fused = QueryRetrievalScores(
        average_precision=np.array([0.8, 0.4, 0.6]),
        rank1_correct=np.array([True, False, True]),
    )
    mamba = QueryRetrievalScores(
        average_precision=np.array([0.7, 0.5, 0.6]),
        rank1_correct=np.array([False, True, True]),
    )

    result = summarize_pairwise_query_outcomes(fused, mamba)

    assert result["fused_ap_wins"] == 1
    assert result["mamba_ap_wins"] == 1
    assert result["ap_ties"] == 1
    assert result["fused_rank1_repairs"] == 1
    assert result["fused_rank1_breaks"] == 1
