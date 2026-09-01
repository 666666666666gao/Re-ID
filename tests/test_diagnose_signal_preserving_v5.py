"""Behavioral tests for read-only V5 collaboration diagnostics."""

import math

import torch


def test_v5_diagnostic_reports_residual_energy_and_router_entropy() -> None:
    from tools.diagnose_signal_preserving_v5 import summarize_collaboration

    baseline = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    suffix = torch.tensor([[0.5, 0.0], [1.0, 0.0]])
    fused = torch.cat((baseline, suffix), dim=1)
    branches = {
        "cnn": torch.cat((baseline, suffix), dim=1),
        "transformer": torch.cat((baseline, suffix), dim=1),
        "mamba": torch.cat((baseline, -suffix), dim=1),
    }
    uniform_router = torch.full((2, 3, 3), 1.0 / 3.0)

    result = summarize_collaboration(
        baseline,
        fused,
        branches,
        uniform_router,
        reliability_r=torch.full((2, 3, 3), 0.75),
        reliability_u=torch.full((2, 3, 3), 0.25),
    )

    assert math.isclose(
        result["fused_suffix_to_baseline_norm_mean"], 0.1, rel_tol=1e-6
    )
    assert math.isclose(result["router_normalized_entropy_mean"], 1.0)
    assert result["branch_suffix_pairwise_cosine_mean"]["cnn__transformer"] == 1.0
    assert result["branch_suffix_pairwise_cosine_mean"]["cnn__mamba"] == -1.0
