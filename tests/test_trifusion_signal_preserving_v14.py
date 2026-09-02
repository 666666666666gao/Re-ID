from __future__ import annotations

import math

import pytest
import torch


def test_v14_cross_camera_risk_matches_worked_example() -> None:
    from modeling.trifusion.signal_preserving_v14 import (
        cross_camera_retrieval_risk,
    )

    embedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])

    output = cross_camera_retrieval_risk(embedding, identities, cameras)

    assert torch.allclose(
        output.per_query_loss,
        torch.full((4,), math.log(2.0)),
        atol=1e-6,
    )
    assert torch.allclose(output.risk, torch.tensor(math.log(2.0)), atol=1e-6)


def test_v14_fold_bound_risk_rejects_rows_from_another_generator() -> None:
    from modeling.trifusion.signal_preserving_v14 import fold_bound_retrieval_risk

    baseline = torch.randn(4, 2)
    residual = torch.randn(4, 3, 3, 2)
    weights = torch.full((4, 3, 3), 1.0 / 9.0)
    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])

    with pytest.raises(ValueError, match="fold-bound"):
        fold_bound_retrieval_risk(
            fold_id=0,
            row_fold_ids=torch.tensor([0, 0, 0, 1]),
            baseline_embedding=baseline,
            modal_residual=residual,
            weights=weights,
            identities=identities,
            cameras=cameras,
        )


def test_v14_source_only_minimax_comparator_uses_worst_fold_risk() -> None:
    from modeling.trifusion.signal_preserving_v14 import select_minimax_fixed_slot

    fixed_slot_risks = torch.tensor(
        [
            [0.10, 0.40, 0.30],
            [0.50, 0.20, 0.30],
        ]
    )

    result = select_minimax_fixed_slot(fixed_slot_risks)

    assert result.slot == 2
    assert torch.allclose(result.worst_fold_risk, torch.tensor(0.30))


def test_v14_fold_bound_risk_reaches_every_router_parameter() -> None:
    from modeling.trifusion.signal_preserving_v13 import DeploymentAlignedRouter
    from modeling.trifusion.signal_preserving_v14 import fold_bound_retrieval_risk

    torch.manual_seed(42)
    router = DeploymentAlignedRouter(
        direct_width=2,
        residual_width=2,
        hidden_width=4,
    )
    direct = torch.randn(6, 3, 2)
    residual = torch.randn(6, 3, 3, 2)
    baseline = torch.randn(6, 4)
    identities = torch.tensor([0, 0, 1, 1, 2, 2])
    cameras = torch.tensor([0, 1, 0, 1, 0, 1])
    mask = torch.ones(6, 3, dtype=torch.bool)

    routing = router(direct, residual, mask)
    output = fold_bound_retrieval_risk(
        fold_id=0,
        row_fold_ids=torch.zeros(6, dtype=torch.long),
        baseline_embedding=baseline,
        modal_residual=residual,
        weights=routing.weights,
        identities=identities,
        cameras=cameras,
    )
    output.risk.backward()

    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().sum() > 0)
        for parameter in router.parameters()
    )

