from __future__ import annotations

import torch


def test_v8_margin_router_has_joint_mass_and_masks_missing_modalities() -> None:
    from modeling.trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
    )

    router = HierarchicalOOFMarginRouter(
        direct_width=4,
        residual_width=3,
        hidden_width=8,
        alpha_max=0.5,
        alpha_init=0.2,
    )
    direct_modal = torch.randn(2, 3, 4)
    modal_residual = torch.randn(2, 3, 3, 3)
    modality_mask = torch.tensor(
        [[True, True, True], [True, False, True]],
        dtype=torch.bool,
    )

    output = router(direct_modal, modal_residual, modality_mask)

    assert output.weights.shape == (2, 3, 3)
    assert torch.allclose(output.weights.sum(dim=(1, 2)), torch.ones(2))
    assert torch.equal(output.modal_probabilities[1, 1], torch.tensor(0.0))
    assert torch.equal(output.weights[1, :, 1], torch.zeros(3))
    assert torch.all(output.alpha > 0.0)
    assert torch.all(output.alpha <= 0.5)


def test_v8_margin_fusion_preserves_signal_prefix_and_bounds_residual_energy() -> None:
    from modeling.trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
        OOFMarginRoutedFusion,
    )

    router = HierarchicalOOFMarginRouter(
        direct_width=4,
        residual_width=3,
        hidden_width=8,
        alpha_max=0.5,
        alpha_init=0.2,
    )
    fusion = OOFMarginRoutedFusion(
        baseline_width=6,
        residual_width=3,
        alpha_max=0.5,
    )
    baseline = torch.randn(2, 6)
    direct_modal = torch.randn(2, 3, 4)
    modal_residual = torch.randn(2, 3, 3, 3)
    modality_mask = torch.ones(2, 3, dtype=torch.bool)
    routing = router(direct_modal, modal_residual, modality_mask)

    output = fusion(baseline, modal_residual, routing)

    suffix = output.fused_embedding[:, baseline.shape[1] :]
    assert torch.equal(output.fused_embedding[:, : baseline.shape[1]], baseline)
    assert output.fused_embedding.shape == (2, 33)
    assert torch.allclose(
        suffix.norm(dim=1),
        baseline.norm(dim=1) * routing.alpha.squeeze(1),
    )
    assert torch.all(suffix.norm(dim=1) <= baseline.norm(dim=1) * 0.5)


def test_v8_oof_margin_loss_prefers_the_highest_margin_slot() -> None:
    from modeling.trifusion.signal_preserving_v8_router import (
        OOFMarginRouterOutput,
        oof_margin_router_loss,
    )

    target_margin = torch.zeros(2, 3, 3)
    target_margin[:, 1, 2] = 1.0
    modality_mask = torch.ones(2, 3, dtype=torch.bool)
    correct_weights = torch.full((2, 3, 3), 0.01 / 8.0)
    correct_weights[:, 1, 2] = 0.99
    wrong_weights = torch.full((2, 3, 3), 0.01 / 8.0)
    wrong_weights[:, 0, 0] = 0.99
    correct = OOFMarginRouterOutput(
        weights=correct_weights,
        modal_probabilities=correct_weights.sum(dim=1),
        expert_probabilities=torch.full((2, 3, 3), 1.0 / 3.0),
        alpha=torch.full((2, 1), 0.5 / 1.1),
    )
    wrong = OOFMarginRouterOutput(
        weights=wrong_weights,
        modal_probabilities=wrong_weights.sum(dim=1),
        expert_probabilities=torch.full((2, 3, 3), 1.0 / 3.0),
        alpha=torch.full((2, 1), 0.1),
    )

    correct_loss = oof_margin_router_loss(
        correct,
        target_margin,
        modality_mask,
        alpha_max=0.5,
        utility_temperature=0.05,
        alpha_gain_scale=0.1,
    )
    wrong_loss = oof_margin_router_loss(
        wrong,
        target_margin,
        modality_mask,
        alpha_max=0.5,
        utility_temperature=0.05,
        alpha_gain_scale=0.1,
    )

    assert correct_loss.total < wrong_loss.total
    assert correct_loss.utility < wrong_loss.utility
    assert torch.allclose(
        correct_loss.alpha_target,
        torch.full((2, 1), 0.5 / 1.1),
    )
