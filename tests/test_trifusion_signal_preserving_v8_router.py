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
