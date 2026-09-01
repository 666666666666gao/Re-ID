from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from modeling.trifusion.state import EXPERT_ORDER, MODALITY_ORDER


def _modal_residual() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.randn(4, 3, 3, 8)


def test_v9_each_expert_receives_orthogonal_messages_from_both_peers() -> None:
    from modeling.trifusion.signal_preserving_v9 import OrthogonalTriadicRelay

    relay = OrthogonalTriadicRelay(
        residual_width=8,
        hidden_width=12,
        relay_depth=2,
    ).eval()
    modal_quality = torch.full((4, 3), 1.0 / 3.0)
    original = _modal_residual()
    output = relay(original, modal_quality)

    assert output.enhanced.shape == (4, 3, 3, 12)
    assert len(output.receiver_inputs) == 2
    assert len(output.orthogonal_messages) == 2
    for receivers, messages in zip(
        output.receiver_inputs,
        output.orthogonal_messages,
        strict=True,
    ):
        cosine = F.cosine_similarity(receivers, messages, dim=-1)
        assert float(cosine.abs().max()) <= 1e-5

    for receiver in range(3):
        for peer in range(3):
            if peer == receiver:
                continue
            changed = original.clone()
            changed[:, peer, :, 0] = changed[:, peer, :, 0] + 0.7
            changed_output = relay(changed, modal_quality)
            assert not torch.allclose(
                output.enhanced[:, receiver],
                changed_output.enhanced[:, receiver],
            )


def test_v9_synthesis_preserves_prefix_and_emits_three_enhanced_experts() -> None:
    from modeling.trifusion.signal_preserving_v9 import OrthogonalTriadicSynthesis

    torch.manual_seed(23)
    synthesis = OrthogonalTriadicSynthesis(
        baseline_width=12,
        prefix_width=20,
        residual_width=8,
        hidden_width=12,
        synergy_modal_width=10,
        relay_depth=2,
        beta_max=0.5,
        beta_init=0.2,
    )
    baseline = torch.randn(4, 12)
    prefix = torch.randn(4, 20)
    modal_residual = _modal_residual()
    modal_quality = torch.full((4, 3), 1.0 / 3.0)

    output = synthesis(
        baseline_embedding=baseline,
        prefix_embedding=prefix,
        modal_residual=modal_residual,
        modal_quality=modal_quality,
    )

    assert torch.equal(output.fused_embedding[:, :20], prefix)
    assert output.fused_embedding.shape == (4, 50)
    assert output.synergy_embedding.shape == (4, 30)
    assert output.beta.shape == (4, 1)
    assert bool((output.beta > 0.0).all())
    assert bool((output.beta <= 0.5).all())
    assert tuple(output.branch_embeddings) == EXPERT_ORDER
    for expert in EXPERT_ORDER:
        branch = output.branch_embeddings[expert]
        assert branch.shape == (4, 42)
        assert torch.equal(branch[:, :12], baseline)

    changed = modal_residual.clone()
    changed[:, 0, :, 0] = changed[:, 0, :, 0] + 0.7
    changed_output = synthesis(
        baseline_embedding=baseline,
        prefix_embedding=prefix,
        modal_residual=changed,
        modal_quality=modal_quality,
    )
    assert not torch.allclose(output.synergy_embedding, changed_output.synergy_embedding)


class _FrozenPhaseA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, return_aux: bool = False):
        assert return_aux
        pooled = torch.stack(
            [batch["images"][name].mean(dim=(1, 2, 3)) for name in MODALITY_ORDER],
            dim=1,
        )
        baseline = pooled.repeat(1, 4) * self.scale
        direct = pooled[..., None].repeat(1, 1, 4) * self.scale
        modal = {
            expert: pooled[..., None].repeat(1, 1, 8) * (index + 1) * self.scale
            for index, expert in enumerate(EXPERT_ORDER)
        }
        return SimpleNamespace(
            baseline_embedding=baseline,
            direct_modal=direct,
            modal_residual_embeddings=modal,
            diagnostics={"all_finite": True},
        )


class _FrozenRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, direct_modal, modal_residual, modality_mask):
        del direct_modal, modal_residual
        quality = modality_mask.float()
        quality = quality / quality.sum(dim=1, keepdim=True)
        return SimpleNamespace(
            modal_probabilities=quality + self.offset * 0.0,
            weights=quality[:, None].repeat(1, 3, 1) / 3.0,
            alpha=torch.full((quality.shape[0], 1), 0.2, device=quality.device),
        )


class _FrozenPhaseBFusion(nn.Module):
    def forward(self, baseline_embedding, modal_residual, routing):
        del routing
        suffix = modal_residual.flatten(1)[:, :8]
        return SimpleNamespace(
            fused_embedding=torch.cat((baseline_embedding, suffix), dim=1)
        )


def test_v9_model_trains_only_collaboration_and_keeps_exact_frozen_prefixes() -> None:
    from modeling.trifusion.signal_preserving_v9 import (
        OrthogonalTriadicSynthesis,
        SignalPreservingCollaborativeV9,
        SignalPreservingV9Criterion,
    )

    synthesis = OrthogonalTriadicSynthesis(
        baseline_width=12,
        prefix_width=20,
        residual_width=8,
        hidden_width=12,
        synergy_modal_width=10,
        relay_depth=2,
        beta_max=0.5,
        beta_init=0.2,
    )
    model = SignalPreservingCollaborativeV9(
        phase_a=_FrozenPhaseA(),
        router=_FrozenRouter(),
        phase_b_fusion=_FrozenPhaseBFusion(),
        synthesis=synthesis,
        num_classes=2,
    )
    model.train()
    batch = {
        "images": {
            modality: torch.randn(4, 3, 4, 4) for modality in MODALITY_ORDER
        },
        "modality_mask": torch.ones(4, 3, dtype=torch.bool),
        "camera_ids": torch.arange(4) % 2,
    }
    labels = torch.tensor([0, 0, 1, 1])
    output = model(batch, return_aux=True)

    assert torch.equal(output.fused_embedding[:, :20], output.phase_b_embedding)
    assert torch.equal(
        output.phase_b_embedding[:, :12],
        output.baseline_embedding,
    )
    assert output.diagnostics["baseline_exact_prefix"]
    assert output.diagnostics["phase_b_exact_prefix"]
    assert not model.phase_a.training
    assert not model.router.training
    assert all(not parameter.requires_grad for parameter in model.phase_a.parameters())
    assert all(not parameter.requires_grad for parameter in model.router.parameters())

    losses = SignalPreservingV9Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
    )(output, labels)
    sum(losses.values()).backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    assert all(parameter.grad is None for parameter in model.phase_a.parameters())
    assert all(parameter.grad is None for parameter in model.router.parameters())


def test_v9_builder_freezes_v8_and_registers_three_paper_mechanisms() -> None:
    from modeling.trifusion.signal_preserving_v9_builder import (
        build_signal_preserving_trifusion_v9,
    )

    build = build_signal_preserving_trifusion_v9(
        _FrozenPhaseA(),
        _FrozenRouter(),
        _FrozenPhaseBFusion(),
        combined_checkpoint_sha256="9" * 64,
        num_classes=2,
        baseline_width=12,
        phase_b_width=20,
        residual_width=8,
        hidden_width=12,
        synergy_modal_width=10,
        relay_depth=2,
        beta_max=0.5,
        beta_init=0.2,
    )

    assert build.provenance["architecture"] == (
        "signal_preserving_collaborative_v9_orthogonal_triadic_synthesis"
    )
    assert build.provenance["relay_depth"] == 2
    assert build.provenance["experts"] == list(EXPERT_ORDER)
    assert build.provenance["paper_mechanisms"] == [
        "pretrained_tail_role_disjoint_experts",
        "receiver_specific_orthogonal_peer_relay",
        "quality_aware_triadic_interaction_synthesis",
    ]
    assert all(
        not parameter.requires_grad for parameter in build.model.phase_a.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in build.model.router.parameters()
    )
    assert any(parameter.requires_grad for parameter in build.model.synthesis.parameters())
