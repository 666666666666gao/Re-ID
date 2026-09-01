from __future__ import annotations

from types import MappingProxyType
from types import SimpleNamespace

import torch
from torch import nn

from modeling.trifusion.state import (
    EXPERT_ORDER,
    ExpertState,
    ExpertStateMap,
    ReliabilityResult,
)


class _FakeVision(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, width, bias=False)

    def forward(self, images, cam_label=None, view_label=None):
        del view_label
        global_embedding = self.projection(images.mean(dim=(2, 3)))
        if cam_label is not None:
            global_embedding = global_embedding + cam_label[:, None]
        return global_embedding[:, None].expand(-1, 4, -1), global_embedding


class _FakeSIM(nn.Module):
    def forward(self, *values):
        return torch.cat(values[-3:], dim=1)


class _FakeSignal(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.clip_vision_encoder = _FakeVision(width)
        self.SIM = _FakeSIM()


def _states(
    anchor_tokens: torch.Tensor,
    reliability: ReliabilityResult,
    *,
    zero_delta: bool,
) -> ExpertStateMap:
    channel_pattern = torch.arange(
        1,
        anchor_tokens.shape[-1] + 1,
        dtype=anchor_tokens.dtype,
    )
    states = {}
    for index, expert in enumerate(EXPERT_ORDER):
        delta = 0.0 if zero_delta else channel_pattern * float(index + 1)
        tokens = anchor_tokens + delta
        states[expert] = ExpertState(
            tokens=tokens,
            global_embedding=tokens.mean(dim=2),
            private_embedding=tokens.mean(dim=2)[..., :1],
            role_payload=MappingProxyType({"summary": tokens.mean(dim=2)[..., :1]}),
            modality_mask=reliability.modality_mask,
            stage=3,
            expert=expert,
        )
    return ExpertStateMap(
        states,
        modality_mask=reliability.modality_mask,
        reliability=reliability,
    )


def _reliability(r: torch.Tensor, mask: torch.Tensor) -> ReliabilityResult:
    return ReliabilityResult(
        alpha=torch.ones_like(r),
        beta=torch.ones_like(r),
        r=r * mask[:, None],
        u=torch.zeros_like(r),
        modality_mask=mask,
    )


def test_v7_matched_token_residual_is_zero_when_expert_adds_nothing() -> None:
    from modeling.trifusion.signal_preserving_v7 import (
        HierarchicalBoundedResidualBankFusion,
    )

    mask = torch.ones(2, 3, dtype=torch.bool)
    reliability = _reliability(torch.full((2, 3, 3), 0.5), mask)
    anchor_tokens = torch.randn(2, 3, 4, 4)
    states = _states(anchor_tokens, reliability, zero_delta=True)
    baseline = torch.randn(2, 24)
    fusion = HierarchicalBoundedResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        baseline_width=24,
        residual_width=2,
        alpha_max=0.5,
        alpha_init=0.2,
    )

    result = fusion(
        states,
        reliability,
        mask,
        baseline_embedding=baseline,
        anchor_tokens=anchor_tokens,
    )

    assert torch.equal(result.fused_embedding[:, :24], baseline)
    assert torch.count_nonzero(result.contribution_embeddings).item() == 0
    assert torch.count_nonzero(result.fused_embedding[:, 24:]).item() == 0


def test_v7_router_has_joint_mass_and_bounded_sample_energy() -> None:
    from modeling.trifusion.signal_preserving_v7 import (
        HierarchicalBoundedResidualBankFusion,
    )

    mask = torch.tensor([[True, True, True], [True, True, False]])
    r = torch.tensor(
        [
            [[0.9, 0.9, 0.001]] * 3,
            [[0.9, 0.8, 0.0]] * 3,
        ]
    )
    reliability = _reliability(r, mask)
    anchor_tokens = torch.randn(2, 3, 4, 4)
    states = _states(anchor_tokens, reliability, zero_delta=False)
    baseline = torch.randn(2, 24)
    fusion = HierarchicalBoundedResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        baseline_width=24,
        residual_width=2,
        alpha_max=0.5,
        alpha_init=0.2,
    )

    result = fusion(
        states,
        reliability,
        mask,
        baseline_embedding=baseline,
        anchor_tokens=anchor_tokens,
    )

    assert torch.allclose(result.weights.sum(dim=(1, 2)), torch.ones(2))
    assert result.modal_probabilities[0, 2] < 0.01
    assert result.weights[1, :, 2].count_nonzero().item() == 0
    assert torch.allclose(result.alpha, torch.full((2, 1), 0.2), atol=1e-6)
    suffix_ratio = (
        result.fused_embedding[:, 24:].norm(dim=1) / baseline.norm(dim=1)
    )
    assert torch.allclose(suffix_ratio, result.alpha[:, 0], atol=1e-6)
    assert bool((result.alpha <= 0.5).all())


def test_v7_router_target_uses_per_expert_modality_marginal_gain() -> None:
    from modeling.trifusion.signal_preserving_v7 import marginal_gain_router_loss

    labels = torch.tensor([0, 0, 1, 1])
    baseline = torch.tensor([[1.0, 0.0]] * 4)
    contributions = torch.zeros(4, 3, 3, 2)
    contributions[:, 0, 0] = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    weights = torch.full((4, 3, 3), 1.0 / 9.0, requires_grad=True)
    alpha = torch.full((4, 1), 0.2, requires_grad=True)

    routing = marginal_gain_router_loss(
        baseline,
        contributions,
        weights,
        alpha,
        labels,
        modality_mask=torch.ones(4, 3, dtype=torch.bool),
        alpha_max=0.5,
        utility_temperature=0.1,
        alpha_gain_scale=0.1,
    )

    assert torch.equal(
        routing.target_weights.flatten(1).argmax(dim=1), torch.zeros(4, dtype=torch.long)
    )
    assert bool((routing.alpha_target > 0).all())
    (routing.router_loss + routing.alpha_loss).backward()
    assert weights.grad is not None and torch.count_nonzero(weights.grad).item() > 0
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad).item() > 0


def test_v7_criterion_supervises_quality_router_and_bounded_alpha() -> None:
    from modeling.trifusion.signal_preserving_v7 import MarginalGainV7Criterion

    labels = torch.tensor([0, 0, 1, 1])
    router_weights = torch.full((4, 3, 3), 1.0 / 9.0, requires_grad=True)
    modal_probabilities = torch.full((4, 3), 1.0 / 3.0, requires_grad=True)
    alpha = torch.full((4, 1), 0.2, requires_grad=True)
    output = SimpleNamespace(
        baseline_embedding=torch.randn(4, 8),
        fused_embedding=torch.randn(4, 14, requires_grad=True),
        branch_embeddings={
            expert: torch.randn(4, 10, requires_grad=True) for expert in EXPERT_ORDER
        },
        residual_embeddings={
            expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
        },
        contribution_embeddings=torch.randn(4, 3, 3, 2),
        fused_logits=torch.randn(4, 2, requires_grad=True),
        branch_logits={
            expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
        },
        residual_logits={
            expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
        },
        modality_mask=torch.ones(4, 3, dtype=torch.bool),
        router_weights=router_weights,
        modal_probabilities=modal_probabilities,
        alpha=alpha,
        alpha_max=0.5,
        quality_targets=torch.tensor([[1.0, 1.0, 0.1]] * 4),
    )

    losses = MarginalGainV7Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
        utility_temperature=0.1,
        alpha_gain_scale=0.1,
    )(output, labels, quality_output=output)

    assert {"peer_logits", "alpha", "reliability"} <= losses.keys()
    assert losses["reliability"].item() > 0
    (losses["peer_logits"] + losses["alpha"] + losses["reliability"]).backward()
    assert router_weights.grad is not None
    assert alpha.grad is not None
    assert modal_probabilities.grad is not None


def test_v7_builder_emits_exact_baseline_and_joint_routing_outputs() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v7_builder import (
        build_signal_preserving_trifusion_v7,
    )

    build = build_signal_preserving_trifusion_v7(
        _FakeSignal(),
        signal_checkpoint_sha256="7" * 64,
        num_classes=2,
        feature_width=4,
        grid_size=(2, 2),
        adapter_width=4,
        residual_width=2,
        relay_rank=2,
        private_width=2,
        reliability_hidden_width=8,
        alpha_max=0.5,
        alpha_init=0.2,
        mamba_mixer_factory=TinySequenceMixer,
    )
    batch = {
        "images": {
            modality: torch.randn(4, 3, 4, 4) for modality in ("RGB", "NI", "TI")
        },
        "modality_mask": torch.ones(4, 3, dtype=torch.bool),
        "camera_ids": torch.arange(4) % 2,
        "modality_quality": torch.tensor([[1.0, 1.0, 0.25]] * 4),
    }

    baseline = build.model(batch, retrieval_output="baseline_only")
    output = build.model(batch, return_aux=True)

    assert build.provenance["architecture"] == "signal_preserving_collaborative_v7"
    assert build.provenance["shared_triplet_geometry"] is True
    assert build.provenance["matched_token_residual"] is True
    assert build.provenance["hierarchical_router"] == "P(modality)*P(expert|modality)"
    assert torch.equal(output.fused_embedding[:, : baseline.shape[1]], baseline)
    assert torch.allclose(output.router_weights.sum(dim=(1, 2)), torch.ones(4))
    assert torch.equal(output.quality_targets, batch["modality_quality"])
    assert len(output.relay_results) == 2
    assert output.diagnostics["bounded_residual_energy"]
