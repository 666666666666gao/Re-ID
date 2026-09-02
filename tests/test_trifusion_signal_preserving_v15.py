from __future__ import annotations

import math

import torch
from torch import nn

from modeling.trifusion.experts.mamba import (
    FourDirectionMambaBlock,
    TinySequenceMixer,
)
from modeling.trifusion.signal_preserving_v8 import (
    ExpertFormationFusion,
    HierarchicalFrozenSignalField,
    PretrainedTailTriExpertEncoder,
)
from modeling.trifusion.state import EXPERT_ORDER


class _IdentityClipTailBlock(nn.Module):
    def forward(
        self,
        sequence: torch.Tensor,
        _attention_mask: object,
        _layer_index: int,
        _prompt: object,
        *,
        prompt_sign: bool,
        adapter_sign: bool,
    ) -> torch.Tensor:
        assert prompt_sign is False
        assert adapter_sign is False
        return sequence


class _ToyFrozenBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch: dict[str, object]) -> HierarchicalFrozenSignalField:
        self.calls += 1
        return batch["field"]  # type: ignore[return-value]


def test_v15_role_delta_exchange_starts_at_exact_no_exchange_parity() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CounterfactualRoleDeltaExchangeStage,
    )

    torch.manual_seed(42)
    exchange = CounterfactualRoleDeltaExchangeStage(
        width=8,
        rank=4,
        grid_size=(2, 1),
        edge_scale_max=0.25,
        mixer_factory=TinySequenceMixer,
    )
    before = {
        expert: torch.randn(2, 3, 3, 8) for expert in EXPERT_ORDER
    }
    after = {
        expert: before[expert] + torch.randn_like(before[expert])
        for expert in EXPERT_ORDER
    }

    output = exchange(before, after)

    assert tuple(output.states) == EXPERT_ORDER
    assert torch.equal(output.edge_scales, torch.zeros(3, 3))
    for expert in EXPERT_ORDER:
        assert torch.equal(output.states[expert], after[expert])


def test_v15_role_delta_exchange_is_synchronous_and_has_no_self_edge() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CounterfactualRoleDeltaExchangeStage,
    )

    torch.manual_seed(7)
    exchange = CounterfactualRoleDeltaExchangeStage(
        width=4,
        rank=2,
        grid_size=(1, 1),
        edge_scale_max=0.25,
        edge_scale_init=0.1,
        mixer_factory=TinySequenceMixer,
    )
    before = {
        expert: torch.zeros(1, 3, 2, 4) for expert in EXPERT_ORDER
    }
    after = {expert: value.clone() for expert, value in before.items()}
    after["cnn"][..., :] = torch.tensor([1.0, 2.0, 4.0, 8.0])

    output = exchange(before, after)

    assert torch.equal(output.states["cnn"], after["cnn"])
    assert not torch.equal(output.states["transformer"], after["transformer"])
    assert not torch.equal(output.states["mamba"], after["mamba"])
    assert torch.equal(output.edge_scales.diag(), torch.zeros(3))


def test_v15_exchange_messages_use_three_role_specific_mixers() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CounterfactualRoleDeltaExchangeStage,
    )

    exchange = CounterfactualRoleDeltaExchangeStage(
        width=8,
        rank=4,
        grid_size=(2, 1),
        edge_scale_max=0.25,
        mixer_factory=TinySequenceMixer,
    )

    assert any(
        isinstance(module, nn.Conv2d)
        for module in exchange.source_mixers["cnn"].modules()
    )
    assert any(
        isinstance(module, nn.MultiheadAttention)
        for module in exchange.source_mixers["transformer"].modules()
    )
    assert any(
        isinstance(module, FourDirectionMambaBlock)
        for module in exchange.source_mixers["mamba"].modules()
    )
    assert hasattr(exchange.source_mixers["mamba"], "modal_mixer")


def test_v15_matched_regret_uses_fixed_weight_and_stops_off_gradient() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        V15_REGRET_WEIGHT,
        matched_retrieval_regret_v15,
    )

    worked = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    on = {
        output: worked.clone().requires_grad_()
        for output in ("fused", *EXPERT_ORDER)
    }
    off = {
        output: worked.clone().requires_grad_()
        for output in ("fused", *EXPERT_ORDER)
    }
    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])

    result = matched_retrieval_regret_v15(on, off, identities, cameras)
    result.total.backward()

    assert V15_REGRET_WEIGHT == 1.0
    assert torch.allclose(result.total, torch.tensor(math.log(2.0)), atol=1e-6)
    assert all(value.grad is not None for value in on.values())
    assert all(value.grad is None for value in off.values())


def test_v15_retrieval_risk_masks_only_queries_without_cross_camera_positive() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        cross_camera_retrieval_risk_v15,
    )

    embedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    identities = torch.tensor([0, 0, 1])
    cameras = torch.tensor([0, 1, 0])

    output = cross_camera_retrieval_risk_v15(
        embedding,
        identities,
        cameras,
    )
    output.risk.backward()

    assert torch.equal(output.valid_query_mask, torch.tensor([True, True, False]))
    assert output.per_query_loss.shape == (2,)
    assert torch.isfinite(output.risk)
    assert embedding.grad is not None


def test_v15_matched_regret_is_zero_when_batch_has_no_valid_query() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        matched_retrieval_regret_v15,
    )

    on = {
        output: torch.randn(4, 3, requires_grad=True)
        for output in ("fused", *EXPERT_ORDER)
    }
    off = {
        output: torch.randn(4, 3, requires_grad=True)
        for output in ("fused", *EXPERT_ORDER)
    }
    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.zeros(4, dtype=torch.long)

    result = matched_retrieval_regret_v15(on, off, identities, cameras)
    result.total.backward()

    assert torch.equal(result.total, torch.tensor(0.0))
    assert all(value.grad is not None for value in on.values())
    assert all(torch.count_nonzero(value.grad) == 0 for value in on.values())
    assert all(value.grad is None for value in off.values())


def test_v15_encoder_zero_exchange_matches_v8_with_only_two_tail_exchanges() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CollaborativeTailTriExpertEncoderV15,
    )

    constructor = dict(
        tail_blocks=tuple(_IdentityClipTailBlock() for _ in range(3)),
        tail_layer_indices=(9, 10, 11),
        semantic_width=8,
        grid_size=(2, 1),
        adapter_width=4,
        expert_modal_width=4,
        mixer_factory=TinySequenceMixer,
        scale_init=0.05,
        gradient_checkpointing=False,
    )
    torch.manual_seed(42)
    v8 = PretrainedTailTriExpertEncoder(**constructor)
    torch.manual_seed(7)
    v15 = CollaborativeTailTriExpertEncoderV15(
        **constructor,
        exchange_rank=4,
        edge_scale_max=0.25,
    )
    missing, unexpected = v15.load_state_dict(v8.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all(name.startswith("exchange_stages.") for name in missing)

    anchor = torch.randn(2, 3, 3, 8)
    reference = torch.randn_like(anchor)
    expected = v8(anchor, reference)
    actual = v15(anchor, reference)

    assert v15.exchange_after_layer_indices == (9, 10)
    assert len(actual.exchange_edge_scales) == 2
    for expert in EXPERT_ORDER:
        assert torch.equal(
            actual.residual_embeddings[expert],
            expected.residual_embeddings[expert],
        )
        assert torch.equal(
            actual.modal_residual_embeddings[expert],
            expected.modal_residual_embeddings[expert],
        )


def test_v15_paired_forward_reuses_one_field_and_detaches_exchange_off() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CollaborativeTailTriExpertEncoderV15,
        SignalPreservingCollaborativeV15,
    )

    class RecordingEncoder(CollaborativeTailTriExpertEncoderV15):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.calls: list[tuple[int, int, bool]] = []

        def forward(
            self,
            anchor_sequence: torch.Tensor,
            reference_sequence: torch.Tensor,
            *,
            exchange_enabled: bool = True,
        ):
            self.calls.append(
                (id(anchor_sequence), id(reference_sequence), exchange_enabled)
            )
            return super().forward(
                anchor_sequence,
                reference_sequence,
                exchange_enabled=exchange_enabled,
            )

    baseline = _ToyFrozenBaseline()
    encoder = RecordingEncoder(
        tail_blocks=tuple(_IdentityClipTailBlock() for _ in range(3)),
        tail_layer_indices=(9, 10, 11),
        semantic_width=8,
        grid_size=(2, 1),
        adapter_width=4,
        expert_modal_width=4,
        mixer_factory=TinySequenceMixer,
        exchange_rank=4,
        edge_scale_max=0.25,
        gradient_checkpointing=False,
    )
    model = SignalPreservingCollaborativeV15(
        baseline=baseline,
        encoder=encoder,
        fusion=ExpertFormationFusion(baseline_width=6, expert_width=12),
        num_classes=2,
    )
    model.train()
    anchor = torch.randn(2, 3, 3, 8)
    reference = torch.randn_like(anchor)
    field = HierarchicalFrozenSignalField(
        baseline_embedding=torch.randn(2, 6),
        direct_modal=torch.randn(2, 3, 2),
        anchor_sequence=anchor,
        reference_sequence=reference,
        modality_mask=torch.ones(2, 3),
    )

    paired = model.forward_paired({"field": field})

    assert baseline.calls == 1
    assert encoder.calls == [
        (id(anchor), id(reference), False),
        (id(anchor), id(reference), True),
    ]
    torch.testing.assert_close(
        paired.exchange_on.fused_embedding,
        paired.exchange_off.fused_embedding,
        rtol=0.0,
        atol=1e-6,
    )
    assert paired.exchange_on.fused_embedding.requires_grad
    assert not paired.exchange_off.fused_embedding.requires_grad
    assert paired.exchange_off.fused_logits is None


def test_v15_criterion_combines_v8_identity_losses_with_fixed_regret() -> None:
    from modeling.trifusion.signal_preserving_v15 import (
        CollaborativeV15Criterion,
        PairedSignalPreservingV15Output,
        SignalPreservingV15Output,
    )

    torch.manual_seed(42)
    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    baseline = torch.randn(4, 2)
    branches = {
        expert: torch.randn(4, 4, requires_grad=True) for expert in EXPERT_ORDER
    }
    residuals = {
        expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
    }
    on = SignalPreservingV15Output(
        fused_embedding=torch.randn(4, 8, requires_grad=True),
        baseline_embedding=baseline,
        direct_modal=torch.randn(4, 3, 2),
        branch_embeddings=branches,
        residual_embeddings=residuals,
        modal_residual_embeddings={
            expert: torch.randn(4, 3, 2) for expert in EXPERT_ORDER
        },
        fused_logits=torch.randn(4, 2, requires_grad=True),
        branch_logits={
            expert: torch.randn(4, 2, requires_grad=True)
            for expert in EXPERT_ORDER
        },
        residual_logits={
            expert: torch.randn(4, 2, requires_grad=True)
            for expert in EXPERT_ORDER
        },
        exchange_edge_scales=(),
        diagnostics={},
    )
    off = SignalPreservingV15Output(
        fused_embedding=on.fused_embedding.detach().clone(),
        baseline_embedding=baseline,
        direct_modal=on.direct_modal,
        branch_embeddings={
            expert: value.detach().clone() for expert, value in branches.items()
        },
        residual_embeddings={},
        modal_residual_embeddings={},
        fused_logits=None,
        branch_logits={},
        residual_logits={},
        exchange_edge_scales=(),
        diagnostics={},
    )
    criterion = CollaborativeV15Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
        id_fused_weight=0.25,
        triplet_fused_weight=1.0,
        id_branch_weight=1.0 / 12.0,
        triplet_branch_weight=0.25,
        id_residual_weight=1.0 / 12.0,
        triplet_residual_weight=0.25,
    )

    losses = criterion(
        PairedSignalPreservingV15Output(exchange_on=on, exchange_off=off),
        identities,
        cameras,
    )
    losses["total"].backward()

    assert "retrieval_regret" in losses
    assert "id_fused" in losses
    assert "triplet_residual_mamba" in losses
    assert torch.isfinite(losses["total"])
    assert on.fused_embedding.grad is not None
    assert on.fused_logits.grad is not None
