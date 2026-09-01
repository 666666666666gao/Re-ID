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

    def forward(
        self,
        images: torch.Tensor,
        cam_label: torch.Tensor | None = None,
        view_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del view_label
        global_embedding = self.projection(images.mean(dim=(2, 3)))
        if cam_label is not None:
            global_embedding = global_embedding + cam_label[:, None]
        return global_embedding[:, None].expand(-1, 4, -1), global_embedding


class _FakeSIM(nn.Module):
    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        globals_by_modality = values[-3:]
        return torch.cat(globals_by_modality, dim=1)


class _FakeSignal(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.clip_vision_encoder = _FakeVision(width)
        self.SIM = _FakeSIM()


def _batch(batch_size: int = 4) -> dict[str, object]:
    return {
        "images": {
            modality: torch.randn(batch_size, 3, 4, 4)
            for modality in ("RGB", "NI", "TI")
        },
        "modality_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "camera_ids": torch.arange(batch_size) % 2,
    }


def _states(direct_modal: torch.Tensor) -> ExpertStateMap:
    mask = torch.ones(direct_modal.shape[:2], dtype=torch.bool)
    channel_pattern = torch.arange(
        1,
        direct_modal.shape[-1] + 1,
        dtype=direct_modal.dtype,
        device=direct_modal.device,
    )
    states = {
        expert: ExpertState(
            tokens=(
                direct_modal + channel_pattern * float(index + 1)
            ).unsqueeze(2),
            global_embedding=direct_modal + channel_pattern * float(index + 1),
            private_embedding=direct_modal[..., :1],
            role_payload=MappingProxyType({"summary": direct_modal[..., :1]}),
            modality_mask=mask,
            stage=3,
            expert=expert,
        )
        for index, expert in enumerate(EXPERT_ORDER)
    }
    reliability = ReliabilityResult(
        alpha=torch.full((direct_modal.shape[0], 3, 3), 2.0),
        beta=torch.full((direct_modal.shape[0], 3, 3), 2.0),
        r=torch.full((direct_modal.shape[0], 3, 3), 0.5),
        u=torch.full((direct_modal.shape[0], 3, 3), 0.5),
        modality_mask=mask,
    )
    return ExpertStateMap(states, modality_mask=mask, reliability=reliability)


def test_v6_activates_residual_geometry_without_rewriting_signal() -> None:
    from modeling.trifusion.signal_preserving_v6 import (
        ComplementarityActivatedResidualBankFusion,
    )

    torch.manual_seed(6)
    baseline = torch.randn(4, 24)
    direct_modal = torch.randn(4, 3, 4)
    states = _states(direct_modal)
    fusion = ComplementarityActivatedResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        baseline_width=24,
        residual_width=2,
    )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        baseline_embedding=baseline,
        direct_modal=direct_modal,
    )

    baseline_norm = baseline.norm(dim=1)
    assert torch.equal(result.fused_embedding[:, :24], baseline)
    assert torch.allclose(
        result.fused_embedding[:, 24:].norm(dim=1), baseline_norm, rtol=1e-5
    )
    assert tuple(result.residual_embeddings) == EXPERT_ORDER
    for expert in EXPERT_ORDER:
        branch = result.branch_embeddings[expert]
        assert torch.equal(branch[:, :24], baseline)
        assert torch.allclose(branch[:, 24:].norm(dim=1), baseline_norm, rtol=1e-5)
        assert torch.equal(branch[:, 24:], result.residual_embeddings[expert])
    assert not hasattr(fusion, "residual_scale_logits")


def test_v6_supervises_each_residual_expert_and_the_utility_router() -> None:
    from modeling.trifusion.signal_preserving_v6 import (
        ComplementarityActivatedV6Criterion,
    )

    labels = torch.tensor([0, 0, 1, 1])
    residual_values = {
        "cnn": [[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [-1.0, 0.1]],
        "transformer": [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        "mamba": [[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0], [-0.5, 0.5]],
    }
    residual_embeddings = {
        expert: torch.tensor(values, requires_grad=True)
        for expert, values in residual_values.items()
    }
    residual_logits = {
        expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
    }
    router_weights = torch.full((4, 3, 3), 1.0 / 3.0, requires_grad=True)
    output = SimpleNamespace(
        fused_embedding=torch.randn(4, 8, requires_grad=True),
        branch_embeddings={
            expert: torch.randn(4, 8, requires_grad=True) for expert in EXPERT_ORDER
        },
        residual_embeddings=residual_embeddings,
        fused_logits=torch.randn(4, 2, requires_grad=True),
        branch_logits={
            expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
        },
        residual_logits=residual_logits,
        reliability=None,
        modality_mask=torch.ones(4, 3, dtype=torch.bool),
        router_weights=router_weights,
        peer_teaching=None,
    )

    losses = ComplementarityActivatedV6Criterion(
        target_cache=None,
        triplet_margin=0.3,
    )(output, labels)

    expected = {
        *(f"id_residual_{expert}" for expert in EXPERT_ORDER),
        *(f"triplet_residual_{expert}" for expert in EXPERT_ORDER),
        "peer_logits",
    }
    assert expected <= losses.keys()
    sum(losses[name] for name in expected).backward()
    assert router_weights.grad is not None
    assert torch.count_nonzero(router_weights.grad).item() > 0
    for expert in EXPERT_ORDER:
        assert residual_embeddings[expert].grad is not None
        assert residual_logits[expert].grad is not None


def test_v6_builder_emits_five_retrieval_outputs_and_residual_supervision() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v6_builder import (
        build_signal_preserving_trifusion_v6,
    )

    build = build_signal_preserving_trifusion_v6(
        _FakeSignal(),
        signal_checkpoint_sha256="6" * 64,
        num_classes=2,
        feature_width=4,
        grid_size=(2, 2),
        adapter_width=4,
        residual_width=2,
        relay_rank=2,
        private_width=2,
        reliability_hidden_width=8,
        mamba_mixer_factory=TinySequenceMixer,
    )
    batch = _batch()
    baseline = build.model(batch, retrieval_output="baseline_only")
    output = build.model(batch, return_aux=True)

    assert build.provenance["architecture"] == "signal_preserving_collaborative_v6"
    assert build.provenance["energy_balance_has_free_scale"] is False
    assert build.provenance["residual_only_supervision"] is True
    assert len(build.provenance["paper_contributions"]) == 3
    assert torch.equal(output.fused_embedding[:, : baseline.shape[1]], baseline)
    assert tuple(output.branch_embeddings) == EXPERT_ORDER
    assert tuple(output.residual_embeddings) == EXPERT_ORDER
    assert tuple(output.residual_logits) == EXPERT_ORDER
    assert len(output.relay_results) == 2
    assert output.diagnostics["residual_energy_activated"]
    assert all(
        not parameter.requires_grad for parameter in build.model.baseline.parameters()
    )
