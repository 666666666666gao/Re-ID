from __future__ import annotations

from types import MappingProxyType

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
        pooled = images.mean(dim=(2, 3))
        global_embedding = self.projection(pooled)
        if cam_label is not None:
            global_embedding = global_embedding + cam_label[:, None].to(
                global_embedding.dtype
            )
        patches = global_embedding[:, None].expand(-1, 4, -1)
        return patches, global_embedding


class _FakeSIM(nn.Module):
    def forward(
        self,
        rgb_patches: torch.Tensor,
        ni_patches: torch.Tensor,
        ti_patches: torch.Tensor,
        rgb_global: torch.Tensor,
        ni_global: torch.Tensor,
        ti_global: torch.Tensor,
    ) -> torch.Tensor:
        del rgb_patches, ni_patches, ti_patches
        return torch.cat(
            (rgb_global + 1.0, ni_global + 2.0, ti_global + 3.0), dim=1
        )


class _FakeSignal(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.clip_vision_encoder = _FakeVision(width)
        self.SIM = _FakeSIM()


def _batch(batch_size: int = 4) -> dict[str, object]:
    images = {
        "RGB": torch.randn(batch_size, 3, 4, 4),
        "NI": torch.randn(batch_size, 3, 4, 4),
        "TI": torch.randn(batch_size, 3, 4, 4),
    }
    return {
        "images": images,
        "modality_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "camera_ids": torch.arange(batch_size) % 2,
    }


def _states(direct_modal: torch.Tensor) -> ExpertStateMap:
    mask = torch.ones(direct_modal.shape[:2], dtype=torch.bool)
    values = {
        "cnn": direct_modal + 1.0,
        "transformer": direct_modal + 2.0,
        "mamba": direct_modal + 3.0,
    }
    states = {
        expert: ExpertState(
            tokens=embedding.unsqueeze(2),
            global_embedding=embedding,
            private_embedding=embedding[..., :1],
            role_payload=MappingProxyType({"summary": embedding[..., :1]}),
            modality_mask=mask,
            stage=3,
            expert=expert,
        )
        for expert, embedding in values.items()
    }
    reliability = ReliabilityResult(
        alpha=torch.full((direct_modal.shape[0], 3, 3), 2.0),
        beta=torch.full((direct_modal.shape[0], 3, 3), 2.0),
        r=torch.full((direct_modal.shape[0], 3, 3), 0.5),
        u=torch.full((direct_modal.shape[0], 3, 3), 0.5),
        modality_mask=mask,
    )
    return ExpertStateMap(states, modality_mask=mask, reliability=reliability)


class _FakeEncoder(nn.Module):
    def forward_token_field(
        self,
        expert_tokens: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> ExpertStateMap:
        batch_size, modalities = modality_mask.shape
        globals_by_expert = {
            expert: tokens.mean(dim=1).reshape(batch_size, modalities, -1)
            + float(index + 1)
            for index, (expert, tokens) in enumerate(expert_tokens.items())
        }
        result = _states(globals_by_expert["cnn"] - 1.0)
        states = {
            expert: ExpertState(
                tokens=globals_by_expert[expert].unsqueeze(2),
                global_embedding=globals_by_expert[expert],
                private_embedding=globals_by_expert[expert][..., :1],
                role_payload=MappingProxyType(
                    {"summary": globals_by_expert[expert][..., :1]}
                ),
                modality_mask=modality_mask,
                stage=3,
                expert=expert,
            )
            for expert in EXPERT_ORDER
        }
        return ExpertStateMap(
            states,
            modality_mask=modality_mask,
            reliability=result.reliability,
        )


def test_frozen_signal_backbone_preserves_the_exact_3072d_contract() -> None:
    from modeling.trifusion.signal_preserving_v5 import FrozenSignalBackbone

    signal = _FakeSignal()
    backbone = FrozenSignalBackbone(signal, feature_width=4)
    batch = _batch()

    field = backbone(batch)

    globals_by_modality = []
    patches_by_modality = []
    for name in ("RGB", "NI", "TI"):
        patches, global_embedding = signal.clip_vision_encoder(
            batch["images"][name], cam_label=batch["camera_ids"]
        )
        patches_by_modality.append(patches)
        globals_by_modality.append(global_embedding)
    expected = torch.cat(
        (
            torch.cat(globals_by_modality, dim=1),
            signal.SIM(*patches_by_modality, *globals_by_modality),
        ),
        dim=1,
    )

    assert torch.equal(field.baseline_embedding, expected)
    assert field.baseline_embedding.shape == (4, 24)
    assert field.direct_modal.shape == (4, 3, 4)
    assert all(not parameter.requires_grad for parameter in signal.parameters())
    backbone.train()
    assert not signal.training


def test_v5_fusion_keeps_baseline_prefix_and_reaches_all_experts() -> None:
    from modeling.trifusion.signal_preserving_v5 import (
        SignalPreservingResidualBankFusion,
    )

    torch.manual_seed(5)
    baseline = torch.randn(4, 24)
    direct_modal = torch.randn(4, 3, 4)
    states = _states(direct_modal)
    fusion = SignalPreservingResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        baseline_width=24,
        residual_width=2,
        residual_scale_init=0.1,
    )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        baseline_embedding=baseline,
        direct_modal=direct_modal,
    )

    assert result.fused_embedding.shape == (4, 42)
    assert torch.equal(result.fused_embedding[:, :24], baseline)
    assert all(value.shape == (4, 30) for value in result.branch_embeddings.values())
    result.fused_embedding[:, 24:].square().sum().backward()
    for expert in EXPERT_ORDER:
        gradient = fusion.residual_projections[expert].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_v5_same_model_emits_exact_baseline_and_fused_outputs() -> None:
    from modeling.trifusion.signal_preserving_v5 import (
        FrozenSignalBackbone,
        SignalPreservingCollaborativeReID,
        SignalPreservingResidualBankFusion,
    )

    signal = _FakeSignal()
    baseline = FrozenSignalBackbone(signal, feature_width=4)
    fusion = SignalPreservingResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        baseline_width=24,
        residual_width=2,
        residual_scale_init=0.1,
    )
    model = SignalPreservingCollaborativeReID(
        baseline=baseline,
        encoder=_FakeEncoder(),
        fusion=fusion,
        num_classes=2,
    )
    batch = _batch()
    frozen_before = {
        name: value.detach().clone() for name, value in signal.state_dict().items()
    }

    baseline_only = model(batch, retrieval_output="baseline_only")
    output = model(batch, return_aux=True)
    assert torch.equal(output.baseline_embedding, baseline_only)
    assert torch.equal(output.fused_embedding[:, :24], baseline_only)
    assert tuple(output.branch_embeddings) == EXPERT_ORDER

    labels = torch.tensor([0, 0, 1, 1])
    loss = torch.nn.functional.cross_entropy(output.fused_logits, labels)
    loss = loss + sum(
        torch.nn.functional.cross_entropy(output.branch_logits[name], labels)
        for name in EXPERT_ORDER
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert all(parameter.grad is None for parameter in signal.parameters())
    assert all(
        torch.equal(frozen_before[name], value)
        for name, value in signal.state_dict().items()
    )


def test_v5_builder_runs_all_experts_and_two_relay_stages() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v5_builder import (
        build_signal_preserving_trifusion_v5,
    )

    result = build_signal_preserving_trifusion_v5(
        _FakeSignal(),
        signal_checkpoint_sha256="1" * 64,
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
    output = result.model(_batch(), return_aux=True)

    assert result.provenance["architecture"] == "signal_preserving_collaborative_v5"
    assert result.provenance["baseline_parameters_frozen"]
    assert result.provenance["experts"] == list(EXPERT_ORDER)
    assert result.provenance["relay_stages"] == [1, 2]
    assert len(result.provenance["paper_contributions"]) == 3
    assert output.baseline_embedding.shape == (4, 24)
    assert output.fused_embedding.shape == (4, 42)
    assert len(output.relay_results) == 2
    assert tuple(output.branch_embeddings) == EXPERT_ORDER
    assert output.diagnostics["baseline_exact_prefix"]
    assert output.diagnostics["baseline_frozen"]
    assert all(
        not parameter.requires_grad
        for expert in result.model.encoder.experts.values()
        for parameter in expert.private_projection.parameters()
    )
