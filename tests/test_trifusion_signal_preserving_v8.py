from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from modeling.trifusion.state import EXPERT_ORDER, MODALITY_ORDER


class _FakeResidualBlock(nn.Module):
    def __init__(self, width: int, offset: float) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.projection.weight)
        self.offset = float(offset)

    def forward(
        self,
        sequence: torch.Tensor,
        modality=None,
        index=None,
        last_prompt=None,
        *,
        prompt_sign: bool,
        adapter_sign: bool,
    ) -> torch.Tensor:
        del modality, index, last_prompt
        assert prompt_sign is False
        assert adapter_sign is False
        return self.projection(sequence) + self.offset


class _FakeClipBase(nn.Module):
    def __init__(self, semantic_width: int, feature_width: int) -> None:
        super().__init__()
        self.transformer = SimpleNamespace(
            resblocks=nn.ModuleList(
                [
                    _FakeResidualBlock(semantic_width, 0.01 * (index + 1))
                    for index in range(4)
                ]
            )
        )
        self.output_projection = nn.Linear(semantic_width, feature_width, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        extra = pooled.mean(dim=1, keepdim=True).expand(-1, 3)
        token = torch.cat((pooled, extra), dim=1)
        patches = token[:, None].expand(-1, 4, -1)
        sequence = torch.cat((token[:, None], patches), dim=1).permute(1, 0, 2)
        for index, block in enumerate(self.transformer.resblocks):
            sequence = block(
                sequence,
                None,
                index,
                None,
                prompt_sign=False,
                adapter_sign=False,
            )
        return sequence.permute(1, 0, 2)


class _FakeVision(nn.Module):
    def __init__(self, semantic_width: int = 6, feature_width: int = 4) -> None:
        super().__init__()
        self.base = _FakeClipBase(semantic_width, feature_width)

    def forward(self, images, cam_label=None, view_label=None):
        del view_label
        sequence = self.base(images)
        projected = self.base.output_projection(sequence)
        if cam_label is not None:
            projected = projected + cam_label[:, None, None]
        return projected[:, 1:], projected[:, 0]


class _FakeSIM(nn.Module):
    def forward(self, *values):
        return torch.cat(values[-3:], dim=1)


class _FakeSignal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip_vision_encoder = _FakeVision()
        self.SIM = _FakeSIM()


def _batch(*, nir_shift: float = 0.0) -> dict[str, object]:
    torch.manual_seed(7)
    images = {
        modality: torch.randn(4, 3, 4, 4) for modality in MODALITY_ORDER
    }
    images["NI"][:, 0] = images["NI"][:, 0] + nir_shift
    return {
        "images": images,
        "modality_mask": torch.ones(4, 3, dtype=torch.bool),
        "camera_ids": torch.arange(4) % 2,
    }


def test_v8_backbone_preserves_exact_signal_and_captures_pretrained_tail() -> None:
    from modeling.trifusion.signal_preserving_v5 import FrozenSignalBackbone
    from modeling.trifusion.signal_preserving_v8 import (
        HierarchicalFrozenSignalBackbone,
    )

    expected_signal = _FakeSignal()
    actual_signal = deepcopy(expected_signal)
    expected = FrozenSignalBackbone(expected_signal, feature_width=4)(_batch())
    backbone = HierarchicalFrozenSignalBackbone(
        actual_signal,
        feature_width=4,
        branch_after_block=0,
    )

    field = backbone(_batch())

    assert torch.equal(field.baseline_embedding, expected.baseline_embedding)
    assert field.anchor_sequence.shape == (4, 3, 5, 6)
    assert field.reference_sequence.shape == (4, 3, 5, 6)
    assert len(backbone.tail_blocks) == 3
    assert backbone.tail_layer_indices == (1, 2, 3)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_v8_builder_emits_three_role_disjoint_pretrained_experts() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v8_builder import (
        build_signal_preserving_trifusion_v8_expert_formation,
    )

    build = build_signal_preserving_trifusion_v8_expert_formation(
        _FakeSignal(),
        signal_checkpoint_sha256="8" * 64,
        num_classes=2,
        feature_width=4,
        semantic_width=6,
        grid_size=(2, 2),
        branch_after_block=0,
        adapter_width=4,
        expert_modal_width=8,
        mamba_mixer_factory=TinySequenceMixer,
    )
    baseline = build.model(_batch(), retrieval_output="baseline_only")
    output = build.model(_batch(), return_aux=True)

    assert build.provenance["architecture"] == (
        "signal_preserving_collaborative_v8_expert_formation"
    )
    assert build.provenance["pretrained_tail_layers"] == [1, 2, 3]
    assert build.provenance["router_enabled"] is False
    assert build.provenance["hfer_enabled"] is False
    assert build.provenance["expert_roles"] == {
        "cnn": "horizontal_local_detail",
        "transformer": "global_cls_relation",
        "mamba": "spatial_and_cross_modal_long_range",
    }
    assert torch.equal(output.fused_embedding[:, : baseline.shape[1]], baseline)
    assert tuple(output.residual_embeddings) == EXPERT_ORDER
    assert all(value.shape == (4, 24) for value in output.residual_embeddings.values())
    assert all(
        output.modal_residual_embeddings[expert].shape == (4, 3, 8)
        for expert in EXPERT_ORDER
    )
    assert not torch.equal(
        output.residual_embeddings["cnn"],
        output.residual_embeddings["transformer"],
    )
    assert not torch.equal(
        output.residual_embeddings["transformer"],
        output.residual_embeddings["mamba"],
    )


def test_v8_phase_a_trains_each_expert_but_not_the_signal_tail() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    from modeling.trifusion.signal_preserving_v8_builder import (
        build_signal_preserving_trifusion_v8_expert_formation,
    )

    build = build_signal_preserving_trifusion_v8_expert_formation(
        _FakeSignal(),
        signal_checkpoint_sha256="8" * 64,
        num_classes=2,
        feature_width=4,
        semantic_width=6,
        grid_size=(2, 2),
        branch_after_block=0,
        adapter_width=4,
        expert_modal_width=8,
        mamba_mixer_factory=TinySequenceMixer,
    )
    tail_before = {
        name: value.detach().clone()
        for name, value in build.model.baseline.signal.state_dict().items()
    }
    output = build.model(_batch(), return_aux=True)
    labels = torch.tensor([0, 0, 1, 1])
    losses = ExpertFormationV8Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
    )(output, labels)
    sum(losses.values()).backward()

    for role in EXPERT_ORDER:
        parameters = [
            parameter
            for name, parameter in build.model.encoder.named_parameters()
            if name.startswith(role) and parameter.requires_grad
        ]
        assert parameters
        assert any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        )
    assert all(
        not parameter.requires_grad for parameter in build.model.baseline.parameters()
    )
    assert all(
        torch.equal(value, tail_before[name])
        for name, value in build.model.baseline.signal.state_dict().items()
    )


def test_v8_mamba_exchanges_modalities_while_cnn_remains_spatial_local() -> None:
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.signal_preserving_v8_builder import (
        build_signal_preserving_trifusion_v8_expert_formation,
    )

    build = build_signal_preserving_trifusion_v8_expert_formation(
        _FakeSignal(),
        signal_checkpoint_sha256="8" * 64,
        num_classes=2,
        feature_width=4,
        semantic_width=6,
        grid_size=(2, 2),
        branch_after_block=0,
        adapter_width=4,
        expert_modal_width=8,
        mamba_mixer_factory=TinySequenceMixer,
    )
    build.model.eval()
    clean = build.model(_batch(), return_aux=True)
    changed = build.model(_batch(nir_shift=2.0), return_aux=True)

    assert torch.allclose(
        clean.modal_residual_embeddings["cnn"][:, 0],
        changed.modal_residual_embeddings["cnn"][:, 0],
    )
    assert not torch.allclose(
        clean.modal_residual_embeddings["mamba"][:, 0],
        changed.modal_residual_embeddings["mamba"][:, 0],
    )
