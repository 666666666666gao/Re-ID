from __future__ import annotations

from types import MappingProxyType
import os
from pathlib import Path

import torch

from modeling.trifusion.state import (
    EXPERT_ORDER,
    ExpertState,
    ExpertStateMap,
    ReliabilityResult,
)


def _state_map(globals_by_expert: dict[str, torch.Tensor]) -> ExpertStateMap:
    first = globals_by_expert[EXPERT_ORDER[0]]
    mask = torch.ones(first.shape[:2], dtype=torch.bool, device=first.device)
    states = {}
    for expert in EXPERT_ORDER:
        values = globals_by_expert[expert]
        states[expert] = ExpertState(
            tokens=values.unsqueeze(2),
            global_embedding=values,
            private_embedding=values[..., :1],
            role_payload=MappingProxyType({"summary": values[..., :1]}),
            modality_mask=mask,
            stage=3,
            expert=expert,
        )
    reliability = ReliabilityResult(
        alpha=torch.full((first.shape[0], 3, 3), 2.0, device=first.device),
        beta=torch.full((first.shape[0], 3, 3), 2.0, device=first.device),
        r=torch.full((first.shape[0], 3, 3), 0.5, device=first.device),
        u=torch.full((first.shape[0], 3, 3), 0.5, device=first.device),
        modality_mask=mask,
    )
    return ExpertStateMap(states, modality_mask=mask, reliability=reliability)


def test_task_anchor_tokenizer_keeps_exact_cls_and_centered_local_field() -> None:
    from modeling.trifusion.task_anchor_v3 import TaskAdaptedAnchorTokenizer

    projection = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)
    torch.nn.init.zeros_(projection.weight)
    tokenizer = TaskAdaptedAnchorTokenizer(
        patch_projection=projection,
        positional_embedding=torch.nn.Parameter(torch.zeros(5, 4)),
        class_embedding=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        pre_norm=torch.nn.Identity(),
        post_norm=torch.nn.Identity(),
        shared_blocks=[torch.nn.Identity()],
        output_projection=torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        ),
        gradient_checkpointing=False,
    )
    images = torch.randn(2, 3, 4, 4)
    modalities = torch.tensor([0, 2], dtype=torch.long)
    field = tokenizer(images, modalities)

    expected_native = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(2, -1)
    expected_projected = torch.tensor([[1.0, 2.0]]).expand(2, -1)
    assert torch.equal(field.anchor_native, expected_native)
    assert torch.equal(field.anchor_projected, expected_projected)
    assert tuple(field.expert_tokens) == EXPERT_ORDER
    for tokens in field.expert_tokens.values():
        assert torch.equal(tokens.mean(dim=1), expected_native)


def test_zero_residual_mode_is_distance_exact_and_expert_invariant() -> None:
    from modeling.trifusion.task_anchor_v3 import AnchorResidualCollaborativeFusion

    torch.manual_seed(7)
    anchor_native = torch.randn(3, 3, 4)
    anchor_projected = torch.randn(3, 3, 2)
    first_states = _state_map(
        {expert: anchor_native + torch.randn_like(anchor_native) for expert in EXPERT_ORDER}
    )
    second_states = _state_map(
        {expert: anchor_native + 100.0 * torch.randn_like(anchor_native) for expert in EXPERT_ORDER}
    )
    fusion = AnchorResidualCollaborativeFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        embedding_width=2,
        residual_scale_init=0.25,
    )

    first = fusion(
        first_states,
        first_states.reliability,
        first_states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
        force_zero_residual=True,
    )
    second = fusion(
        second_states,
        second_states.reliability,
        second_states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
        force_zero_residual=True,
    )

    anchor = anchor_projected.flatten(1)
    expected = torch.cat((anchor, torch.zeros_like(anchor)), dim=1)
    assert torch.equal(first.fused_embedding, expected)
    assert torch.equal(second.fused_embedding, expected)
    anchor_distance = torch.cdist(torch.nn.functional.normalize(anchor, dim=1),
                                  torch.nn.functional.normalize(anchor, dim=1))
    fused_distance = torch.cdist(
        torch.nn.functional.normalize(first.fused_embedding, dim=1),
        torch.nn.functional.normalize(first.fused_embedding, dim=1),
    )
    assert torch.allclose(fused_distance, anchor_distance, atol=1e-6, rtol=1e-6)


def test_default_residual_fusion_backpropagates_to_all_three_experts() -> None:
    from modeling.trifusion.task_anchor_v3 import AnchorResidualCollaborativeFusion

    torch.manual_seed(11)
    anchor_native = torch.randn(2, 3, 4)
    anchor_projected = torch.randn(2, 3, 2)
    states = _state_map(
        {expert: anchor_native + torch.randn_like(anchor_native) for expert in EXPERT_ORDER}
    )
    fusion = AnchorResidualCollaborativeFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        embedding_width=2,
        residual_scale_init=0.25,
    )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
    )
    result.fused_embedding[:, anchor_projected.numel() // anchor_projected.shape[0] :].square().sum().backward()

    for expert in EXPERT_ORDER:
        gradient = fusion.residual_projections[expert].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_each_expert_residual_is_norm_bounded_by_its_anchor() -> None:
    from modeling.trifusion.task_anchor_v3 import AnchorResidualCollaborativeFusion

    torch.manual_seed(13)
    anchor_native = torch.randn(2, 3, 4)
    anchor_projected = torch.randn(2, 3, 2)
    states = _state_map(
        {
            expert: anchor_native + 1_000.0 * torch.randn_like(anchor_native)
            for expert in EXPERT_ORDER
        }
    )
    fusion = AnchorResidualCollaborativeFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        embedding_width=2,
        residual_scale_init=0.25,
    )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
    )

    anchor_norm = anchor_projected.norm(dim=-1)
    scales = torch.sigmoid(fusion.residual_scale_logits)
    residual_norm = result.contribution_embeddings.norm(dim=-1)
    bound = scales.view(1, -1, 1) * anchor_norm[:, None]
    assert torch.all(residual_norm <= bound + 1e-6)


def test_supervised_cross_modal_alignment_rewards_identity_consistency() -> None:
    from modeling.trifusion.task_anchor_v3 import supervised_cross_modal_alignment

    labels = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4, 3, dtype=torch.bool)
    prototypes = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    aligned = torch.stack((prototypes, prototypes, prototypes), dim=1)
    misaligned = aligned.clone()
    misaligned[:, 1] = misaligned[torch.tensor([2, 3, 0, 1]), 1]

    good = supervised_cross_modal_alignment(aligned, labels, mask, temperature=0.1)
    bad = supervised_cross_modal_alignment(misaligned, labels, mask, temperature=0.1)

    assert torch.isfinite(good)
    assert good < bad


def test_real_clip_builder_exposes_task_anchor_and_three_collaborative_experts() -> None:
    checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
    if not checkpoint_value:
        import pytest

        pytest.skip("TRIFUSION_CLIP_CHECKPOINT is not configured")
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.task_anchor_v3_builder import (
        build_task_anchored_trifusion_v3_from_clip,
    )

    result = build_task_anchored_trifusion_v3_from_clip(
        Path(checkpoint_value),
        num_classes=141,
        architecture="task_anchored_collaborative_v3",
        reliability_mode="joint_beta",
        mamba_mixer_factory=TinySequenceMixer,
    )
    model = result.model

    assert result.provenance["architecture"] == "task_anchored_collaborative_v3"
    assert result.provenance["anchor_path"] == "exact_projected_clip_cls_concat_rgb_ni_ti"
    assert result.provenance["reliability_refresh_stages"] == [1, 2, 3]
    assert result.provenance["loss_slot_contract"] == {
        "reliability": "anchor_id_quarter_plus_anchor_triplet",
        "private_diversity": "supervised_cross_modal_identity_alignment",
    }
    assert result.provenance["signal_license"] == "MIT"
    assert model.anchor_embedding_width == 1536
    assert model.fused_embedding_width == 3072
    assert tuple(model.encoder.experts) == EXPERT_ORDER
    assert result.provenance["parameter_budget_pass"]
