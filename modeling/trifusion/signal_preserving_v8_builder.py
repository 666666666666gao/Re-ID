"""Build the pretrained-tail expert-formation phase of TriFusion V8."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .builder import TriFusionBuildResult
from .experts.mamba import production_mamba_factory
from .signal_preserving_v8 import (
    ExpertFormationFusion,
    HierarchicalFrozenSignalBackbone,
    PretrainedTailTriExpertEncoder,
    SignalPreservingExpertFormationV8,
)
from .state import EXPERT_ORDER, MODALITY_ORDER


def build_signal_preserving_trifusion_v8_expert_formation(
    signal_model: nn.Module,
    *,
    signal_checkpoint_sha256: str,
    num_classes: int,
    feature_width: int = 512,
    semantic_width: int = 768,
    grid_size: tuple[int, int] = (16, 8),
    branch_after_block: int = 8,
    adapter_width: int = 128,
    expert_modal_width: int = 512,
    scale_init: float = 0.05,
    gradient_checkpointing: bool = True,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    if len(signal_checkpoint_sha256) != 64:
        raise ValueError("Signal checkpoint SHA-256 must have 64 hex characters")
    baseline = HierarchicalFrozenSignalBackbone(
        signal_model,
        feature_width=feature_width,
        branch_after_block=branch_after_block,
    )
    encoder = PretrainedTailTriExpertEncoder(
        tail_blocks=baseline.tail_blocks,
        tail_layer_indices=baseline.tail_layer_indices,
        semantic_width=semantic_width,
        grid_size=grid_size,
        adapter_width=adapter_width,
        expert_modal_width=expert_modal_width,
        mixer_factory=mamba_mixer_factory,
        scale_init=scale_init,
        gradient_checkpointing=gradient_checkpointing,
    )
    expert_width = len(MODALITY_ORDER) * int(expert_modal_width)
    fusion = ExpertFormationFusion(
        baseline_width=baseline.baseline_width,
        expert_width=expert_width,
    )
    model = SignalPreservingExpertFormationV8(
        baseline=baseline,
        encoder=encoder,
        fusion=fusion,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    provenance = {
        "architecture": "signal_preserving_collaborative_v8_expert_formation",
        "phase": "A_expert_formation",
        "signal_checkpoint_sha256": signal_checkpoint_sha256,
        "signal_baseline_width": baseline.baseline_width,
        "baseline_parameters_frozen": True,
        "branch_after_pretrained_clip_block": int(branch_after_block),
        "pretrained_tail_layers": list(baseline.tail_layer_indices),
        "pretrained_tail_shared_and_frozen": True,
        "expert_adapter_depths": [1, 1, 1],
        "expert_roles": {
            "cnn": "horizontal_local_detail",
            "transformer": "global_cls_relation",
            "mamba": "spatial_and_cross_modal_long_range",
        },
        "matched_residual": "expert_tail_output_minus_frozen_clip_tail_output",
        "experts": list(EXPERT_ORDER),
        "router_enabled": False,
        "hfer_enabled": False,
        "fusion_during_formation": "fixed_equal_energy_diagnostic_bank",
        "paper_hypotheses": [
            "pretrained_tail_rebranching_preserves_strong_semantics_for_all_experts",
            "role_disjoint_structured_heads_create_identity_complementarity",
            "staged_router_and_typed_exchange_are_deferred_until_experts_pass_oracle_gate",
        ],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=signal_checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v8_expert_formation"]
