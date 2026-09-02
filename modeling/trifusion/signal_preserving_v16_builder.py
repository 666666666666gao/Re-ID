"""Build TriFusion V16 with training-only Signal-anchored triadic repair."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .builder import TriFusionBuildResult
from .experts.mamba import production_mamba_factory
from .signal_preserving_v8_builder import (
    build_signal_preserving_trifusion_v8_expert_formation,
)
from .signal_preserving_v16 import (
    V16_PROTECTION_THRESHOLD,
    V16_PROTECTION_TOLERANCE,
    V16_PROTECTION_WEIGHT,
    V16_RELATION_GAP,
    V16_REPAIR_WEIGHT,
)


def build_signal_preserving_trifusion_v16(
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
    base = build_signal_preserving_trifusion_v8_expert_formation(
        signal_model,
        signal_checkpoint_sha256=signal_checkpoint_sha256,
        num_classes=num_classes,
        feature_width=feature_width,
        semantic_width=semantic_width,
        grid_size=grid_size,
        branch_after_block=branch_after_block,
        adapter_width=adapter_width,
        expert_modal_width=expert_modal_width,
        scale_init=scale_init,
        gradient_checkpointing=gradient_checkpointing,
        mamba_mixer_factory=mamba_mixer_factory,
    )
    provenance = dict(base.provenance)
    provenance.update(
        {
            "architecture": "signal_anchored_triadic_repair_v16",
            "phase": "matched_endpoint_training",
            "training_collaboration": "signal_anchored_triadic_relation_repair",
            "inference_collaboration": "none",
            "new_trainable_inference_parameters": 0,
            "router_enabled": False,
            "hfer_enabled": False,
            "crde_enabled": False,
            "relation_gap": V16_RELATION_GAP,
            "protection_threshold": V16_PROTECTION_THRESHOLD,
            "protection_tolerance": V16_PROTECTION_TOLERANCE,
            "repair_weight": V16_REPAIR_WEIGHT,
            "protection_weight": V16_PROTECTION_WEIGHT,
        }
    )
    return TriFusionBuildResult(
        model=base.model,
        checkpoint_sha256=base.checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v16"]
