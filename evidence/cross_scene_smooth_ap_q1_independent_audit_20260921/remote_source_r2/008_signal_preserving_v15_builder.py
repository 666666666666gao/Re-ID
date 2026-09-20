"""Build Counterfactual Role-Delta Exchange over frozen V8 experts."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .builder import TriFusionBuildResult
from .experts.mamba import production_mamba_factory
from .signal_preserving_v8 import (
    ExpertFormationFusion,
    HierarchicalFrozenSignalBackbone,
)
from .signal_preserving_v15 import (
    CollaborativeTailTriExpertEncoderV15,
    SignalPreservingCollaborativeV15,
    V15_REGRET_WEIGHT,
)
from .state import EXPERT_ORDER, MODALITY_ORDER


def build_signal_preserving_trifusion_v15(
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
    exchange_rank: int = 64,
    edge_scale_max: float = 0.25,
    regret_weight: float = 1.0,
    gradient_checkpointing: bool = True,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    if len(signal_checkpoint_sha256) != 64:
        raise ValueError("Signal checkpoint SHA-256 must have 64 hex characters")
    if float(regret_weight) != V15_REGRET_WEIGHT:
        raise ValueError("V15 regret weight is frozen to 1.0")
    baseline = HierarchicalFrozenSignalBackbone(
        signal_model,
        feature_width=feature_width,
        branch_after_block=branch_after_block,
    )
    encoder = CollaborativeTailTriExpertEncoderV15(
        tail_blocks=baseline.tail_blocks,
        tail_layer_indices=baseline.tail_layer_indices,
        semantic_width=semantic_width,
        grid_size=grid_size,
        adapter_width=adapter_width,
        expert_modal_width=expert_modal_width,
        mixer_factory=mamba_mixer_factory,
        exchange_rank=exchange_rank,
        edge_scale_max=edge_scale_max,
        scale_init=scale_init,
        gradient_checkpointing=gradient_checkpointing,
    )
    expert_width = len(MODALITY_ORDER) * int(expert_modal_width)
    fusion = ExpertFormationFusion(
        baseline_width=baseline.baseline_width,
        expert_width=expert_width,
    )
    model = SignalPreservingCollaborativeV15(
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
        "architecture": "signal_preserving_collaborative_v15_crde",
        "phase": "counterfactual_role_delta_exchange",
        "signal_checkpoint_sha256": signal_checkpoint_sha256,
        "signal_baseline_width": baseline.baseline_width,
        "baseline_parameters_frozen": True,
        "base_experts_frozen": True,
        "branch_after_pretrained_clip_block": int(branch_after_block),
        "pretrained_tail_layers": list(baseline.tail_layer_indices),
        "exchange_after_tail_layers": list(baseline.tail_layer_indices[:-1]),
        "exchange_count": 2,
        "exchange_rank": int(exchange_rank),
        "edge_scale_max": float(edge_scale_max),
        "edge_scale_init": 0.0,
        "regret_weight": V15_REGRET_WEIGHT,
        "experts": list(EXPERT_ORDER),
        "router_enabled": False,
        "late_fusion_mlp_enabled": False,
        "final_fusion": "exact_signal_plus_equal_energy_normalized_residual_bank",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=signal_checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v15"]
