"""Build TriFusion V9 from the frozen V8 Phase-A-plus-Router checkpoint."""

from __future__ import annotations

from torch import nn

from .builder import TriFusionBuildResult
from .signal_preserving_v9 import (
    OrthogonalTriadicSynthesis,
    SignalPreservingCollaborativeV9,
)
from .state import EXPERT_ORDER


def build_signal_preserving_trifusion_v9(
    phase_a_model: nn.Module,
    router: nn.Module,
    phase_b_fusion: nn.Module,
    *,
    combined_checkpoint_sha256: str,
    num_classes: int,
    baseline_width: int,
    phase_b_width: int,
    residual_width: int,
    hidden_width: int,
    synergy_modal_width: int,
    relay_depth: int,
    beta_max: float,
    beta_init: float,
) -> TriFusionBuildResult:
    if len(combined_checkpoint_sha256) != 64:
        raise ValueError("combined checkpoint SHA-256 must have 64 characters")
    synthesis = OrthogonalTriadicSynthesis(
        baseline_width=baseline_width,
        prefix_width=phase_b_width,
        residual_width=residual_width,
        hidden_width=hidden_width,
        synergy_modal_width=synergy_modal_width,
        relay_depth=relay_depth,
        beta_max=beta_max,
        beta_init=beta_init,
    )
    model = SignalPreservingCollaborativeV9(
        phase_a=phase_a_model,
        router=router,
        phase_b_fusion=phase_b_fusion,
        synthesis=synthesis,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    provenance = {
        "architecture": (
            "signal_preserving_collaborative_v9_orthogonal_triadic_synthesis"
        ),
        "combined_v8_checkpoint_sha256": combined_checkpoint_sha256,
        "exact_signal_frozen": True,
        "phase_a_experts_frozen": True,
        "phase_b_router_frozen": True,
        "phase_b_embedding_exact_prefix": True,
        "experts": list(EXPERT_ORDER),
        "relay_depth": int(relay_depth),
        "hidden_width": int(hidden_width),
        "synergy_modal_width": int(synergy_modal_width),
        "beta_max": float(beta_max),
        "paper_mechanisms": [
            "pretrained_tail_role_disjoint_experts",
            "receiver_specific_orthogonal_peer_relay",
            "quality_aware_triadic_interaction_synthesis",
        ],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=combined_checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v9"]
