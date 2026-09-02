"""Build V17 DTRED over one hash-bound frozen V8 endpoint."""

from __future__ import annotations

from torch import nn

from .builder import TriFusionBuildResult
from .signal_preserving_v17 import (
    SignalPreservingCollaborativeV17,
    V17_ENVELOPE_WEIGHT,
    V17_PROTECTION_THRESHOLD,
    V17_PROTECTION_TOLERANCE,
    V17_PROTECTION_WEIGHT,
)
from .state import EXPERT_ORDER


def build_signal_preserving_trifusion_v17(
    base_v8: nn.Module,
    *,
    signal_checkpoint_sha256: str,
    v8_checkpoint_sha256: str,
    num_classes: int,
    adapter_width: int = 256,
) -> TriFusionBuildResult:
    if len(signal_checkpoint_sha256) != 64:
        raise ValueError("Signal checkpoint SHA-256 must have 64 hex characters")
    if len(v8_checkpoint_sha256) != 64:
        raise ValueError("V8 checkpoint SHA-256 must have 64 hex characters")

    model = SignalPreservingCollaborativeV17(
        base_v8=base_v8,
        num_classes=num_classes,
        adapter_width=adapter_width,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    provenance = {
        "architecture": "signal_preserving_v17_dtred",
        "phase": "dense_triadic_relation_envelope_distillation",
        "signal_checkpoint_sha256": signal_checkpoint_sha256,
        "v8_checkpoint_sha256": v8_checkpoint_sha256,
        "base_v8_frozen": True,
        "adapter_width": int(adapter_width),
        "experts": list(EXPERT_ORDER),
        "triadic_interaction": "receiver_plus_peer_hadamard_intersection_plus_mean",
        "relation_envelope_weight": V17_ENVELOPE_WEIGHT,
        "signal_protection_weight": V17_PROTECTION_WEIGHT,
        "signal_protection_threshold": V17_PROTECTION_THRESHOLD,
        "signal_protection_tolerance": V17_PROTECTION_TOLERANCE,
        "router_enabled": False,
        "runtime_fallback_enabled": False,
        "reranking_enabled": False,
        "final_fusion": "exact_signal_plus_equal_energy_corrected_residual_bank",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=v8_checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v17"]
