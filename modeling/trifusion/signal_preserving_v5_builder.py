"""Build V5 around a strictly loaded, frozen Signal checkpoint."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .builder import TriFusionBuildResult
from .experts.mamba import production_mamba_factory
from .experts.semantic_residual import (
    SemanticCNNExpert,
    SemanticMambaExpert,
    SemanticTransformerExpert,
)
from .reliability import ReliabilityPosterior
from .relay import HeterogeneousRelay
from .signal_preserving_v5 import (
    FrozenSignalBackbone,
    SignalPreservingCollaborativeReID,
    SignalPreservingResidualBankFusion,
)
from .state import EXPERT_ORDER
from .task_anchor_v3 import TaskAnchoredTriBranchEncoder


def build_signal_preserving_trifusion_v5(
    signal_model: nn.Module,
    *,
    signal_checkpoint_sha256: str,
    num_classes: int,
    feature_width: int = 512,
    grid_size: tuple[int, int] = (16, 8),
    adapter_width: int = 128,
    residual_width: int = 256,
    relay_rank: int = 64,
    private_width: int = 64,
    reliability_hidden_width: int = 128,
    residual_scale_init: float = 0.1,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    """Attach three collaborative residual experts to the exact Signal feature."""

    if len(signal_checkpoint_sha256) != 64:
        raise ValueError("Signal checkpoint SHA-256 must have 64 hex characters")
    stage_depths = (1, 1, 1)
    scale_init = 1e-3
    baseline = FrozenSignalBackbone(signal_model, feature_width=feature_width)
    experts = {
        "cnn": SemanticCNNExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=scale_init,
        ),
        "transformer": SemanticTransformerExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=scale_init,
        ),
        "mamba": SemanticMambaExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
            scale_init=scale_init,
        ),
    }
    expert_widths = {expert: feature_width for expert in EXPERT_ORDER}
    reliability = ReliabilityPosterior(
        expert_widths=expert_widths,
        hidden_width=reliability_hidden_width,
        heads=4,
        kappa_min=2.0,
    )
    relay = HeterogeneousRelay(
        expert_widths=expert_widths,
        relay_rank=relay_rank,
        token_grid=grid_size,
        gamma_init=0.05,
    )
    encoder = TaskAnchoredTriBranchEncoder(
        experts,
        reliability_gate=reliability,
        collaborator=relay,
    )
    fusion = SignalPreservingResidualBankFusion(
        expert_widths=expert_widths,
        baseline_width=baseline.baseline_width,
        residual_width=residual_width,
        residual_scale_init=residual_scale_init,
    )
    model = SignalPreservingCollaborativeReID(
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
        "architecture": "signal_preserving_collaborative_v5",
        "signal_checkpoint_sha256": signal_checkpoint_sha256,
        "signal_baseline_width": baseline.baseline_width,
        "signal_feature": "direct_3x512_plus_SIM_3x512_with_camera_SIE",
        "baseline_parameters_frozen": True,
        "experts": list(EXPERT_ORDER),
        "expert_stage_depths": list(stage_depths),
        "relay_stages": [1, 2],
        "reliability_refresh_stages": [1, 2, 3],
        "fusion": "exact_signal_prefix_plus_quality_routed_tri_expert_residual_bank",
        "branch_embedding_width": fusion.branch_embedding_width,
        "residual_bank_width": fusion.residual_bank_width,
        "fused_embedding_width": fusion.fused_embedding_width,
        "residual_scale_init": residual_scale_init,
        "paper_contributions": [
            "signal_preserving_shared_semantic_expertization",
            "stagewise_bidirectional_heterogeneous_feature_exchange",
            "identity_utility_quality_routed_non_destructive_residual_bank",
        ],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_budget_pass": total_parameters <= 120_000_000,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=signal_checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_signal_preserving_trifusion_v5"]
