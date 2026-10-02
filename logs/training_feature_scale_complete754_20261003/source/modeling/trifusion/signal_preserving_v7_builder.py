"""Build geometry-aligned marginal-gain-routed V7 around exact Signal."""

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
from .signal_preserving_v5 import FrozenSignalBackbone
from .signal_preserving_v7 import (
    HierarchicalBoundedResidualBankFusion,
    SignalPreservingCollaborativeReIDV7,
)
from .state import EXPERT_ORDER
from .task_anchor_v3 import TaskAnchoredTriBranchEncoder


def build_signal_preserving_trifusion_v7(
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
    alpha_max: float = 0.5,
    alpha_init: float = 0.2,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    if len(signal_checkpoint_sha256) != 64:
        raise ValueError("Signal checkpoint SHA-256 must have 64 hex characters")
    stage_depths = (1, 1, 1)
    baseline = FrozenSignalBackbone(signal_model, feature_width=feature_width)
    experts = {
        "cnn": SemanticCNNExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=1e-3,
        ),
        "transformer": SemanticTransformerExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=1e-3,
        ),
        "mamba": SemanticMambaExpert(
            width=feature_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
            scale_init=1e-3,
        ),
    }
    for expert in experts.values():
        for parameter in expert.private_projection.parameters():
            parameter.requires_grad_(False)
    expert_widths = {expert: feature_width for expert in EXPERT_ORDER}
    encoder = TaskAnchoredTriBranchEncoder(
        experts,
        reliability_gate=ReliabilityPosterior(
            expert_widths=expert_widths,
            hidden_width=reliability_hidden_width,
            heads=4,
            kappa_min=2.0,
        ),
        collaborator=HeterogeneousRelay(
            expert_widths=expert_widths,
            relay_rank=relay_rank,
            token_grid=grid_size,
            gamma_init=0.05,
        ),
    )
    fusion = HierarchicalBoundedResidualBankFusion(
        expert_widths=expert_widths,
        baseline_width=baseline.baseline_width,
        residual_width=residual_width,
        alpha_max=alpha_max,
        alpha_init=alpha_init,
    )
    model = SignalPreservingCollaborativeReIDV7(
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
        "architecture": "signal_preserving_collaborative_v7",
        "signal_checkpoint_sha256": signal_checkpoint_sha256,
        "signal_baseline_width": baseline.baseline_width,
        "baseline_parameters_frozen": True,
        "shared_triplet_geometry": True,
        "matched_token_residual": True,
        "hierarchical_router": "P(modality)*P(expert|modality)",
        "bounded_sample_alpha": {"max": alpha_max, "init": alpha_init},
        "router_utility_source": "per_expert_modality_marginal_identity_gain",
        "quality_supervision": "controlled_modality_degradation",
        "experts": list(EXPERT_ORDER),
        "expert_stage_depths": list(stage_depths),
        "relay_stages": [1, 2],
        "reliability_refresh_stages": [1, 2, 3],
        "paper_contributions": [
            "geometry_aligned_matched_residual_expertization",
            "stagewise_bidirectional_heterogeneous_feature_exchange",
            "hierarchical_marginal_gain_routing_with_bounded_energy",
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


__all__ = ["build_signal_preserving_trifusion_v7"]
