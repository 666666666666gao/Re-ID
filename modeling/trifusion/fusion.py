"""Uncertainty-aware collaborative fusion over all expert-modality entries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from .state import EXPERT_ORDER, ExpertStateMap, ReliabilityResult


@dataclass(frozen=True, eq=False)
class FusionResult:
    fused_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    weights: torch.Tensor
    modality_mask: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_embeddings",
            MappingProxyType(dict(self.branch_embeddings)),
        )


class CollaborativeFusion(nn.Module):
    """Project, hard-mask and reliability-normalize nine contributions."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        embedding_width: int = 512,
        residual_floor: float = 0.0,
        use_uncertainty_multiplier: bool = False,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        if embedding_width <= 0 or residual_floor < 0:
            raise ValueError("embedding_width must be positive and floor nonnegative")
        self.residual_floor = float(residual_floor)
        self.use_uncertainty_multiplier = use_uncertainty_multiplier
        self.contribution_projections = nn.ModuleDict(
            {
                expert: nn.Linear(
                    expert_widths[expert], embedding_width, bias=False
                )
                for expert in EXPERT_ORDER
            }
        )
        self.branch_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(embedding_width) for expert in EXPERT_ORDER}
        )
        self.fused_norm = nn.LayerNorm(embedding_width)

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
    ) -> FusionResult:
        if not torch.equal(states.modality_mask, modality_mask):
            raise ValueError("states and fusion must use the same modality mask")
        if not torch.equal(reliability.modality_mask, modality_mask):
            raise ValueError("reliability and fusion must use the same modality mask")

        valid = modality_mask[:, None, :].expand_as(reliability.r)
        valid_float = valid.to(dtype=reliability.r.dtype)
        contributions = torch.stack(
            [
                self.contribution_projections[expert](
                    states[expert].global_embedding
                )
                for expert in EXPERT_ORDER
            ],
            dim=1,
        )
        contributions = contributions * valid_float[..., None]

        scores = reliability.r
        if self.use_uncertainty_multiplier:
            scores = scores * (1.0 - reliability.u)
        scores = (scores + self.residual_floor) * valid_float
        total = scores.sum(dim=(1, 2), keepdim=True)
        fallback = valid_float / valid_float.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1.0)
        weights = torch.where(total > 0, scores / total.clamp_min(1e-12), fallback)
        weights = weights * valid_float
        fused_embedding = self.fused_norm(
            (contributions * weights[..., None]).sum(dim=(1, 2))
        )

        branch_embeddings = {}
        for expert_index, expert in enumerate(EXPERT_ORDER):
            branch_scores = scores[:, expert_index]
            branch_total = branch_scores.sum(dim=1, keepdim=True)
            modality_fallback = valid_float[:, expert_index] / valid_float[
                :, expert_index
            ].sum(dim=1, keepdim=True).clamp_min(1.0)
            branch_weights = torch.where(
                branch_total > 0,
                branch_scores / branch_total.clamp_min(1e-12),
                modality_fallback,
            )
            branch_embeddings[expert] = self.branch_norms[expert](
                (
                    contributions[:, expert_index]
                    * branch_weights[..., None]
                ).sum(dim=1)
            )

        return FusionResult(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=contributions,
            weights=weights,
            modality_mask=modality_mask,
        )


__all__ = ["CollaborativeFusion", "FusionResult"]
