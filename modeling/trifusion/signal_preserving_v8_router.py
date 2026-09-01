"""Frozen-expert hierarchical routing for TriFusion V8 Phase B."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .state import EXPERT_ORDER, MODALITY_ORDER


@dataclass(frozen=True, eq=False)
class OOFMarginRouterOutput:
    weights: torch.Tensor
    modal_probabilities: torch.Tensor
    expert_probabilities: torch.Tensor
    alpha: torch.Tensor


class HierarchicalOOFMarginRouter(nn.Module):
    """Predict P(modality) P(expert|modality) from frozen V8 features."""

    def __init__(
        self,
        *,
        direct_width: int,
        residual_width: int,
        hidden_width: int,
        alpha_max: float,
        alpha_init: float,
    ) -> None:
        super().__init__()
        if direct_width <= 0 or residual_width <= 0 or hidden_width <= 0:
            raise ValueError("router widths must be positive")
        if not 0.0 < alpha_init < alpha_max <= 1.0:
            raise ValueError("alpha requires 0 < init < max <= 1")
        self.direct_width = int(direct_width)
        self.residual_width = int(residual_width)
        self.hidden_width = int(hidden_width)
        self.alpha_max = float(alpha_max)
        self.direct_projection = nn.Sequential(
            nn.LayerNorm(self.direct_width),
            nn.Linear(self.direct_width, self.hidden_width, bias=False),
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(self.residual_width),
            nn.Linear(self.residual_width, self.hidden_width, bias=False),
        )
        self.modal_head = nn.Sequential(
            nn.Linear(3 * self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )
        self.expert_head = nn.Sequential(
            nn.Linear(3 * self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(
            self.alpha_head[-1].bias,
            math.log((alpha_init / alpha_max) / (1.0 - alpha_init / alpha_max)),
        )

    def forward(
        self,
        direct_modal: torch.Tensor,
        modal_residual: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> OOFMarginRouterOutput:
        batch_size = direct_modal.shape[0]
        if direct_modal.shape != (
            batch_size,
            len(MODALITY_ORDER),
            self.direct_width,
        ):
            raise ValueError("direct_modal must have shape B,M,D")
        if modal_residual.shape != (
            batch_size,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
            self.residual_width,
        ):
            raise ValueError("modal_residual must have shape B,E,M,R")
        if modality_mask.dtype != torch.bool or modality_mask.shape != (
            batch_size,
            len(MODALITY_ORDER),
        ):
            raise ValueError("modality_mask must have bool shape B,M")
        if bool((~modality_mask).all(dim=1).any()):
            raise ValueError("router input cannot have an all-missing row")

        direct = self.direct_projection(direct_modal)
        residual = self.residual_projection(modal_residual)
        valid = modality_mask[..., None].to(direct.dtype)
        context = (direct * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        context_by_modality = context[:, None].expand(-1, len(MODALITY_ORDER), -1)
        modal_features = torch.cat(
            (direct, residual.mean(dim=1), context_by_modality),
            dim=-1,
        )
        modal_logits = self.modal_head(modal_features).squeeze(-1)
        modal_logits = modal_logits.masked_fill(~modality_mask, -torch.inf)
        modal_probabilities = torch.softmax(modal_logits, dim=1)

        direct_by_expert = direct[:, None].expand(-1, len(EXPERT_ORDER), -1, -1)
        context_by_slot = context[:, None, None].expand(
            -1,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
            -1,
        )
        expert_features = torch.cat(
            (direct_by_expert, residual, context_by_slot),
            dim=-1,
        )
        expert_logits = self.expert_head(expert_features).squeeze(-1)
        expert_probabilities = torch.softmax(expert_logits, dim=1)
        weights = expert_probabilities * modal_probabilities[:, None]

        direct_context = (direct * modal_probabilities[..., None]).sum(dim=1)
        residual_context = (residual * weights[..., None]).sum(dim=(1, 2))
        alpha = self.alpha_max * torch.sigmoid(
            self.alpha_head(torch.cat((direct_context, residual_context), dim=1))
        )
        return OOFMarginRouterOutput(
            weights=weights,
            modal_probabilities=modal_probabilities,
            expert_probabilities=expert_probabilities,
            alpha=alpha,
        )


__all__ = ["HierarchicalOOFMarginRouter", "OOFMarginRouterOutput"]
