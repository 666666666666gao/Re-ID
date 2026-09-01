"""Frozen-expert hierarchical routing for TriFusion V8 Phase B."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .state import EXPERT_ORDER, MODALITY_ORDER


@dataclass(frozen=True, eq=False)
class OOFMarginRouterOutput:
    weights: torch.Tensor
    modal_probabilities: torch.Tensor
    expert_probabilities: torch.Tensor
    alpha: torch.Tensor


@dataclass(frozen=True, eq=False)
class OOFMarginFusionOutput:
    fused_embedding: torch.Tensor


@dataclass(frozen=True, eq=False)
class OOFMarginRouterLoss:
    total: torch.Tensor
    utility: torch.Tensor
    alpha: torch.Tensor
    target_weights: torch.Tensor
    alpha_target: torch.Tensor


def oof_margin_router_loss(
    output: OOFMarginRouterOutput,
    target_identity_margin: torch.Tensor,
    modality_mask: torch.Tensor,
    *,
    alpha_max: float,
    utility_temperature: float,
    alpha_gain_scale: float,
) -> OOFMarginRouterLoss:
    """Fit hierarchical routing and bounded energy to continuous OOF margins."""

    if target_identity_margin.shape != output.weights.shape:
        raise ValueError("target margins must match router weights")
    if modality_mask.dtype != torch.bool or modality_mask.shape != (
        output.weights.shape[0],
        len(MODALITY_ORDER),
    ):
        raise ValueError("modality_mask must have bool shape B,M")
    if utility_temperature <= 0.0 or alpha_gain_scale <= 0.0:
        raise ValueError("margin loss scales must be positive")
    valid_slots = modality_mask[:, None].expand_as(output.weights)
    target_logits = (target_identity_margin / float(utility_temperature)).masked_fill(
        ~valid_slots,
        -torch.inf,
    )
    target_weights = torch.softmax(target_logits.flatten(1), dim=1).reshape_as(
        output.weights
    )
    predicted_weights = output.weights * valid_slots.to(output.weights.dtype)
    predicted_weights = predicted_weights / predicted_weights.sum(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1e-12)
    utility = F.kl_div(
        predicted_weights.flatten(1).clamp_min(1e-12).log(),
        target_weights.flatten(1),
        reduction="batchmean",
    )
    best_margin = target_identity_margin.masked_fill(
        ~valid_slots,
        -torch.inf,
    ).flatten(1).max(dim=1).values
    positive_margin = best_margin.clamp_min(0.0)
    alpha_target = (
        float(alpha_max)
        * positive_margin
        / (positive_margin + float(alpha_gain_scale))
    ).unsqueeze(1)
    alpha = F.mse_loss(output.alpha, alpha_target)
    return OOFMarginRouterLoss(
        total=utility + alpha,
        utility=utility,
        alpha=alpha,
        target_weights=target_weights,
        alpha_target=alpha_target,
    )


def modality_quality_loss(
    modal_probabilities: torch.Tensor,
    modality_quality: torch.Tensor,
    modality_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise modality mass from controlled corruption quality labels."""

    if modal_probabilities.shape != modality_quality.shape:
        raise ValueError("modal probabilities and quality targets must match")
    if modality_mask.dtype != torch.bool or modality_mask.shape != modality_quality.shape:
        raise ValueError("modality_mask must match quality targets")
    valid = modality_mask.to(modal_probabilities.dtype)
    target = modality_quality * valid
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    predicted = modal_probabilities * valid
    predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(
        predicted.clamp_min(1e-12).log(),
        target,
        reduction="batchmean",
    )


class OOFMarginRoutedFusion(nn.Module):
    """Append one bounded, jointly routed residual bank to exact Signal."""

    def __init__(
        self,
        *,
        baseline_width: int,
        residual_width: int,
        alpha_max: float,
    ) -> None:
        super().__init__()
        if baseline_width <= 0 or residual_width <= 0:
            raise ValueError("fusion widths must be positive")
        if not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must lie in (0,1]")
        self.baseline_width = int(baseline_width)
        self.residual_width = int(residual_width)
        self.alpha_max = float(alpha_max)
        self.residual_bank_width = (
            len(EXPERT_ORDER) * len(MODALITY_ORDER) * self.residual_width
        )
        self.fused_embedding_width = self.baseline_width + self.residual_bank_width

    def forward(
        self,
        baseline_embedding: torch.Tensor,
        modal_residual: torch.Tensor,
        routing: OOFMarginRouterOutput,
    ) -> OOFMarginFusionOutput:
        batch_size = baseline_embedding.shape[0]
        if baseline_embedding.shape != (batch_size, self.baseline_width):
            raise ValueError("baseline embedding has the wrong shape")
        if modal_residual.shape != (
            batch_size,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
            self.residual_width,
        ):
            raise ValueError("modal_residual must have shape B,E,M,R")
        if routing.weights.shape != modal_residual.shape[:3]:
            raise ValueError("router weights do not match residual slots")
        if routing.alpha.shape != (batch_size, 1):
            raise ValueError("router alpha must have shape B,1")
        if bool((routing.alpha > self.alpha_max).any()):
            raise ValueError("router alpha exceeds the fusion bound")

        routed = (modal_residual * routing.weights[..., None]).flatten(1)
        baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
        activated = F.normalize(routed, dim=1) * baseline_norm * routing.alpha
        return OOFMarginFusionOutput(
            fused_embedding=torch.cat((baseline_embedding, activated), dim=1)
        )


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


__all__ = [
    "HierarchicalOOFMarginRouter",
    "OOFMarginFusionOutput",
    "OOFMarginRoutedFusion",
    "OOFMarginRouterLoss",
    "OOFMarginRouterOutput",
    "modality_quality_loss",
    "oof_margin_router_loss",
]
