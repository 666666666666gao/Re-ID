"""Geometry-aligned, marginal-gain-routed Signal-preserving TriFusion V7."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import _batch_hard_triplet
from .signal_preserving_v5 import FrozenSignalBackbone
from .state import EXPERT_ORDER, MODALITY_ORDER, ExpertStateMap, ReliabilityResult
from .task_anchor_v4 import batch_hard_identity_gap_per_sample


@dataclass(frozen=True, eq=False)
class HierarchicalBoundedFusionResult:
    fused_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    weights: torch.Tensor
    modal_probabilities: torch.Tensor
    expert_probabilities: torch.Tensor
    alpha: torch.Tensor
    modality_mask: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_embeddings", MappingProxyType(dict(self.branch_embeddings))
        )
        object.__setattr__(
            self,
            "residual_embeddings",
            MappingProxyType(dict(self.residual_embeddings)),
        )


@dataclass(frozen=True, eq=False)
class MarginalGainRoutingResult:
    router_loss: torch.Tensor
    alpha_loss: torch.Tensor
    target_weights: torch.Tensor
    predicted_weights: torch.Tensor
    utilities: torch.Tensor
    alpha_target: torch.Tensor
    valid_samples: torch.Tensor


def marginal_gain_router_loss(
    baseline_embedding: torch.Tensor,
    contributions: torch.Tensor,
    router_weights: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    *,
    modality_mask: torch.Tensor,
    alpha_max: float,
    utility_temperature: float,
    alpha_gain_scale: float,
) -> MarginalGainRoutingResult:
    """Supervise every expert-modality slot by gain over the exact baseline."""

    batch_size = baseline_embedding.shape[0]
    expected = (batch_size, len(EXPERT_ORDER), len(MODALITY_ORDER))
    if contributions.shape[:3] != expected or router_weights.shape != expected:
        raise ValueError("contributions and router weights must have shape B,E,M")
    if alpha.shape != (batch_size, 1) or labels.shape != (batch_size,):
        raise ValueError("alpha and labels have incompatible shapes")
    baseline_gap, baseline_valid = batch_hard_identity_gap_per_sample(
        F.normalize(baseline_embedding, dim=1), labels
    )
    baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
    slot_scale = baseline_norm * float(alpha_max)
    utilities = []
    validities = []
    for expert_index in range(len(EXPERT_ORDER)):
        expert_utilities = []
        expert_validities = []
        for modality_index in range(len(MODALITY_ORDER)):
            candidate = torch.cat(
                (
                    baseline_embedding,
                    contributions[:, expert_index, modality_index] * slot_scale,
                ),
                dim=1,
            )
            candidate_gap, candidate_valid = batch_hard_identity_gap_per_sample(
                F.normalize(candidate, dim=1), labels
            )
            expert_utilities.append(candidate_gap - baseline_gap)
            expert_validities.append(candidate_valid & baseline_valid)
        utilities.append(torch.stack(expert_utilities, dim=1))
        validities.append(torch.stack(expert_validities, dim=1))
    utility_tensor = torch.stack(utilities, dim=1).detach()
    validity_tensor = torch.stack(validities, dim=1)
    valid_slots = modality_mask[:, None].expand_as(router_weights) & validity_tensor
    logits = (utility_tensor / float(utility_temperature)).masked_fill(
        ~valid_slots, -torch.inf
    )
    target_weights = F.softmax(logits.flatten(1), dim=1).reshape_as(router_weights)
    predicted_weights = router_weights * valid_slots.to(router_weights.dtype)
    predicted_weights = predicted_weights / predicted_weights.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-12)
    valid_samples = valid_slots.flatten(1).any(dim=1)
    per_sample = F.kl_div(
        predicted_weights.flatten(1).clamp_min(1e-12).log(),
        target_weights.flatten(1),
        reduction="none",
    ).sum(dim=1)
    router_loss = per_sample[valid_samples].mean()
    best_gain = utility_tensor.masked_fill(~valid_slots, -torch.inf).flatten(1).max(
        dim=1
    ).values
    positive_gain = best_gain.clamp_min(0.0)
    alpha_target = (
        float(alpha_max)
        * positive_gain
        / (positive_gain + float(alpha_gain_scale))
    ).unsqueeze(1)
    alpha_loss = F.mse_loss(alpha[valid_samples], alpha_target[valid_samples])
    return MarginalGainRoutingResult(
        router_loss=router_loss,
        alpha_loss=alpha_loss,
        target_weights=target_weights,
        predicted_weights=predicted_weights,
        utilities=utility_tensor,
        alpha_target=alpha_target,
        valid_samples=valid_samples,
    )


class HierarchicalBoundedResidualBankFusion(nn.Module):
    """Route matched token residuals with joint mass and bounded sample energy."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        baseline_width: int,
        residual_width: int = 256,
        alpha_max: float = 0.5,
        alpha_init: float = 0.2,
        alpha_hidden_width: int = 32,
    ) -> None:
        super().__init__()
        if tuple(expert_widths) != EXPERT_ORDER:
            raise ValueError(f"expert widths must follow {EXPERT_ORDER}")
        widths = {int(value) for value in expert_widths.values()}
        if len(widths) != 1 or baseline_width <= 0 or residual_width <= 0:
            raise ValueError("fusion widths must be positive and shared")
        if not 0.0 < alpha_init < alpha_max <= 1.0:
            raise ValueError("alpha requires 0 < init < max <= 1")
        expert_width = widths.pop()
        self.baseline_width = int(baseline_width)
        self.residual_width = int(residual_width)
        self.alpha_max = float(alpha_max)
        self.expert_residual_width = len(MODALITY_ORDER) * self.residual_width
        self.branch_embedding_width = self.baseline_width + self.expert_residual_width
        self.residual_bank_width = len(EXPERT_ORDER) * self.expert_residual_width
        self.fused_embedding_width = self.baseline_width + self.residual_bank_width
        self.residual_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(expert_width) for expert in EXPERT_ORDER}
        )
        self.residual_projections = nn.ModuleDict(
            {
                expert: nn.Linear(expert_width, self.residual_width, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.alpha_predictor = nn.Sequential(
            nn.Linear(
                2 * len(EXPERT_ORDER) * len(MODALITY_ORDER),
                alpha_hidden_width,
            ),
            nn.GELU(),
            nn.Linear(alpha_hidden_width, 1),
        )
        final = self.alpha_predictor[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            math.log((alpha_init / alpha_max) / (1.0 - alpha_init / alpha_max)),
        )

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        *,
        baseline_embedding: torch.Tensor,
        anchor_tokens: torch.Tensor,
    ) -> HierarchicalBoundedFusionResult:
        batch_size = modality_mask.shape[0]
        if baseline_embedding.shape != (batch_size, self.baseline_width):
            raise ValueError("baseline embedding has the wrong shape")
        if anchor_tokens.ndim != 4 or anchor_tokens.shape[:2] != modality_mask.shape:
            raise ValueError("anchor tokens must have shape B,M,N,D")
        if reliability.r.shape != (
            batch_size,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
        ):
            raise ValueError("reliability has the wrong shape")

        valid = modality_mask[:, None, :, None].to(anchor_tokens.dtype)
        projected = []
        for expert in EXPERT_ORDER:
            token_delta = states[expert].tokens - anchor_tokens
            pooled_delta = self.residual_norms[expert](token_delta).mean(dim=2)
            residual = self.residual_projections[expert](pooled_delta)
            projected.append(F.normalize(residual, dim=-1))
        contributions = torch.stack(projected, dim=1) * valid

        confidence = reliability.r * (1.0 - reliability.u)
        confidence = confidence * modality_mask[:, None].to(confidence.dtype)
        modal_confidence = confidence.mean(dim=1)
        modal_probabilities = modal_confidence / modal_confidence.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        expert_probabilities = confidence / confidence.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        weights = expert_probabilities * modal_probabilities[:, None]

        alpha_features = torch.cat((reliability.r, reliability.u), dim=1).flatten(1)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_predictor(alpha_features))
        baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
        routed_bank = (contributions * weights[..., None]).flatten(1)
        activated_bank = F.normalize(routed_bank, dim=1) * baseline_norm * alpha
        residual_embeddings = {
            expert: F.normalize(contributions[:, index].flatten(1), dim=1)
            * baseline_norm
            for index, expert in enumerate(EXPERT_ORDER)
        }
        branch_embeddings = {
            expert: torch.cat(
                (baseline_embedding, residual_embeddings[expert] * alpha), dim=1
            )
            for expert in EXPERT_ORDER
        }
        return HierarchicalBoundedFusionResult(
            fused_embedding=torch.cat((baseline_embedding, activated_bank), dim=1),
            branch_embeddings=branch_embeddings,
            residual_embeddings=residual_embeddings,
            contribution_embeddings=contributions,
            weights=weights,
            modal_probabilities=modal_probabilities,
            expert_probabilities=expert_probabilities,
            alpha=alpha,
            modality_mask=modality_mask,
        )


class MarginalGainV7Criterion(nn.Module):
    """Train normalized retrieval, slot utility, alpha and quality routing."""

    def __init__(
        self,
        *,
        triplet_margin: float,
        label_smoothing: float,
        utility_temperature: float,
        alpha_gain_scale: float,
    ) -> None:
        super().__init__()
        self.triplet_margin = float(triplet_margin)
        self.label_smoothing = float(label_smoothing)
        self.utility_temperature = float(utility_temperature)
        self.alpha_gain_scale = float(alpha_gain_scale)

    def _identity_losses(
        self,
        logits: torch.Tensor,
        embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        triplet = _batch_hard_triplet(
            F.normalize(embedding, dim=1),
            labels,
            self.triplet_margin,
        )
        return identity, triplet

    def forward(
        self,
        output: object,
        labels: torch.Tensor,
        *,
        quality_output: object,
    ) -> dict[str, torch.Tensor]:
        if quality_output.quality_targets.shape != quality_output.modality_mask.shape:
            raise ValueError("V7 quality targets must have shape B,M")
        id_fused, triplet_fused = self._identity_losses(
            output.fused_logits, output.fused_embedding, labels
        )
        losses = {"id_fused": id_fused, "triplet_fused": triplet_fused}
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"], losses[f"triplet_{expert}"] = (
                self._identity_losses(
                    output.branch_logits[expert],
                    output.branch_embeddings[expert],
                    labels,
                )
            )
            (
                losses[f"id_residual_{expert}"],
                losses[f"triplet_residual_{expert}"],
            ) = self._identity_losses(
                output.residual_logits[expert],
                output.residual_embeddings[expert],
                labels,
            )
        routing = marginal_gain_router_loss(
            output.baseline_embedding,
            output.contribution_embeddings,
            output.router_weights,
            output.alpha,
            labels,
            modality_mask=output.modality_mask,
            alpha_max=output.alpha_max,
            utility_temperature=self.utility_temperature,
            alpha_gain_scale=self.alpha_gain_scale,
        )
        losses["peer_logits"] = routing.router_loss
        losses["alpha"] = routing.alpha_loss
        quality_target = quality_output.quality_targets * quality_output.modality_mask.to(
            quality_output.quality_targets.dtype
        )
        quality_target = quality_target / quality_target.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        losses["reliability"] = F.kl_div(
            quality_output.modal_probabilities.clamp_min(1e-12).log(),
            quality_target,
            reduction="batchmean",
        )
        return losses


@dataclass(frozen=True, eq=False)
class SignalPreservingV7Output:
    fused_embedding: torch.Tensor
    baseline_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    reliability: ReliabilityResult
    relay_results: tuple[Any, ...]
    peer_teaching: None
    fused_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    residual_logits: Mapping[str, torch.Tensor]
    modality_mask: torch.Tensor
    router_weights: torch.Tensor
    modal_probabilities: torch.Tensor
    expert_probabilities: torch.Tensor
    alpha: torch.Tensor
    alpha_max: float
    quality_targets: torch.Tensor | None
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        for name in (
            "branch_embeddings",
            "residual_embeddings",
            "branch_logits",
            "residual_logits",
            "diagnostics",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class SignalPreservingCollaborativeReIDV7(nn.Module):
    """Exact Signal prefix plus matched residual experts and bounded routing."""

    def __init__(
        self,
        *,
        baseline: FrozenSignalBackbone,
        encoder: nn.Module,
        fusion: HierarchicalBoundedResidualBankFusion,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.encoder = encoder
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.baseline_embedding_width = baseline.baseline_width
        self.branch_embedding_width = fusion.branch_embedding_width
        self.residual_embedding_width = fusion.expert_residual_width
        self.fused_embedding_width = fusion.fused_embedding_width
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {expert: self._make_neck(self.branch_embedding_width) for expert in EXPERT_ORDER}
        )
        self.residual_necks = nn.ModuleDict(
            {
                expert: self._make_neck(self.residual_embedding_width)
                for expert in EXPERT_ORDER
            }
        )
        if num_classes:
            self.fused_classifier = nn.Linear(
                self.fused_embedding_width, num_classes, bias=False
            )
            self.branch_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(self.branch_embedding_width, num_classes, bias=False)
                    for expert in EXPERT_ORDER
                }
            )
            self.residual_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(
                        self.residual_embedding_width, num_classes, bias=False
                    )
                    for expert in EXPERT_ORDER
                }
            )
            for classifier in (
                self.fused_classifier,
                *self.branch_classifiers.values(),
                *self.residual_classifiers.values(),
            ):
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.branch_classifiers = nn.ModuleDict()
            self.residual_classifiers = nn.ModuleDict()

    @staticmethod
    def _make_neck(width: int) -> nn.BatchNorm1d:
        neck = nn.BatchNorm1d(width)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        neck.bias.requires_grad_(False)
        return neck

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
        retrieval_output: str = "fused",
    ) -> torch.Tensor | SignalPreservingV7Output:
        del targets
        field = self.baseline(batch)
        if retrieval_output == "baseline_only" and not return_aux:
            return field.baseline_embedding
        states = self.encoder.forward_token_field(
            field.expert_tokens, field.modality_mask
        )
        if states.reliability is None:
            raise RuntimeError("V7 collaborative encoder did not emit reliability")
        packed_anchor = field.expert_tokens[EXPERT_ORDER[0]]
        anchor_tokens = packed_anchor.reshape(
            field.modality_mask.shape[0],
            len(MODALITY_ORDER),
            packed_anchor.shape[1],
            packed_anchor.shape[2],
        )
        fusion = self.fusion(
            states,
            states.reliability,
            field.modality_mask,
            baseline_embedding=field.baseline_embedding,
            anchor_tokens=anchor_tokens,
        )
        if not return_aux:
            if retrieval_output == "fused":
                return fusion.fused_embedding
            if retrieval_output in EXPERT_ORDER:
                return fusion.branch_embeddings[retrieval_output]
            raise ValueError("retrieval output must be baseline_only, fused, or an expert")

        fused_logits = None
        branch_logits: dict[str, torch.Tensor] = {}
        residual_logits: dict[str, torch.Tensor] = {}
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(self.fused_neck(fusion.fused_embedding))
            branch_logits = {
                expert: self.branch_classifiers[expert](
                    self.branch_necks[expert](fusion.branch_embeddings[expert])
                )
                for expert in EXPERT_ORDER
            }
            residual_logits = {
                expert: self.residual_classifiers[expert](
                    self.residual_necks[expert](fusion.residual_embeddings[expert])
                )
                for expert in EXPERT_ORDER
            }
        finite = [
            fusion.fused_embedding,
            fusion.contribution_embeddings,
            fusion.weights,
            fusion.alpha,
            *fusion.branch_embeddings.values(),
            *fusion.residual_embeddings.values(),
        ]
        if fused_logits is not None:
            finite.extend((fused_logits, *branch_logits.values(), *residual_logits.values()))
        suffix_norm = fusion.fused_embedding[:, self.baseline_embedding_width :].norm(
            dim=1
        )
        baseline_norm = field.baseline_embedding.norm(dim=1)
        return SignalPreservingV7Output(
            fused_embedding=fusion.fused_embedding,
            baseline_embedding=field.baseline_embedding,
            branch_embeddings=fusion.branch_embeddings,
            residual_embeddings=fusion.residual_embeddings,
            contribution_embeddings=fusion.contribution_embeddings,
            reliability=states.reliability,
            relay_results=states.relay_results,
            peer_teaching=None,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            residual_logits=residual_logits,
            modality_mask=field.modality_mask,
            router_weights=fusion.weights,
            modal_probabilities=fusion.modal_probabilities,
            expert_probabilities=fusion.expert_probabilities,
            alpha=fusion.alpha,
            alpha_max=self.fusion.alpha_max,
            quality_targets=batch.get("modality_quality"),
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    fusion.fused_embedding[:, : self.baseline_embedding_width],
                    field.baseline_embedding,
                ),
                "baseline_frozen": all(
                    not parameter.requires_grad for parameter in self.baseline.parameters()
                ),
                "has_all_three_experts": tuple(fusion.branch_embeddings) == EXPERT_ORDER,
                "joint_router_mass": bool(
                    torch.allclose(
                        fusion.weights.sum(dim=(1, 2)),
                        torch.ones_like(fusion.weights.sum(dim=(1, 2))),
                        rtol=1e-4,
                        atol=1e-6,
                    )
                ),
                "bounded_residual_energy": bool(
                    (suffix_norm <= baseline_norm * self.fusion.alpha_max + 1e-5).all()
                ),
                "matched_token_residual": True,
            },
        )


__all__ = [
    "HierarchicalBoundedFusionResult",
    "HierarchicalBoundedResidualBankFusion",
    "MarginalGainRoutingResult",
    "MarginalGainV7Criterion",
    "SignalPreservingCollaborativeReIDV7",
    "SignalPreservingV7Output",
    "marginal_gain_router_loss",
]
