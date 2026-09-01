"""Energy-balanced non-destructive expert fusion for task-anchored TriFusion V4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .fusion import FusionResult
from .state import EXPERT_ORDER, MODALITY_ORDER, ExpertStateMap, ReliabilityResult
from .task_anchor_v3 import (
    TaskAnchoredCollaborativeReID,
    TaskAnchoredV3Criterion,
    TaskAnchoredV3Output,
)


@dataclass(frozen=True, eq=False)
class IdentityUtilityRoutingResult:
    """Detached identity utility target and differentiable router loss."""

    loss: torch.Tensor
    target_weights: torch.Tensor
    predicted_weights: torch.Tensor
    identity_gaps: torch.Tensor
    valid_samples: torch.Tensor


def batch_hard_identity_gap_per_sample(
    embedding: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-negative minus farthest-positive distance per sample."""

    if embedding.ndim != 2 or labels.shape != (embedding.shape[0],):
        raise ValueError("embedding must be B,D and labels must be length B")
    if embedding.shape[0] == 0:
        return embedding.new_zeros((0,)), labels.new_zeros((0,), dtype=torch.bool)
    normalized = F.normalize(embedding.float(), dim=1)
    distances = torch.cdist(normalized, normalized)
    same_identity = labels[:, None] == labels[None, :]
    same_identity.fill_diagonal_(False)
    different_identity = labels[:, None] != labels[None, :]
    valid = same_identity.any(dim=1) & different_identity.any(dim=1)
    hardest_positive = distances.masked_fill(~same_identity, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(~different_identity, torch.inf).amin(dim=1)
    gap = hardest_negative - hardest_positive
    gap = torch.where(valid, gap, torch.zeros_like(gap))
    return gap, valid


def identity_utility_router_loss(
    branch_embeddings: Mapping[str, torch.Tensor],
    router_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    modality_mask: torch.Tensor,
) -> IdentityUtilityRoutingResult:
    """Teach routing from each expert branch's detached batch-hard identity utility."""

    if tuple(branch_embeddings) != EXPERT_ORDER:
        raise ValueError(f"branch embeddings must follow expert order {EXPERT_ORDER}")
    first = branch_embeddings[EXPERT_ORDER[0]]
    expected_weights = (first.shape[0], len(EXPERT_ORDER), len(MODALITY_ORDER))
    if router_weights.shape != expected_weights:
        raise ValueError(f"router_weights must have shape {expected_weights}")
    if modality_mask.dtype != torch.bool or modality_mask.shape != expected_weights[::2]:
        raise ValueError("modality_mask must have shape B,M and bool dtype")
    if labels.shape != (first.shape[0],):
        raise ValueError("labels must have shape B")
    gaps = []
    validity = []
    for expert in EXPERT_ORDER:
        expert_embedding = branch_embeddings[expert]
        if expert_embedding.ndim != 2 or expert_embedding.shape[0] != first.shape[0]:
            raise ValueError("every expert embedding must have shape B,D")
        gap, valid = batch_hard_identity_gap_per_sample(expert_embedding, labels)
        gaps.append(gap)
        validity.append(valid)
    identity_gaps = torch.stack(gaps, dim=1).detach()
    valid_samples = torch.stack(validity, dim=1).all(dim=1)

    centered = identity_gaps - identity_gaps.mean(dim=1, keepdim=True)
    standardized = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(
        1e-6
    )
    target_weights = F.softmax(standardized, dim=1)

    valid_modalities = modality_mask[:, None].to(router_weights.dtype)
    predicted_weights = (router_weights * valid_modalities).sum(dim=2)
    predicted_weights = predicted_weights / valid_modalities.sum(
        dim=2
    ).clamp_min(1.0)
    predicted_weights = predicted_weights / predicted_weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    per_sample = F.kl_div(
        predicted_weights.clamp_min(1e-12).log(),
        target_weights,
        reduction="none",
    ).sum(dim=1)
    if bool(valid_samples.any()):
        loss = per_sample[valid_samples].mean()
    else:
        loss = router_weights.sum() * 0.0
    return IdentityUtilityRoutingResult(
        loss=loss,
        target_weights=target_weights,
        predicted_weights=predicted_weights,
        identity_gaps=identity_gaps,
        valid_samples=valid_samples,
    )


class EnergyBalancedResidualBankFusion(nn.Module):
    """Keep all expert residual blocks and match their joint energy to the anchor."""

    baseline_safe_zero_mode = True
    non_destructive_expert_bank = True
    energy_balance_has_free_scale = False

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        embedding_width: int = 512,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        widths = {int(value) for value in expert_widths.values()}
        if len(widths) != 1 or embedding_width <= 0:
            raise ValueError("V4 requires one positive shared semantic width")
        semantic_width = widths.pop()
        self.embedding_width = int(embedding_width)
        self.anchor_embedding_width = len(MODALITY_ORDER) * self.embedding_width
        self.residual_bank_width = len(EXPERT_ORDER) * self.anchor_embedding_width
        self.fused_embedding_width = self.anchor_embedding_width + self.residual_bank_width
        self.branch_embedding_width = 2 * self.anchor_embedding_width
        self.residual_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(semantic_width) for expert in EXPERT_ORDER}
        )
        self.residual_projections = nn.ModuleDict(
            {
                expert: nn.Linear(semantic_width, self.embedding_width, bias=False)
                for expert in EXPERT_ORDER
            }
        )

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        *,
        anchor_native: torch.Tensor,
        anchor_projected: torch.Tensor,
        force_zero_residual: bool = False,
    ) -> FusionResult:
        if anchor_native.shape[:2] != modality_mask.shape:
            raise ValueError("native anchor must have shape B,M,D")
        if anchor_projected.shape != (
            modality_mask.shape[0],
            modality_mask.shape[1],
            self.embedding_width,
        ):
            raise ValueError("projected anchor has the wrong shape")
        if reliability.r.shape[:1] + reliability.r.shape[2:] != modality_mask.shape:
            raise ValueError("reliability and modality mask shapes disagree")

        valid = modality_mask[..., None].to(anchor_projected.dtype)
        anchor_projected = anchor_projected * valid
        modal_anchor_norm = anchor_projected.detach().norm(dim=-1, keepdim=True)
        projected = []
        for expert in EXPERT_ORDER:
            delta = states[expert].global_embedding - anchor_native
            contribution = self.residual_projections[expert](
                self.residual_norms[expert](delta)
            )
            contribution = F.normalize(contribution, dim=-1) * modal_anchor_norm
            projected.append(contribution * valid)
        contributions = torch.stack(projected, dim=1)

        confidence = reliability.r * (1.0 - reliability.u).clamp_min(0.05)
        confidence = confidence * modality_mask[:, None].to(confidence.dtype)
        weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weighted_bank = contributions * weights[..., None]
        anchor_flat = anchor_projected.flatten(1)
        bank_flat = weighted_bank.flatten(1)
        anchor_norm = anchor_flat.detach().norm(dim=1, keepdim=True)
        bank_norm = bank_flat.norm(dim=1, keepdim=True)
        bank_scale = torch.where(
            bank_norm > 1e-12,
            anchor_norm / bank_norm.clamp_min(1e-12),
            torch.zeros_like(bank_norm),
        )
        calibrated_bank = bank_flat * bank_scale

        if force_zero_residual:
            contributions_for_output = torch.zeros_like(contributions)
            calibrated_bank = torch.zeros_like(calibrated_bank)
        else:
            contributions_for_output = contributions
        fused_embedding = torch.cat((anchor_flat, calibrated_bank), dim=1)
        branch_embeddings = {
            expert: torch.cat(
                (anchor_flat, contributions_for_output[:, index].flatten(1)),
                dim=1,
            )
            for index, expert in enumerate(EXPERT_ORDER)
        }
        return FusionResult(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=contributions_for_output,
            weights=weights,
            modality_mask=modality_mask,
        )


@dataclass(frozen=True, eq=False)
class TaskAnchoredV4Output(TaskAnchoredV3Output):
    """V3 training contract extended with the actual fusion router weights."""

    router_weights: torch.Tensor


class TaskAnchoredCollaborativeReIDV4(TaskAnchoredCollaborativeReID):
    """Task-anchored ReID model with an energy-balanced tri-expert residual bank."""

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | TaskAnchoredV4Output:
        del targets
        if "images" not in batch or "modality_mask" not in batch:
            raise ValueError("batch must contain images and modality_mask")
        images = batch["images"]
        mask = batch["modality_mask"]
        self._validate_batch(images, mask)
        if bool((~mask).all(dim=1).any()):
            raise ValueError("every row requires at least one modality")
        stacked = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
        modality_indices = torch.arange(3, device=mask.device).view(1, 3).expand_as(mask)
        field = self.tokenizer(stacked[mask], modality_indices[mask])
        states = self.encoder.forward_token_field(field.expert_tokens, mask)
        if states.reliability is None:
            raise RuntimeError("V4 collaborative encoder did not emit reliability")
        anchor_native = self._scatter(field.anchor_native, mask)
        anchor_projected = self._scatter(field.anchor_projected, mask)
        fusion = self.fusion(
            states,
            states.reliability,
            mask,
            anchor_native=anchor_native,
            anchor_projected=anchor_projected,
            force_zero_residual=bool(batch.get("force_zero_residual", False)),
        )
        anchor_embedding = anchor_projected.flatten(1)
        fused_neck = self.fused_neck(fusion.fused_embedding)
        anchor_neck = self.anchor_neck(anchor_embedding)
        branch_necks = {
            expert: self.branch_necks[expert](fusion.branch_embeddings[expert])
            for expert in EXPERT_ORDER
        }
        fused_logits = None
        anchor_logits = None
        branch_logits: dict[str, torch.Tensor] = {}
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(fused_neck)
            anchor_logits = self.anchor_classifier(anchor_neck)
            branch_logits = {
                expert: self.branch_classifiers[expert](branch_necks[expert])
                for expert in EXPERT_ORDER
            }
        if not return_aux:
            return fusion.fused_embedding
        finite = [
            fusion.fused_embedding,
            anchor_embedding,
            anchor_projected,
            fusion.contribution_embeddings,
            fusion.weights,
            states.reliability.r,
            states.reliability.u,
            *fusion.branch_embeddings.values(),
        ]
        if fused_logits is not None:
            finite.extend((fused_logits, anchor_logits, *branch_logits.values()))
        return TaskAnchoredV4Output(
            fused_embedding=fusion.fused_embedding,
            branch_embeddings=fusion.branch_embeddings,
            contribution_embeddings=fusion.contribution_embeddings,
            reliability=states.reliability,
            relay_results=states.relay_results,
            peer_teaching=None,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            modality_mask=mask,
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "has_all_three_experts": tuple(branch_logits or fusion.branch_embeddings)
                == EXPERT_ORDER,
                "anchor_direct_to_retrieval": True,
                "residual_bank_non_destructive": True,
                "anchor_residual_equal_energy": True,
            },
            anchor_embedding=anchor_embedding,
            anchor_modal=anchor_projected,
            anchor_logits=anchor_logits,
            router_weights=fusion.weights,
        )


class TaskAnchoredV4Criterion(TaskAnchoredV3Criterion):
    """V3 identity objectives plus identity-utility supervision for routing."""

    def forward(
        self,
        output: Any,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        losses = super().forward(output, labels, **kwargs)
        routing = identity_utility_router_loss(
            output.branch_embeddings,
            output.router_weights,
            labels,
            modality_mask=output.modality_mask,
        )
        losses["peer_logits"] = routing.loss
        return losses


__all__ = [
    "EnergyBalancedResidualBankFusion",
    "IdentityUtilityRoutingResult",
    "TaskAnchoredCollaborativeReIDV4",
    "TaskAnchoredV4Criterion",
    "TaskAnchoredV4Output",
    "batch_hard_identity_gap_per_sample",
    "identity_utility_router_loss",
]
