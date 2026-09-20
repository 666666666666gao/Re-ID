"""Signal-anchored triadic relation repair for TriFusion V16."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.nn.functional as F
from torch import nn

from .signal_preserving_v8 import ExpertFormationV8Criterion, SignalPreservingV8Output
from .state import EXPERT_ORDER


V16_RELATION_GAP = 0.05
V16_PROTECTION_THRESHOLD = 0.30
V16_PROTECTION_TOLERANCE = 0.02
V16_REPAIR_WEIGHT = 1.0
V16_PROTECTION_WEIGHT = 0.25
V16_ID_FUSED_WEIGHT = 0.25
V16_TRIPLET_FUSED_WEIGHT = 1.0
V16_ID_BRANCH_WEIGHT = 1.0 / 12.0
V16_TRIPLET_BRANCH_WEIGHT = 0.25
V16_ID_RESIDUAL_WEIGHT = 1.0 / 12.0
V16_TRIPLET_RESIDUAL_WEIGHT = 0.25


@dataclass(frozen=True, eq=False)
class SignalHardPairsV16:
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor
    valid_query_mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class SATRRelationObjectiveV16:
    total: torch.Tensor
    repair_total: torch.Tensor
    protection_loss: torch.Tensor
    repair_losses: Mapping[str, torch.Tensor]
    eligible_masks: Mapping[str, torch.Tensor]
    coverages: Mapping[str, torch.Tensor]
    protection_mask: torch.Tensor
    pairs: SignalHardPairsV16

    def __post_init__(self) -> None:
        for name in ("repair_losses", "eligible_masks", "coverages"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )


def select_signal_hard_pairs_v16(
    baseline_embedding: torch.Tensor,
    identities: torch.Tensor,
    physical_cameras: torch.Tensor,
) -> SignalHardPairsV16:
    """Select one exact-Signal hard cross-camera relation per valid query."""

    if baseline_embedding.ndim != 2:
        raise ValueError("baseline embedding must have shape B,D")
    batch_size = baseline_embedding.shape[0]
    if identities.shape != (batch_size,) or physical_cameras.shape != (batch_size,):
        raise ValueError("identities and physical cameras must have shape B")

    normalized = F.normalize(baseline_embedding, dim=1)
    similarity = normalized @ normalized.transpose(0, 1)
    same_identity = identities[:, None] == identities[None, :]
    different_camera = physical_cameras[:, None] != physical_cameras[None, :]
    positive_mask = same_identity & different_camera
    negative_mask = ~same_identity
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)

    positive_indices = similarity.masked_fill(~positive_mask, torch.inf).argmin(dim=1)
    negative_indices = similarity.masked_fill(~negative_mask, -torch.inf).argmax(dim=1)
    invalid_index = torch.full_like(positive_indices, -1)
    positive_indices = torch.where(valid, positive_indices, invalid_index)
    negative_indices = torch.where(valid, negative_indices, invalid_index)
    return SignalHardPairsV16(
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        valid_query_mask=valid,
    )


def _signal_pair_margins_v16(
    embedding: torch.Tensor,
    pairs: SignalHardPairsV16,
) -> torch.Tensor:
    normalized = F.normalize(embedding, dim=1)
    rows = torch.arange(embedding.shape[0], device=embedding.device)
    positive = pairs.positive_indices.clamp_min(0)
    negative = pairs.negative_indices.clamp_min(0)
    positive_similarity = (normalized[rows] * normalized[positive]).sum(dim=1)
    negative_similarity = (normalized[rows] * normalized[negative]).sum(dim=1)
    margin = positive_similarity - negative_similarity
    return torch.where(pairs.valid_query_mask, margin, torch.zeros_like(margin))


def satr_relation_objective_v16(
    baseline_embedding: torch.Tensor,
    fused_embedding: torch.Tensor,
    branch_embeddings: Mapping[str, torch.Tensor],
    identities: torch.Tensor,
    physical_cameras: torch.Tensor,
) -> SATRRelationObjectiveV16:
    """Compute the fixed SATR repair and fused Signal-protection objective."""

    if tuple(branch_embeddings) != EXPERT_ORDER:
        raise ValueError(f"branch embeddings must follow {EXPERT_ORDER}")
    pairs = select_signal_hard_pairs_v16(
        baseline_embedding,
        identities,
        physical_cameras,
    )
    baseline_margin = _signal_pair_margins_v16(baseline_embedding, pairs)
    fused_margin = _signal_pair_margins_v16(fused_embedding, pairs)
    branch_margins = {
        expert: _signal_pair_margins_v16(branch_embeddings[expert], pairs)
        for expert in EXPERT_ORDER
    }
    valid_count = pairs.valid_query_mask.sum().clamp_min(1)
    repair_losses: dict[str, torch.Tensor] = {}
    eligible_masks: dict[str, torch.Tensor] = {}
    coverages: dict[str, torch.Tensor] = {}
    for receiver in EXPERT_ORDER:
        peers = tuple(expert for expert in EXPERT_ORDER if expert != receiver)
        teacher = torch.minimum(
            branch_margins[peers[0]].detach(),
            branch_margins[peers[1]].detach(),
        )
        receiver_margin = branch_margins[receiver]
        eligible = (
            pairs.valid_query_mask
            & (teacher > 0.0)
            & (
                teacher
                >= torch.maximum(
                    baseline_margin.detach(),
                    receiver_margin.detach(),
                )
                + V16_RELATION_GAP
            )
        )
        gap = torch.relu(teacher - receiver_margin)
        repair_losses[receiver] = (0.5 * gap.square() * eligible).sum() / eligible.sum().clamp_min(1)
        eligible_masks[receiver] = eligible
        coverages[receiver] = eligible.sum() / valid_count

    protection_mask = (
        pairs.valid_query_mask
        & (baseline_margin.detach() >= V16_PROTECTION_THRESHOLD)
    )
    protection_gap = torch.relu(
        baseline_margin.detach() - V16_PROTECTION_TOLERANCE - fused_margin
    )
    protection_loss = (
        protection_gap.square() * protection_mask
    ).sum() / protection_mask.sum().clamp_min(1)
    repair_total = torch.stack(tuple(repair_losses.values())).sum()
    total = (
        V16_REPAIR_WEIGHT * repair_total
        + V16_PROTECTION_WEIGHT * protection_loss
    )
    return SATRRelationObjectiveV16(
        total=total,
        repair_total=repair_total,
        protection_loss=protection_loss,
        repair_losses=repair_losses,
        eligible_masks=eligible_masks,
        coverages=coverages,
        protection_mask=protection_mask,
        pairs=pairs,
    )


class SATRV16Criterion(nn.Module):
    """Registered V8 ReID supervision plus the fixed V16 SATR objective."""

    def __init__(
        self,
        *,
        triplet_margin: float,
        label_smoothing: float,
        satr_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.satr_enabled = bool(satr_enabled)
        self.reid = ExpertFormationV8Criterion(
            triplet_margin=triplet_margin,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        output: SignalPreservingV8Output,
        labels: torch.Tensor,
        physical_cameras: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = self.reid(output, labels)
        supervised_total = (
            V16_ID_FUSED_WEIGHT * losses["id_fused"]
            + V16_TRIPLET_FUSED_WEIGHT * losses["triplet_fused"]
        )
        for expert in EXPERT_ORDER:
            supervised_total = (
                supervised_total
                + V16_ID_BRANCH_WEIGHT * losses[f"id_{expert}"]
                + V16_TRIPLET_BRANCH_WEIGHT * losses[f"triplet_{expert}"]
                + V16_ID_RESIDUAL_WEIGHT * losses[f"id_residual_{expert}"]
                + V16_TRIPLET_RESIDUAL_WEIGHT
                * losses[f"triplet_residual_{expert}"]
            )
        relation = satr_relation_objective_v16(
            output.baseline_embedding,
            output.fused_embedding,
            output.branch_embeddings,
            labels,
            physical_cameras,
        )
        losses["supervised_total"] = supervised_total
        for expert in EXPERT_ORDER:
            losses[f"satr_{expert}"] = relation.repair_losses[expert]
            losses[f"coverage_{expert}"] = relation.coverages[expert]
        losses["satr_repair_total"] = relation.repair_total
        losses["satr_protection"] = relation.protection_loss
        satr_total = relation.total if self.satr_enabled else relation.total * 0.0
        losses["satr_total"] = satr_total
        losses["total"] = supervised_total + satr_total
        return losses


__all__ = [
    "SATRV16Criterion",
    "SATRRelationObjectiveV16",
    "SignalHardPairsV16",
    "V16_ID_BRANCH_WEIGHT",
    "V16_ID_FUSED_WEIGHT",
    "V16_ID_RESIDUAL_WEIGHT",
    "V16_PROTECTION_THRESHOLD",
    "V16_PROTECTION_TOLERANCE",
    "V16_PROTECTION_WEIGHT",
    "V16_RELATION_GAP",
    "V16_REPAIR_WEIGHT",
    "V16_TRIPLET_BRANCH_WEIGHT",
    "V16_TRIPLET_FUSED_WEIGHT",
    "V16_TRIPLET_RESIDUAL_WEIGHT",
    "satr_relation_objective_v16",
    "select_signal_hard_pairs_v16",
]
