"""Fold-bound retrieval objective for TriFusion V14."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .signal_preserving_v13 import compose_v13_fusion


@dataclass(frozen=True, eq=False)
class RetrievalRiskOutput:
    risk: torch.Tensor
    per_query_loss: torch.Tensor
    hardest_positive_distance: torch.Tensor
    nearest_negative_distance: torch.Tensor


@dataclass(frozen=True, eq=False)
class MinimaxFixedSlotOutput:
    slot: int
    worst_fold_risk: torch.Tensor


def cross_camera_retrieval_risk(
    embedding: torch.Tensor,
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> RetrievalRiskOutput:
    """Compute smooth batch-hard risk with cross-camera positives."""

    embedding = F.normalize(embedding, dim=1)
    distances = torch.cdist(embedding, embedding)
    same_identity = identities[:, None] == identities[None, :]
    different_camera = cameras[:, None] != cameras[None, :]
    positives = same_identity & different_camera
    negatives = ~same_identity
    if not bool(positives.any(dim=1).all()) or not bool(negatives.any(dim=1).all()):
        raise ValueError("retrieval risk requires cross-camera positives and negatives")
    hardest_positive = distances.masked_fill(~positives, -torch.inf).max(dim=1).values
    nearest_negative = distances.masked_fill(~negatives, torch.inf).min(dim=1).values
    per_query = F.softplus(hardest_positive - nearest_negative)
    return RetrievalRiskOutput(
        risk=per_query.mean(),
        per_query_loss=per_query,
        hardest_positive_distance=hardest_positive,
        nearest_negative_distance=nearest_negative,
    )


def fold_bound_retrieval_risk(
    *,
    fold_id: int,
    row_fold_ids: torch.Tensor,
    baseline_embedding: torch.Tensor,
    modal_residual: torch.Tensor,
    weights: torch.Tensor,
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> RetrievalRiskOutput:
    """Compose and score rows registered to one OOF teacher generator."""

    if not bool((row_fold_ids == int(fold_id)).all()):
        raise ValueError("fold-bound risk received rows from another generator")
    fused = compose_v13_fusion(
        baseline_embedding,
        modal_residual,
        weights,
    )
    return cross_camera_retrieval_risk(
        fused.retrieval_embedding,
        identities,
        cameras,
    )


def select_minimax_fixed_slot(
    fixed_slot_risks: torch.Tensor,
) -> MinimaxFixedSlotOutput:
    """Select one slot whose worst source-fold retrieval risk is minimal."""

    worst_by_slot = fixed_slot_risks.max(dim=0).values
    slot = int(worst_by_slot.argmin())
    return MinimaxFixedSlotOutput(
        slot=slot,
        worst_fold_risk=worst_by_slot[slot],
    )


__all__ = [
    "MinimaxFixedSlotOutput",
    "RetrievalRiskOutput",
    "cross_camera_retrieval_risk",
    "fold_bound_retrieval_risk",
    "select_minimax_fixed_slot",
]

