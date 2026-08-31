"""Named TriFusion training objective with immutable CIRC supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .intervention_targets import CIRCTargetCache
from .model import TriFusionOutput
from .standalone import SingleBranchOutput
from .state import EXPERT_ORDER


def _batch_hard_triplet(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float
) -> torch.Tensor:
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    same_identity = labels[:, None] == labels[None, :]
    diagonal = torch.eye(
        labels.shape[0], dtype=torch.bool, device=labels.device
    )
    positives = same_identity & ~diagonal
    negatives = ~same_identity
    valid_rows = positives.any(dim=1) & negatives.any(dim=1)
    if not bool(valid_rows.any()):
        return embeddings.sum() * 0.0
    hardest_positive = distances.masked_fill(~positives, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negatives, float("inf")).min(dim=1).values
    return F.relu(
        hardest_positive[valid_rows] - hardest_negative[valid_rows] + margin
    ).mean()


class TriFusionCriterion(nn.Module):
    """Compute explicit paper-facing losses; no tuple-position inference."""

    def __init__(
        self,
        *,
        target_cache: CIRCTargetCache | None,
        triplet_margin: float = 0.3,
        brier_weight: float = 1.0,
        evidence_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if triplet_margin < 0 or brier_weight < 0 or evidence_weight < 0:
            raise ValueError("criterion weights and margin must be nonnegative")
        self.target_cache = target_cache
        self.triplet_margin = triplet_margin
        self.brier_weight = brier_weight
        self.evidence_weight = evidence_weight

    def forward(
        self,
        output: TriFusionOutput,
        labels: torch.Tensor,
        *,
        sample_keys: Sequence[str] | None = None,
        conditions: Sequence[Mapping[str, object]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if output.fused_logits is None:
            raise ValueError("supervised criterion requires fused logits")
        if set(output.branch_logits) != set(EXPERT_ORDER):
            raise ValueError("supervised criterion requires all branch logits")
        if labels.ndim != 1 or labels.shape[0] != output.fused_embedding.shape[0]:
            raise ValueError("labels must be a length-B tensor")

        losses = {
            "id_fused": F.cross_entropy(output.fused_logits, labels),
            "triplet_fused": _batch_hard_triplet(
                output.fused_embedding, labels, self.triplet_margin
            ),
        }
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"] = F.cross_entropy(
                output.branch_logits[expert], labels
            )
            losses[f"triplet_{expert}"] = _batch_hard_triplet(
                output.branch_embeddings[expert], labels, self.triplet_margin
            )

        zero = output.fused_embedding.sum() * 0.0
        if self.target_cache is None:
            if sample_keys is not None or conditions is not None:
                raise ValueError("CIRC metadata was supplied without a target cache")
            losses["reliability"] = zero
        else:
            if sample_keys is None or conditions is None:
                raise ValueError("CIRC target cache requires sample_keys and conditions")
            circ = self.target_cache.lookup(
                sample_keys,
                conditions,
                device=output.reliability.r.device,
                allow_missing=True,
            )
            available = output.modality_mask[:, None, :].expand_as(circ.valid_mask)
            valid = circ.valid_mask & available
            if bool(valid.any()):
                predicted = output.reliability.r[valid].clamp(1e-6, 1.0 - 1e-6)
                helpful = circ.helpful_targets[valid]
                uncertainty = output.reliability.u[valid]
                bce = F.binary_cross_entropy(predicted, helpful)
                brier = (predicted - helpful).square().mean()
                detached_error = (predicted - helpful).abs().detach()
                evidence_regularizer = (uncertainty - detached_error).square().mean()
                losses["reliability"] = (
                    bce
                    + self.brier_weight * brier
                    + self.evidence_weight * evidence_regularizer
                )
            else:
                losses["reliability"] = zero

        if output.peer_teaching is None:
            losses["peer_logits"] = zero
            losses["peer_role"] = zero
            losses["private_diversity"] = zero
        else:
            losses["peer_logits"] = output.peer_teaching.logit_kl
            losses["peer_role"] = output.peer_teaching.role_loss
            losses["private_diversity"] = output.peer_teaching.private_diversity
        return losses


class SingleBranchCriterion(nn.Module):
    """ID plus batch-hard metric loss for one true standalone expert."""

    def __init__(self, *, triplet_margin: float = 0.3) -> None:
        super().__init__()
        if triplet_margin < 0:
            raise ValueError("triplet margin must be nonnegative")
        self.triplet_margin = float(triplet_margin)

    def forward(
        self, output: SingleBranchOutput, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if output.logits is None:
            raise ValueError("standalone supervised criterion requires logits")
        if labels.ndim != 1 or labels.shape[0] != output.embedding.shape[0]:
            raise ValueError("labels must be a length-B tensor")
        return {
            f"id_{output.expert}": F.cross_entropy(output.logits, labels),
            f"triplet_{output.expert}": _batch_hard_triplet(
                output.embedding, labels, self.triplet_margin
            ),
        }


__all__ = ["SingleBranchCriterion", "TriFusionCriterion"]
