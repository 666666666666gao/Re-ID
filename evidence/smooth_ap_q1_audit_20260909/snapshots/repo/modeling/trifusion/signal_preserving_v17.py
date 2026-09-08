"""Dense triadic relation-envelope distillation for TriFusion V17."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import _batch_hard_triplet
from .signal_preserving_v16 import select_signal_hard_pairs_v16
from .state import EXPERT_ORDER


V17_FUSED_ENVELOPE_WEIGHT = 0.5
V17_BRANCH_ENVELOPE_WEIGHT = 1.0 / 6.0
V17_BRANCH_SUPERVISION_WEIGHT = 1.0 / 3.0
V17_ENVELOPE_WEIGHT = 1.0
V17_PROTECTION_THRESHOLD = 0.30
V17_PROTECTION_TOLERANCE = 0.02
V17_PROTECTION_WEIGHT = 0.25


@dataclass(frozen=True, eq=False)
class TriadicCorrectionOutputV17:
    corrected_residuals: Mapping[str, torch.Tensor]
    fused_residual: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "corrected_residuals",
            MappingProxyType(dict(self.corrected_residuals)),
        )


class TriadicCorrectionV17(nn.Module):
    """Shared low-rank correction driven by the other two expert views."""

    def __init__(self, *, residual_width: int, adapter_width: int) -> None:
        super().__init__()
        self.residual_width = int(residual_width)
        self.input_norm = nn.LayerNorm(residual_width)
        self.input_projection = nn.Linear(residual_width, adapter_width)
        self.shared_correction = nn.Sequential(
            nn.Linear(3 * adapter_width, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, adapter_width),
        )
        self.receiver_projections = nn.ModuleDict(
            {
                expert: nn.Linear(adapter_width, residual_width)
                for expert in EXPERT_ORDER
            }
        )
        for projection in self.receiver_projections.values():
            nn.init.normal_(projection.weight, std=1e-3)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        residual_embeddings: Mapping[str, torch.Tensor],
    ) -> TriadicCorrectionOutputV17:
        if tuple(residual_embeddings) != EXPERT_ORDER:
            raise ValueError(f"residual embeddings must follow {EXPERT_ORDER}")

        projected = {
            expert: self.input_projection(
                self.input_norm(residual_embeddings[expert])
            )
            for expert in EXPERT_ORDER
        }
        triadic_mean = torch.stack(tuple(projected.values())).mean(dim=0)
        corrected: dict[str, torch.Tensor] = {}
        for receiver in EXPERT_ORDER:
            peers = tuple(expert for expert in EXPERT_ORDER if expert != receiver)
            peer_intersection = projected[peers[0]] * projected[peers[1]]
            message = self.shared_correction(
                torch.cat(
                    (projected[receiver], peer_intersection, triadic_mean),
                    dim=1,
                )
            )
            delta = self.receiver_projections[receiver](message)
            corrected[receiver] = F.normalize(
                residual_embeddings[receiver] + delta,
                dim=1,
            )
        fused = F.normalize(torch.cat(tuple(corrected.values()), dim=1), dim=1)
        return TriadicCorrectionOutputV17(
            corrected_residuals=corrected,
            fused_residual=fused,
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV17Output:
    fused_embedding: torch.Tensor
    baseline_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    teacher_residual_embeddings: Mapping[str, torch.Tensor]
    fused_logits: torch.Tensor | None
    residual_logits: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        for name in (
            "branch_embeddings",
            "residual_embeddings",
            "teacher_residual_embeddings",
            "residual_logits",
            "diagnostics",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class SignalPreservingCollaborativeV17(nn.Module):
    """Correct frozen V8 expert residuals while retaining exact Signal features."""

    def __init__(
        self,
        *,
        base_v8: nn.Module,
        num_classes: int,
        adapter_width: int,
    ) -> None:
        super().__init__()
        self.base_v8 = base_v8
        self.num_classes = int(num_classes)
        self.baseline_embedding_width = int(base_v8.baseline_embedding_width)
        self.residual_embedding_width = int(base_v8.residual_embedding_width)
        self.branch_embedding_width = (
            self.baseline_embedding_width + self.residual_embedding_width
        )
        self.fused_embedding_width = (
            self.baseline_embedding_width
            + len(EXPERT_ORDER) * self.residual_embedding_width
        )
        for parameter in self.base_v8.parameters():
            parameter.requires_grad_(False)

        self.correction = TriadicCorrectionV17(
            residual_width=self.residual_embedding_width,
            adapter_width=adapter_width,
        )
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.residual_necks = nn.ModuleDict(
            {
                expert: self._make_neck(self.residual_embedding_width)
                for expert in EXPERT_ORDER
            }
        )
        if self.num_classes:
            self.fused_classifier = nn.Linear(
                self.fused_embedding_width,
                self.num_classes,
                bias=False,
            )
            self.residual_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(
                        self.residual_embedding_width,
                        self.num_classes,
                        bias=False,
                    )
                    for expert in EXPERT_ORDER
                }
            )
            for classifier in (
                self.fused_classifier,
                *self.residual_classifiers.values(),
            ):
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.residual_classifiers = nn.ModuleDict()
        self.train(self.training)

    @staticmethod
    def _make_neck(width: int) -> nn.BatchNorm1d:
        neck = nn.BatchNorm1d(width)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        neck.bias.requires_grad_(False)
        return neck

    def train(self, mode: bool = True) -> "SignalPreservingCollaborativeV17":
        super().train(mode)
        self.base_v8.eval()
        return self

    def _forward_output(
        self,
        batch: Mapping[str, Any],
        *,
        with_heads: bool,
    ) -> SignalPreservingV17Output:
        with torch.no_grad():
            base_output = self.base_v8(batch, return_aux=True)
        teachers = {
            expert: base_output.residual_embeddings[expert].detach()
            for expert in EXPERT_ORDER
        }
        correction = self.correction(teachers)
        baseline = base_output.baseline_embedding
        baseline_norm = baseline.norm(dim=1, keepdim=True)
        branches = {
            expert: torch.cat(
                (
                    baseline,
                    correction.corrected_residuals[expert] * baseline_norm,
                ),
                dim=1,
            )
            for expert in EXPERT_ORDER
        }
        fused = torch.cat(
            (baseline, correction.fused_residual * baseline_norm),
            dim=1,
        )

        fused_logits = None
        residual_logits: dict[str, torch.Tensor] = {}
        if with_heads and self.fused_classifier is not None:
            fused_logits = self.fused_classifier(self.fused_neck(fused))
            residual_logits = {
                expert: self.residual_classifiers[expert](
                    self.residual_necks[expert](
                        correction.corrected_residuals[expert]
                    )
                )
                for expert in EXPERT_ORDER
            }
        finite = (
            fused,
            *branches.values(),
            *correction.corrected_residuals.values(),
        )
        return SignalPreservingV17Output(
            fused_embedding=fused,
            baseline_embedding=baseline,
            branch_embeddings=branches,
            residual_embeddings=correction.corrected_residuals,
            teacher_residual_embeddings=teachers,
            fused_logits=fused_logits,
            residual_logits=residual_logits,
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    fused[:, : self.baseline_embedding_width], baseline
                ),
                "v8_frozen": all(
                    not parameter.requires_grad
                    for parameter in self.base_v8.parameters()
                ),
                "router_enabled": False,
                "reranking_enabled": False,
            },
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
        retrieval_output: str = "fused",
    ) -> torch.Tensor | SignalPreservingV17Output:
        del targets
        output = self._forward_output(batch, with_heads=return_aux)
        if return_aux:
            return output
        if retrieval_output == "baseline_only":
            return output.baseline_embedding
        if retrieval_output == "fused":
            return output.fused_embedding
        if retrieval_output in EXPERT_ORDER:
            return output.branch_embeddings[retrieval_output]
        raise ValueError("retrieval output must be baseline_only, fused, or an expert")


@dataclass(frozen=True, eq=False)
class RelationEnvelopeObjectiveV17:
    total: torch.Tensor
    fused_positive: torch.Tensor
    fused_negative: torch.Tensor
    branch_positive: Mapping[str, torch.Tensor]
    branch_negative: Mapping[str, torch.Tensor]
    positive_source_counts: Mapping[str, int]
    negative_source_counts: Mapping[str, int]
    positive_tie_count: int
    negative_tie_count: int

    def __post_init__(self) -> None:
        for name in (
            "branch_positive",
            "branch_negative",
            "positive_source_counts",
            "negative_source_counts",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, dim=1)
    return normalized @ normalized.transpose(0, 1)


def _one_sided_envelope_losses(
    embeddings: torch.Tensor,
    positive_target: torch.Tensor,
    negative_target: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    similarity = _cosine_similarity_matrix(embeddings)
    positive = torch.relu(positive_target - similarity)[positive_mask].square().mean()
    negative = torch.relu(similarity - negative_target)[negative_mask].square().mean()
    return positive, negative


def relation_envelope_objective_v17(
    teacher_residuals: Mapping[str, torch.Tensor],
    fused_residual: torch.Tensor,
    corrected_residuals: Mapping[str, torch.Tensor],
    identities: torch.Tensor,
    physical_cameras: torch.Tensor,
) -> RelationEnvelopeObjectiveV17:
    """Match the best positive and best negative relations of frozen experts."""

    if tuple(teacher_residuals) != EXPERT_ORDER:
        raise ValueError(f"teacher residuals must follow {EXPERT_ORDER}")
    if tuple(corrected_residuals) != EXPERT_ORDER:
        raise ValueError(f"corrected residuals must follow {EXPERT_ORDER}")

    same_identity = identities[:, None] == identities[None, :]
    positive_mask = same_identity & (
        physical_cameras[:, None] != physical_cameras[None, :]
    )
    negative_mask = ~same_identity
    if not positive_mask.any() or not negative_mask.any():
        raise ValueError("relation envelope requires positive and negative pairs")

    teacher_similarities = torch.stack(
        tuple(
            _cosine_similarity_matrix(teacher_residuals[expert]).detach()
            for expert in EXPERT_ORDER
        )
    )
    positive_target, positive_source = teacher_similarities.max(dim=0)
    negative_target, negative_source = teacher_similarities.min(dim=0)

    fused_positive, fused_negative = _one_sided_envelope_losses(
        fused_residual,
        positive_target,
        negative_target,
        positive_mask,
        negative_mask,
    )
    branch_positive: dict[str, torch.Tensor] = {}
    branch_negative: dict[str, torch.Tensor] = {}
    for expert in EXPERT_ORDER:
        positive, negative = _one_sided_envelope_losses(
            corrected_residuals[expert],
            positive_target,
            negative_target,
            positive_mask,
            negative_mask,
        )
        branch_positive[expert] = positive
        branch_negative[expert] = negative

    fused_envelope = 0.5 * (fused_positive + fused_negative)
    branch_envelope_sum = sum(
        0.5 * (branch_positive[expert] + branch_negative[expert])
        for expert in EXPERT_ORDER
    )
    total = (
        V17_FUSED_ENVELOPE_WEIGHT * fused_envelope
        + V17_BRANCH_ENVELOPE_WEIGHT * branch_envelope_sum
    )
    positive_source_counts = {
        expert: int(((positive_source == index) & positive_mask).sum().item())
        for index, expert in enumerate(EXPERT_ORDER)
    }
    negative_source_counts = {
        expert: int(((negative_source == index) & negative_mask).sum().item())
        for index, expert in enumerate(EXPERT_ORDER)
    }
    positive_tie_count = int(
        (
            (teacher_similarities == positive_target.unsqueeze(0)).sum(dim=0) > 1
        )[positive_mask].sum().item()
    )
    negative_tie_count = int(
        (
            (teacher_similarities == negative_target.unsqueeze(0)).sum(dim=0) > 1
        )[negative_mask].sum().item()
    )
    return RelationEnvelopeObjectiveV17(
        total=total,
        fused_positive=fused_positive,
        fused_negative=fused_negative,
        branch_positive=branch_positive,
        branch_negative=branch_negative,
        positive_source_counts=positive_source_counts,
        negative_source_counts=negative_source_counts,
        positive_tie_count=positive_tie_count,
        negative_tie_count=negative_tie_count,
    )


def _signal_pair_margins_v17(
    embeddings: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    valid_query_mask: torch.Tensor,
) -> torch.Tensor:
    normalized = F.normalize(embeddings, dim=1)
    rows = torch.arange(embeddings.shape[0], device=embeddings.device)
    positive = positive_indices.clamp_min(0)
    negative = negative_indices.clamp_min(0)
    margins = (
        (normalized[rows] * normalized[positive]).sum(dim=1)
        - (normalized[rows] * normalized[negative]).sum(dim=1)
    )
    return torch.where(valid_query_mask, margins, torch.zeros_like(margins))


def _signal_protection_loss_v17(
    baseline_embedding: torch.Tensor,
    fused_embedding: torch.Tensor,
    identities: torch.Tensor,
    physical_cameras: torch.Tensor,
) -> torch.Tensor:
    pairs = select_signal_hard_pairs_v16(
        baseline_embedding,
        identities,
        physical_cameras,
    )
    baseline_margin = _signal_pair_margins_v17(
        baseline_embedding,
        pairs.positive_indices,
        pairs.negative_indices,
        pairs.valid_query_mask,
    ).detach()
    fused_margin = _signal_pair_margins_v17(
        fused_embedding,
        pairs.positive_indices,
        pairs.negative_indices,
        pairs.valid_query_mask,
    )
    protected = (
        pairs.valid_query_mask
        & (baseline_margin >= V17_PROTECTION_THRESHOLD)
    )
    hinge = torch.relu(
        baseline_margin - V17_PROTECTION_TOLERANCE - fused_margin
    )
    return (hinge * protected).sum() / protected.sum().clamp_min(1)


class DenseTriadicV17Criterion(nn.Module):
    """Registered ReID, dense relation-envelope and Signal safety objective."""

    def __init__(
        self,
        *,
        triplet_margin: float,
        label_smoothing: float,
        envelope_enabled: bool,
    ) -> None:
        super().__init__()
        self.triplet_margin = float(triplet_margin)
        self.label_smoothing = float(label_smoothing)
        self.envelope_enabled = bool(envelope_enabled)

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
        output: SignalPreservingV17Output,
        labels: torch.Tensor,
        physical_cameras: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if output.fused_logits is None:
            raise ValueError("V17 training requires identity heads")
        id_fused, triplet_fused = self._identity_losses(
            output.fused_logits,
            output.fused_embedding,
            labels,
        )
        losses: dict[str, torch.Tensor] = {
            "id_fused": id_fused,
            "triplet_fused": triplet_fused,
        }
        supervised_total = id_fused + triplet_fused
        for expert in EXPERT_ORDER:
            identity, triplet = self._identity_losses(
                output.residual_logits[expert],
                output.residual_embeddings[expert],
                labels,
            )
            losses[f"id_{expert}"] = identity
            losses[f"triplet_{expert}"] = triplet
            supervised_total = supervised_total + (
                V17_BRANCH_SUPERVISION_WEIGHT * (identity + triplet)
            )

        envelope = relation_envelope_objective_v17(
            output.teacher_residual_embeddings,
            output.fused_embedding[:, output.baseline_embedding.shape[1] :],
            output.residual_embeddings,
            labels,
            physical_cameras,
        )
        protection = _signal_protection_loss_v17(
            output.baseline_embedding,
            output.fused_embedding,
            labels,
            physical_cameras,
        )
        envelope_total = (
            V17_ENVELOPE_WEIGHT * envelope.total
            if self.envelope_enabled
            else envelope.total * 0.0
        )
        losses["supervised_total"] = supervised_total
        losses["envelope_raw"] = envelope.total
        losses["envelope_total"] = envelope_total
        losses["envelope_fused_positive"] = envelope.fused_positive
        losses["envelope_fused_negative"] = envelope.fused_negative
        for expert in EXPERT_ORDER:
            losses[f"envelope_positive_{expert}"] = envelope.branch_positive[expert]
            losses[f"envelope_negative_{expert}"] = envelope.branch_negative[expert]
            losses[f"teacher_positive_source_{expert}"] = envelope.total.new_tensor(
                envelope.positive_source_counts[expert]
            )
            losses[f"teacher_negative_source_{expert}"] = envelope.total.new_tensor(
                envelope.negative_source_counts[expert]
            )
        losses["teacher_positive_ties"] = envelope.total.new_tensor(
            envelope.positive_tie_count
        )
        losses["teacher_negative_ties"] = envelope.total.new_tensor(
            envelope.negative_tie_count
        )
        losses["signal_protection"] = protection
        losses["total"] = (
            supervised_total
            + envelope_total
            + V17_PROTECTION_WEIGHT * protection
        )
        return losses


__all__ = [
    "DenseTriadicV17Criterion",
    "RelationEnvelopeObjectiveV17",
    "SignalPreservingCollaborativeV17",
    "SignalPreservingV17Output",
    "TriadicCorrectionOutputV17",
    "TriadicCorrectionV17",
    "V17_BRANCH_ENVELOPE_WEIGHT",
    "V17_BRANCH_SUPERVISION_WEIGHT",
    "V17_ENVELOPE_WEIGHT",
    "V17_FUSED_ENVELOPE_WEIGHT",
    "V17_PROTECTION_THRESHOLD",
    "V17_PROTECTION_TOLERANCE",
    "V17_PROTECTION_WEIGHT",
    "relation_envelope_objective_v17",
]
