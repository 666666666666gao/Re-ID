"""Optional role-directed peer teaching (RDPT) auxiliary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .state import EXPERT_ORDER, ExpertStateMap, ReliabilityResult


@dataclass(frozen=True, eq=False)
class PeerTeachingResult:
    direction_gates: torch.Tensor
    logit_kl: torch.Tensor
    role_loss: torch.Tensor
    private_diversity: torch.Tensor
    rejection_rate: torch.Tensor
    direction_frequency: torch.Tensor


class RoleDirectedPeerTeaching(nn.Module):
    """Teach only from detached, sufficiently better expert-role payloads."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        num_classes: int,
        role_width: int = 64,
        quality_delta: float = 0.1,
        minimum_teacher_quality: float = 0.6,
        private_cosine_margin: float = 0.2,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        if num_classes <= 0 or role_width <= 0:
            raise ValueError("num_classes and role_width must be positive")
        self.quality_delta = quality_delta
        self.minimum_teacher_quality = minimum_teacher_quality
        self.private_cosine_margin = private_cosine_margin
        self.classifiers = nn.ModuleDict(
            {
                expert: nn.Linear(expert_widths[expert], num_classes, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.role_encoders = nn.ModuleDict(
            {expert: nn.Linear(1, role_width) for expert in EXPERT_ORDER}
        )
        self.directional_adapters = nn.ModuleDict(
            {
                f"{student}__{teacher}": nn.Linear(
                    role_width, role_width, bias=False
                )
                for student in EXPERT_ORDER
                for teacher in EXPERT_ORDER
                if student != teacher
            }
        )

    @staticmethod
    def _masked_modality_mean(
        values: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        denominator = modality_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (
            values * modality_mask.to(dtype=values.dtype)[..., None]
        ).sum(dim=1) / denominator

    def _role_descriptor(
        self,
        expert: str,
        states: ExpertStateMap,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        summaries = []
        for payload in states[expert].role_payload.values():
            if payload.shape[:2] != modality_mask.shape:
                raise ValueError("role payload must lead with B,M")
            reduced = payload
            if reduced.ndim > 2:
                reduced = reduced.mean(dim=tuple(range(2, reduced.ndim)))
            summaries.append(
                (
                    reduced * modality_mask.to(dtype=reduced.dtype)
                ).sum(dim=1, keepdim=True)
                / modality_mask.sum(dim=1, keepdim=True).clamp_min(1)
            )
        if not summaries:
            raise ValueError("each expert must expose a role payload")
        scalar_role = torch.stack(summaries, dim=-1).mean(dim=-1)
        return self.role_encoders[expert](scalar_role)

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        labels: torch.Tensor,
    ) -> PeerTeachingResult:
        modality_mask = states.modality_mask
        if not torch.equal(reliability.modality_mask, modality_mask):
            raise ValueError("peer teaching requires the shared modality mask")
        if labels.ndim != 1 or labels.shape[0] != modality_mask.shape[0]:
            raise ValueError("labels must be a length-B tensor")

        mask_float = modality_mask.to(dtype=reliability.r.dtype)
        denominator = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        quality = (reliability.r * mask_float[:, None, :]).sum(dim=2) / denominator
        detached_quality = quality.detach()
        student_quality = detached_quality[:, :, None]
        teacher_quality = detached_quality[:, None, :]
        gates = (
            (teacher_quality > student_quality + self.quality_delta)
            & (teacher_quality >= self.minimum_teacher_quality)
        )
        no_self = ~torch.eye(
            len(EXPERT_ORDER), dtype=torch.bool, device=gates.device
        )
        gates = gates & no_self[None]

        logits = {}
        roles = {}
        private_embeddings = {}
        for expert in EXPERT_ORDER:
            pooled_global = self._masked_modality_mean(
                states[expert].global_embedding, modality_mask
            )
            logits[expert] = self.classifiers[expert](pooled_global)
            roles[expert] = self._role_descriptor(
                expert, states, modality_mask
            )
            private_embeddings[expert] = self._masked_modality_mean(
                states[expert].private_embedding, modality_mask
            )

        logit_terms = []
        role_terms = []
        active_terms = []
        for student_index, student in enumerate(EXPERT_ORDER):
            for teacher_index, teacher in enumerate(EXPERT_ORDER):
                if student == teacher:
                    continue
                active = gates[:, student_index, teacher_index]
                if not bool(active.any()):
                    continue
                teacher_probabilities = torch.softmax(
                    logits[teacher].detach(), dim=-1
                )
                logit_term = F.kl_div(
                    torch.log_softmax(logits[student], dim=-1),
                    teacher_probabilities,
                    reduction="none",
                ).sum(dim=-1)
                adapted_student = self.directional_adapters[
                    f"{student}__{teacher}"
                ](roles[student])
                role_term = (
                    adapted_student - roles[teacher].detach()
                ).square().mean(dim=-1)
                logit_terms.append(logit_term)
                role_terms.append(role_term)
                active_terms.append(active)

        zero = sum(value.sum() for value in logits.values()) * 0.0
        if not active_terms:
            logit_kl = zero
            role_loss = zero
        else:
            active_stack = torch.stack(active_terms, dim=1)
            logit_stack = torch.stack(logit_terms, dim=1)
            role_stack = torch.stack(role_terms, dim=1)
            active_float = active_stack.to(dtype=logit_stack.dtype)
            count = active_float.sum().clamp_min(1.0)
            logit_kl = (logit_stack * active_float).sum() / count
            role_loss = (role_stack * active_float).sum() / count

        diversity_terms = []
        for left_index, left in enumerate(EXPERT_ORDER):
            for right in EXPERT_ORDER[left_index + 1 :]:
                cosine = F.cosine_similarity(
                    private_embeddings[left], private_embeddings[right], dim=-1
                )
                diversity_terms.append(
                    F.relu(cosine - self.private_cosine_margin)
                )
        private_diversity = torch.stack(diversity_terms, dim=1).mean()
        rejection_rate = (~gates.any(dim=2)).to(dtype=quality.dtype).mean()
        direction_frequency = gates.to(dtype=quality.dtype).mean(dim=0)
        return PeerTeachingResult(
            direction_gates=gates,
            logit_kl=logit_kl,
            role_loss=role_loss,
            private_diversity=private_diversity,
            rejection_rate=rejection_rate,
            direction_frequency=direction_frequency,
        )


__all__ = ["PeerTeachingResult", "RoleDirectedPeerTeaching"]
