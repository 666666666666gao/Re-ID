"""Complementarity-activated Signal-preserving TriFusion V6."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import TriFusionCriterion, _batch_hard_triplet
from .state import (
    EXPERT_ORDER,
    MODALITY_ORDER,
    ExpertStateMap,
    ReliabilityResult,
)
from .task_anchor_v4 import identity_utility_router_loss
from .signal_preserving_v5 import FrozenSignalBackbone


@dataclass(frozen=True, eq=False)
class ComplementarityActivatedFusionResult:
    fused_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    weights: torch.Tensor
    modality_mask: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_embeddings",
            MappingProxyType(dict(self.branch_embeddings)),
        )
        object.__setattr__(
            self,
            "residual_embeddings",
            MappingProxyType(dict(self.residual_embeddings)),
        )


class ComplementarityActivatedResidualBankFusion(nn.Module):
    """Preserve Signal and give the routed residual bank equal retrieval energy."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        baseline_width: int,
        residual_width: int = 256,
    ) -> None:
        super().__init__()
        if tuple(expert_widths) != EXPERT_ORDER:
            raise ValueError(f"expert widths must follow {EXPERT_ORDER}")
        widths = {int(value) for value in expert_widths.values()}
        if len(widths) != 1 or baseline_width <= 0 or residual_width <= 0:
            raise ValueError("fusion widths must be positive and expert widths shared")
        expert_width = widths.pop()
        self.baseline_width = int(baseline_width)
        self.residual_width = int(residual_width)
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

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        *,
        baseline_embedding: torch.Tensor,
        direct_modal: torch.Tensor,
    ) -> ComplementarityActivatedFusionResult:
        batch_size = modality_mask.shape[0]
        if baseline_embedding.shape != (batch_size, self.baseline_width):
            raise ValueError("baseline embedding has the wrong shape")
        if direct_modal.shape[:2] != modality_mask.shape:
            raise ValueError("direct modal features must have shape B,3,D")
        if reliability.r.shape != (
            batch_size,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
        ):
            raise ValueError("reliability has the wrong shape")

        valid = modality_mask[:, None, :, None].to(direct_modal.dtype)
        direct_norm = direct_modal.detach().norm(dim=-1, keepdim=True)
        projected = []
        for expert in EXPERT_ORDER:
            delta = states[expert].global_embedding - direct_modal
            residual = self.residual_projections[expert](
                self.residual_norms[expert](delta)
            )
            projected.append(F.normalize(residual, dim=-1) * direct_norm)
        contributions = torch.stack(projected, dim=1) * valid

        baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
        residual_embeddings = {
            expert: F.normalize(contributions[:, index].flatten(1), dim=1)
            * baseline_norm
            for index, expert in enumerate(EXPERT_ORDER)
        }
        confidence = reliability.r * (1.0 - reliability.u).clamp_min(0.05)
        confidence = confidence * modality_mask[:, None].to(confidence.dtype)
        weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
        routed_bank = (contributions * weights[..., None]).flatten(1)
        activated_bank = F.normalize(routed_bank, dim=1) * baseline_norm

        fused_embedding = torch.cat((baseline_embedding, activated_bank), dim=1)
        branch_embeddings = {
            expert: torch.cat((baseline_embedding, residual_embeddings[expert]), dim=1)
            for expert in EXPERT_ORDER
        }
        return ComplementarityActivatedFusionResult(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            residual_embeddings=residual_embeddings,
            contribution_embeddings=contributions,
            weights=weights,
            modality_mask=modality_mask,
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV6Output:
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
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in (
            "branch_embeddings",
            "residual_embeddings",
            "branch_logits",
            "residual_logits",
            "diagnostics",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


class SignalPreservingCollaborativeReIDV6(nn.Module):
    """Exact Signal output plus complementarity-activated collaborative outputs."""

    def __init__(
        self,
        *,
        baseline: FrozenSignalBackbone,
        encoder: nn.Module,
        fusion: ComplementarityActivatedResidualBankFusion,
        num_classes: int,
    ) -> None:
        super().__init__()
        if num_classes < 0:
            raise ValueError("num classes must be nonnegative")
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
            nn.init.normal_(self.fused_classifier.weight, std=0.001)
            for classifiers in (self.branch_classifiers, self.residual_classifiers):
                for classifier in classifiers.values():
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
    ) -> torch.Tensor | SignalPreservingV6Output:
        del targets
        field = self.baseline(batch)
        if retrieval_output == "baseline_only" and not return_aux:
            return field.baseline_embedding
        states = self.encoder.forward_token_field(
            field.expert_tokens, field.modality_mask
        )
        if states.reliability is None:
            raise RuntimeError("V6 collaborative encoder did not emit reliability")
        fusion = self.fusion(
            states,
            states.reliability,
            field.modality_mask,
            baseline_embedding=field.baseline_embedding,
            direct_modal=field.direct_modal,
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
            field.baseline_embedding,
            fusion.fused_embedding,
            fusion.contribution_embeddings,
            fusion.weights,
            states.reliability.r,
            states.reliability.u,
            *fusion.branch_embeddings.values(),
            *fusion.residual_embeddings.values(),
        ]
        if fused_logits is not None:
            finite.extend(
                (fused_logits, *branch_logits.values(), *residual_logits.values())
            )
        suffix_norm = fusion.fused_embedding[:, self.baseline_embedding_width :].norm(
            dim=1
        )
        baseline_norm = field.baseline_embedding.norm(dim=1)
        return SignalPreservingV6Output(
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
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    fusion.fused_embedding[:, : self.baseline_embedding_width],
                    field.baseline_embedding,
                ),
                "baseline_frozen": all(
                    not parameter.requires_grad
                    for parameter in self.baseline.signal.parameters()
                ),
                "has_all_three_experts": tuple(fusion.branch_embeddings)
                == EXPERT_ORDER,
                "residual_energy_activated": bool(
                    torch.allclose(suffix_norm, baseline_norm, rtol=1e-3, atol=1e-5)
                ),
            },
        )


class ComplementarityActivatedV6Criterion(TriFusionCriterion):
    """Train residual experts directly and route by their identity utility."""

    def forward(
        self,
        output: object,
        labels: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        losses = super().forward(output, labels, **kwargs)
        for expert in EXPERT_ORDER:
            losses[f"id_residual_{expert}"] = F.cross_entropy(
                output.residual_logits[expert], labels
            )
            losses[f"triplet_residual_{expert}"] = _batch_hard_triplet(
                output.residual_embeddings[expert], labels, self.triplet_margin
            )
        routing = identity_utility_router_loss(
            output.residual_embeddings,
            output.router_weights,
            labels,
            modality_mask=output.modality_mask,
        )
        losses["peer_logits"] = routing.loss
        return losses


__all__ = [
    "ComplementarityActivatedFusionResult",
    "ComplementarityActivatedResidualBankFusion",
    "ComplementarityActivatedV6Criterion",
    "SignalPreservingCollaborativeReIDV6",
    "SignalPreservingV6Output",
]
