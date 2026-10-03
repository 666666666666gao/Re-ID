"""End-to-end TriFusion-ReID model seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from .fusion import CollaborativeFusion
from .interventions import FullNetworkIntervention
from .state import EXPERT_ORDER, ReliabilityResult


@dataclass(frozen=True, eq=False)
class TriFusionOutput:
    fused_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    reliability: ReliabilityResult
    relay_results: tuple[Any, ...]
    peer_teaching: Any
    fused_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    modality_mask: torch.Tensor
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_embeddings",
            MappingProxyType(dict(self.branch_embeddings)),
        )
        object.__setattr__(
            self, "branch_logits", MappingProxyType(dict(self.branch_logits))
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


class TriFusionReID(nn.Module):
    """Run deep collaboration once and expose fused plus named branch outputs."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        fusion: CollaborativeFusion,
        embedding_width: int = 512,
        num_classes: int = 0,
        peer_teaching: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if embedding_width <= 0 or num_classes < 0:
            raise ValueError("embedding_width must be positive and classes nonnegative")
        self.encoder = encoder
        self.fusion = fusion
        self.peer_teaching = peer_teaching
        self.num_classes = num_classes
        self.fused_neck = self._make_neck(embedding_width)
        self.branch_necks = nn.ModuleDict(
            {expert: self._make_neck(embedding_width) for expert in EXPERT_ORDER}
        )
        if num_classes:
            self.fused_classifier = nn.Linear(
                embedding_width, num_classes, bias=False
            )
            self.branch_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(embedding_width, num_classes, bias=False)
                    for expert in EXPERT_ORDER
                }
            )
            nn.init.normal_(self.fused_classifier.weight, std=0.001)
            for classifier in self.branch_classifiers.values():
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.branch_classifiers = nn.ModuleDict()

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
    ) -> torch.Tensor | TriFusionOutput:
        if "images" not in batch or "modality_mask" not in batch:
            raise ValueError("batch must contain images and modality_mask")
        images = batch["images"]
        modality_mask = batch["modality_mask"]
        if targets is None:
            targets = batch.get("targets")

        intervention = None
        if "intervention" in batch:
            if self.training:
                raise RuntimeError("full-network interventions are evaluation-only")
            intervention = FullNetworkIntervention.from_value(batch["intervention"])
            intervention.validate_modality_mask(modality_mask)

        if intervention is None:
            states = self.encoder(images, modality_mask)
        else:
            states = self.encoder._forward_intervened(
                images,
                modality_mask,
                intervention,
            )
        if states.reliability is None:
            raise RuntimeError("TriFusionReID requires the collaborative encoder posterior")
        if intervention is None:
            fusion_result = self.fusion(
                states, states.reliability, modality_mask
            )
        else:
            fusion_result = self.fusion._forward_intervened(
                states,
                states.reliability,
                modality_mask,
                intervention,
            )
        fused_embedding = self.fused_neck(fusion_result.fused_embedding)
        branch_embeddings = {
            expert: self.branch_necks[expert](
                fusion_result.branch_embeddings[expert]
            )
            for expert in EXPERT_ORDER
        }

        fused_logits = None
        branch_logits = {}
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(fused_embedding)
            branch_logits = {
                expert: self.branch_classifiers[expert](branch_embeddings[expert])
                for expert in EXPERT_ORDER
            }

        peer_result = None
        if self.training and targets is not None and self.peer_teaching is not None:
            peer_result = self.peer_teaching(
                states, states.reliability, targets
            )

        if not return_aux:
            return fused_embedding

        finite_tensors = [
            fused_embedding,
            fusion_result.contribution_embeddings,
            states.reliability.r,
            states.reliability.u,
            *branch_embeddings.values(),
        ]
        if fused_logits is not None:
            finite_tensors.append(fused_logits)
            finite_tensors.extend(branch_logits.values())
        diagnostics = {
            "all_finite": all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in finite_tensors
            ),
            "has_all_three_experts": tuple(branch_embeddings) == EXPERT_ORDER,
        }
        return TriFusionOutput(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=fusion_result.contribution_embeddings,
            reliability=states.reliability,
            relay_results=states.relay_results,
            peer_teaching=peer_result,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            modality_mask=modality_mask,
            diagnostics=diagnostics,
        )


__all__ = ["TriFusionReID", "TriFusionOutput"]
