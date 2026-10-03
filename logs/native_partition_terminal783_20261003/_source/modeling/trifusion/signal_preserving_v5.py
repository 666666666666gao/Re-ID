"""Signal-preserving CNN/Transformer/Mamba collaborative retrieval model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import TriFusionCriterion
from .fusion import FusionResult
from .state import EXPERT_ORDER, MODALITY_ORDER, ExpertStateMap, ReliabilityResult
from .task_anchor_v4 import identity_utility_router_loss


@dataclass(frozen=True, eq=False)
class FrozenSignalField:
    """Exact Signal retrieval feature plus the frozen semantic token field."""

    baseline_embedding: torch.Tensor
    direct_modal: torch.Tensor
    expert_tokens: Mapping[str, torch.Tensor]
    modality_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.baseline_embedding.ndim != 2:
            raise ValueError("baseline embedding must have shape B,D")
        if self.direct_modal.ndim != 3 or self.direct_modal.shape[:2] != (
            self.baseline_embedding.shape[0],
            len(MODALITY_ORDER),
        ):
            raise ValueError("direct modal features must have shape B,3,D")
        if tuple(self.expert_tokens) != EXPERT_ORDER:
            raise ValueError(f"expert tokens must follow {EXPERT_ORDER}")
        object.__setattr__(
            self,
            "expert_tokens",
            MappingProxyType(dict(self.expert_tokens)),
        )


class FrozenSignalBackbone(nn.Module):
    """Expose Signal's exact direct+SIM feature without updating its checkpoint."""

    def __init__(self, signal: nn.Module, *, feature_width: int = 512) -> None:
        super().__init__()
        if feature_width <= 0:
            raise ValueError("feature width must be positive")
        if not hasattr(signal, "clip_vision_encoder") or not hasattr(signal, "SIM"):
            raise ValueError("Signal model must expose clip_vision_encoder and SIM")
        self.signal = signal
        self.feature_width = int(feature_width)
        self.baseline_width = len(MODALITY_ORDER) * 2 * self.feature_width
        for parameter in self.signal.parameters():
            parameter.requires_grad_(False)
        self.signal.eval()

    def train(self, mode: bool = True) -> FrozenSignalBackbone:
        super().train(mode)
        self.signal.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> FrozenSignalField:
        if "images" not in batch or "modality_mask" not in batch:
            raise ValueError("batch must contain images and modality_mask")
        images = batch["images"]
        modality_mask = batch["modality_mask"]
        if tuple(images) != MODALITY_ORDER:
            raise ValueError(f"images must follow {MODALITY_ORDER}")
        if (
            modality_mask.dtype != torch.bool
            or modality_mask.ndim != 2
            or modality_mask.shape[1] != len(MODALITY_ORDER)
            or not bool(modality_mask.all())
        ):
            raise ValueError("V5 Signal baseline requires all RGB/NI/TI modalities")
        camera_ids = batch.get("camera_ids")
        if camera_ids is None or camera_ids.shape != (modality_mask.shape[0],):
            raise ValueError("camera_ids must have shape B")

        patches = []
        globals_by_modality = []
        with torch.no_grad():
            for modality in MODALITY_ORDER:
                patch, global_embedding = self.signal.clip_vision_encoder(
                    images[modality],
                    cam_label=camera_ids,
                    view_label=None,
                )
                patches.append(patch)
                globals_by_modality.append(global_embedding)
            sim = self.signal.SIM(*patches, *globals_by_modality)
            direct_modal = torch.stack(globals_by_modality, dim=1)
            baseline_embedding = torch.cat((direct_modal.flatten(1), sim), dim=1)
            semantic_tokens = torch.stack(
                [
                    patch + global_embedding[:, None]
                    for patch, global_embedding in zip(
                        patches, globals_by_modality, strict=True
                    )
                ],
                dim=1,
            )

        expected = (modality_mask.shape[0], self.baseline_width)
        if baseline_embedding.shape != expected:
            raise ValueError(f"Signal baseline feature must have shape {expected}")
        packed_tokens = semantic_tokens.flatten(0, 1)
        return FrozenSignalField(
            baseline_embedding=baseline_embedding,
            direct_modal=direct_modal,
            expert_tokens={expert: packed_tokens for expert in EXPERT_ORDER},
            modality_mask=modality_mask,
        )


class SignalPreservingResidualBankFusion(nn.Module):
    """Append quality-routed expert residuals without rewriting Signal bytes."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        baseline_width: int,
        residual_width: int = 256,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(expert_widths) != EXPERT_ORDER:
            raise ValueError(f"expert widths must follow {EXPERT_ORDER}")
        widths = {int(value) for value in expert_widths.values()}
        if len(widths) != 1 or baseline_width <= 0 or residual_width <= 0:
            raise ValueError("fusion widths must be positive and expert widths shared")
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError("residual scale init must be between zero and one")
        expert_width = widths.pop()
        self.baseline_width = int(baseline_width)
        self.residual_width = int(residual_width)
        self.branch_embedding_width = (
            self.baseline_width + len(MODALITY_ORDER) * self.residual_width
        )
        self.residual_bank_width = (
            len(EXPERT_ORDER) * len(MODALITY_ORDER) * self.residual_width
        )
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
        initial_logit = math.log(residual_scale_init / (1.0 - residual_scale_init))
        self.residual_scale_logits = nn.Parameter(
            torch.full((len(EXPERT_ORDER),), initial_logit)
        )

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        *,
        baseline_embedding: torch.Tensor,
        direct_modal: torch.Tensor,
    ) -> FusionResult:
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
        scales = torch.sigmoid(self.residual_scale_logits)
        projected = []
        for index, expert in enumerate(EXPERT_ORDER):
            delta = states[expert].global_embedding - direct_modal
            residual = self.residual_projections[expert](
                self.residual_norms[expert](delta)
            )
            residual = F.normalize(residual, dim=-1) * direct_norm
            projected.append(residual * scales[index])
        contributions = torch.stack(projected, dim=1) * valid

        confidence = reliability.r * (1.0 - reliability.u).clamp_min(0.05)
        confidence = confidence * modality_mask[:, None].to(confidence.dtype)
        weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weighted_bank = contributions * weights[..., None]
        fused_embedding = torch.cat((baseline_embedding, weighted_bank.flatten(1)), dim=1)
        branch_embeddings = {
            expert: torch.cat(
                (baseline_embedding, contributions[:, index].flatten(1)), dim=1
            )
            for index, expert in enumerate(EXPERT_ORDER)
        }
        return FusionResult(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=contributions,
            weights=weights,
            modality_mask=modality_mask,
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV5Output:
    fused_embedding: torch.Tensor
    baseline_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    contribution_embeddings: torch.Tensor
    reliability: ReliabilityResult
    relay_results: tuple[Any, ...]
    peer_teaching: None
    fused_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    modality_mask: torch.Tensor
    router_weights: torch.Tensor
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_embeddings", MappingProxyType(dict(self.branch_embeddings))
        )
        object.__setattr__(self, "branch_logits", MappingProxyType(dict(self.branch_logits)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class SignalPreservingCollaborativeReID(nn.Module):
    """One checkpoint with exact baseline-only and collaborative fused outputs."""

    def __init__(
        self,
        *,
        baseline: FrozenSignalBackbone,
        encoder: nn.Module,
        fusion: SignalPreservingResidualBankFusion,
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
        self.fused_embedding_width = fusion.fused_embedding_width
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {expert: self._make_neck(self.branch_embedding_width) for expert in EXPERT_ORDER}
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
        retrieval_output: str = "fused",
    ) -> torch.Tensor | SignalPreservingV5Output:
        del targets
        field = self.baseline(batch)
        if retrieval_output == "baseline_only" and not return_aux:
            return field.baseline_embedding
        states = self.encoder.forward_token_field(
            field.expert_tokens, field.modality_mask
        )
        if states.reliability is None:
            raise RuntimeError("V5 collaborative encoder did not emit reliability")
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
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(self.fused_neck(fusion.fused_embedding))
            branch_logits = {
                expert: self.branch_classifiers[expert](
                    self.branch_necks[expert](fusion.branch_embeddings[expert])
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
        ]
        if fused_logits is not None:
            finite.extend((fused_logits, *branch_logits.values()))
        return SignalPreservingV5Output(
            fused_embedding=fusion.fused_embedding,
            baseline_embedding=field.baseline_embedding,
            branch_embeddings=fusion.branch_embeddings,
            contribution_embeddings=fusion.contribution_embeddings,
            reliability=states.reliability,
            relay_results=states.relay_results,
            peer_teaching=None,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
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
            },
        )


class SignalPreservingV5Criterion(TriFusionCriterion):
    """Train fused/branch retrieval and the identity-utility quality router."""

    def forward(
        self,
        output: SignalPreservingV5Output,
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
    "FrozenSignalBackbone",
    "FrozenSignalField",
    "SignalPreservingCollaborativeReID",
    "SignalPreservingResidualBankFusion",
    "SignalPreservingV5Criterion",
    "SignalPreservingV5Output",
]
