"""Task-adapted CLIP anchor with bounded collaborative expert residuals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .cascade_v2 import StageUpdatedTriBranchEncoder
from .criterion import TriFusionCriterion, _batch_hard_triplet
from .fusion import FusionResult
from .state import EXPERT_ORDER, MODALITY_ORDER, ExpertStateMap, ReliabilityResult


@dataclass(frozen=True, eq=False)
class AnchorTokenField:
    """One exact CLIP anchor and one centered local field per valid modality."""

    anchor_native: torch.Tensor
    anchor_projected: torch.Tensor
    expert_tokens: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.anchor_native.ndim != 2 or self.anchor_projected.ndim != 2:
            raise ValueError("anchor tensors must be rank two")
        if self.anchor_native.shape[0] != self.anchor_projected.shape[0]:
            raise ValueError("native and projected anchors must share a batch")
        if set(self.expert_tokens) != set(EXPERT_ORDER):
            raise ValueError(f"expert token fields must contain exactly {EXPERT_ORDER}")
        ordered = {expert: self.expert_tokens[expert] for expert in EXPERT_ORDER}
        for tokens in ordered.values():
            if tokens.ndim != 3 or tokens.shape[0] != self.anchor_native.shape[0]:
                raise ValueError("expert tokens must have shape Nv,N,D")
            if tokens.shape[-1] != self.anchor_native.shape[-1]:
                raise ValueError("expert tokens and native anchor widths must match")
        object.__setattr__(self, "expert_tokens", MappingProxyType(ordered))


class TaskAdaptedAnchorTokenizer(nn.Module):
    """Run one shared CLIP trunk and retain its projected CLS without rewriting."""

    center_patch_residuals = True
    exact_projected_cls = True

    def __init__(
        self,
        *,
        patch_projection: nn.Conv2d,
        positional_embedding: nn.Parameter,
        class_embedding: nn.Parameter,
        pre_norm: nn.Module,
        post_norm: nn.Module,
        shared_blocks: Sequence[nn.Module],
        output_projection: nn.Parameter,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        width = int(patch_projection.out_channels)
        if positional_embedding.ndim != 2 or positional_embedding.shape[1] != width:
            raise ValueError("CLIP positional embedding has the wrong width")
        if class_embedding.shape != (width,):
            raise ValueError("CLIP class embedding has the wrong width")
        if output_projection.ndim != 2 or output_projection.shape[0] != width:
            raise ValueError("CLIP output projection must have shape D,Dout")
        if not shared_blocks:
            raise ValueError("at least one shared CLIP block is required")
        self.width = width
        self.output_width = int(output_projection.shape[1])
        self.patch_projection = patch_projection
        self.positional_embedding = positional_embedding
        self.class_embedding = class_embedding
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.shared_blocks = nn.ModuleList(list(shared_blocks))
        self.output_projection = output_projection
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def _run_block(self, block: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(block, sequence, use_reentrant=False)
        return block(sequence)

    def forward(
        self,
        packed_images: torch.Tensor,
        packed_modalities: torch.Tensor,
    ) -> AnchorTokenField:
        if packed_images.ndim != 4 or packed_images.shape[1] != 3:
            raise ValueError("packed_images must have shape Nv,3,H,W")
        expected = (packed_images.shape[0],)
        if packed_modalities.dtype != torch.long or packed_modalities.shape != expected:
            raise ValueError("packed_modalities must be a length-Nv long tensor")
        if packed_modalities.numel() and (
            int(packed_modalities.min()) < 0
            or int(packed_modalities.max()) >= len(MODALITY_ORDER)
        ):
            raise ValueError("packed modality indices are out of range")
        dtype = self.patch_projection.weight.dtype
        patches = self.patch_projection(packed_images.to(dtype=dtype))
        patches = patches.flatten(2).transpose(1, 2)
        token_count = int(patches.shape[1])
        if self.positional_embedding.shape != (token_count + 1, self.width):
            raise ValueError("CLIP position table does not match the patch grid")
        class_tokens = self.class_embedding.to(
            device=patches.device, dtype=patches.dtype
        ).view(1, 1, -1).expand(patches.shape[0], 1, -1)
        sequence = torch.cat((class_tokens, patches), dim=1)
        sequence = sequence + self.positional_embedding.to(
            device=sequence.device, dtype=sequence.dtype
        ).unsqueeze(0)
        sequence = self.pre_norm(sequence)
        for block in self.shared_blocks:
            sequence = self._run_block(block, sequence)
        sequence = self.post_norm(sequence)

        anchor_native = sequence[:, 0]
        anchor_projected = anchor_native @ self.output_projection.to(
            device=anchor_native.device, dtype=anchor_native.dtype
        )
        local = sequence[:, 1:] - sequence[:, 1:].mean(dim=1, keepdim=True)
        tokens = anchor_native[:, None] + local
        return AnchorTokenField(
            anchor_native=anchor_native,
            anchor_projected=anchor_projected,
            expert_tokens={expert: tokens for expert in EXPERT_ORDER},
        )


class TaskAnchoredTriBranchEncoder(StageUpdatedTriBranchEncoder):
    """Run the existing three-stage experts from an externally retained token field."""

    def __init__(
        self,
        experts: Mapping[str, nn.Module],
        *,
        reliability_gate: nn.Module,
        collaborator: nn.Module,
    ) -> None:
        super().__init__(
            experts,
            tokenizer=nn.Identity(),
            reliability_gate=reliability_gate,
            collaborator=collaborator,
            refresh_final_reliability=True,
        )

    def forward_token_field(
        self,
        expert_tokens: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> ExpertStateMap:
        return self._forward_collaborative(
            expert_tokens,
            modality_mask,
            intervention=None,
        )

    def forward(self, *_args: Any, **_kwargs: Any) -> ExpertStateMap:
        raise RuntimeError("TaskAnchoredTriBranchEncoder requires forward_token_field")


class AnchorResidualCollaborativeFusion(nn.Module):
    """Keep the anchor intact and append one norm-bounded, quality-routed residual."""

    baseline_safe_zero_mode = True

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        embedding_width: int = 512,
        residual_scale_init: float = 0.25,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        widths = {int(value) for value in expert_widths.values()}
        if len(widths) != 1 or embedding_width <= 0:
            raise ValueError("V3 requires one positive shared semantic width")
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError("residual_scale_init must be strictly between zero and one")
        semantic_width = widths.pop()
        self.embedding_width = int(embedding_width)
        self.anchor_embedding_width = len(MODALITY_ORDER) * self.embedding_width
        self.fused_embedding_width = 2 * self.anchor_embedding_width
        self.branch_embedding_width = self.fused_embedding_width
        self.residual_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(semantic_width) for expert in EXPERT_ORDER}
        )
        self.residual_projections = nn.ModuleDict(
            {
                expert: nn.Linear(semantic_width, self.embedding_width, bias=False)
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
        anchor_norm = anchor_projected.detach().norm(dim=-1, keepdim=True)
        projected = []
        scales = torch.sigmoid(self.residual_scale_logits)
        for index, expert in enumerate(EXPERT_ORDER):
            delta = states[expert].global_embedding - anchor_native
            contribution = self.residual_projections[expert](
                self.residual_norms[expert](delta)
            )
            relative_cap = (
                anchor_norm / contribution.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ).clamp_max(1.0)
            projected.append(
                contribution * relative_cap * valid * scales[index]
            )
        contributions = torch.stack(projected, dim=1)

        confidence = reliability.r * (1.0 - reliability.u).clamp_min(0.05)
        confidence = confidence * modality_mask[:, None].to(confidence.dtype)
        weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused_residual = (contributions * weights[..., None]).sum(dim=1)
        if force_zero_residual:
            contributions_for_output = torch.zeros_like(contributions)
            fused_residual = torch.zeros_like(fused_residual)
        else:
            contributions_for_output = contributions

        anchor_flat = anchor_projected.flatten(1)
        fused_embedding = torch.cat((anchor_flat, fused_residual.flatten(1)), dim=1)
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
class TaskAnchoredV3Output:
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
    anchor_embedding: torch.Tensor
    anchor_modal: torch.Tensor
    anchor_logits: torch.Tensor | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_embeddings", MappingProxyType(dict(self.branch_embeddings))
        )
        object.__setattr__(self, "branch_logits", MappingProxyType(dict(self.branch_logits)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class TaskAnchoredCollaborativeReID(nn.Module):
    """End-to-end V3 model with a direct Signal-style CLIP retrieval anchor."""

    retrieval_before_neck = True

    def __init__(
        self,
        *,
        tokenizer: TaskAdaptedAnchorTokenizer,
        encoder: TaskAnchoredTriBranchEncoder,
        fusion: AnchorResidualCollaborativeFusion,
        num_classes: int,
    ) -> None:
        super().__init__()
        if num_classes < 0:
            raise ValueError("num_classes must be nonnegative")
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.anchor_embedding_width = fusion.anchor_embedding_width
        self.fused_embedding_width = fusion.fused_embedding_width
        self.branch_embedding_width = fusion.branch_embedding_width
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.anchor_neck = self._make_neck(self.anchor_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {expert: self._make_neck(self.branch_embedding_width) for expert in EXPERT_ORDER}
        )
        if num_classes:
            self.fused_classifier = nn.Linear(self.fused_embedding_width, num_classes, bias=False)
            self.anchor_classifier = nn.Linear(self.anchor_embedding_width, num_classes, bias=False)
            self.branch_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(self.branch_embedding_width, num_classes, bias=False)
                    for expert in EXPERT_ORDER
                }
            )
            for classifier in (
                self.fused_classifier,
                self.anchor_classifier,
                *self.branch_classifiers.values(),
            ):
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.anchor_classifier = None
            self.branch_classifiers = nn.ModuleDict()

    @staticmethod
    def _make_neck(width: int) -> nn.BatchNorm1d:
        neck = nn.BatchNorm1d(width)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        neck.bias.requires_grad_(False)
        return neck

    @staticmethod
    def _validate_batch(images: Mapping[str, torch.Tensor], mask: torch.Tensor) -> None:
        if tuple(images) != MODALITY_ORDER:
            raise ValueError(f"images must follow modality order {MODALITY_ORDER}")
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != 3:
            raise ValueError("modality_mask must have shape B,3 and bool dtype")
        for image in images.values():
            if image.ndim != 4 or image.shape[0] != mask.shape[0]:
                raise ValueError("each image tensor must have shape B,C,H,W")

    @staticmethod
    def _scatter(packed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = packed.new_zeros((mask.shape[0], mask.shape[1], *packed.shape[1:]))
        output[mask] = packed
        return output

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | TaskAnchoredV3Output:
        if "images" not in batch or "modality_mask" not in batch:
            raise ValueError("batch must contain images and modality_mask")
        images = batch["images"]
        mask = batch["modality_mask"]
        self._validate_batch(images, mask)
        if bool((~mask).all(dim=1).any()):
            raise ValueError("every row requires at least one modality")
        stacked = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
        modality_indices = torch.arange(3, device=mask.device).view(1, 3).expand_as(mask)
        field = self.tokenizer(
            stacked[mask],
            modality_indices[mask],
        )
        states = self.encoder.forward_token_field(field.expert_tokens, mask)
        if states.reliability is None:
            raise RuntimeError("V3 collaborative encoder did not emit reliability")
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
            states.reliability.r,
            states.reliability.u,
            *fusion.branch_embeddings.values(),
        ]
        if fused_logits is not None:
            finite.extend((fused_logits, anchor_logits, *branch_logits.values()))
        return TaskAnchoredV3Output(
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
            },
            anchor_embedding=anchor_embedding,
            anchor_modal=anchor_projected,
            anchor_logits=anchor_logits,
        )


def supervised_cross_modal_alignment(
    modal_embeddings: torch.Tensor,
    labels: torch.Tensor,
    modality_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-positive identity alignment across each ordered modality pair."""

    if modal_embeddings.ndim != 3 or modal_embeddings.shape[:2] != modality_mask.shape:
        raise ValueError("modal embeddings must have shape B,M,D")
    if labels.shape != (modal_embeddings.shape[0],) or temperature <= 0.0:
        raise ValueError("alignment labels or temperature are invalid")
    normalized = F.normalize(modal_embeddings.float(), dim=-1)
    losses = []
    for source in range(modal_embeddings.shape[1]):
        for target in range(modal_embeddings.shape[1]):
            if source == target:
                continue
            source_valid = modality_mask[:, source]
            target_valid = modality_mask[:, target]
            if not bool(source_valid.any()) or not bool(target_valid.any()):
                continue
            source_features = normalized[source_valid, source]
            target_features = normalized[target_valid, target]
            source_labels = labels[source_valid]
            target_labels = labels[target_valid]
            positives = source_labels[:, None] == target_labels[None, :]
            usable = positives.any(dim=1)
            if not bool(usable.any()):
                continue
            log_probabilities = F.log_softmax(
                source_features @ target_features.T / temperature,
                dim=1,
            )
            positive_weights = positives.to(log_probabilities.dtype)
            positive_weights = positive_weights / positive_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            losses.append(
                -(log_probabilities * positive_weights).sum(dim=1)[usable].mean()
            )
    if not losses:
        return modal_embeddings.sum() * 0.0
    return torch.stack(losses).mean()


class TaskAnchoredV3Criterion(TriFusionCriterion):
    """Signal-style anchor/fused retrieval losses plus identity-aware alignment."""

    def __init__(
        self,
        *,
        target_cache: Any,
        triplet_margin: float = 0.3,
        brier_weight: float = 1.0,
        evidence_weight: float = 0.1,
        label_smoothing: float = 0.1,
        alignment_temperature: float = 0.07,
    ) -> None:
        if not 0.0 <= label_smoothing < 1.0 or alignment_temperature <= 0.0:
            raise ValueError("V3 criterion configuration is invalid")
        super().__init__(
            target_cache=target_cache,
            triplet_margin=triplet_margin,
            brier_weight=brier_weight,
            evidence_weight=evidence_weight,
        )
        self.label_smoothing = float(label_smoothing)
        self.alignment_temperature = float(alignment_temperature)

    def forward(
        self,
        output: TaskAnchoredV3Output,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        losses = super().forward(output, labels, **kwargs)
        losses["id_fused"] = F.cross_entropy(
            output.fused_logits, labels, label_smoothing=self.label_smoothing
        )
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"] = F.cross_entropy(
                output.branch_logits[expert],
                labels,
                label_smoothing=self.label_smoothing,
            )
        if output.anchor_logits is None:
            raise ValueError("anchor supervision requires anchor logits")
        anchor_identity = F.cross_entropy(
            output.anchor_logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        anchor_triplet = _batch_hard_triplet(
            output.anchor_embedding,
            labels,
            self.triplet_margin,
        )
        losses["reliability"] = 0.25 * anchor_identity + anchor_triplet
        losses["private_diversity"] = supervised_cross_modal_alignment(
            output.anchor_modal,
            labels,
            output.modality_mask,
            temperature=self.alignment_temperature,
        )
        return losses


__all__ = [
    "AnchorResidualCollaborativeFusion",
    "AnchorTokenField",
    "TaskAdaptedAnchorTokenizer",
    "TaskAnchoredCollaborativeReID",
    "TaskAnchoredTriBranchEncoder",
    "TaskAnchoredV3Criterion",
    "TaskAnchoredV3Output",
    "supervised_cross_modal_alignment",
]
