"""Isolated V2 collaboration modules that preserve the frozen V1 source tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import TriFusionCriterion
from .encoder import TriBranchEncoder
from .fusion import FusionResult
from .intervention_targets import CIRCTargetCache
from .interventions import FullNetworkIntervention
from .model import TriFusionOutput
from .semantic_tokenizer import SharedCLIPSemanticTokenizer
from .state import (
    EXPERT_ORDER,
    MODALITY_ORDER,
    ExpertStateMap,
    ReliabilityResult,
)


class AnchorPreservingSemanticTokenizer(SharedCLIPSemanticTokenizer):
    """Decompose each CLIP field into its exact CLS anchor plus local residuals."""

    center_patch_residuals = True

    def forward(
        self,
        packed_images: torch.Tensor,
        packed_modalities: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        if packed_images.ndim != 4 or packed_images.shape[1] != 3:
            raise ValueError("packed_images must have shape Nv,3,H,W")
        if (
            packed_modalities.dtype != torch.long
            or packed_modalities.ndim != 1
            or packed_modalities.shape[0] != packed_images.shape[0]
        ):
            raise ValueError("packed_modalities must be a length-Nv long tensor")
        if packed_modalities.numel() and (
            int(packed_modalities.min().item()) < 0
            or int(packed_modalities.max().item()) >= len(MODALITY_ORDER)
        ):
            raise ValueError("packed modality indices must be in RGB, NI, TI range")

        projection_dtype = self.patch_projection.weight.dtype
        patches = self.patch_projection(
            packed_images.to(dtype=projection_dtype)
        )
        patches = patches.flatten(2).transpose(1, 2)
        token_count = int(patches.shape[1])
        if self.positional_embedding.shape != (token_count + 1, self.width):
            raise ValueError("CLIP position table does not match the patch grid")

        positions = self.positional_embedding.to(
            device=patches.device,
            dtype=patches.dtype,
        )
        class_tokens = self.class_embedding.to(
            device=patches.device,
            dtype=patches.dtype,
        ).view(1, 1, -1).expand(patches.shape[0], 1, -1)
        sequence = torch.cat((class_tokens, patches), dim=1)
        sequence = sequence + positions.unsqueeze(0)
        modality_offsets = self.modality_embedding(packed_modalities).to(
            dtype=sequence.dtype
        )
        sequence = self.pre_norm(sequence + modality_offsets.unsqueeze(1))
        for block in self.shared_blocks:
            sequence = self._run_block(block, sequence)
        sequence = self.post_norm(sequence)

        patch_residuals = sequence[:, 1:]
        patch_residuals = patch_residuals - patch_residuals.mean(
            dim=1, keepdim=True
        )
        semantic_tokens = sequence[:, :1] + patch_residuals
        return MappingProxyType(
            {expert: semantic_tokens for expert in EXPERT_ORDER}
        )


class StageUpdatedTriBranchEncoder(TriBranchEncoder):
    """Refresh quality after stages one, two, and three before each consumer."""

    refresh_reliability_each_stage = True

    def _forward_collaborative(
        self,
        expert_inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention | None,
    ) -> ExpertStateMap:
        runtimes = {
            expert: self.experts[expert].initialize(expert_inputs[expert])
            for expert in EXPERT_ORDER
        }
        reliability = None
        relay_results = []
        final_states = None
        for stage in (1, 2, 3):
            packed_outputs = {}
            for expert in EXPERT_ORDER:
                runtimes[expert] = self.experts[expert].run_stage(
                    runtimes[expert], stage
                )
                packed_outputs[expert] = self.experts[expert].summarize(
                    runtimes[expert], stage
                )
            stage_states = ExpertStateMap(
                {
                    expert: self._scatter_expert_output(
                        packed_outputs[expert], modality_mask, expert
                    )
                    for expert in EXPERT_ORDER
                },
                modality_mask=modality_mask,
            )
            reliability = self.reliability_gate(stage_states, modality_mask)
            if stage < 3:
                if intervention is None:
                    relay_result = self.collaborator(
                        stage_states, reliability, stage
                    )
                else:
                    relay_result = self.collaborator._forward_intervened(
                        stage_states,
                        reliability,
                        stage,
                        intervention,
                    )
                relay_results.append(relay_result)
                for expert in EXPERT_ORDER:
                    runtimes[expert] = self.experts[expert].inject(
                        runtimes[expert],
                        relay_result.states[expert].tokens[modality_mask],
                    )
            else:
                final_states = stage_states

        if reliability is None or final_states is None:
            raise RuntimeError("collaborative encoder did not complete its schedule")
        return ExpertStateMap(
            {expert: final_states[expert] for expert in EXPERT_ORDER},
            modality_mask=modality_mask,
            reliability=reliability,
            relay_results=tuple(relay_results),
        )


class InformationPreservingFusion(nn.Module):
    """Quality-scale and retain all nine expert-modality blocks plus interaction."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        embedding_width: int = 512,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        semantic_widths = {int(width) for width in expert_widths.values()}
        if len(semantic_widths) != 1:
            raise ValueError(
                "information-preserving fusion requires one shared semantic width"
            )
        if embedding_width <= 0 or residual_scale_init <= 0:
            raise ValueError("embedding width and residual scale must be positive")
        semantic_width = semantic_widths.pop()
        self.embedding_width = int(embedding_width)
        self.fused_embedding_width = self.embedding_width * (
            len(EXPERT_ORDER) * len(MODALITY_ORDER) + 1
        )
        self.branch_embedding_width = self.embedding_width * (
            len(MODALITY_ORDER) + 1
        )
        self.semantic_projection = nn.Linear(
            semantic_width, self.embedding_width, bias=False
        )
        self.residual_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(semantic_width) for expert in EXPERT_ORDER}
        )
        self.residual_projections = nn.ModuleDict(
            {
                expert: nn.Linear(
                    semantic_width, self.embedding_width, bias=False
                )
                for expert in EXPERT_ORDER
            }
        )
        for projection in self.residual_projections.values():
            nn.init.zeros_(projection.weight)
        self.residual_scales = nn.Parameter(
            torch.full((len(EXPERT_ORDER),), float(residual_scale_init))
        )

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
    ) -> FusionResult:
        return self._forward(states, reliability, modality_mask, intervention=None)

    def _forward_intervened(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention,
    ) -> FusionResult:
        return self._forward(
            states,
            reliability,
            modality_mask,
            intervention=intervention,
        )

    def _forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        modality_mask: torch.Tensor,
        intervention: FullNetworkIntervention | None,
    ) -> FusionResult:
        if not torch.equal(states.modality_mask, modality_mask):
            raise ValueError("states and fusion must use the same modality mask")
        if not torch.equal(reliability.modality_mask, modality_mask):
            raise ValueError("reliability and fusion must use the same modality mask")

        valid = modality_mask[:, None, :].expand_as(reliability.r)
        valid_float = valid.to(dtype=reliability.r.dtype)
        contributions = []
        for expert_index, expert in enumerate(EXPERT_ORDER):
            global_embedding = states[expert].global_embedding
            anchor = self.semantic_projection(global_embedding)
            residual = self.residual_projections[expert](
                self.residual_norms[expert](global_embedding)
            )
            contributions.append(
                anchor
                + residual
                * self.residual_scales[expert_index].to(dtype=residual.dtype)
            )
        contribution_tensor = torch.stack(contributions, dim=1)
        contribution_tensor = contribution_tensor * valid_float[..., None]

        scores = reliability.r.clamp_min(0.0) * valid_float
        suppression = (
            intervention.fusion_suppression()
            if intervention is not None
            else None
        )
        if suppression is not None:
            expert, modality = suppression
            allowed = torch.ones_like(scores)
            allowed[
                :, EXPERT_ORDER.index(expert), MODALITY_ORDER.index(modality)
            ] = 0
            scores = scores * allowed
        total = scores.sum(dim=(1, 2), keepdim=True)
        fallback = valid_float / valid_float.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1.0)
        weights = torch.where(
            total > 0,
            scores / total.clamp_min(1e-12),
            fallback,
        ) * valid_float

        valid_count = valid_float.sum(dim=(1, 2), keepdim=True)
        block_scales = (weights * valid_count).clamp_min(0.0).sqrt()
        preserved_blocks = contribution_tensor * block_scales[..., None]
        interaction = (
            contribution_tensor * weights[..., None]
        ).sum(dim=(1, 2)) * valid_count.squeeze(-1).sqrt()
        fused_embedding = torch.cat(
            (preserved_blocks.flatten(1), interaction), dim=1
        )

        branch_embeddings = {}
        for expert_index, expert in enumerate(EXPERT_ORDER):
            branch_scores = scores[:, expert_index]
            branch_total = branch_scores.sum(dim=1, keepdim=True)
            branch_valid = valid_float[:, expert_index]
            branch_fallback = branch_valid / branch_valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            branch_weights = torch.where(
                branch_total > 0,
                branch_scores / branch_total.clamp_min(1e-12),
                branch_fallback,
            ) * branch_valid
            branch_count = branch_valid.sum(dim=1, keepdim=True)
            branch_scales = (
                branch_weights * branch_count
            ).clamp_min(0.0).sqrt()
            branch_contributions = contribution_tensor[:, expert_index]
            branch_interaction = (
                branch_contributions * branch_weights[..., None]
            ).sum(dim=1) * branch_count.sqrt()
            branch_embeddings[expert] = torch.cat(
                (
                    (
                        branch_contributions
                        * branch_scales[..., None]
                    ).flatten(1),
                    branch_interaction,
                ),
                dim=1,
            )

        return FusionResult(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=contribution_tensor,
            weights=weights,
            modality_mask=modality_mask,
        )


class CascadeV2ReID(nn.Module):
    """Use wide raw retrieval features while retaining BN-neck classifiers."""

    retrieval_before_neck = True

    def __init__(
        self,
        *,
        encoder: nn.Module,
        fusion: InformationPreservingFusion,
        num_classes: int = 0,
        peer_teaching: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 0:
            raise ValueError("classes must be nonnegative")
        self.encoder = encoder
        self.fusion = fusion
        self.peer_teaching = peer_teaching
        self.num_classes = num_classes
        self.fused_embedding_width = fusion.fused_embedding_width
        self.branch_embedding_width = fusion.branch_embedding_width
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {
                expert: self._make_neck(self.branch_embedding_width)
                for expert in EXPERT_ORDER
            }
        )
        if num_classes:
            self.fused_classifier = nn.Linear(
                self.fused_embedding_width, num_classes, bias=False
            )
            self.branch_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(
                        self.branch_embedding_width,
                        num_classes,
                        bias=False,
                    )
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
            intervention = FullNetworkIntervention.from_value(
                batch["intervention"]
            )
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
            raise RuntimeError("CascadeV2ReID requires a collaborative posterior")
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

        fused_neck = self.fused_neck(fusion_result.fused_embedding)
        branch_necks = {
            expert: self.branch_necks[expert](
                fusion_result.branch_embeddings[expert]
            )
            for expert in EXPERT_ORDER
        }
        fused_logits = None
        branch_logits = {}
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(fused_neck)
            branch_logits = {
                expert: self.branch_classifiers[expert](branch_necks[expert])
                for expert in EXPERT_ORDER
            }

        peer_result = None
        if self.training and targets is not None and self.peer_teaching is not None:
            peer_result = self.peer_teaching(
                states, states.reliability, targets
            )
        if not return_aux:
            return fusion_result.fused_embedding

        finite_tensors = [
            fusion_result.fused_embedding,
            fusion_result.contribution_embeddings,
            states.reliability.r,
            states.reliability.u,
            *fusion_result.branch_embeddings.values(),
        ]
        if fused_logits is not None:
            finite_tensors.append(fused_logits)
            finite_tensors.extend(branch_logits.values())
        return TriFusionOutput(
            fused_embedding=fusion_result.fused_embedding,
            branch_embeddings=fusion_result.branch_embeddings,
            contribution_embeddings=fusion_result.contribution_embeddings,
            reliability=states.reliability,
            relay_results=states.relay_results,
            peer_teaching=peer_result,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            modality_mask=modality_mask,
            diagnostics={
                "all_finite": all(
                    bool(torch.isfinite(tensor).all().item())
                    for tensor in finite_tensors
                ),
                "has_all_three_experts": (
                    tuple(fusion_result.branch_embeddings) == EXPERT_ORDER
                ),
            },
        )


class CascadeV2Criterion(TriFusionCriterion):
    """Add smoothed identity supervision and signed-effect reliability ranking."""

    def __init__(
        self,
        *,
        target_cache: CIRCTargetCache | None,
        triplet_margin: float = 0.3,
        brier_weight: float = 1.0,
        evidence_weight: float = 0.1,
        effect_rank_weight: float = 0.0,
        effect_rank_margin: float = 0.05,
        effect_rank_min_gap: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        values = (
            effect_rank_weight,
            effect_rank_margin,
            effect_rank_min_gap,
            label_smoothing,
        )
        if any(value < 0 for value in values):
            raise ValueError("V2 criterion weights must be nonnegative")
        if label_smoothing >= 1.0:
            raise ValueError("label smoothing must be smaller than one")
        super().__init__(
            target_cache=target_cache,
            triplet_margin=triplet_margin,
            brier_weight=brier_weight,
            evidence_weight=evidence_weight,
        )
        self.effect_rank_weight = effect_rank_weight
        self.effect_rank_margin = effect_rank_margin
        self.effect_rank_min_gap = effect_rank_min_gap
        self.label_smoothing = label_smoothing

    def forward(
        self,
        output: TriFusionOutput,
        labels: torch.Tensor,
        *,
        sample_keys: Sequence[str] | None = None,
        conditions: Sequence[Mapping[str, object]] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().forward(
            output,
            labels,
            sample_keys=sample_keys,
            conditions=conditions,
        )
        losses["id_fused"] = F.cross_entropy(
            output.fused_logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"] = F.cross_entropy(
                output.branch_logits[expert],
                labels,
                label_smoothing=self.label_smoothing,
            )

        if self.target_cache is None or self.effect_rank_weight == 0.0:
            return losses
        if sample_keys is None or conditions is None:
            raise ValueError("effect ranking requires CIRC metadata")
        circ = self.target_cache.lookup(
            sample_keys,
            conditions,
            device=output.reliability.r.device,
            allow_missing=True,
        )
        available = output.modality_mask[:, None, :].expand_as(circ.valid_mask)
        valid = (circ.valid_mask & available).reshape(
            output.reliability.r.shape[0], -1
        )
        effects = circ.signed_total_effects.reshape(valid.shape)
        valid = valid & torch.isfinite(effects)
        predicted = output.reliability.r.reshape(valid.shape)
        effect_difference = effects[:, :, None] - effects[:, None, :]
        prediction_difference = predicted[:, :, None] - predicted[:, None, :]
        ordered_pairs = (
            valid[:, :, None]
            & valid[:, None, :]
            & (effect_difference > self.effect_rank_min_gap)
        )
        if bool(ordered_pairs.any()):
            rank_loss = F.relu(
                self.effect_rank_margin - prediction_difference
            )[ordered_pairs].mean()
            losses["reliability"] = losses["reliability"] + (
                self.effect_rank_weight * rank_loss
            )
        return losses


__all__ = [
    "AnchorPreservingSemanticTokenizer",
    "CascadeV2Criterion",
    "CascadeV2ReID",
    "InformationPreservingFusion",
    "StageUpdatedTriBranchEncoder",
]
