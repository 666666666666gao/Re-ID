"""Pretrained-tail expert formation for Signal-preserving TriFusion V8."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .criterion import _batch_hard_triplet
from .experts.mamba import FourDirectionMambaBlock
from .experts.semantic_residual import (
    GlobalAttentionResidualBlock,
    HighFrequencyResidualBlock,
)
from .state import EXPERT_ORDER, MODALITY_ORDER


@dataclass(frozen=True, eq=False)
class HierarchicalFrozenSignalField:
    baseline_embedding: torch.Tensor
    direct_modal: torch.Tensor
    anchor_sequence: torch.Tensor
    reference_sequence: torch.Tensor
    modality_mask: torch.Tensor


class HierarchicalFrozenSignalBackbone(nn.Module):
    """Keep exact Signal output and expose its matched pretrained tail boundary."""

    def __init__(
        self,
        signal: nn.Module,
        *,
        feature_width: int = 512,
        branch_after_block: int = 8,
    ) -> None:
        super().__init__()
        if not hasattr(signal, "clip_vision_encoder") or not hasattr(signal, "SIM"):
            raise ValueError("Signal model must expose clip_vision_encoder and SIM")
        vision = signal.clip_vision_encoder
        if not hasattr(vision, "base") or not hasattr(vision.base, "transformer"):
            raise ValueError("Signal CLIP encoder must expose base.transformer")
        blocks = vision.base.transformer.resblocks
        if len(blocks) - int(branch_after_block) - 1 != 3:
            raise ValueError("V8 expert formation requires exactly three CLIP tail blocks")
        if feature_width <= 0:
            raise ValueError("feature width must be positive")

        self.signal = signal
        self.feature_width = int(feature_width)
        self.baseline_width = len(MODALITY_ORDER) * 2 * self.feature_width
        self.branch_after_block = int(branch_after_block)
        self.tail_layer_indices = tuple(range(self.branch_after_block + 1, len(blocks)))
        self._tail_blocks = tuple(blocks[index] for index in self.tail_layer_indices)
        self._split_block = blocks[self.branch_after_block]
        self._reference_block = blocks[-1]
        for parameter in self.signal.parameters():
            parameter.requires_grad_(False)
        for block in (self._split_block, self._reference_block, *self._tail_blocks):
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self.signal.eval()

    @property
    def tail_blocks(self) -> tuple[nn.Module, ...]:
        return self._tail_blocks

    def train(self, mode: bool = True) -> HierarchicalFrozenSignalBackbone:
        super().train(mode)
        self.signal.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> HierarchicalFrozenSignalField:
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
            raise ValueError("V8 Signal baseline requires all RGB/NI/TI modalities")
        camera_ids = batch.get("camera_ids")
        if camera_ids is None or camera_ids.shape != (modality_mask.shape[0],):
            raise ValueError("camera_ids must have shape B")

        anchors: list[torch.Tensor] = []
        references: list[torch.Tensor] = []

        def capture_anchor(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            anchors.append(output.detach())

        def capture_reference(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            references.append(output.detach())

        patches = []
        globals_by_modality = []
        anchor_handle = self._split_block.register_forward_hook(capture_anchor)
        reference_handle = self._reference_block.register_forward_hook(capture_reference)
        with anchor_handle, reference_handle, torch.no_grad():
            for modality in MODALITY_ORDER:
                patch, global_embedding = self.signal.clip_vision_encoder(
                    images[modality],
                    cam_label=camera_ids,
                    view_label=None,
                )
                patches.append(patch)
                globals_by_modality.append(global_embedding)

        if len(anchors) != len(MODALITY_ORDER) or len(references) != len(MODALITY_ORDER):
            raise RuntimeError("Signal CLIP hooks did not capture all three modalities")
        sim = self.signal.SIM(*patches, *globals_by_modality)
        direct_modal = torch.stack(globals_by_modality, dim=1)
        baseline_embedding = torch.cat((direct_modal.flatten(1), sim), dim=1)
        expected = (modality_mask.shape[0], self.baseline_width)
        if baseline_embedding.shape != expected:
            raise ValueError(f"Signal baseline feature must have shape {expected}")

        anchor_sequence = torch.stack(
            [sequence.permute(1, 0, 2) for sequence in anchors],
            dim=1,
        )
        reference_sequence = torch.stack(
            [sequence.permute(1, 0, 2) for sequence in references],
            dim=1,
        )
        return HierarchicalFrozenSignalField(
            baseline_embedding=baseline_embedding,
            direct_modal=direct_modal,
            anchor_sequence=anchor_sequence,
            reference_sequence=reference_sequence,
            modality_mask=modality_mask,
        )


class CrossModalMambaResidualStage(nn.Module):
    """Mix spatial scans and aligned RGB/NI/TI states at every CLIP tail stage."""

    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        mixer_factory: Callable[[int], nn.Module],
        scale_init: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.adapter_width = int(adapter_width)
        self.grid_size = grid_size
        self.input_norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, adapter_width, bias=False)
        self.spatial = FourDirectionMambaBlock(
            adapter_width,
            grid_size,
            mixer_factory,
        )
        self.modal_norm = nn.LayerNorm(adapter_width)
        self.modal_mixer = mixer_factory(adapter_width)
        self.up = nn.Linear(adapter_width, width, bias=False)
        self.layer_scale = nn.Parameter(torch.full((width,), float(scale_init)))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1] != len(MODALITY_ORDER):
            raise ValueError("Mamba stage patches must have shape B,3,N,D")
        batch_size, modality_count, token_count, width = patches.shape
        if width != self.width or token_count != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("Mamba stage patch grid or width is invalid")
        reduced = self.down(self.input_norm(patches)).flatten(0, 1)
        spatial = self.spatial(reduced).reshape(
            batch_size,
            modality_count,
            token_count,
            self.adapter_width,
        )
        modal_sequence = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * token_count,
            modality_count,
            self.adapter_width,
        )
        modal_sequence = self.modal_norm(modal_sequence)
        modal_forward = self.modal_mixer(modal_sequence)
        modal_reverse = self.modal_mixer(modal_sequence.flip(dims=(1,))).flip(dims=(1,))
        cross_modal = 0.5 * (modal_forward + modal_reverse)
        cross_modal = cross_modal.reshape(
            batch_size,
            token_count,
            modality_count,
            self.adapter_width,
        ).permute(0, 2, 1, 3)
        residual = self.up(spatial + cross_modal)
        return patches + residual * self.layer_scale.to(dtype=residual.dtype)


class _CNNRoleHead(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        grid_size: tuple[int, int],
        output_width: int,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.part_count = min(4, grid_size[0])
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(self.part_count * width, output_width, bias=False)

    def forward(self, patch_delta: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, width = patch_delta.shape
        height, grid_width = self.grid_size
        if token_count != height * grid_width:
            raise ValueError("CNN role head received the wrong patch grid")
        grid = self.norm(patch_delta).reshape(
            batch_size * modality_count,
            height,
            grid_width,
            width,
        ).permute(0, 3, 1, 2)
        parts = F.adaptive_avg_pool2d(grid, (self.part_count, 1)).squeeze(-1)
        parts = parts.flatten(1)
        return self.projection(parts).reshape(batch_size, modality_count, -1)


class _GlobalRoleHead(nn.Module):
    def __init__(self, *, width: int, output_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, output_width, bias=False)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(delta))


@dataclass(frozen=True, eq=False)
class ExpertFormationRepresentations:
    residual_embeddings: Mapping[str, torch.Tensor]
    modal_residual_embeddings: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        for name in ("residual_embeddings", "modal_residual_embeddings"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class PretrainedTailTriExpertEncoder(nn.Module):
    """Continue a shared frozen CLIP tail with three role-specific deep paths."""

    def __init__(
        self,
        *,
        tail_blocks: Sequence[nn.Module],
        tail_layer_indices: Sequence[int],
        semantic_width: int,
        grid_size: tuple[int, int],
        adapter_width: int,
        expert_modal_width: int,
        mixer_factory: Callable[[int], nn.Module],
        scale_init: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if len(tail_blocks) != 3 or len(tail_layer_indices) != 3:
            raise ValueError("V8 encoder requires three pretrained tail stages")
        self._tail_blocks = tuple(tail_blocks)
        self.tail_layer_indices = tuple(int(index) for index in tail_layer_indices)
        self.semantic_width = int(semantic_width)
        self.grid_size = grid_size
        self.expert_modal_width = int(expert_modal_width)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        for block in self._tail_blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(False)

        self.cnn_stages = nn.ModuleList(
            [
                HighFrequencyResidualBlock(
                    width=semantic_width,
                    adapter_width=adapter_width,
                    grid_size=grid_size,
                    dilation=1 + index % 2,
                    scale_init=scale_init,
                )
                for index in range(3)
            ]
        )
        self.transformer_stages = nn.ModuleList(
            [
                GlobalAttentionResidualBlock(
                    width=semantic_width,
                    adapter_width=adapter_width,
                    scale_init=scale_init,
                )
                for _ in range(3)
            ]
        )
        self.mamba_stages = nn.ModuleList(
            [
                CrossModalMambaResidualStage(
                    width=semantic_width,
                    adapter_width=adapter_width,
                    grid_size=grid_size,
                    mixer_factory=mixer_factory,
                    scale_init=scale_init,
                )
                for _ in range(3)
            ]
        )
        self.cnn_head = _CNNRoleHead(
            width=semantic_width,
            grid_size=grid_size,
            output_width=expert_modal_width,
        )
        self.transformer_head = _GlobalRoleHead(
            width=semantic_width,
            output_width=expert_modal_width,
        )
        self.mamba_head = _GlobalRoleHead(
            width=semantic_width,
            output_width=expert_modal_width,
        )

    @staticmethod
    def _call_tail_block(
        block: nn.Module,
        sequence: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        return block(
            sequence,
            None,
            layer_index,
            None,
            prompt_sign=False,
            adapter_sign=False,
        )

    def _run_tail_block(
        self,
        block: nn.Module,
        sequence: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(
                lambda value: self._call_tail_block(block, value, layer_index),
                sequence,
                use_reentrant=False,
            )
        return self._call_tail_block(block, sequence, layer_index)

    @staticmethod
    def _to_lbd(sequence: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, width = sequence.shape
        return sequence.permute(2, 0, 1, 3).reshape(
            token_count,
            batch_size * modality_count,
            width,
        )

    @staticmethod
    def _from_lbd(
        sequence: torch.Tensor,
        *,
        batch_size: int,
        modality_count: int,
    ) -> torch.Tensor:
        token_count, _packed_batch, width = sequence.shape
        return sequence.reshape(token_count, batch_size, modality_count, width).permute(
            1, 2, 0, 3
        )

    def forward(
        self,
        anchor_sequence: torch.Tensor,
        reference_sequence: torch.Tensor,
    ) -> ExpertFormationRepresentations:
        if anchor_sequence.shape != reference_sequence.shape:
            raise ValueError("anchor and reference sequences must match")
        if (
            anchor_sequence.ndim != 4
            or anchor_sequence.shape[1] != len(MODALITY_ORDER)
            or anchor_sequence.shape[2] != 1 + self.grid_size[0] * self.grid_size[1]
            or anchor_sequence.shape[3] != self.semantic_width
        ):
            raise ValueError("V8 CLIP sequences have the wrong shape")
        batch_size, modality_count = anchor_sequence.shape[:2]
        runtimes = {expert: anchor_sequence for expert in EXPERT_ORDER}
        for stage_index, (tail_block, layer_index) in enumerate(
            zip(self._tail_blocks, self.tail_layer_indices, strict=True)
        ):
            for expert in EXPERT_ORDER:
                sequence = self._to_lbd(runtimes[expert])
                sequence = self._run_tail_block(tail_block, sequence, layer_index)
                runtimes[expert] = self._from_lbd(
                    sequence,
                    batch_size=batch_size,
                    modality_count=modality_count,
                )

            cnn = runtimes["cnn"]
            cnn_patches = self.cnn_stages[stage_index](cnn[:, :, 1:].flatten(0, 1))
            runtimes["cnn"] = torch.cat(
                (
                    cnn[:, :, :1],
                    cnn_patches.reshape(
                        batch_size,
                        modality_count,
                        -1,
                        self.semantic_width,
                    ),
                ),
                dim=2,
            )

            transformer = runtimes["transformer"]
            transformer = self.transformer_stages[stage_index](
                transformer.flatten(0, 1)
            )
            runtimes["transformer"] = transformer.reshape_as(
                runtimes["transformer"]
            )

            mamba = runtimes["mamba"]
            mamba_patches = self.mamba_stages[stage_index](mamba[:, :, 1:])
            runtimes["mamba"] = torch.cat((mamba[:, :, :1], mamba_patches), dim=2)

        deltas = {
            expert: runtimes[expert] - reference_sequence for expert in EXPERT_ORDER
        }
        modal = {
            "cnn": self.cnn_head(deltas["cnn"][:, :, 1:]),
            "transformer": self.transformer_head(deltas["transformer"][:, :, 0]),
            "mamba": self.mamba_head(deltas["mamba"][:, :, 1:].mean(dim=2)),
        }
        modal = {
            expert: F.normalize(value, dim=-1) for expert, value in modal.items()
        }
        residual = {
            expert: F.normalize(value.flatten(1), dim=1)
            for expert, value in modal.items()
        }
        return ExpertFormationRepresentations(
            residual_embeddings=residual,
            modal_residual_embeddings=modal,
        )


@dataclass(frozen=True, eq=False)
class ExpertFormationFusionResult:
    fused_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    modal_residual_embeddings: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        for name in (
            "branch_embeddings",
            "residual_embeddings",
            "modal_residual_embeddings",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class ExpertFormationFusion(nn.Module):
    """Expose equal-energy expert capacity without a learned Router in phase A."""

    def __init__(self, *, baseline_width: int, expert_width: int) -> None:
        super().__init__()
        self.baseline_width = int(baseline_width)
        self.expert_width = int(expert_width)
        self.branch_embedding_width = self.baseline_width + self.expert_width
        self.residual_bank_width = len(EXPERT_ORDER) * self.expert_width
        self.fused_embedding_width = self.baseline_width + self.residual_bank_width

    def forward(
        self,
        baseline_embedding: torch.Tensor,
        representations: ExpertFormationRepresentations,
    ) -> ExpertFormationFusionResult:
        baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
        residuals = representations.residual_embeddings
        if tuple(residuals) != EXPERT_ORDER:
            raise ValueError(f"expert residuals must follow {EXPERT_ORDER}")
        branch_embeddings = {
            expert: torch.cat(
                (baseline_embedding, residuals[expert] * baseline_norm),
                dim=1,
            )
            for expert in EXPERT_ORDER
        }
        bank = torch.cat([residuals[expert] for expert in EXPERT_ORDER], dim=1)
        bank = F.normalize(bank, dim=1) * baseline_norm
        return ExpertFormationFusionResult(
            fused_embedding=torch.cat((baseline_embedding, bank), dim=1),
            branch_embeddings=branch_embeddings,
            residual_embeddings=residuals,
            modal_residual_embeddings=representations.modal_residual_embeddings,
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV8Output:
    fused_embedding: torch.Tensor
    baseline_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    modal_residual_embeddings: Mapping[str, torch.Tensor]
    fused_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    residual_logits: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        for name in (
            "branch_embeddings",
            "residual_embeddings",
            "modal_residual_embeddings",
            "branch_logits",
            "residual_logits",
            "diagnostics",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class SignalPreservingExpertFormationV8(nn.Module):
    def __init__(
        self,
        *,
        baseline: HierarchicalFrozenSignalBackbone,
        encoder: PretrainedTailTriExpertEncoder,
        fusion: ExpertFormationFusion,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.encoder = encoder
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.baseline_embedding_width = fusion.baseline_width
        self.fused_embedding_width = fusion.fused_embedding_width
        self.branch_embedding_width = fusion.branch_embedding_width
        self.residual_embedding_width = fusion.expert_width
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
                self.fused_embedding_width,
                num_classes,
                bias=False,
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
                        self.residual_embedding_width,
                        num_classes,
                        bias=False,
                    )
                    for expert in EXPERT_ORDER
                }
            )
            for classifier in (
                self.fused_classifier,
                *self.branch_classifiers.values(),
                *self.residual_classifiers.values(),
            ):
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
    ) -> torch.Tensor | SignalPreservingV8Output:
        del targets
        field = self.baseline(batch)
        if retrieval_output == "baseline_only" and not return_aux:
            return field.baseline_embedding
        representations = self.encoder(
            field.anchor_sequence,
            field.reference_sequence,
        )
        fusion = self.fusion(field.baseline_embedding, representations)
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
            fusion.fused_embedding,
            *fusion.branch_embeddings.values(),
            *fusion.residual_embeddings.values(),
            *fusion.modal_residual_embeddings.values(),
        ]
        return SignalPreservingV8Output(
            fused_embedding=fusion.fused_embedding,
            baseline_embedding=field.baseline_embedding,
            branch_embeddings=fusion.branch_embeddings,
            residual_embeddings=fusion.residual_embeddings,
            modal_residual_embeddings=fusion.modal_residual_embeddings,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            residual_logits=residual_logits,
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    fusion.fused_embedding[:, : self.baseline_embedding_width],
                    field.baseline_embedding,
                ),
                "baseline_frozen": all(
                    not parameter.requires_grad for parameter in self.baseline.parameters()
                ),
                "pretrained_tail_frozen": all(
                    not parameter.requires_grad
                    for block in self.baseline.tail_blocks
                    for parameter in block.parameters()
                ),
                "router_disabled_during_expert_formation": True,
                "hfer_disabled_during_expert_formation": True,
            },
        )


class ExpertFormationV8Criterion(nn.Module):
    def __init__(self, *, triplet_margin: float, label_smoothing: float) -> None:
        super().__init__()
        self.triplet_margin = float(triplet_margin)
        self.label_smoothing = float(label_smoothing)

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
        output: SignalPreservingV8Output,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if output.fused_logits is None:
            raise ValueError("V8 expert formation requires identity classifiers")
        id_fused, triplet_fused = self._identity_losses(
            output.fused_logits,
            output.fused_embedding,
            labels,
        )
        losses = {"id_fused": id_fused, "triplet_fused": triplet_fused}
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"], losses[f"triplet_{expert}"] = (
                self._identity_losses(
                    output.branch_logits[expert],
                    output.branch_embeddings[expert],
                    labels,
                )
            )
            losses[f"id_residual_{expert}"], losses[f"triplet_residual_{expert}"] = (
                self._identity_losses(
                    output.residual_logits[expert],
                    output.residual_embeddings[expert],
                    labels,
                )
            )
        return losses


__all__ = [
    "CrossModalMambaResidualStage",
    "ExpertFormationFusion",
    "ExpertFormationRepresentations",
    "ExpertFormationV8Criterion",
    "HierarchicalFrozenSignalBackbone",
    "HierarchicalFrozenSignalField",
    "PretrainedTailTriExpertEncoder",
    "SignalPreservingExpertFormationV8",
    "SignalPreservingV8Output",
]
