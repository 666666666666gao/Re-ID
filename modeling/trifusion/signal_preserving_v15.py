"""Counterfactual role-delta exchange for Signal-preserving TriFusion V15."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import _batch_hard_triplet
from .experts.mamba import FourDirectionMambaBlock
from .signal_preserving_v8 import (
    ExpertFormationRepresentations,
    PretrainedTailTriExpertEncoder,
)
from .state import EXPERT_ORDER, MODALITY_ORDER

V15_REGRET_WEIGHT = 1.0
V15_OUTPUT_ORDER = ("fused", *EXPERT_ORDER)


@dataclass(frozen=True, eq=False)
class CrossCameraRetrievalRiskV15:
    risk: torch.Tensor
    per_query_loss: torch.Tensor
    hardest_positive_distance: torch.Tensor
    nearest_negative_distance: torch.Tensor
    valid_query_mask: torch.Tensor


def cross_camera_retrieval_risk_v15(
    embedding: torch.Tensor,
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> CrossCameraRetrievalRiskV15:
    """Score valid cross-camera queries while retaining the full gallery."""

    embedding = F.normalize(embedding, dim=1)
    distances = torch.cdist(embedding, embedding)
    same_identity = identities[:, None] == identities[None, :]
    positives = same_identity & (cameras[:, None] != cameras[None, :])
    negatives = ~same_identity
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not bool(valid.any()):
        zero = embedding.sum() * 0.0
        empty = distances.new_empty((0,))
        return CrossCameraRetrievalRiskV15(
            risk=zero,
            per_query_loss=empty,
            hardest_positive_distance=empty,
            nearest_negative_distance=empty,
            valid_query_mask=valid,
        )
    hardest_positive = distances.masked_fill(~positives, -torch.inf).max(dim=1).values[
        valid
    ]
    nearest_negative = distances.masked_fill(~negatives, torch.inf).min(dim=1).values[
        valid
    ]
    per_query = F.softplus(hardest_positive - nearest_negative)
    return CrossCameraRetrievalRiskV15(
        risk=per_query.mean(),
        per_query_loss=per_query,
        hardest_positive_distance=hardest_positive,
        nearest_negative_distance=nearest_negative,
        valid_query_mask=valid,
    )


@dataclass(frozen=True, eq=False)
class MatchedRetrievalRegretV15:
    total: torch.Tensor
    on_risks: Mapping[str, torch.Tensor]
    off_risks: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "on_risks", MappingProxyType(dict(self.on_risks)))
        object.__setattr__(self, "off_risks", MappingProxyType(dict(self.off_risks)))


def matched_retrieval_regret_v15(
    on_embeddings: Mapping[str, torch.Tensor],
    off_embeddings: Mapping[str, torch.Tensor],
    identities: torch.Tensor,
    cameras: torch.Tensor,
) -> MatchedRetrievalRegretV15:
    """Penalize exchange-on risk relative to a state-clean matched comparator."""

    if tuple(on_embeddings) != V15_OUTPUT_ORDER or tuple(off_embeddings) != V15_OUTPUT_ORDER:
        raise ValueError(f"V15 retrieval outputs must follow {V15_OUTPUT_ORDER}")
    on_outputs = {
        output: cross_camera_retrieval_risk_v15(
            on_embeddings[output], identities, cameras
        )
        for output in V15_OUTPUT_ORDER
    }
    off_outputs = {
        output: cross_camera_retrieval_risk_v15(
            off_embeddings[output].detach(), identities, cameras
        )
        for output in V15_OUTPUT_ORDER
    }
    on_risks = {output: value.risk for output, value in on_outputs.items()}
    off_risks = {
        output: value.risk.detach() for output, value in off_outputs.items()
    }
    total = V15_REGRET_WEIGHT * torch.stack(
        [
            (
                F.softplus(on_risks[output] - off_risks[output])
                if bool(on_outputs[output].valid_query_mask.any())
                else on_risks[output]
            )
            for output in V15_OUTPUT_ORDER
        ]
    ).mean()
    return MatchedRetrievalRegretV15(
        total=total,
        on_risks=on_risks,
        off_risks=off_risks,
    )


@dataclass(frozen=True, eq=False)
class RoleDeltaExchangeOutput:
    states: Mapping[str, torch.Tensor]
    messages: Mapping[str, torch.Tensor]
    edge_scales: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(self, "messages", MappingProxyType(dict(self.messages)))


class _CNNRoleDeltaMixerV15(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        rank: int,
        grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.depthwise = nn.Conv2d(
            rank,
            rank,
            kernel_size=3,
            padding=1,
            groups=rank,
            bias=False,
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, _width = delta.shape
        height, grid_width = self.grid_size
        patches = self.down(self.norm(delta[:, :, 1:]))
        grid = patches.reshape(
            batch_size * modality_count,
            height,
            grid_width,
            -1,
        ).permute(0, 3, 1, 2)
        high_frequency = grid - F.avg_pool2d(
            grid,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        mixed = self.depthwise(high_frequency).permute(0, 2, 3, 1).reshape(
            batch_size,
            modality_count,
            token_count - 1,
            -1,
        )
        return torch.cat((mixed.new_zeros(*mixed.shape[:2], 1, mixed.shape[-1]), mixed), dim=2)


class _TransformerRoleDeltaMixerV15(nn.Module):
    def __init__(self, *, width: int, rank: int) -> None:
        super().__init__()
        heads = 4 if rank % 4 == 0 else 1
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.attention_norm = nn.LayerNorm(rank)
        self.attention = nn.MultiheadAttention(rank, heads, batch_first=True)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        reduced = self.down(self.norm(delta)).flatten(0, 1)
        normalized = self.attention_norm(reduced)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return (reduced + attended).reshape(*delta.shape[:3], -1)


class _MambaRoleDeltaMixerV15(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        rank: int,
        grid_size: tuple[int, int],
        mixer_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.spatial_mixer = FourDirectionMambaBlock(
            rank,
            grid_size,
            mixer_factory,
        )
        self.modal_norm = nn.LayerNorm(rank)
        self.modal_mixer = mixer_factory(rank)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, _width = delta.shape
        patches = self.down(self.norm(delta[:, :, 1:]))
        spatial = self.spatial_mixer(patches.flatten(0, 1)).reshape_as(patches)
        aligned = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * (token_count - 1),
            modality_count,
            -1,
        )
        aligned = self.modal_norm(aligned)
        modal = 0.5 * (
            self.modal_mixer(aligned)
            + self.modal_mixer(aligned.flip(dims=(1,))).flip(dims=(1,))
        )
        modal = modal.reshape(
            batch_size,
            token_count - 1,
            modality_count,
            -1,
        ).permute(0, 2, 1, 3)
        mixed = spatial + modal
        return torch.cat((mixed.new_zeros(*mixed.shape[:2], 1, mixed.shape[-1]), mixed), dim=2)


class CounterfactualRoleDeltaExchangeStage(nn.Module):
    """Synchronously inject bounded peer role deltas without self-edges."""

    def __init__(
        self,
        *,
        width: int,
        rank: int,
        grid_size: tuple[int, int],
        edge_scale_max: float,
        mixer_factory: Callable[[int], nn.Module],
        edge_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.rank = int(rank)
        self.grid_size = tuple(int(value) for value in grid_size)
        self.edge_scale_max = float(edge_scale_max)
        if not 0.0 <= float(edge_scale_init) < self.edge_scale_max:
            raise ValueError("edge scale requires 0 <= init < max")
        self.source_mixers = nn.ModuleDict(
            {
                "cnn": _CNNRoleDeltaMixerV15(
                    width=self.width,
                    rank=self.rank,
                    grid_size=self.grid_size,
                ),
                "transformer": _TransformerRoleDeltaMixerV15(
                    width=self.width,
                    rank=self.rank,
                ),
                "mamba": _MambaRoleDeltaMixerV15(
                    width=self.width,
                    rank=self.rank,
                    grid_size=self.grid_size,
                    mixer_factory=mixer_factory,
                ),
            }
        )
        self.edge_projections = nn.ModuleDict(
            {
                f"{source}__{target}": nn.Linear(
                    self.rank,
                    self.width,
                    bias=False,
                )
                for source in EXPERT_ORDER
                for target in EXPERT_ORDER
                if source != target
            }
        )
        initial_logit = math.atanh(float(edge_scale_init) / self.edge_scale_max)
        self.edge_logits = nn.Parameter(
            torch.full(
                (len(EXPERT_ORDER), len(EXPERT_ORDER)),
                initial_logit,
            )
        )

    def forward(
        self,
        before: Mapping[str, torch.Tensor],
        after: Mapping[str, torch.Tensor],
    ) -> RoleDeltaExchangeOutput:
        if tuple(before) != EXPERT_ORDER or tuple(after) != EXPERT_ORDER:
            raise ValueError(f"role states must follow {EXPERT_ORDER}")
        reference_shape = after[EXPERT_ORDER[0]].shape
        expected_tokens = 1 + self.grid_size[0] * self.grid_size[1]
        if (
            len(reference_shape) != 4
            or reference_shape[1] != len(MODALITY_ORDER)
            or reference_shape[2] != expected_tokens
            or reference_shape[3] != self.width
        ):
            raise ValueError("role states have the wrong B,M,L,D shape")
        for expert in EXPERT_ORDER:
            if before[expert].shape != reference_shape or after[expert].shape != reference_shape:
                raise ValueError("all role states must share one shape")

        deltas = {
            expert: after[expert] - before[expert] for expert in EXPERT_ORDER
        }
        messages = {
            expert: self.source_mixers[expert](deltas[expert])
            * deltas[expert]
            .ne(0)
            .any(dim=(1, 2, 3), keepdim=True)
            .to(dtype=deltas[expert].dtype)
            for expert in EXPERT_ORDER
        }
        scales = self.edge_scale_max * torch.tanh(self.edge_logits)
        no_self = ~torch.eye(
            len(EXPERT_ORDER),
            dtype=torch.bool,
            device=scales.device,
        )
        scales = scales * no_self.to(dtype=scales.dtype)
        states = {}
        for target_index, target in enumerate(EXPERT_ORDER):
            incoming = torch.zeros_like(after[target])
            for source_index, source in enumerate(EXPERT_ORDER):
                if source == target:
                    continue
                projected = self.edge_projections[f"{source}__{target}"](
                    messages[source]
                )
                incoming = incoming + scales[source_index, target_index] * projected
            states[target] = after[target] + incoming
        return RoleDeltaExchangeOutput(
            states=states,
            messages=messages,
            edge_scales=scales,
        )


@dataclass(frozen=True, eq=False)
class CollaborativeExpertFormationRepresentationsV15(
    ExpertFormationRepresentations
):
    exchange_edge_scales: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "exchange_edge_scales",
            tuple(self.exchange_edge_scales),
        )


class CollaborativeTailTriExpertEncoderV15(PretrainedTailTriExpertEncoder):
    """Run two bounded role-delta exchanges inside the frozen CLIP tail."""

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
        exchange_rank: int,
        edge_scale_max: float,
        scale_init: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            tail_blocks=tail_blocks,
            tail_layer_indices=tail_layer_indices,
            semantic_width=semantic_width,
            grid_size=grid_size,
            adapter_width=adapter_width,
            expert_modal_width=expert_modal_width,
            mixer_factory=mixer_factory,
            scale_init=scale_init,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.exchange_after_layer_indices = self.tail_layer_indices[:-1]
        self.exchange_stages = nn.ModuleList(
            [
                CounterfactualRoleDeltaExchangeStage(
                    width=semantic_width,
                    rank=exchange_rank,
                    grid_size=grid_size,
                    edge_scale_max=edge_scale_max,
                    mixer_factory=mixer_factory,
                )
                for _ in self.exchange_after_layer_indices
            ]
        )

    def forward(
        self,
        anchor_sequence: torch.Tensor,
        reference_sequence: torch.Tensor,
        *,
        exchange_enabled: bool = True,
    ) -> CollaborativeExpertFormationRepresentationsV15:
        if anchor_sequence.shape != reference_sequence.shape:
            raise ValueError("anchor and reference sequences must match")
        if (
            anchor_sequence.ndim != 4
            or anchor_sequence.shape[1] != len(MODALITY_ORDER)
            or anchor_sequence.shape[2] != 1 + self.grid_size[0] * self.grid_size[1]
            or anchor_sequence.shape[3] != self.semantic_width
        ):
            raise ValueError("V15 CLIP sequences have the wrong shape")
        batch_size, modality_count = anchor_sequence.shape[:2]
        runtimes = {expert: anchor_sequence for expert in EXPERT_ORDER}
        exchange_edge_scales = []
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
            before = dict(runtimes)

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
            runtimes["mamba"] = torch.cat(
                (mamba[:, :, :1], mamba_patches),
                dim=2,
            )

            if exchange_enabled and stage_index < len(self.exchange_stages):
                exchanged = self.exchange_stages[stage_index](before, runtimes)
                runtimes = dict(exchanged.states)
                exchange_edge_scales.append(exchanged.edge_scales)

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
        return CollaborativeExpertFormationRepresentationsV15(
            residual_embeddings=residual,
            modal_residual_embeddings=modal,
            exchange_edge_scales=tuple(exchange_edge_scales),
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV15Output:
    fused_embedding: torch.Tensor
    baseline_embedding: torch.Tensor
    direct_modal: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    residual_embeddings: Mapping[str, torch.Tensor]
    modal_residual_embeddings: Mapping[str, torch.Tensor]
    fused_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    residual_logits: Mapping[str, torch.Tensor]
    exchange_edge_scales: tuple[torch.Tensor, ...]
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
        object.__setattr__(
            self,
            "exchange_edge_scales",
            tuple(self.exchange_edge_scales),
        )


@dataclass(frozen=True, eq=False)
class PairedSignalPreservingV15Output:
    exchange_on: SignalPreservingV15Output
    exchange_off: SignalPreservingV15Output


class SignalPreservingCollaborativeV15(nn.Module):
    """Train only CRDE and source-local heads over one frozen Signal field."""

    def __init__(
        self,
        *,
        baseline: nn.Module,
        encoder: CollaborativeTailTriExpertEncoderV15,
        fusion: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.encoder = encoder
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.baseline_embedding_width = int(fusion.baseline_width)
        self.fused_embedding_width = int(fusion.fused_embedding_width)
        self.branch_embedding_width = int(fusion.branch_embedding_width)
        self.residual_embedding_width = int(fusion.expert_width)

        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for exchange in self.encoder.exchange_stages:
            for parameter in exchange.parameters():
                parameter.requires_grad_(True)

        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {
                expert: self._make_neck(self.branch_embedding_width)
                for expert in EXPERT_ORDER
            }
        )
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
            self.branch_classifiers = nn.ModuleDict(
                {
                    expert: nn.Linear(
                        self.branch_embedding_width,
                        self.num_classes,
                        bias=False,
                    )
                    for expert in EXPERT_ORDER
                }
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
                *self.branch_classifiers.values(),
                *self.residual_classifiers.values(),
            ):
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.branch_classifiers = nn.ModuleDict()
            self.residual_classifiers = nn.ModuleDict()
        self.train(self.training)

    @staticmethod
    def _make_neck(width: int) -> nn.BatchNorm1d:
        neck = nn.BatchNorm1d(width)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        neck.bias.requires_grad_(False)
        return neck

    def train(self, mode: bool = True) -> "SignalPreservingCollaborativeV15":
        super().train(mode)
        self.baseline.eval()
        self.encoder.eval()
        self.encoder.exchange_stages.train(mode)
        return self

    def _forward_field(
        self,
        field: Any,
        *,
        exchange_enabled: bool,
        with_heads: bool,
    ) -> SignalPreservingV15Output:
        representations = self.encoder(
            field.anchor_sequence,
            field.reference_sequence,
            exchange_enabled=exchange_enabled,
        )
        fusion = self.fusion(field.baseline_embedding, representations)
        fused_logits = None
        branch_logits: dict[str, torch.Tensor] = {}
        residual_logits: dict[str, torch.Tensor] = {}
        if with_heads and self.fused_classifier is not None:
            fused_logits = self.fused_classifier(
                self.fused_neck(fusion.fused_embedding)
            )
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
        finite = (
            fusion.fused_embedding,
            *fusion.branch_embeddings.values(),
            *fusion.residual_embeddings.values(),
        )
        return SignalPreservingV15Output(
            fused_embedding=fusion.fused_embedding,
            baseline_embedding=field.baseline_embedding,
            direct_modal=field.direct_modal,
            branch_embeddings=fusion.branch_embeddings,
            residual_embeddings=fusion.residual_embeddings,
            modal_residual_embeddings=fusion.modal_residual_embeddings,
            fused_logits=fused_logits,
            branch_logits=branch_logits,
            residual_logits=residual_logits,
            exchange_edge_scales=representations.exchange_edge_scales,
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    fusion.fused_embedding[:, : self.baseline_embedding_width],
                    field.baseline_embedding,
                ),
                "baseline_frozen": all(
                    not parameter.requires_grad for parameter in self.baseline.parameters()
                ),
                "base_experts_frozen": all(
                    not parameter.requires_grad
                    for name, parameter in self.encoder.named_parameters()
                    if not name.startswith("exchange_stages.")
                ),
                "exchange_enabled": bool(exchange_enabled),
            },
        )

    def forward_paired(
        self,
        batch: Mapping[str, Any],
        *,
        with_on_heads: bool = True,
    ) -> PairedSignalPreservingV15Output:
        field = self.baseline(batch)
        with torch.no_grad():
            exchange_off = self._forward_field(
                field,
                exchange_enabled=False,
                with_heads=False,
            )
        exchange_on = self._forward_field(
            field,
            exchange_enabled=True,
            with_heads=with_on_heads,
        )
        return PairedSignalPreservingV15Output(
            exchange_on=exchange_on,
            exchange_off=exchange_off,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
        retrieval_output: str = "fused",
    ) -> torch.Tensor | SignalPreservingV15Output:
        del targets
        field = self.baseline(batch)
        if retrieval_output == "baseline_only" and not return_aux:
            return field.baseline_embedding
        output = self._forward_field(
            field,
            exchange_enabled=True,
            with_heads=return_aux,
        )
        if return_aux:
            return output
        if retrieval_output == "fused":
            return output.fused_embedding
        if retrieval_output in EXPERT_ORDER:
            return output.branch_embeddings[retrieval_output]
        raise ValueError("retrieval output must be baseline_only, fused, or an expert")


class CollaborativeV15Criterion(nn.Module):
    def __init__(
        self,
        *,
        triplet_margin: float,
        label_smoothing: float,
        id_fused_weight: float,
        triplet_fused_weight: float,
        id_branch_weight: float,
        triplet_branch_weight: float,
        id_residual_weight: float,
        triplet_residual_weight: float,
    ) -> None:
        super().__init__()
        self.triplet_margin = float(triplet_margin)
        self.label_smoothing = float(label_smoothing)
        self.id_fused_weight = float(id_fused_weight)
        self.triplet_fused_weight = float(triplet_fused_weight)
        self.id_branch_weight = float(id_branch_weight)
        self.triplet_branch_weight = float(triplet_branch_weight)
        self.id_residual_weight = float(id_residual_weight)
        self.triplet_residual_weight = float(triplet_residual_weight)

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
        paired: PairedSignalPreservingV15Output,
        labels: torch.Tensor,
        cameras: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        on = paired.exchange_on
        off = paired.exchange_off
        if on.fused_logits is None:
            raise ValueError("V15 collaboration training requires identity heads")
        id_fused, triplet_fused = self._identity_losses(
            on.fused_logits,
            on.fused_embedding,
            labels,
        )
        losses = {
            "id_fused": id_fused,
            "triplet_fused": triplet_fused,
        }
        supervised_total = (
            self.id_fused_weight * id_fused
            + self.triplet_fused_weight * triplet_fused
        )
        for expert in EXPERT_ORDER:
            id_branch, triplet_branch = self._identity_losses(
                on.branch_logits[expert],
                on.branch_embeddings[expert],
                labels,
            )
            id_residual, triplet_residual = self._identity_losses(
                on.residual_logits[expert],
                on.residual_embeddings[expert],
                labels,
            )
            losses[f"id_{expert}"] = id_branch
            losses[f"triplet_{expert}"] = triplet_branch
            losses[f"id_residual_{expert}"] = id_residual
            losses[f"triplet_residual_{expert}"] = triplet_residual
            supervised_total = (
                supervised_total
                + self.id_branch_weight * id_branch
                + self.triplet_branch_weight * triplet_branch
                + self.id_residual_weight * id_residual
                + self.triplet_residual_weight * triplet_residual
            )
        regret = matched_retrieval_regret_v15(
            {
                "fused": on.fused_embedding,
                **{
                    expert: on.branch_embeddings[expert]
                    for expert in EXPERT_ORDER
                },
            },
            {
                "fused": off.fused_embedding,
                **{
                    expert: off.branch_embeddings[expert]
                    for expert in EXPERT_ORDER
                },
            },
            labels,
            cameras,
        )
        losses["supervised_total"] = supervised_total
        losses["retrieval_regret"] = regret.total
        losses["total"] = supervised_total + regret.total
        return losses


__all__ = [
    "CollaborativeExpertFormationRepresentationsV15",
    "CollaborativeTailTriExpertEncoderV15",
    "CollaborativeV15Criterion",
    "CounterfactualRoleDeltaExchangeStage",
    "CrossCameraRetrievalRiskV15",
    "MatchedRetrievalRegretV15",
    "PairedSignalPreservingV15Output",
    "RoleDeltaExchangeOutput",
    "SignalPreservingCollaborativeV15",
    "SignalPreservingV15Output",
    "V15_OUTPUT_ORDER",
    "V15_REGRET_WEIGHT",
    "cross_camera_retrieval_risk_v15",
    "matched_retrieval_regret_v15",
]
