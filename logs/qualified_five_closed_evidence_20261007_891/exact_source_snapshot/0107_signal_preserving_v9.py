"""Orthogonal triadic relay synthesis for Signal-preserving TriFusion V9."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .criterion import _batch_hard_triplet
from .state import EXPERT_ORDER, MODALITY_ORDER


@dataclass(frozen=True, eq=False)
class OrthogonalTriadicRelayOutput:
    enhanced: torch.Tensor
    receiver_inputs: tuple[torch.Tensor, ...]
    orthogonal_messages: tuple[torch.Tensor, ...]


@dataclass(frozen=True, eq=False)
class OrthogonalTriadicSynthesisOutput:
    fused_embedding: torch.Tensor
    synergy_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    beta: torch.Tensor
    relay: OrthogonalTriadicRelayOutput

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_embeddings",
            MappingProxyType(dict(self.branch_embeddings)),
        )


@dataclass(frozen=True, eq=False)
class SignalPreservingV9Output:
    baseline_embedding: torch.Tensor
    phase_b_embedding: torch.Tensor
    fused_embedding: torch.Tensor
    synergy_embedding: torch.Tensor
    branch_embeddings: Mapping[str, torch.Tensor]
    fused_logits: torch.Tensor | None
    synergy_logits: torch.Tensor | None
    branch_logits: Mapping[str, torch.Tensor]
    beta: torch.Tensor
    relay: OrthogonalTriadicRelayOutput
    diagnostics: Mapping[str, bool]

    def __post_init__(self) -> None:
        for name in ("branch_embeddings", "branch_logits", "diagnostics"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class _OrthogonalPeerRelayRound(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.message_heads = nn.ModuleDict(
            {
                expert: nn.Linear(3 * hidden_width, hidden_width, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.gate_heads = nn.ModuleDict(
            {
                expert: nn.Linear(2 * hidden_width, hidden_width)
                for expert in EXPERT_ORDER
            }
        )
        self.output_norms = nn.ModuleDict(
            {expert: nn.LayerNorm(hidden_width) for expert in EXPERT_ORDER}
        )

    def forward(
        self,
        states: torch.Tensor,
        modal_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enhanced = []
        messages = []
        for receiver_index, receiver_name in enumerate(EXPERT_ORDER):
            peer_indices = [
                index for index in range(len(EXPERT_ORDER)) if index != receiver_index
            ]
            receiver = states[:, receiver_index]
            first_peer = states[:, peer_indices[0]]
            second_peer = states[:, peer_indices[1]]
            message = self.message_heads[receiver_name](
                torch.cat(
                    (first_peer, second_peer, first_peer * second_peer),
                    dim=-1,
                )
            )
            projection = (
                (message * receiver).sum(dim=-1, keepdim=True)
                / receiver.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ) * receiver
            orthogonal = message - projection
            gate = torch.sigmoid(
                self.gate_heads[receiver_name](
                    torch.cat((receiver, orthogonal), dim=-1)
                )
            )
            enhanced.append(
                self.output_norms[receiver_name](
                    receiver + modal_quality[..., None] * gate * orthogonal
                )
            )
            messages.append(orthogonal)
        return torch.stack(enhanced, dim=1), torch.stack(messages, dim=1)


class OrthogonalTriadicRelay(nn.Module):
    """Let every expert absorb only peer information orthogonal to itself."""

    def __init__(
        self,
        *,
        residual_width: int,
        hidden_width: int,
        relay_depth: int,
    ) -> None:
        super().__init__()
        self.residual_width = int(residual_width)
        self.hidden_width = int(hidden_width)
        self.relay_depth = int(relay_depth)
        self.input_projections = nn.ModuleDict(
            {
                expert: nn.Sequential(
                    nn.LayerNorm(self.residual_width),
                    nn.Linear(self.residual_width, self.hidden_width, bias=False),
                )
                for expert in EXPERT_ORDER
            }
        )
        self.rounds = nn.ModuleList(
            [
                _OrthogonalPeerRelayRound(self.hidden_width)
                for _ in range(self.relay_depth)
            ]
        )

    def forward(
        self,
        modal_residual: torch.Tensor,
        modal_quality: torch.Tensor,
    ) -> OrthogonalTriadicRelayOutput:
        batch_size = modal_residual.shape[0]
        expected = (
            batch_size,
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
            self.residual_width,
        )
        if modal_residual.shape != expected:
            raise ValueError(f"modal residual must have shape {expected}")
        if modal_quality.shape != (batch_size, len(MODALITY_ORDER)):
            raise ValueError("modal quality must have shape B,M")
        states = torch.stack(
            [
                self.input_projections[expert](modal_residual[:, index])
                for index, expert in enumerate(EXPERT_ORDER)
            ],
            dim=1,
        )
        receiver_inputs = []
        orthogonal_messages = []
        for relay_round in self.rounds:
            receiver_inputs.append(states)
            states, messages = relay_round(states, modal_quality)
            orthogonal_messages.append(messages)
        return OrthogonalTriadicRelayOutput(
            enhanced=states,
            receiver_inputs=tuple(receiver_inputs),
            orthogonal_messages=tuple(orthogonal_messages),
        )


class OrthogonalTriadicSynthesis(nn.Module):
    """Synthesize a new identity residual from mutually enhanced experts."""

    def __init__(
        self,
        *,
        baseline_width: int,
        prefix_width: int,
        residual_width: int,
        hidden_width: int,
        synergy_modal_width: int,
        relay_depth: int,
        beta_max: float,
        beta_init: float,
    ) -> None:
        super().__init__()
        self.baseline_width = int(baseline_width)
        self.prefix_width = int(prefix_width)
        self.synergy_modal_width = int(synergy_modal_width)
        self.beta_max = float(beta_max)
        self.relay = OrthogonalTriadicRelay(
            residual_width=residual_width,
            hidden_width=hidden_width,
            relay_depth=relay_depth,
        )
        self.branch_projections = nn.ModuleDict(
            {
                expert: nn.Linear(hidden_width, synergy_modal_width, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.synergy_head = nn.Sequential(
            nn.Linear(6 * hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, synergy_modal_width, bias=False),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.constant_(
            self.beta_head[-1].bias,
            math.log((beta_init / beta_max) / (1.0 - beta_init / beta_max)),
        )

    def forward(
        self,
        *,
        baseline_embedding: torch.Tensor,
        prefix_embedding: torch.Tensor,
        modal_residual: torch.Tensor,
        modal_quality: torch.Tensor,
    ) -> OrthogonalTriadicSynthesisOutput:
        if baseline_embedding.shape[1] != self.baseline_width:
            raise ValueError("baseline embedding width is invalid")
        if prefix_embedding.shape[1] != self.prefix_width:
            raise ValueError("prefix embedding width is invalid")
        relay = self.relay(modal_residual, modal_quality)
        enhanced = relay.enhanced
        branch_residuals = {
            expert: F.normalize(
                self.branch_projections[expert](enhanced[:, index]).flatten(1),
                dim=1,
            )
            for index, expert in enumerate(EXPERT_ORDER)
        }
        baseline_norm = baseline_embedding.detach().norm(dim=1, keepdim=True)
        branch_embeddings = {
            expert: torch.cat(
                (baseline_embedding, branch_residuals[expert] * baseline_norm),
                dim=1,
            )
            for expert in EXPERT_ORDER
        }
        cnn, transformer, mamba = enhanced.unbind(dim=1)
        synergy_features = torch.cat(
            (
                cnn,
                transformer,
                mamba,
                cnn * transformer,
                cnn * mamba,
                transformer * mamba,
            ),
            dim=-1,
        )
        synergy_embedding = F.normalize(
            self.synergy_head(synergy_features).flatten(1),
            dim=1,
        )
        beta = self.beta_max * torch.sigmoid(
            self.beta_head(enhanced.mean(dim=(1, 2)))
        )
        activated_synergy = (
            synergy_embedding
            * prefix_embedding.detach().norm(dim=1, keepdim=True)
            * beta
        )
        return OrthogonalTriadicSynthesisOutput(
            fused_embedding=torch.cat((prefix_embedding, activated_synergy), dim=1),
            synergy_embedding=synergy_embedding,
            branch_embeddings=branch_embeddings,
            beta=beta,
            relay=relay,
        )


class SignalPreservingCollaborativeV9(nn.Module):
    """Freeze V8 and learn only peer-conditioned identity synthesis."""

    def __init__(
        self,
        *,
        phase_a: nn.Module,
        router: nn.Module,
        phase_b_fusion: nn.Module,
        synthesis: OrthogonalTriadicSynthesis,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.phase_a = phase_a
        self.router = router
        self.phase_b_fusion = phase_b_fusion
        self.synthesis = synthesis
        for module in (self.phase_a, self.router, self.phase_b_fusion):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.baseline_embedding_width = synthesis.baseline_width
        self.phase_b_embedding_width = synthesis.prefix_width
        self.synergy_embedding_width = (
            len(MODALITY_ORDER) * synthesis.synergy_modal_width
        )
        self.fused_embedding_width = (
            self.phase_b_embedding_width + self.synergy_embedding_width
        )
        self.branch_embedding_width = (
            self.baseline_embedding_width + self.synergy_embedding_width
        )
        self.fused_neck = self._make_neck(self.fused_embedding_width)
        self.synergy_neck = self._make_neck(self.synergy_embedding_width)
        self.branch_necks = nn.ModuleDict(
            {
                expert: self._make_neck(self.branch_embedding_width)
                for expert in EXPERT_ORDER
            }
        )
        if num_classes:
            self.fused_classifier = nn.Linear(
                self.fused_embedding_width,
                num_classes,
                bias=False,
            )
            self.synergy_classifier = nn.Linear(
                self.synergy_embedding_width,
                num_classes,
                bias=False,
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
            for classifier in (
                self.fused_classifier,
                self.synergy_classifier,
                *self.branch_classifiers.values(),
            ):
                nn.init.normal_(classifier.weight, std=0.001)
        else:
            self.fused_classifier = None
            self.synergy_classifier = None
            self.branch_classifiers = nn.ModuleDict()

    @staticmethod
    def _make_neck(width: int) -> nn.BatchNorm1d:
        neck = nn.BatchNorm1d(width)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        neck.bias.requires_grad_(False)
        return neck

    def train(self, mode: bool = True) -> SignalPreservingCollaborativeV9:
        super().train(mode)
        self.phase_a.eval()
        self.router.eval()
        self.phase_b_fusion.eval()
        return self

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_aux: bool = False,
        retrieval_output: str = "fused",
    ) -> torch.Tensor | SignalPreservingV9Output:
        with torch.no_grad():
            phase = self.phase_a(batch, return_aux=True)
            modal_residual = torch.stack(
                [
                    phase.modal_residual_embeddings[expert]
                    for expert in EXPERT_ORDER
                ],
                dim=1,
            )
            routing = self.router(
                phase.direct_modal,
                modal_residual,
                batch["modality_mask"],
            )
            phase_b = self.phase_b_fusion(
                phase.baseline_embedding,
                modal_residual,
                routing,
            )
        synthesis = self.synthesis(
            baseline_embedding=phase.baseline_embedding,
            prefix_embedding=phase_b.fused_embedding,
            modal_residual=modal_residual,
            modal_quality=routing.modal_probabilities,
        )
        if not return_aux:
            if retrieval_output == "baseline_only":
                return phase.baseline_embedding
            if retrieval_output == "phase_b":
                return phase_b.fused_embedding
            if retrieval_output == "fused":
                return synthesis.fused_embedding
            if retrieval_output in EXPERT_ORDER:
                return synthesis.branch_embeddings[retrieval_output]
            raise ValueError("retrieval output must be baseline_only, phase_b, fused, or an expert")

        fused_logits = None
        synergy_logits = None
        branch_logits: dict[str, torch.Tensor] = {}
        if self.fused_classifier is not None:
            fused_logits = self.fused_classifier(
                self.fused_neck(synthesis.fused_embedding)
            )
            synergy_logits = self.synergy_classifier(
                self.synergy_neck(synthesis.synergy_embedding)
            )
            branch_logits = {
                expert: self.branch_classifiers[expert](
                    self.branch_necks[expert](synthesis.branch_embeddings[expert])
                )
                for expert in EXPERT_ORDER
            }
        finite = (
            synthesis.fused_embedding,
            synthesis.synergy_embedding,
            synthesis.beta,
            *synthesis.branch_embeddings.values(),
        )
        return SignalPreservingV9Output(
            baseline_embedding=phase.baseline_embedding,
            phase_b_embedding=phase_b.fused_embedding,
            fused_embedding=synthesis.fused_embedding,
            synergy_embedding=synthesis.synergy_embedding,
            branch_embeddings=synthesis.branch_embeddings,
            fused_logits=fused_logits,
            synergy_logits=synergy_logits,
            branch_logits=branch_logits,
            beta=synthesis.beta,
            relay=synthesis.relay,
            diagnostics={
                "all_finite": all(bool(torch.isfinite(value).all()) for value in finite),
                "baseline_exact_prefix": torch.equal(
                    phase_b.fused_embedding[:, : self.baseline_embedding_width],
                    phase.baseline_embedding,
                ),
                "phase_b_exact_prefix": torch.equal(
                    synthesis.fused_embedding[:, : self.phase_b_embedding_width],
                    phase_b.fused_embedding,
                ),
                "phase_a_frozen": all(
                    not parameter.requires_grad for parameter in self.phase_a.parameters()
                ),
                "router_frozen": all(
                    not parameter.requires_grad for parameter in self.router.parameters()
                ),
            },
        )


class SignalPreservingV9Criterion(nn.Module):
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
        output: SignalPreservingV9Output,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if output.fused_logits is None or output.synergy_logits is None:
            raise ValueError("V9 collaboration training requires identity classifiers")
        id_fused, triplet_fused = self._identity_losses(
            output.fused_logits,
            output.fused_embedding,
            labels,
        )
        id_synergy, triplet_synergy = self._identity_losses(
            output.synergy_logits,
            output.synergy_embedding,
            labels,
        )
        losses = {
            "id_fused": id_fused,
            "triplet_fused": triplet_fused,
            "id_synergy": id_synergy,
            "triplet_synergy": triplet_synergy,
        }
        for expert in EXPERT_ORDER:
            losses[f"id_{expert}"], losses[f"triplet_{expert}"] = (
                self._identity_losses(
                    output.branch_logits[expert],
                    output.branch_embeddings[expert],
                    labels,
                )
            )
        return losses


__all__ = [
    "OrthogonalTriadicRelay",
    "OrthogonalTriadicRelayOutput",
    "OrthogonalTriadicSynthesis",
    "OrthogonalTriadicSynthesisOutput",
    "SignalPreservingCollaborativeV9",
    "SignalPreservingV9Criterion",
    "SignalPreservingV9Output",
]
