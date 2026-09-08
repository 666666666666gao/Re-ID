"""Joint intervention-calibrated reliability posterior."""

from __future__ import annotations

from collections.abc import Mapping
from math import log

import torch
import torch.nn.functional as F
from torch import nn

from .state import EXPERT_ORDER, MODALITY_ORDER, ExpertStateMap, ReliabilityResult


class UniformReliabilityGate(nn.Module):
    """Return one parameter-free equal prior for HFER target generators."""

    def forward(
        self, states: ExpertStateMap, modality_mask: torch.Tensor
    ) -> ReliabilityResult:
        if not torch.equal(states.modality_mask, modality_mask):
            raise ValueError("states and uniform gate must use the same modality mask")
        valid = modality_mask[:, None, :].expand(
            modality_mask.shape[0],
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
        )
        reference = states["cnn"].tokens
        valid_float = valid.to(dtype=reference.dtype, device=reference.device)
        beta_prior = torch.full_like(valid_float, 2.0)
        return ReliabilityResult(
            alpha=beta_prior,
            beta=beta_prior.clone(),
            r=valid_float * 0.5,
            u=valid_float * 0.5,
            modality_mask=modality_mask,
        )


class ReliabilityPosterior(nn.Module):
    """Predict all nine Beta posteriors with one masked joint set function."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        hidden_width: int = 128,
        heads: int = 4,
        kappa_min: float = 2.0,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        if hidden_width <= 0 or heads <= 0 or hidden_width % heads:
            raise ValueError("hidden_width must be positive and divisible by heads")
        if kappa_min <= 0:
            raise ValueError("kappa_min must be positive")
        self.kappa_min = float(kappa_min)
        self.entry_projections = nn.ModuleDict(
            {
                expert: nn.Linear(expert_widths[expert], hidden_width, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.quality_projection = nn.Linear(5, hidden_width, bias=False)
        self.expert_embedding = nn.Parameter(
            torch.empty(len(EXPERT_ORDER), hidden_width)
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(len(MODALITY_ORDER), hidden_width)
        )
        self.set_attention = nn.MultiheadAttention(
            hidden_width, heads, batch_first=True
        )
        self.set_norm = nn.LayerNorm(hidden_width)
        self.set_mlp = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width * 2),
            nn.GELU(),
            nn.Linear(hidden_width * 2, hidden_width),
        )
        self.shared_output = nn.Linear(hidden_width, 2)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.expert_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self, states: ExpertStateMap, modality_mask: torch.Tensor
    ) -> ReliabilityResult:
        if not torch.equal(states.modality_mask, modality_mask):
            raise ValueError("states and posterior must use the same modality mask")
        batch_size, modality_count = modality_mask.shape
        mask_scalar = modality_mask.to(dtype=states["cnn"].tokens.dtype)

        projected_entries = []
        normalized_entries = []
        base_statistics = []
        for expert in EXPERT_ORDER:
            state = states[expert]
            projected = self.entry_projections[expert](state.global_embedding)
            projected = projected * mask_scalar[:, :, None]
            normalized = F.normalize(projected, dim=-1)

            token_energy = state.tokens.square().mean(dim=-1)
            token_probabilities = torch.softmax(token_energy, dim=-1)
            entropy = -(
                token_probabilities
                * token_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1) / max(log(state.tokens.shape[2]), 1.0)
            variance = state.tokens.var(dim=(2, 3), unbiased=False)
            norm = state.global_embedding.norm(dim=-1)

            denominator = modality_mask.sum(dim=1, keepdim=True).clamp_min(1)
            modal_consensus = normalized.sum(dim=1) / denominator
            cross_modal_agreement = (
                normalized * F.normalize(modal_consensus, dim=-1)[:, None]
            ).sum(dim=-1)

            projected_entries.append(projected)
            normalized_entries.append(normalized)
            base_statistics.append((variance, norm, entropy, cross_modal_agreement))

        projected_stack = torch.stack(projected_entries, dim=1)
        normalized_stack = torch.stack(normalized_entries, dim=1)
        expert_consensus = F.normalize(normalized_stack.mean(dim=1), dim=-1)
        cross_expert_agreement = (
            normalized_stack * expert_consensus[:, None]
        ).sum(dim=-1)

        statistics = []
        for expert_index, (variance, norm, entropy, cross_modal) in enumerate(
            base_statistics
        ):
            statistics.append(
                torch.stack(
                    (
                        variance,
                        norm,
                        entropy,
                        cross_modal,
                        cross_expert_agreement[:, expert_index],
                    ),
                    dim=-1,
                )
            )
        statistic_stack = torch.stack(statistics, dim=1)
        entries = projected_stack + self.quality_projection(statistic_stack)
        entries = (
            entries
            + self.expert_embedding[None, :, None, :]
            + self.modality_embedding[None, None, :, :]
        )

        valid_entries = modality_mask[:, None, :].expand(
            batch_size, len(EXPERT_ORDER), modality_count
        )
        flat_valid = valid_entries.reshape(batch_size, -1)
        flat_entries = entries.reshape(batch_size, -1, entries.shape[-1])
        attended = self.set_attention(
            flat_entries,
            flat_entries,
            flat_entries,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )[0]
        mixed = self.set_norm(flat_entries + attended)
        mixed = mixed + self.set_mlp(mixed)
        raw = self.shared_output(mixed).reshape(
            batch_size, len(EXPERT_ORDER), modality_count, 2
        )

        temperature = F.softplus(self.log_temperature) + 1e-4
        mu = torch.sigmoid(raw[..., 0] / temperature)
        kappa = self.kappa_min + F.softplus(raw[..., 1])
        alpha = 1.0 + mu * kappa
        beta = 1.0 + (1.0 - mu) * kappa
        valid_float = valid_entries.to(dtype=alpha.dtype)
        r = alpha / (alpha + beta) * valid_float
        u = 2.0 / (alpha + beta) * valid_float
        return ReliabilityResult(
            alpha=alpha,
            beta=beta,
            r=r,
            u=u,
            modality_mask=modality_mask,
        )


__all__ = ["ReliabilityPosterior", "UniformReliabilityGate"]
