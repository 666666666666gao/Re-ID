"""Heterogeneous Feature Exchange Relay (HFER)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from .state import (
    EXPERT_ORDER,
    ExpertState,
    ExpertStateMap,
    ReliabilityResult,
)


class _CNNRoleMixer(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.local = nn.Conv1d(rank, rank, kernel_size=3, padding=1, groups=rank)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, rank = tokens.shape
        flat = tokens.reshape(batch_size * modality_count, token_count, rank)
        mixed = self.local(flat.transpose(1, 2)).transpose(1, 2)
        return mixed.reshape(batch_size, modality_count, token_count, rank)


class _TransformerRoleMixer(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        heads = 4 if rank % 4 == 0 else 1
        self.attention = nn.MultiheadAttention(rank, heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, rank = tokens.shape
        flat = tokens.reshape(batch_size * modality_count, token_count, rank)
        mixed = self.attention(flat, flat, flat, need_weights=False)[0]
        return mixed.reshape(batch_size, modality_count, token_count, rank)


class _MambaRoleMixer(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.bidirectional = nn.Conv1d(
            rank, rank, kernel_size=3, padding=1, groups=rank
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, token_count, rank = tokens.shape
        flat = tokens.reshape(batch_size * modality_count, token_count, rank)
        forward = self.bidirectional(flat.transpose(1, 2)).transpose(1, 2)
        reverse = self.bidirectional(
            flat.flip(dims=(1,)).transpose(1, 2)
        ).transpose(1, 2).flip(dims=(1,))
        return (0.5 * (forward + reverse)).reshape(
            batch_size, modality_count, token_count, rank
        )


@dataclass(frozen=True, eq=False)
class RelayResult:
    states: ExpertStateMap
    gates: torch.Tensor
    private_energy: Mapping[str, torch.Tensor]
    reliability: ReliabilityResult
    stage: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "private_energy", MappingProxyType(dict(self.private_energy))
        )


class HeterogeneousRelay(nn.Module):
    """Synchronous no-self role-specific exchange in a shared low-rank space."""

    def __init__(
        self,
        *,
        expert_widths: Mapping[str, int],
        relay_rank: int,
        token_grid: tuple[int, int],
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        if relay_rank <= 0 or token_grid[0] * token_grid[1] <= 0:
            raise ValueError("relay rank and token grid must be positive")
        self.token_grid = token_grid
        self.norms = nn.ModuleDict(
            {expert: nn.LayerNorm(expert_widths[expert]) for expert in EXPERT_ORDER}
        )
        self.to_shared = nn.ModuleDict(
            {
                expert: nn.Linear(expert_widths[expert], relay_rank, bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.from_shared = nn.ModuleDict(
            {
                expert: nn.Linear(relay_rank, expert_widths[expert], bias=False)
                for expert in EXPERT_ORDER
            }
        )
        self.role_mixers = nn.ModuleDict(
            {
                "cnn": _CNNRoleMixer(relay_rank),
                "transformer": _TransformerRoleMixer(relay_rank),
                "mamba": _MambaRoleMixer(relay_rank),
            }
        )
        self.gamma = nn.Parameter(torch.full((len(EXPERT_ORDER),), gamma_init))

    def forward(
        self,
        states: ExpertStateMap,
        reliability: ReliabilityResult,
        stage: int,
    ) -> RelayResult:
        if stage not in (1, 2):
            raise ValueError("relay stage must be 1 or 2")
        if not torch.equal(states.modality_mask, reliability.modality_mask):
            raise ValueError("states and reliability must use the same modality mask")
        modality_mask = states.modality_mask
        batch_size, modality_count = modality_mask.shape

        shared_states = {}
        private_energy = {}
        role_messages = {}
        mask4 = modality_mask[:, :, None, None]
        for expert in EXPERT_ORDER:
            native = states[expert].tokens
            if native.shape[2] != self.token_grid[0] * self.token_grid[1]:
                raise ValueError("state token count does not match relay token grid")
            normalized = self.norms[expert](native)
            shared = self.to_shared[expert](normalized) * mask4
            reconstructed = self.from_shared[expert](shared)
            private = (native - reconstructed) * mask4
            shared_states[expert] = shared
            private_energy[expert] = private.square().mean()

            denominator = modality_mask.sum(dim=1, keepdim=True).clamp_min(1)
            consensus = shared.sum(dim=1) / denominator[:, :, None]
            enriched = shared + consensus[:, None]
            role_messages[expert] = self.role_mixers[expert](enriched) * mask4

        source_scores = reliability.r.clamp_min(0.0)
        raw_gates = source_scores[:, None].expand(
            batch_size, len(EXPERT_ORDER), len(EXPERT_ORDER), modality_count
        )
        no_self = ~torch.eye(
            len(EXPERT_ORDER), dtype=torch.bool, device=raw_gates.device
        )
        raw_gates = raw_gates * no_self[None, :, :, None]
        raw_gates = raw_gates * modality_mask[:, None, None, :]
        denominator = raw_gates.sum(dim=2, keepdim=True)
        gates = torch.where(
            denominator > 0,
            raw_gates / denominator.clamp_min(torch.finfo(raw_gates.dtype).eps),
            torch.zeros_like(raw_gates),
        )

        output_states = {}
        for target_index, target in enumerate(EXPERT_ORDER):
            incoming = torch.zeros_like(shared_states[target])
            for source_index, source in enumerate(EXPERT_ORDER):
                source_gate = gates[:, target_index, source_index]
                incoming = incoming + source_gate[:, :, None, None] * role_messages[
                    source
                ]
            native_message = self.from_shared[target](incoming)
            updated_tokens = (
                states[target].tokens
                + self.gamma[target_index].to(dtype=native_message.dtype)
                * native_message
            ) * mask4
            output_states[target] = ExpertState(
                tokens=updated_tokens,
                global_embedding=updated_tokens.mean(dim=2),
                private_embedding=states[target].private_embedding,
                role_payload=states[target].role_payload,
                modality_mask=modality_mask,
                stage=stage,
                expert=target,
            )
        return RelayResult(
            states=ExpertStateMap(output_states, modality_mask=modality_mask),
            gates=gates,
            private_energy=private_energy,
            reliability=reliability,
            stage=stage,
        )


__all__ = ["HeterogeneousRelay", "RelayResult"]
