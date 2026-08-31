"""Immutable public result types shared by TriFusion components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any

import torch


EXPERT_ORDER = ("cnn", "transformer", "mamba")
MODALITY_ORDER = ("RGB", "NI", "TI")


@dataclass(frozen=True, eq=False)
class PackedExpertOutput:
    """Internal packed-slot result shared by tiny and production experts."""

    tokens: torch.Tensor
    global_embedding: torch.Tensor
    private_embedding: torch.Tensor
    role_payload: Mapping[str, torch.Tensor]
    stage: int = 3


@dataclass(frozen=True, eq=False)
class ReliabilityResult:
    """Joint Beta posterior values for B x expert x modality entries."""

    alpha: torch.Tensor
    beta: torch.Tensor
    r: torch.Tensor
    u: torch.Tensor
    modality_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.modality_mask.dtype != torch.bool or self.modality_mask.ndim != 2:
            raise ValueError("modality_mask must be a rank-2 bool tensor")
        expected_shape = (
            self.modality_mask.shape[0],
            len(EXPERT_ORDER),
            len(MODALITY_ORDER),
        )
        for name in ("alpha", "beta", "r", "u"):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape B,E,M")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, eq=False)
class ExpertState:
    """Observable state emitted by one complete heterogeneous expert."""

    tokens: torch.Tensor
    global_embedding: torch.Tensor
    private_embedding: torch.Tensor
    role_payload: Mapping[str, torch.Tensor]
    modality_mask: torch.Tensor
    stage: int
    expert: str

    def __post_init__(self) -> None:
        if self.expert not in EXPERT_ORDER:
            raise ValueError(f"unknown expert {self.expert!r}")
        if self.stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        if self.modality_mask.dtype != torch.bool or self.modality_mask.ndim != 2:
            raise ValueError("modality_mask must be a rank-2 bool tensor")
        batch_size, modality_count = self.modality_mask.shape
        if modality_count != len(MODALITY_ORDER):
            raise ValueError("modality_mask columns must be RGB, NI, TI")
        if self.tokens.ndim != 4 or self.tokens.shape[:2] != (
            batch_size,
            modality_count,
        ):
            raise ValueError("tokens must have shape B,M,N,D")
        if self.global_embedding.ndim != 3 or self.global_embedding.shape[:2] != (
            batch_size,
            modality_count,
        ):
            raise ValueError("global_embedding must have shape B,M,D")
        if self.private_embedding.ndim != 3 or self.private_embedding.shape[:2] != (
            batch_size,
            modality_count,
        ):
            raise ValueError("private_embedding must have shape B,M,Dp")
        object.__setattr__(
            self, "role_payload", MappingProxyType(dict(self.role_payload))
        )


@dataclass(frozen=True, eq=False)
class ExpertStateMap(Mapping[str, ExpertState]):
    """Read-only, expert-keyed states plus non-key metadata properties."""

    states: InitVar[Mapping[str, ExpertState]]
    modality_mask: torch.Tensor
    reliability: Any = None
    relay_results: tuple[Any, ...] = ()
    stage_traces: tuple[Any, ...] = ()
    _states: Mapping[str, ExpertState] = field(init=False, repr=False)

    def __post_init__(self, states: Mapping[str, ExpertState]) -> None:
        if set(states) != set(EXPERT_ORDER) or len(states) != len(EXPERT_ORDER):
            raise ValueError(f"states must contain exactly {EXPERT_ORDER}")
        if self.modality_mask.dtype != torch.bool or self.modality_mask.ndim != 2:
            raise ValueError("modality_mask must be a rank-2 bool tensor")
        ordered = {expert: states[expert] for expert in EXPERT_ORDER}
        for expert, state in ordered.items():
            if state.expert != expert:
                raise ValueError(f"state key {expert!r} does not match its expert")
            if not torch.equal(state.modality_mask, self.modality_mask):
                raise ValueError("all states must carry the exact modality mask values")
        object.__setattr__(self, "_states", MappingProxyType(ordered))

    def __getitem__(self, expert: str) -> ExpertState:
        return self._states[expert]

    def __iter__(self) -> Iterator[str]:
        return iter(EXPERT_ORDER)

    def __len__(self) -> int:
        return len(EXPERT_ORDER)


__all__ = [
    "EXPERT_ORDER",
    "MODALITY_ORDER",
    "ExpertState",
    "ExpertStateMap",
    "ReliabilityResult",
]
