"""Validated full-network interventions used to score CIRC targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .state import EXPERT_ORDER, MODALITY_ORDER


@dataclass(frozen=True)
class FullNetworkIntervention:
    """Remove one expert-modality contribution from selected network paths."""

    kind: str
    modality: str
    expert: str | None = None
    stage: int | None = None
    source: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_ORDER:
            raise ValueError(
                f"intervention modality must be one of {MODALITY_ORDER}"
            )
        if self.kind == "edge":
            if self.expert is not None:
                raise ValueError("edge intervention must not define expert")
            if self.stage not in (1, 2):
                raise ValueError("edge intervention stage must be 1 or 2")
            if self.source not in EXPERT_ORDER or self.target not in EXPERT_ORDER:
                raise ValueError(f"edge experts must be drawn from {EXPERT_ORDER}")
            if self.source == self.target:
                raise ValueError("edge intervention must be no-self")
            return
        if self.kind not in ("direct", "relay", "total"):
            raise ValueError("intervention kind must be direct, relay, total, or edge")
        if self.expert not in EXPERT_ORDER:
            raise ValueError(f"intervention expert must be one of {EXPERT_ORDER}")
        if any(value is not None for value in (self.stage, self.source, self.target)):
            raise ValueError("contribution intervention must not define edge fields")

    @classmethod
    def from_value(cls, value: object) -> FullNetworkIntervention:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("intervention must be a mapping or FullNetworkIntervention")
        kind = str(value.get("kind", ""))
        expected = (
            {"kind", "stage", "source", "target", "modality"}
            if kind == "edge"
            else {"kind", "expert", "modality"}
        )
        if set(value) != expected:
            raise ValueError(f"intervention must contain exactly {sorted(expected)}")
        typed: Mapping[str, Any] = value
        if kind == "edge":
            return cls(
                kind=kind,
                modality=str(typed["modality"]),
                stage=int(typed["stage"]),
                source=str(typed["source"]),
                target=str(typed["target"]),
            )
        return cls(
            kind=kind,
            modality=str(typed["modality"]),
            expert=str(typed["expert"]),
        )

    @property
    def suppresses_relay(self) -> bool:
        return self.kind in ("relay", "total", "edge")

    @property
    def suppresses_fusion(self) -> bool:
        return self.kind in ("direct", "total")

    def validate_modality_mask(self, modality_mask: torch.Tensor) -> None:
        if self.kind != "edge":
            return
        modality_index = MODALITY_ORDER.index(self.modality)
        invalid_rows = (~modality_mask[:, modality_index]).nonzero(
            as_tuple=False
        ).flatten()
        if invalid_rows.numel():
            rows = invalid_rows.detach().cpu().tolist()
            raise ValueError(f"edge intervention is invalid for modality rows: {rows}")

    def relay_suppression(
        self, stage: int
    ) -> tuple[str, str, str | None] | None:
        if not self.suppresses_relay:
            return None
        if self.kind == "edge":
            if self.stage != stage:
                return None
            if self.source is None or self.target is None:
                raise RuntimeError("edge intervention was not validated")
            return self.source, self.modality, self.target
        if self.expert is None:
            raise RuntimeError("contribution intervention was not validated")
        return self.expert, self.modality, None

    def fusion_suppression(self) -> tuple[str, str] | None:
        if not self.suppresses_fusion:
            return None
        if self.expert is None:
            raise RuntimeError("contribution intervention was not validated")
        return self.expert, self.modality


__all__ = ["FullNetworkIntervention"]
