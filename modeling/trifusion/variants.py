"""Frozen experiment identities that prevent name-only ablations."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


_VARIANTS: dict[str, dict[str, Any]] = {
    "core_pre_circ": {
        "variant": "core_pre_circ",
        "family": "collaborative",
        "claim_role": "pre-CIRC training gate",
        "active_experts": ["cnn", "transformer", "mamba"],
        "collaborator": "hfer",
        "reliability": "joint_beta_observational",
        "fusion": "reliability_weighted",
        "peer_mode": "none",
        "circ_targets_required": False,
        "evaluation_outputs": ["fused", "cnn", "transformer", "mamba"],
    },
    "cnn_standalone": {
        "variant": "cnn_standalone",
        "family": "standalone",
        "claim_role": "R020",
        "active_experts": ["cnn"],
        "collaborator": "none",
        "reliability": "none",
        "fusion": "single_expert",
        "peer_mode": "none",
        "circ_targets_required": False,
        "evaluation_outputs": ["cnn"],
    },
    "transformer_standalone": {
        "variant": "transformer_standalone",
        "family": "standalone",
        "claim_role": "R021",
        "active_experts": ["transformer"],
        "collaborator": "none",
        "reliability": "none",
        "fusion": "single_expert",
        "peer_mode": "none",
        "circ_targets_required": False,
        "evaluation_outputs": ["transformer"],
    },
    "mamba_standalone": {
        "variant": "mamba_standalone",
        "family": "standalone",
        "claim_role": "R022",
        "active_experts": ["mamba"],
        "collaborator": "none",
        "reliability": "none",
        "fusion": "single_expert",
        "peer_mode": "none",
        "circ_targets_required": False,
        "evaluation_outputs": ["mamba"],
    },
}


def variant_names() -> tuple[str, ...]:
    return tuple(_VARIANTS)


def resolve_variant(name: str) -> Mapping[str, Any]:
    if name not in _VARIANTS:
        raise ValueError(f"unknown TriFusion experiment variant: {name}")
    return MappingProxyType(dict(_VARIANTS[name]))


def variant_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["resolve_variant", "variant_names", "variant_sha256"]
