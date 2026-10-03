"""Fail-closed HFER-uniform to CIRC-posterior warm-start transition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def load_hfer_uniform_warm_start(
    model: torch.nn.Module,
    checkpoint: Path | str,
    *,
    allow_classifier_reinitialization: bool = False,
    classifier_prefixes: tuple[str, ...] = (
        "fused_classifier.",
        "branch_classifiers.",
    ),
) -> dict[str, Any]:
    """Load all shared HFER weights while allowing only the new posterior to be absent."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not hasattr(model, "encoder") or not hasattr(
        model.encoder, "reliability_gate"
    ):
        raise TypeError("full CIRC model must expose encoder.reliability_gate")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("HFER-uniform checkpoint must contain a nonempty state dictionary")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise ValueError("HFER-uniform checkpoint state must contain only named tensors")

    model_keys = set(model.state_dict())
    posterior_keys = {
        f"encoder.reliability_gate.{name}"
        for name in model.encoder.reliability_gate.state_dict()
    }
    state = dict(state)
    model_state = model.state_dict()
    reinitialized_classifier_keys = sorted(
        name
        for name in model_keys
        if any(name.startswith(prefix) for prefix in classifier_prefixes)
        and (
            name not in state
            or tuple(state[name].shape) != tuple(model_state[name].shape)
        )
    )
    if reinitialized_classifier_keys and not allow_classifier_reinitialization:
        raise ValueError(
            "HFER warm-start tensor mismatch in identity classifiers: "
            f"{reinitialized_classifier_keys}"
        )
    for name in reinitialized_classifier_keys:
        state.pop(name, None)
    state_keys = set(state)
    allowed_missing = posterior_keys | set(reinitialized_classifier_keys)
    nonposterior_missing = sorted((model_keys - state_keys) - allowed_missing)
    foreign = sorted(state_keys - model_keys)
    if nonposterior_missing or foreign:
        raise ValueError(
            "HFER warm start has non-posterior gaps or foreign tensors: "
            f"missing={nonposterior_missing}, unexpected={foreign}"
        )
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise ValueError(f"HFER warm-start tensor mismatch: {error}") from error
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            "HFER warm start may omit only the CIRC posterior: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": str(checkpoint),
        "loaded_tensor_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "posterior_initialized_from_scratch": True,
        "classifier_reinitialized": bool(reinitialized_classifier_keys),
        "reinitialized_classifier_keys": reinitialized_classifier_keys,
    }


__all__ = ["load_hfer_uniform_warm_start"]
