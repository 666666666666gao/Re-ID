"""Frozen phase policy for CIRC router warm-up and joint optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CIRCTrainingPhase:
    name: str
    epoch: int
    parameter_scope: str
    loss_scope: str


def resolve_training_phase(
    *,
    epoch: int,
    circ_enabled: bool,
    router_warm_epochs: int,
    schedule_horizon_epochs: int,
) -> CIRCTrainingPhase:
    if epoch < 1 or epoch > schedule_horizon_epochs:
        raise ValueError("epoch is outside the configured training horizon")
    if router_warm_epochs < 0 or router_warm_epochs >= schedule_horizon_epochs:
        raise ValueError("router warm-up must leave at least one joint epoch")
    if circ_enabled and router_warm_epochs == 0:
        raise ValueError("CIRC training requires a nonempty router-only warm-up")
    if circ_enabled and epoch <= router_warm_epochs:
        return CIRCTrainingPhase(
            name="router_only",
            epoch=epoch,
            parameter_scope="encoder.reliability_gate",
            loss_scope="immutable_circ_reliability_only",
        )
    return CIRCTrainingPhase(
        name="joint_hfer_urgc" if circ_enabled else "joint_retrieval",
        epoch=epoch,
        parameter_scope="all_except_private_projection",
        loss_scope="full_registered_objective",
    )


def parameter_trainable_in_phase(name: str, phase: CIRCTrainingPhase) -> bool:
    if "private_projection" in name:
        return False
    if phase.name == "router_only":
        return name.startswith("encoder.reliability_gate.")
    return True


def active_loss_weights(
    weights: Mapping[str, float], phase: CIRCTrainingPhase
) -> dict[str, float]:
    registered = {str(name): float(weight) for name, weight in weights.items()}
    if phase.name != "router_only":
        return registered
    if "reliability" not in registered or registered["reliability"] <= 0.0:
        raise ValueError("router-only warm-up requires positive immutable reliability loss")
    return {"reliability": registered["reliability"]}


__all__ = [
    "CIRCTrainingPhase",
    "active_loss_weights",
    "parameter_trainable_in_phase",
    "resolve_training_phase",
]
