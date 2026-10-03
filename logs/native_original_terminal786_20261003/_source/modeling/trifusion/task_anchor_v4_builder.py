"""Build task-anchored TriFusion V4 from the frozen CLIP source checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .builder import TriFusionBuildResult
from .experts.mamba import production_mamba_factory
from .task_anchor_v3_builder import build_task_anchored_trifusion_v3_from_clip
from .task_anchor_v4 import (
    EnergyBalancedResidualBankFusion,
    TaskAnchoredCollaborativeReIDV4,
)


def build_task_anchored_trifusion_v4_from_clip(
    checkpoint: Path | str,
    *,
    num_classes: int,
    image_size: tuple[int, int] = (256, 128),
    patch_size: int = 16,
    cnn_width: int = 256,
    mamba_width: int = 256,
    relay_rank: int = 64,
    embedding_width: int = 512,
    private_width: int = 64,
    reliability_mode: str = "joint_beta",
    architecture: str = "task_anchored_collaborative_v4",
    adapter_width: int = 192,
    gradient_checkpointing: bool = True,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    """Reuse V3's audited CLIP/expert trunk and replace only its failed fusion seam."""

    if architecture != "task_anchored_collaborative_v4":
        raise ValueError("V4 builder accepts only task_anchored_collaborative_v4")
    base = build_task_anchored_trifusion_v3_from_clip(
        checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        cnn_width=cnn_width,
        mamba_width=mamba_width,
        relay_rank=relay_rank,
        embedding_width=embedding_width,
        private_width=private_width,
        reliability_mode=reliability_mode,
        architecture="task_anchored_collaborative_v3",
        adapter_width=adapter_width,
        gradient_checkpointing=gradient_checkpointing,
        residual_scale_init=0.25,
        mamba_mixer_factory=mamba_mixer_factory,
    )
    semantic_width = int(base.model.tokenizer.width)
    fusion = EnergyBalancedResidualBankFusion(
        expert_widths={expert: semantic_width for expert in base.model.encoder.experts},
        embedding_width=embedding_width,
    )
    with torch.no_grad():
        for expert, projection in fusion.residual_projections.items():
            projection.weight.copy_(base.model.fusion.residual_projections[expert].weight)
    model = TaskAnchoredCollaborativeReIDV4(
        tokenizer=base.model.tokenizer,
        encoder=base.model.encoder,
        fusion=fusion,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = dict(base.provenance)
    provenance.update(
        {
            "architecture": architecture,
            "anchor_embedding_width": fusion.anchor_embedding_width,
            "residual_bank_width": fusion.residual_bank_width,
            "branch_embedding_width": fusion.branch_embedding_width,
            "fused_embedding_width": fusion.fused_embedding_width,
            "fusion": "energy_balanced_utility_routed_tri_expert_residual_bank",
            "energy_balance": "joint_residual_bank_l2_equals_detached_anchor_l2",
            "residual_bank_layout": "cnn_rgb_ni_ti__transformer_rgb_ni_ti__mamba_rgb_ni_ti",
            "router_supervision": "detached_per_sample_batch_hard_identity_gap_kl",
            "loss_slot_contract": {
                **dict(base.provenance["loss_slot_contract"]),
                "peer_logits": "identity_utility_router_kl",
            },
            "total_parameters": total_parameters,
            "parameter_budget_pass": total_parameters <= 120_000_000,
        }
    )
    provenance.pop("residual_scale_init", None)
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=base.checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_task_anchored_trifusion_v4_from_clip"]
