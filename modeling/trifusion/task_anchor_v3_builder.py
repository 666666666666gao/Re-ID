"""Build the isolated task-anchored V3 model from the bound CLIP checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .builder import (
    TriFusionBuildResult,
    _DeMoCLIPBlockAdapter,
    _load_resized_clip_visual,
    _sha256_file,
)
from .experts.mamba import production_mamba_factory
from .experts.semantic_residual import (
    SemanticCNNExpert,
    SemanticMambaExpert,
    SemanticTransformerExpert,
)
from .reliability import ReliabilityPosterior, UniformReliabilityGate
from .relay import HeterogeneousRelay
from .task_anchor_v3 import (
    AnchorResidualCollaborativeFusion,
    TaskAdaptedAnchorTokenizer,
    TaskAnchoredCollaborativeReID,
    TaskAnchoredTriBranchEncoder,
)


def build_task_anchored_trifusion_v3_from_clip(
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
    architecture: str = "task_anchored_collaborative_v3",
    adapter_width: int = 192,
    gradient_checkpointing: bool = True,
    residual_scale_init: float = 0.25,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    del cnn_width, mamba_width
    if architecture != "task_anchored_collaborative_v3":
        raise ValueError("V3 builder accepts only task_anchored_collaborative_v3")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if image_size[0] % patch_size or image_size[1] % patch_size:
        raise ValueError("image dimensions must be divisible by patch size")
    if adapter_width <= 0:
        raise ValueError("adapter width must be positive")
    checkpoint_sha256 = _sha256_file(checkpoint)
    visual = _load_resized_clip_visual(
        checkpoint,
        image_size=image_size,
        patch_size=patch_size,
    )
    if visual.proj is None:
        raise ValueError("V3 task anchor requires the CLIP visual projection")
    semantic_width = int(visual.conv1.out_channels)
    if visual.proj.shape != (semantic_width, embedding_width):
        raise ValueError("configured embedding width must match CLIP visual projection")
    grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
    tokenizer = TaskAdaptedAnchorTokenizer(
        patch_projection=visual.conv1,
        positional_embedding=visual.positional_embedding,
        class_embedding=visual.class_embedding,
        pre_norm=visual.ln_pre,
        post_norm=visual.ln_post,
        shared_blocks=[
            _DeMoCLIPBlockAdapter(block) for block in visual.transformer.resblocks
        ],
        output_projection=visual.proj,
        gradient_checkpointing=gradient_checkpointing,
    )
    stage_depths = (1, 1, 1)
    scale_init = 1e-3
    experts = {
        "cnn": SemanticCNNExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=scale_init,
        ),
        "transformer": SemanticTransformerExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=scale_init,
        ),
        "mamba": SemanticMambaExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
            scale_init=scale_init,
        ),
    }
    expert_widths = {expert: semantic_width for expert in experts}
    if reliability_mode == "uniform":
        posterior: nn.Module = UniformReliabilityGate()
    elif reliability_mode == "joint_beta":
        posterior = ReliabilityPosterior(
            expert_widths=expert_widths,
            hidden_width=128,
            heads=4,
            kappa_min=2.0,
        )
    else:
        raise ValueError("reliability_mode must be uniform or joint_beta")
    relay = HeterogeneousRelay(
        expert_widths=expert_widths,
        relay_rank=relay_rank,
        token_grid=grid_size,
        gamma_init=0.05,
    )
    encoder = TaskAnchoredTriBranchEncoder(
        experts,
        reliability_gate=posterior,
        collaborator=relay,
    )
    fusion = AnchorResidualCollaborativeFusion(
        expert_widths=expert_widths,
        embedding_width=embedding_width,
        residual_scale_init=residual_scale_init,
    )
    with torch.no_grad():
        pretrained_projection = visual.proj.detach().T.float()
        for projection in fusion.residual_projections.values():
            projection.weight.copy_(pretrained_projection)
    model = TaskAnchoredCollaborativeReID(
        tokenizer=tokenizer,
        encoder=encoder,
        fusion=fusion,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = {
        "architecture": architecture,
        "clip_checkpoint": str(checkpoint),
        "clip_checkpoint_sha256": checkpoint_sha256,
        "shared_clip_layers": len(tokenizer.shared_blocks),
        "image_size": list(image_size),
        "token_grid": list(grid_size),
        "embedding_width": embedding_width,
        "anchor_embedding_width": fusion.anchor_embedding_width,
        "fused_embedding_width": fusion.fused_embedding_width,
        "adapter_width": adapter_width,
        "adapter_stage_depths": list(stage_depths),
        "adapter_scale_init": scale_init,
        "relay_rank": relay_rank,
        "relay_gamma_init": 0.05,
        "reliability_mode": reliability_mode,
        "reliability_refresh_stages": [1, 2, 3],
        "residual_scale_init": residual_scale_init,
        "anchor_path": "exact_projected_clip_cls_concat_rgb_ni_ti",
        "fusion": "anchor_plus_quality_routed_bounded_tri_expert_residual_3072",
        "loss_slot_contract": {
            "reliability": "anchor_id_quarter_plus_anchor_triplet",
            "private_diversity": "supervised_cross_modal_identity_alignment",
        },
        "signal_mechanism_source_commit": "cd1b0a672d1fe642e7608731cb4899a19dda7d51",
        "signal_license": "MIT",
        "gradient_checkpointing": bool(gradient_checkpointing),
        "total_parameters": total_parameters,
        "parameter_budget_pass": total_parameters <= 120_000_000,
        "mamba_mixer_factory": getattr(
            mamba_mixer_factory,
            "__name__",
            type(mamba_mixer_factory).__name__,
        ),
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        provenance=provenance,
    )


__all__ = ["build_task_anchored_trifusion_v3_from_clip"]
