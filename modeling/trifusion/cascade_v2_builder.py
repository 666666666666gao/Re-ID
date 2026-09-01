"""Builder for the isolated anchor-preserving TriFusion cascade V2."""

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
from .cascade_v2 import (
    AnchorPreservingSemanticTokenizer,
    CascadeV2ReID,
    InformationPreservingFusion,
    StageUpdatedTriBranchEncoder,
)
from .experts.mamba import production_mamba_factory
from .experts.semantic_residual import (
    SemanticCNNExpert,
    SemanticMambaExpert,
    SemanticTransformerExpert,
)
from .reliability import ReliabilityPosterior, UniformReliabilityGate
from .relay import HeterogeneousRelay


def build_trifusion_cascade_v2_from_clip(
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
    architecture: str = "shared_semantic_cascade_v2",
    adapter_width: int = 192,
    gradient_checkpointing: bool = False,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    del cnn_width, mamba_width
    if architecture != "shared_semantic_cascade_v2":
        raise ValueError("cascade V2 builder requires its registered architecture")
    if adapter_width <= 0:
        raise ValueError("adapter_width must be positive")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if image_size[0] % patch_size or image_size[1] % patch_size:
        raise ValueError("image dimensions must be divisible by patch size")
    grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
    checkpoint_sha256 = _sha256_file(checkpoint)
    visual = _load_resized_clip_visual(
        checkpoint, image_size=image_size, patch_size=patch_size
    )
    semantic_width = int(visual.conv1.out_channels)
    expert_widths = {
        expert: semantic_width
        for expert in ("cnn", "transformer", "mamba")
    }
    shared_blocks = [
        _DeMoCLIPBlockAdapter(block)
        for block in visual.transformer.resblocks
    ]
    tokenizer = AnchorPreservingSemanticTokenizer(
        patch_projection=visual.conv1,
        positional_embedding=visual.positional_embedding,
        class_embedding=visual.class_embedding,
        pre_norm=visual.ln_pre,
        post_norm=visual.ln_post,
        shared_blocks=shared_blocks,
        gradient_checkpointing=gradient_checkpointing,
    )
    stage_depths = (1, 1, 1)
    residual_scale_init = 1e-3
    experts = {
        "cnn": SemanticCNNExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=residual_scale_init,
        ),
        "transformer": SemanticTransformerExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            scale_init=residual_scale_init,
        ),
        "mamba": SemanticMambaExpert(
            width=semantic_width,
            adapter_width=adapter_width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
            scale_init=residual_scale_init,
        ),
    }
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

    relay_gamma_init = 0.1
    encoder = StageUpdatedTriBranchEncoder(
        experts,
        tokenizer=tokenizer,
        reliability_gate=posterior,
        collaborator=HeterogeneousRelay(
            expert_widths=expert_widths,
            relay_rank=relay_rank,
            token_grid=grid_size,
            gamma_init=relay_gamma_init,
        ),
        refresh_final_reliability=True,
    )
    fusion = InformationPreservingFusion(
        expert_widths=expert_widths,
        embedding_width=embedding_width,
    )
    if visual.proj is not None and visual.proj.shape == (
        semantic_width,
        embedding_width,
    ):
        with torch.no_grad():
            fusion.semantic_projection.weight.copy_(
                visual.proj.detach().T.float()
            )
    model = CascadeV2ReID(
        encoder=encoder,
        fusion=fusion,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = {
        "architecture": "shared_semantic_cascade_v2",
        "clip_checkpoint": str(checkpoint),
        "clip_checkpoint_sha256": checkpoint_sha256,
        "shared_clip_layers": len(shared_blocks),
        "image_size": list(image_size),
        "token_grid": list(grid_size),
        "expert_widths": expert_widths,
        "adapter_width": adapter_width,
        "adapter_stage_depths": list(stage_depths),
        "adapter_scale_init": residual_scale_init,
        "relay_rank": relay_rank,
        "relay_gamma_init": relay_gamma_init,
        "reliability_mode": reliability_mode,
        "semantic_decomposition": "centered_patch_exact_cls",
        "reliability_refresh_stages": [1, 2, 3],
        "fusion": f"quality_gated_blockwise_{fusion.fused_embedding_width}",
        "retrieval_before_neck": True,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "embedding_width": embedding_width,
        "shared_patch_projection_instances": 1,
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


__all__ = ["build_trifusion_cascade_v2_from_clip"]
