"""Production TriFusion builder from a hash-bound CLIP ViT-B/16 checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn

from config import cfg as default_cfg
from modeling.clip.model import build_model as build_clip_model

from .encoder import TriBranchEncoder
from .experts.cnn import CNNExpert
from .experts.mamba import MambaExpert, production_mamba_factory
from .experts.semantic_residual import (
    SemanticCNNExpert,
    SemanticMambaExpert,
    SemanticTransformerExpert,
)
from .experts.transformer import TransformerExpert
from .fusion import CollaborativeFusion
from .model import TriFusionReID
from .reliability import ReliabilityPosterior, UniformReliabilityGate
from .relay import HeterogeneousRelay
from .semantic_tokenizer import SharedCLIPSemanticTokenizer
from .standalone import SingleBranchReID, SingleExpertTokenizer
from .tokenizer import SharedCLIPTokenizer
from .variants import resolve_variant, variant_sha256


class _DeMoCLIPBlockAdapter(nn.Module):
    """Expose a batch-first block while retaining the exact pretrained module."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_first = sequence.permute(1, 0, 2)
        output = self.block(
            sequence_first,
            modality=None,
            index=None,
            last_prompt=None,
            prompt_sign=False,
            adapter_sign=False,
        )
        return output.permute(1, 0, 2)


@dataclass(frozen=True, eq=False)
class TriFusionBuildResult:
    model: nn.Module
    checkpoint_sha256: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resized_clip_visual(
    checkpoint: Path, *, image_size: tuple[int, int], patch_size: int
) -> nn.Module:
    scripted = torch.jit.load(str(checkpoint), map_location="cpu").eval()
    state_dict = dict(scripted.state_dict())
    clip_cfg = default_cfg.clone()
    clip_cfg.defrost()
    clip_cfg.MODEL.PROMPT = False
    clip_cfg.MODEL.ADAPTER = False
    clip_cfg.INPUT.SIZE_TRAIN = list(image_size)
    clip_cfg.MODEL.STRIDE_SIZE = [patch_size, patch_size]
    clip_cfg.freeze()
    clip_model = build_clip_model(
        clip_cfg,
        state_dict,
        image_size[0] // patch_size,
        image_size[1] // patch_size,
        patch_size,
    ).float()
    return clip_model.visual

def _build_shared_semantic_residual(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    visual: nn.Module,
    num_classes: int,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    adapter_width: int,
    relay_rank: int,
    embedding_width: int,
    private_width: int,
    reliability_mode: str,
    gradient_checkpointing: bool,
    mamba_mixer_factory: Callable[[int], nn.Module],
) -> TriFusionBuildResult:
    if adapter_width <= 0:
        raise ValueError("adapter_width must be positive")
    semantic_width = int(visual.conv1.out_channels)
    expert_widths = {
        expert: semantic_width
        for expert in ("cnn", "transformer", "mamba")
    }
    shared_blocks = [
        _DeMoCLIPBlockAdapter(block)
        for block in visual.transformer.resblocks
    ]
    tokenizer = SharedCLIPSemanticTokenizer(
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
    relay = HeterogeneousRelay(
        expert_widths=expert_widths,
        relay_rank=relay_rank,
        token_grid=grid_size,
        gamma_init=relay_gamma_init,
    )
    encoder = TriBranchEncoder(
        experts,
        tokenizer=tokenizer,
        reliability_gate=posterior,
        collaborator=relay,
        refresh_final_reliability=True,
    )
    fusion = CollaborativeFusion(
        expert_widths=expert_widths,
        embedding_width=embedding_width,
    )
    if visual.proj is not None and visual.proj.shape == (
        semantic_width,
        embedding_width,
    ):
        with torch.no_grad():
            pretrained_projection = visual.proj.detach().T.float()
            for expert in ("cnn", "transformer", "mamba"):
                fusion.contribution_projections[expert].weight.copy_(
                    pretrained_projection
                )

    model = TriFusionReID(
        encoder=encoder,
        fusion=fusion,
        embedding_width=embedding_width,
        num_classes=num_classes,
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    provenance = {
        "architecture": "shared_semantic_residual",
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
        "final_reliability_refresh": True,
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


def build_trifusion_from_clip(
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
    architecture: str = "legacy_parallel",
    adapter_width: int = 192,
    gradient_checkpointing: bool = False,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    """Build the full three-stage, same-posterior collaborative model."""

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
    if architecture == "shared_semantic_residual":
        return _build_shared_semantic_residual(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            visual=visual,
            num_classes=num_classes,
            image_size=image_size,
            grid_size=grid_size,
            adapter_width=adapter_width,
            relay_rank=relay_rank,
            embedding_width=embedding_width,
            private_width=private_width,
            reliability_mode=reliability_mode,
            gradient_checkpointing=gradient_checkpointing,
            mamba_mixer_factory=mamba_mixer_factory,
        )
    if architecture != "legacy_parallel":
        raise ValueError("unknown TriFusion architecture")

    transformer_width = int(visual.conv1.out_channels)
    expert_widths = {
        "cnn": cnn_width,
        "transformer": transformer_width,
        "mamba": mamba_width,
    }

    tokenizer = SharedCLIPTokenizer(
        patch_projection=visual.conv1,
        positional_embedding=visual.positional_embedding,
        expert_widths=expert_widths,
    )
    transformer = TransformerExpert(
        width=transformer_width,
        grid_size=grid_size,
        blocks=[
            _DeMoCLIPBlockAdapter(block)
            for block in visual.transformer.resblocks
        ],
        class_embedding=visual.class_embedding,
        class_position=nn.Parameter(
            visual.positional_embedding[0].detach().clone()
        ),
        pre_norm=visual.ln_pre,
        post_norm=visual.ln_post,
        output_projection=nn.Identity(),
        private_width=private_width,
    )
    experts = {
        "cnn": CNNExpert(
            width=cnn_width,
            grid_size=grid_size,
            stage_depths=(3, 3, 3),
            private_width=private_width,
        ),
        "transformer": transformer,
        "mamba": MambaExpert(
            width=mamba_width,
            grid_size=grid_size,
            stage_depths=(3, 3, 3),
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
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
    relay = HeterogeneousRelay(
        expert_widths=expert_widths,
        relay_rank=relay_rank,
        token_grid=grid_size,
        gamma_init=0.0,
    )
    encoder = TriBranchEncoder(
        experts,
        tokenizer=tokenizer,
        reliability_gate=posterior,
        collaborator=relay,
    )
    fusion = CollaborativeFusion(
        expert_widths=expert_widths,
        embedding_width=embedding_width,
    )
    if visual.proj is not None and visual.proj.shape == (
        transformer_width,
        embedding_width,
    ):
        with torch.no_grad():
            fusion.contribution_projections["transformer"].weight.copy_(
                visual.proj.detach().T.float()
            )
    model = TriFusionReID(
        encoder=encoder,
        fusion=fusion,
        embedding_width=embedding_width,
        num_classes=num_classes,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = {
        "clip_checkpoint": str(checkpoint),
        "clip_checkpoint_sha256": checkpoint_sha256,
        "clip_visual_layers": len(transformer.blocks),
        "image_size": list(image_size),
        "token_grid": list(grid_size),
        "expert_widths": expert_widths,
        "relay_rank": relay_rank,
        "reliability_mode": reliability_mode,
        "embedding_width": embedding_width,
        "cnn_blocks": sum(len(stage) for stage in experts["cnn"].stages),
        "mamba_blocks": sum(len(stage) for stage in experts["mamba"].stages),
        "shared_patch_projection_instances": 1,
        "total_parameters": total_parameters,
        "parameter_budget_pass": total_parameters <= 120_000_000,
        "mamba_mixer_factory": getattr(
            mamba_mixer_factory, "__name__", type(mamba_mixer_factory).__name__
        ),
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        provenance=provenance,
    )


def build_single_branch_from_clip(
    checkpoint: Path | str,
    *,
    expert_name: str,
    num_classes: int,
    image_size: tuple[int, int] = (256, 128),
    patch_size: int = 16,
    cnn_width: int = 256,
    mamba_width: int = 256,
    embedding_width: int = 512,
    private_width: int = 64,
    mamba_mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
) -> TriFusionBuildResult:
    """Build one complete expert without dormant peer branches or routers."""

    if expert_name not in ("cnn", "transformer", "mamba"):
        raise ValueError(f"unknown standalone expert: {expert_name}")
    variant_name = f"{expert_name}_standalone"
    variant_contract = resolve_variant(variant_name)
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
    transformer_width = int(visual.conv1.out_channels)
    expert_width = {
        "cnn": cnn_width,
        "transformer": transformer_width,
        "mamba": mamba_width,
    }[expert_name]
    if expert_name == "cnn":
        expert_module: nn.Module = CNNExpert(
            width=cnn_width,
            grid_size=grid_size,
            stage_depths=(3, 3, 3),
            private_width=private_width,
        )
    elif expert_name == "transformer":
        expert_module = TransformerExpert(
            width=transformer_width,
            grid_size=grid_size,
            blocks=[
                _DeMoCLIPBlockAdapter(block)
                for block in visual.transformer.resblocks
            ],
            class_embedding=visual.class_embedding,
            class_position=nn.Parameter(
                visual.positional_embedding[0].detach().clone()
            ),
            pre_norm=visual.ln_pre,
            post_norm=visual.ln_post,
            output_projection=nn.Identity(),
            private_width=private_width,
        )
    else:
        expert_module = MambaExpert(
            width=mamba_width,
            grid_size=grid_size,
            stage_depths=(3, 3, 3),
            private_width=private_width,
            mixer_factory=mamba_mixer_factory,
        )
    tokenizer = SingleExpertTokenizer(
        expert=expert_name,
        patch_projection=visual.conv1,
        positional_embedding=visual.positional_embedding,
        output_width=expert_width,
    )
    model = SingleBranchReID(
        expert_name=expert_name,
        tokenizer=tokenizer,
        expert=expert_module,
        expert_width=expert_width,
        embedding_width=embedding_width,
        num_classes=num_classes,
    )
    if (
        expert_name == "transformer"
        and visual.proj is not None
        and visual.proj.shape == (transformer_width, embedding_width)
    ):
        with torch.no_grad():
            model.embedding_projection.weight.copy_(visual.proj.detach().T.float())
    for name, parameter in model.named_parameters():
        if "private_projection" in name:
            parameter.requires_grad_(False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    provenance = {
        "variant": variant_name,
        "variant_contract_sha256": variant_sha256(variant_contract),
        "architecture": "single_branch",
        "active_experts": [expert_name],
        "dormant_experts": [],
        "collaborator": "none",
        "reliability": "none",
        "fusion": "single_expert_uniform_modality_mean",
        "clip_checkpoint": str(checkpoint),
        "clip_checkpoint_sha256": checkpoint_sha256,
        "image_size": list(image_size),
        "token_grid": list(grid_size),
        "expert_width": expert_width,
        "embedding_width": embedding_width,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_budget_pass": total_parameters <= 120_000_000,
        "shared_patch_projection_instances": 1,
    }
    return TriFusionBuildResult(
        model=model,
        checkpoint_sha256=checkpoint_sha256,
        provenance=provenance,
    )


__all__ = [
    "TriFusionBuildResult",
    "build_single_branch_from_clip",
    "build_trifusion_from_clip",
]
