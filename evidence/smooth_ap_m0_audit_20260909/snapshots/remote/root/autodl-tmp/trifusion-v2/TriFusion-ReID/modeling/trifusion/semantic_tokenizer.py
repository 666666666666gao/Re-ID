"""One full pretrained CLIP visual trunk shared by every residual expert."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .state import EXPERT_ORDER, MODALITY_ORDER


class SharedCLIPSemanticTokenizer(nn.Module):
    """Emit one strong CLIP semantic token field to all heterogeneous experts."""

    def __init__(
        self,
        *,
        patch_projection: nn.Conv2d,
        positional_embedding: nn.Parameter,
        class_embedding: nn.Parameter,
        pre_norm: nn.Module,
        post_norm: nn.Module,
        shared_blocks: Sequence[nn.Module],
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        native_width = int(patch_projection.out_channels)
        if positional_embedding.ndim != 2:
            raise ValueError("positional_embedding must be rank two")
        if positional_embedding.shape[1] != native_width:
            raise ValueError("positional embedding width must match CLIP")
        if class_embedding.shape != (native_width,):
            raise ValueError("class embedding width must match CLIP")
        if not shared_blocks:
            raise ValueError("the shared CLIP trunk requires at least one block")

        self.width = native_width
        self.patch_projection = patch_projection
        self.positional_embedding = positional_embedding
        self.class_embedding = class_embedding
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.shared_blocks = nn.ModuleList(list(shared_blocks))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.modality_embedding = nn.Embedding(len(MODALITY_ORDER), native_width)
        nn.init.zeros_(self.modality_embedding.weight)

    def _run_block(
        self,
        block: nn.Module,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return activation_checkpoint(block, sequence, use_reentrant=False)
        return block(sequence)

    def forward(
        self,
        packed_images: torch.Tensor,
        packed_modalities: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        if packed_images.ndim != 4 or packed_images.shape[1] != 3:
            raise ValueError("packed_images must have shape Nv,3,H,W")
        if (
            packed_modalities.dtype != torch.long
            or packed_modalities.ndim != 1
            or packed_modalities.shape[0] != packed_images.shape[0]
        ):
            raise ValueError("packed_modalities must be a length-Nv long tensor")
        if packed_modalities.numel() and (
            int(packed_modalities.min().item()) < 0
            or int(packed_modalities.max().item()) >= len(MODALITY_ORDER)
        ):
            raise ValueError("packed modality indices must be in RGB, NI, TI range")

        projection_dtype = self.patch_projection.weight.dtype
        patches = self.patch_projection(
            packed_images.to(dtype=projection_dtype)
        )
        patches = patches.flatten(2).transpose(1, 2)
        token_count = int(patches.shape[1])
        if self.positional_embedding.shape != (token_count + 1, self.width):
            raise ValueError("CLIP position table does not match the patch grid")

        positions = self.positional_embedding.to(
            device=patches.device,
            dtype=patches.dtype,
        )
        class_tokens = self.class_embedding.to(
            device=patches.device,
            dtype=patches.dtype,
        ).view(1, 1, -1).expand(patches.shape[0], 1, -1)
        sequence = torch.cat((class_tokens, patches), dim=1)
        sequence = sequence + positions.unsqueeze(0)
        modality_offsets = self.modality_embedding(packed_modalities).to(
            dtype=sequence.dtype
        )
        sequence = self.pre_norm(sequence + modality_offsets.unsqueeze(1))
        for block in self.shared_blocks:
            sequence = self._run_block(block, sequence)
        sequence = self.post_norm(sequence)

        # Broadcasting the pretrained CLS state into every patch retains the
        # CLIP global identity prior while leaving local differences available
        # to the CNN high-pass path.
        semantic_tokens = sequence[:, 1:] + sequence[:, :1]
        return MappingProxyType(
            {expert: semantic_tokens for expert in EXPERT_ORDER}
        )


__all__ = ["SharedCLIPSemanticTokenizer"]
