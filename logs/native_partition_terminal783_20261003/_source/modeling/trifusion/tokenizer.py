"""Shared CLIP-compatible patch tokenization for all three experts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import nn

from .state import EXPERT_ORDER, MODALITY_ORDER


class SharedCLIPTokenizer(nn.Module):
    """Apply one patch projection before expert-specific token projections."""

    def __init__(
        self,
        *,
        patch_projection: nn.Conv2d,
        positional_embedding: nn.Parameter,
        expert_widths: Mapping[str, int],
    ) -> None:
        super().__init__()
        if set(expert_widths) != set(EXPERT_ORDER):
            raise ValueError(f"expert_widths must contain exactly {EXPERT_ORDER}")
        native_width = patch_projection.out_channels
        if positional_embedding.ndim != 2 or positional_embedding.shape[1] != native_width:
            raise ValueError("positional_embedding width must match patch projection")
        if expert_widths["transformer"] != native_width:
            raise ValueError("Transformer token width must equal the CLIP native width")

        self.patch_projection = patch_projection
        self.positional_embedding = positional_embedding
        self.modality_embedding = nn.Embedding(len(MODALITY_ORDER), native_width)
        self.expert_embedding = nn.Parameter(
            torch.empty(len(EXPERT_ORDER), native_width)
        )
        self.projections = nn.ModuleDict(
            {
                expert: (
                    nn.Identity()
                    if expert_widths[expert] == native_width
                    else nn.Linear(native_width, expert_widths[expert], bias=False)
                )
                for expert in EXPERT_ORDER
            }
        )
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.expert_embedding, std=0.02)

    def forward(
        self, packed_images: torch.Tensor, packed_modalities: torch.Tensor
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
            packed_modalities.min() < 0
            or packed_modalities.max() >= len(MODALITY_ORDER)
        ):
            raise ValueError("packed modality indices must be in RGB, NI, TI range")

        projection_dtype = self.patch_projection.weight.dtype
        patches = self.patch_projection(packed_images.to(dtype=projection_dtype))
        patches = patches.flatten(2).transpose(1, 2)
        token_count = patches.shape[1]
        if self.positional_embedding.shape[0] != token_count + 1:
            raise ValueError(
                "positional_embedding must contain one class plus every patch position"
            )
        patch_positions = self.positional_embedding[1:].to(
            device=patches.device, dtype=patches.dtype
        )
        modality_offsets = self.modality_embedding(packed_modalities).to(
            dtype=patches.dtype
        )
        common_tokens = patches + patch_positions.unsqueeze(0)

        outputs = {}
        for expert_index, expert in enumerate(EXPERT_ORDER):
            expert_offset = self.expert_embedding[expert_index].to(
                device=patches.device, dtype=patches.dtype
            )
            enriched = (
                common_tokens
                + modality_offsets.unsqueeze(1)
                + expert_offset.view(1, 1, -1)
            )
            outputs[expert] = self.projections[expert](enriched)
        return MappingProxyType(outputs)


__all__ = ["SharedCLIPTokenizer"]
