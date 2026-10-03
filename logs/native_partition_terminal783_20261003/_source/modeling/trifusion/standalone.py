"""True one-expert baselines with no dormant collaborative branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .state import EXPERT_ORDER, MODALITY_ORDER


class SingleExpertTokenizer(nn.Module):
    """CLIP patch tokenizer containing only the selected expert projection."""

    def __init__(
        self,
        *,
        expert: str,
        patch_projection: nn.Conv2d,
        positional_embedding: nn.Parameter,
        output_width: int,
    ) -> None:
        super().__init__()
        if expert not in EXPERT_ORDER:
            raise ValueError(f"unknown expert: {expert}")
        native_width = int(patch_projection.out_channels)
        if positional_embedding.ndim != 2 or positional_embedding.shape[1] != native_width:
            raise ValueError("positional embedding must match CLIP patch width")
        self.expert = expert
        self.patch_projection = patch_projection
        self.positional_embedding = positional_embedding
        self.modality_embedding = nn.Embedding(len(MODALITY_ORDER), native_width)
        self.expert_embedding = nn.Parameter(torch.empty(native_width))
        self.projection = (
            nn.Identity()
            if output_width == native_width
            else nn.Linear(native_width, output_width, bias=False)
        )
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.expert_embedding, std=0.02)

    def forward(
        self, packed_images: torch.Tensor, packed_modalities: torch.Tensor
    ) -> torch.Tensor:
        if packed_images.ndim != 4 or packed_images.shape[1] != 3:
            raise ValueError("packed_images must have shape Nv,3,H,W")
        if (
            packed_modalities.dtype != torch.long
            or packed_modalities.ndim != 1
            or packed_modalities.shape[0] != packed_images.shape[0]
        ):
            raise ValueError("packed_modalities must be a length-Nv long tensor")
        patches = self.patch_projection(
            packed_images.to(dtype=self.patch_projection.weight.dtype)
        ).flatten(2).transpose(1, 2)
        if self.positional_embedding.shape[0] != patches.shape[1] + 1:
            raise ValueError("CLIP position table does not match the patch grid")
        native = (
            patches
            + self.positional_embedding[1:].to(patches).unsqueeze(0)
            + self.modality_embedding(packed_modalities).to(patches).unsqueeze(1)
            + self.expert_embedding.to(patches).view(1, 1, -1)
        )
        return self.projection(native)


@dataclass(frozen=True, eq=False)
class SingleBranchOutput:
    embedding: torch.Tensor
    logits: torch.Tensor | None
    expert: str
    modality_mask: torch.Tensor
    all_finite: bool


class SingleBranchReID(nn.Module):
    """One complete CNN, Transformer, or Mamba branch over all three modalities."""

    def __init__(
        self,
        *,
        expert_name: str,
        tokenizer: SingleExpertTokenizer,
        expert: nn.Module,
        expert_width: int,
        embedding_width: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if expert_name not in EXPERT_ORDER:
            raise ValueError(f"unknown expert: {expert_name}")
        self.expert_name = expert_name
        self.tokenizer = tokenizer
        self.expert = expert
        self.embedding_projection = nn.Linear(
            expert_width, embedding_width, bias=False
        )
        self.embedding_norm = nn.LayerNorm(embedding_width)
        self.neck = nn.BatchNorm1d(embedding_width)
        nn.init.ones_(self.neck.weight)
        nn.init.zeros_(self.neck.bias)
        self.neck.bias.requires_grad_(False)
        self.classifier = (
            nn.Linear(embedding_width, num_classes, bias=False)
            if num_classes
            else None
        )
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(
        self,
        batch: Mapping[str, Any],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | SingleBranchOutput:
        del targets
        if "images" not in batch or "modality_mask" not in batch:
            raise ValueError("batch must contain images and modality_mask")
        images = batch["images"]
        modality_mask = batch["modality_mask"]
        if tuple(images) != MODALITY_ORDER:
            raise ValueError(f"images must follow modality order {MODALITY_ORDER}")
        if modality_mask.dtype != torch.bool or modality_mask.ndim != 2:
            raise ValueError("modality_mask must be a rank-2 bool tensor")
        if modality_mask.shape[1] != len(MODALITY_ORDER):
            raise ValueError("modality_mask columns must be RGB, NI, TI")
        if bool((~modality_mask).all(dim=1).any()):
            raise ValueError("single-branch input contains an all-missing row")
        stacked = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
        packed_images = stacked[modality_mask]
        modality_indices = torch.arange(
            len(MODALITY_ORDER), device=modality_mask.device
        ).view(1, -1).expand_as(modality_mask)
        tokens = self.tokenizer(packed_images, modality_indices[modality_mask])
        packed = self.expert(tokens)
        batch_size, modality_count = modality_mask.shape
        modal = packed.global_embedding.new_zeros(
            batch_size, modality_count, packed.global_embedding.shape[-1]
        )
        modal[modality_mask] = packed.global_embedding
        weights = modality_mask.to(modal).unsqueeze(-1)
        pooled = (modal * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        embedding = self.neck(
            self.embedding_norm(self.embedding_projection(pooled))
        )
        logits = self.classifier(embedding) if self.classifier is not None else None
        if not return_aux:
            return embedding
        finite = bool(torch.isfinite(embedding).all().item())
        if logits is not None:
            finite = finite and bool(torch.isfinite(logits).all().item())
        return SingleBranchOutput(
            embedding=embedding,
            logits=logits,
            expert=self.expert_name,
            modality_mask=modality_mask,
            all_finite=finite,
        )


__all__ = ["SingleBranchOutput", "SingleBranchReID", "SingleExpertTokenizer"]
