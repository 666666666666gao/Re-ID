"""Global Transformer expert with a retained class token."""

from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F
from torch import nn

from ..state import PackedExpertOutput


class TransformerResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(sequence)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        sequence = sequence + attended
        return sequence + self.mlp(self.norm_mlp(sequence))


class TransformerExpert(nn.Module):
    """A full global-context expert; production accepts the twelve CLIP blocks."""

    def __init__(
        self,
        *,
        width: int,
        grid_size: tuple[int, int],
        blocks: list[nn.Module],
        class_embedding: nn.Parameter,
        class_position: nn.Parameter,
        pre_norm: nn.Module,
        post_norm: nn.Module,
        output_projection: nn.Module,
        private_width: int,
    ) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("Transformer expert requires at least one block")
        if len(blocks) < 3:
            raise ValueError("Transformer expert requires at least three stage blocks")
        self.width = width
        self.grid_size = grid_size
        self.blocks = nn.ModuleList(blocks)
        self.class_embedding = class_embedding
        self.class_position = class_position
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.output_projection = output_projection
        self.private_projection = nn.Linear(width, private_width)
        base_depth, remainder = divmod(len(blocks), 3)
        self.stage_depths = tuple(
            base_depth + (1 if stage < remainder else 0) for stage in range(3)
        )

    @classmethod
    def from_scratch(
        cls,
        *,
        width: int,
        grid_size: tuple[int, int],
        layers: int,
        heads: int,
        private_width: int,
    ) -> "TransformerExpert":
        if layers <= 0:
            raise ValueError("layers must be positive")
        scale = width**-0.5
        return cls(
            width=width,
            grid_size=grid_size,
            blocks=[TransformerResidualBlock(width, heads) for _ in range(layers)],
            class_embedding=nn.Parameter(scale * torch.randn(width)),
            class_position=nn.Parameter(scale * torch.randn(width)),
            pre_norm=nn.LayerNorm(width),
            post_norm=nn.LayerNorm(width),
            output_projection=nn.Linear(width, width, bias=False),
            private_width=private_width,
        )

    def _validate_patch_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            prod(self.grid_size),
            self.width,
        ):
            raise ValueError("Transformer tokens must match the configured N,D")

    def initialize(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_patch_tokens(tokens)
        batch_size = tokens.shape[0]
        class_token = (
            self.class_embedding + self.class_position
        ).to(device=tokens.device, dtype=tokens.dtype)
        class_tokens = class_token.view(1, 1, -1).expand(batch_size, 1, -1)
        return self.pre_norm(torch.cat((class_tokens, tokens), dim=1))

    def run_stage(self, runtime: torch.Tensor, stage: int) -> torch.Tensor:
        if stage not in (1, 2, 3):
            raise ValueError("Transformer stage must be 1, 2, or 3")
        if runtime.ndim != 3 or runtime.shape[1:] != (
            prod(self.grid_size) + 1,
            self.width,
        ):
            raise ValueError("Transformer runtime must contain class plus patches")
        start = sum(self.stage_depths[: stage - 1])
        stop = start + self.stage_depths[stage - 1]
        for block in self.blocks[start:stop]:
            runtime = block(runtime)
        return runtime

    def summarize(self, runtime: torch.Tensor, stage: int) -> PackedExpertOutput:
        sequence = self.post_norm(runtime) if stage == 3 else runtime
        output_tokens = sequence[:, 1:]
        global_embedding = self.output_projection(sequence[:, 0])
        private_embedding = self.private_projection(global_embedding)
        global_to_patch = F.cosine_similarity(
            global_embedding.unsqueeze(1), output_tokens, dim=-1
        ).unsqueeze(-1)
        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=private_embedding,
            role_payload={"global_to_patch": global_to_patch},
            stage=stage,
        )

    def inject(self, runtime: torch.Tensor, relayed_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_patch_tokens(relayed_tokens)
        return torch.cat((runtime[:, :1], relayed_tokens), dim=1)

    def forward(self, tokens: torch.Tensor) -> PackedExpertOutput:
        runtime = self.initialize(tokens)
        for stage in (1, 2, 3):
            runtime = self.run_stage(runtime, stage)
        return self.summarize(runtime, 3)


__all__ = ["TransformerExpert", "TransformerResidualBlock"]
