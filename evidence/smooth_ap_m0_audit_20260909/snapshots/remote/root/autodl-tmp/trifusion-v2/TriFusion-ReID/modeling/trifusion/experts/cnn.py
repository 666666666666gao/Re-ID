"""Local multi-scale CNN expert on the shared patch grid."""

from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F
from torch import nn

from ..state import PackedExpertOutput


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


class LocalMixBlock(nn.Module):
    """Gated pointwise expansion plus one depthwise local receptive field."""

    def __init__(
        self,
        width: int,
        expansion: int,
        *,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        hidden_width = width * expansion
        padding = dilation * (kernel_size // 2)
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.expand = nn.Conv2d(width, hidden_width * 2, kernel_size=1)
        self.local_mix = nn.Conv2d(
            hidden_width,
            hidden_width,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_width,
        )
        self.project = nn.Conv2d(hidden_width, width, kernel_size=1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(self.norm(grid))
        values, gates = expanded.chunk(2, dim=1)
        mixed = self.local_mix(values) * torch.sigmoid(gates)
        return grid + self.project(F.silu(mixed))


class CNNExpert(nn.Module):
    """Nine-block production CNN, parameterized down for CPU seam tests."""

    def __init__(
        self,
        *,
        width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int] = (3, 3, 3),
        private_width: int = 64,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 3 or min(stage_depths) <= 0:
            raise ValueError("stage_depths must contain three positive depths")
        if prod(grid_size) <= 0:
            raise ValueError("grid_size must be positive")
        self.width = width
        self.grid_size = grid_size
        receptive_fields = ((3, 1), (5, 1), (3, 2))
        self.stages = nn.ModuleList()
        block_index = 0
        for depth in stage_depths:
            blocks = []
            for _ in range(depth):
                kernel_size, dilation = receptive_fields[block_index % 3]
                blocks.append(
                    LocalMixBlock(
                        width,
                        expansion,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    )
                )
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
        self.output_norm = nn.LayerNorm(width)
        self.private_projection = nn.Linear(width, private_width)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            prod(self.grid_size),
            self.width,
        ):
            raise ValueError("CNN tokens must match the configured N,D")

    def initialize(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(tokens)
        return tokens

    def run_stage(self, runtime: torch.Tensor, stage: int) -> torch.Tensor:
        if stage not in (1, 2, 3):
            raise ValueError("CNN stage must be 1, 2, or 3")
        self._validate_tokens(runtime)
        tokens = runtime
        batch_size = tokens.shape[0]
        height, width = self.grid_size
        grid = tokens.transpose(1, 2).reshape(batch_size, self.width, height, width)
        grid = self.stages[stage - 1](grid)
        return grid.flatten(2).transpose(1, 2)

    def summarize(self, runtime: torch.Tensor, stage: int) -> PackedExpertOutput:
        self._validate_tokens(runtime)
        batch_size = runtime.shape[0]
        height, width = self.grid_size
        output_tokens = self.output_norm(runtime) if stage == 3 else runtime
        global_embedding = output_tokens.mean(dim=1)
        private_embedding = self.private_projection(global_embedding)

        normalized_grid = F.normalize(
            output_tokens.reshape(batch_size, height, width, self.width), dim=-1
        )
        right = (normalized_grid[:, :, :-1] * normalized_grid[:, :, 1:]).sum(-1)
        down = (normalized_grid[:, :-1] * normalized_grid[:, 1:]).sum(-1)
        right = F.pad(right, (0, 1)).reshape(batch_size, -1, 1)
        down = F.pad(down, (0, 0, 0, 1)).reshape(batch_size, -1, 1)
        local_neighbors = torch.cat((right, down), dim=-1)
        role_grid = output_tokens.transpose(1, 2).reshape(
            batch_size, self.width, height, width
        )
        horizontal_parts = (
            F.adaptive_avg_pool2d(role_grid, (4, 1)).squeeze(-1).transpose(1, 2)
        )

        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=private_embedding,
            role_payload={
                "local_neighbors": local_neighbors,
                "horizontal_parts": horizontal_parts,
            },
            stage=stage,
        )

    def inject(self, runtime: torch.Tensor, relayed_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(relayed_tokens)
        return relayed_tokens

    def forward(self, tokens: torch.Tensor) -> PackedExpertOutput:
        runtime = self.initialize(tokens)
        for stage in (1, 2, 3):
            runtime = self.run_stage(runtime, stage)
        return self.summarize(runtime, 3)


__all__ = ["CNNExpert"]
