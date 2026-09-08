"""Pretrained-semantic residual CNN, Transformer, and Mamba experts."""

from __future__ import annotations

from collections.abc import Callable
from math import prod

import torch
import torch.nn.functional as F
from torch import nn

from ..state import PackedExpertOutput
from .mamba import FourDirectionMambaBlock, production_mamba_factory


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


def _validate_stage_depths(stage_depths: tuple[int, int, int]) -> None:
    if len(stage_depths) != 3 or min(stage_depths) <= 0:
        raise ValueError("stage_depths must contain three positive depths")


class HighFrequencyResidualBlock(nn.Module):
    """Add a small local high-frequency delta to a strong semantic token field."""

    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        dilation: int,
        scale_init: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.adapter_width = adapter_width
        self.grid_size = grid_size
        self.input_norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, adapter_width, bias=False)
        self.local_norm = nn.GroupNorm(
            _group_count(adapter_width),
            adapter_width,
        )
        self.depthwise = nn.Conv2d(
            adapter_width,
            adapter_width,
            kernel_size=3,
            padding=1,
            groups=adapter_width,
        )
        self.dilated = nn.Conv2d(
            adapter_width,
            adapter_width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=adapter_width,
        )
        self.gate = nn.Conv2d(adapter_width, adapter_width, kernel_size=1)
        self.up = nn.Linear(adapter_width, width, bias=False)
        self.layer_scale = nn.Parameter(torch.full((width,), scale_init))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        height, width = self.grid_size
        reduced = self.down(self.input_norm(tokens))
        grid = reduced.transpose(1, 2).reshape(
            batch_size,
            self.adapter_width,
            height,
            width,
        )
        low_frequency = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
        high_frequency = grid - low_frequency
        normalized = self.local_norm(high_frequency)
        mixed = self.depthwise(normalized) + self.dilated(normalized)
        mixed = F.silu(mixed) * torch.sigmoid(self.gate(grid))
        residual = self.up(mixed.flatten(2).transpose(1, 2))
        return tokens + residual * self.layer_scale.to(dtype=residual.dtype)


class GlobalAttentionResidualBlock(nn.Module):
    """Add a bottleneck global-attention delta without replacing CLIP semantics."""

    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        scale_init: float,
    ) -> None:
        super().__init__()
        heads = 4 if adapter_width % 4 == 0 else 1
        self.input_norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, adapter_width, bias=False)
        self.attention_norm = nn.LayerNorm(adapter_width)
        self.attention = nn.MultiheadAttention(
            adapter_width,
            heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(adapter_width)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_width, adapter_width * 2),
            nn.GELU(),
            nn.Linear(adapter_width * 2, adapter_width),
        )
        self.up = nn.Linear(adapter_width, width, bias=False)
        self.layer_scale = nn.Parameter(torch.full((width,), scale_init))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        reduced = self.down(self.input_norm(tokens))
        normalized = self.attention_norm(reduced)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        reduced = reduced + attended
        reduced = reduced + self.mlp(self.mlp_norm(reduced))
        residual = self.up(reduced)
        return tokens + residual * self.layer_scale.to(dtype=residual.dtype)


class LongRangeMambaResidualBlock(nn.Module):
    """Add a four-direction state-space delta in a compact latent width."""

    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        mixer_factory: Callable[[int], nn.Module],
        scale_init: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, adapter_width, bias=False)
        self.mixer = FourDirectionMambaBlock(
            adapter_width,
            grid_size,
            mixer_factory,
        )
        self.up = nn.Linear(adapter_width, width, bias=False)
        self.layer_scale = nn.Parameter(torch.full((width,), scale_init))

    def forward_with_context(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.down(self.input_norm(tokens))
        mixed, direction_context = self.mixer.forward_with_context(reduced)
        residual = self.up(mixed)
        output = tokens + residual * self.layer_scale.to(dtype=residual.dtype)
        return output, direction_context


class _ResidualTokenExpert(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        grid_size: tuple[int, int],
        stages: list[nn.Sequential],
        private_width: int,
    ) -> None:
        super().__init__()
        if prod(grid_size) <= 0:
            raise ValueError("grid_size must be positive")
        self.width = width
        self.grid_size = grid_size
        self.stages = nn.ModuleList(stages)
        self.output_norm = nn.LayerNorm(width)
        self.private_projection = nn.Linear(width, private_width)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            prod(self.grid_size),
            self.width,
        ):
            raise ValueError("semantic tokens must match the configured N,D")

    def initialize(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(tokens)
        return tokens

    def run_stage(self, runtime: torch.Tensor, stage: int) -> torch.Tensor:
        if stage not in (1, 2, 3):
            raise ValueError("semantic expert stage must be 1, 2, or 3")
        self._validate_tokens(runtime)
        return self.stages[stage - 1](runtime)

    def inject(
        self,
        runtime: torch.Tensor,
        relayed_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_tokens(relayed_tokens)
        return relayed_tokens

    def _output_tokens(self, runtime: torch.Tensor, stage: int) -> torch.Tensor:
        return self.output_norm(runtime) if stage == 3 else runtime


class SemanticCNNExpert(_ResidualTokenExpert):
    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int] = (1, 1, 1),
        private_width: int = 64,
        scale_init: float = 1e-3,
    ) -> None:
        _validate_stage_depths(stage_depths)
        block_index = 0
        stages = []
        for depth in stage_depths:
            blocks = []
            for _ in range(depth):
                blocks.append(
                    HighFrequencyResidualBlock(
                        width=width,
                        adapter_width=adapter_width,
                        grid_size=grid_size,
                        dilation=1 + block_index % 2,
                        scale_init=scale_init,
                    )
                )
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        super().__init__(
            width=width,
            grid_size=grid_size,
            stages=stages,
            private_width=private_width,
        )

    def summarize(self, runtime: torch.Tensor, stage: int) -> PackedExpertOutput:
        self._validate_tokens(runtime)
        output_tokens = self._output_tokens(runtime, stage)
        batch_size = output_tokens.shape[0]
        height, width = self.grid_size
        global_embedding = output_tokens.mean(dim=1)
        normalized_grid = F.normalize(
            output_tokens.reshape(batch_size, height, width, self.width),
            dim=-1,
        )
        right = (normalized_grid[:, :, :-1] * normalized_grid[:, :, 1:]).sum(-1)
        down = (normalized_grid[:, :-1] * normalized_grid[:, 1:]).sum(-1)
        right = F.pad(right, (0, 1)).reshape(batch_size, -1, 1)
        down = F.pad(down, (0, 0, 0, 1)).reshape(batch_size, -1, 1)
        role_grid = output_tokens.transpose(1, 2).reshape(
            batch_size,
            self.width,
            height,
            width,
        )
        horizontal_parts = (
            F.adaptive_avg_pool2d(role_grid, (4, 1))
            .squeeze(-1)
            .transpose(1, 2)
        )
        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=self.private_projection(global_embedding),
            role_payload={
                "local_neighbors": torch.cat((right, down), dim=-1),
                "horizontal_parts": horizontal_parts,
            },
            stage=stage,
        )

    def forward(self, tokens: torch.Tensor) -> PackedExpertOutput:
        runtime = self.initialize(tokens)
        for stage in (1, 2, 3):
            runtime = self.run_stage(runtime, stage)
        return self.summarize(runtime, 3)


class SemanticTransformerExpert(_ResidualTokenExpert):
    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int] = (1, 1, 1),
        private_width: int = 64,
        scale_init: float = 1e-3,
    ) -> None:
        _validate_stage_depths(stage_depths)
        stages = [
            nn.Sequential(
                *[
                    GlobalAttentionResidualBlock(
                        width=width,
                        adapter_width=adapter_width,
                        scale_init=scale_init,
                    )
                    for _ in range(depth)
                ]
            )
            for depth in stage_depths
        ]
        super().__init__(
            width=width,
            grid_size=grid_size,
            stages=stages,
            private_width=private_width,
        )

    def summarize(self, runtime: torch.Tensor, stage: int) -> PackedExpertOutput:
        self._validate_tokens(runtime)
        output_tokens = self._output_tokens(runtime, stage)
        global_embedding = output_tokens.mean(dim=1)
        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=self.private_projection(global_embedding),
            role_payload={
                "global_to_patch": F.cosine_similarity(
                    global_embedding.unsqueeze(1),
                    output_tokens,
                    dim=-1,
                ).unsqueeze(-1)
            },
            stage=stage,
        )

    def forward(self, tokens: torch.Tensor) -> PackedExpertOutput:
        runtime = self.initialize(tokens)
        for stage in (1, 2, 3):
            runtime = self.run_stage(runtime, stage)
        return self.summarize(runtime, 3)


class SemanticMambaExpert(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        adapter_width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int] = (1, 1, 1),
        private_width: int = 64,
        mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
        scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        _validate_stage_depths(stage_depths)
        self.width = width
        self.grid_size = grid_size
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LongRangeMambaResidualBlock(
                            width=width,
                            adapter_width=adapter_width,
                            grid_size=grid_size,
                            mixer_factory=mixer_factory,
                            scale_init=scale_init,
                        )
                        for _ in range(depth)
                    ]
                )
                for depth in stage_depths
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.private_projection = nn.Linear(width, private_width)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            prod(self.grid_size),
            self.width,
        ):
            raise ValueError("semantic Mamba tokens must match N,D")

    def initialize(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(tokens)
        return tokens, None

    def run_stage(
        self,
        runtime: tuple[torch.Tensor, torch.Tensor | None],
        stage: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage not in (1, 2, 3):
            raise ValueError("semantic Mamba stage must be 1, 2, or 3")
        tokens, direction_context = runtime
        self._validate_tokens(tokens)
        for block in self.stages[stage - 1]:
            tokens, direction_context = block.forward_with_context(tokens)
        if direction_context is None:
            raise RuntimeError("semantic Mamba stage executed no blocks")
        return tokens, direction_context

    def summarize(
        self,
        runtime: tuple[torch.Tensor, torch.Tensor | None],
        stage: int,
    ) -> PackedExpertOutput:
        tokens, direction_context = runtime
        self._validate_tokens(tokens)
        if direction_context is None:
            raise RuntimeError("semantic Mamba context is unavailable")
        output_tokens = self.output_norm(tokens) if stage == 3 else tokens
        global_embedding = output_tokens.mean(dim=1)
        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=self.private_projection(global_embedding),
            role_payload={"direction_context": direction_context},
            stage=stage,
        )

    def inject(
        self,
        runtime: tuple[torch.Tensor, torch.Tensor | None],
        relayed_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(relayed_tokens)
        return relayed_tokens, runtime[1]

    def forward(self, tokens: torch.Tensor) -> PackedExpertOutput:
        runtime = self.initialize(tokens)
        for stage in (1, 2, 3):
            runtime = self.run_stage(runtime, stage)
        return self.summarize(runtime, 3)


__all__ = [
    "GlobalAttentionResidualBlock",
    "HighFrequencyResidualBlock",
    "LongRangeMambaResidualBlock",
    "SemanticCNNExpert",
    "SemanticMambaExpert",
    "SemanticTransformerExpert",
]
