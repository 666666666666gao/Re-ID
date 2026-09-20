"""Four-direction 2D Mamba expert with a shared core per block."""

from __future__ import annotations

from collections.abc import Callable
from math import prod

import torch
from torch import nn

from ..state import PackedExpertOutput


class TinySequenceMixer(nn.Module):
    """Order-sensitive CPU substitute for public algebra tests."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.pointwise = nn.Linear(width, width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mixed = self.depthwise(sequence.transpose(1, 2)).transpose(1, 2)
        return self.pointwise(torch.nn.functional.silu(mixed))


def production_mamba_factory(width: int) -> nn.Module:
    from mamba_ssm import Mamba

    return Mamba(d_model=width, d_state=16, d_conv=4, expand=2)


class FourDirectionMambaBlock(nn.Module):
    def __init__(
        self,
        width: int,
        grid_size: tuple[int, int],
        mixer_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.width = width
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(width)
        self.core = mixer_factory(width)
        self.direction_gate = nn.Linear(width, 4)
        self.output_projection = nn.Linear(width, width, bias=False)

    def _to_column_order(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size = sequence.shape[0]
        height, width = self.grid_size
        return (
            sequence.reshape(batch_size, height, width, self.width)
            .transpose(1, 2)
            .reshape(batch_size, height * width, self.width)
        )

    def _from_column_order(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size = sequence.shape[0]
        height, width = self.grid_size
        return (
            sequence.reshape(batch_size, width, height, self.width)
            .transpose(1, 2)
            .reshape(batch_size, height * width, self.width)
        )

    def forward_with_context(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(tokens)
        column_order = self._to_column_order(normalized)
        row_forward = self.core(normalized)
        row_reverse = self.core(normalized.flip(dims=(1,))).flip(dims=(1,))
        column_forward = self._from_column_order(self.core(column_order))
        column_reverse = self._from_column_order(
            self.core(column_order.flip(dims=(1,))).flip(dims=(1,))
        )
        directional = torch.stack(
            (row_forward, row_reverse, column_forward, column_reverse), dim=2
        )
        gates = torch.softmax(self.direction_gate(normalized), dim=-1).unsqueeze(-1)
        mixed = (directional * gates).sum(dim=2)
        output = tokens + self.output_projection(mixed)
        context = torch.stack(
            (
                (row_forward - row_reverse).square().mean(dim=-1),
                (column_forward - column_reverse).square().mean(dim=-1),
                0.5 * (row_forward + row_reverse).mean(dim=-1),
                0.5 * (column_forward + column_reverse).mean(dim=-1),
            ),
            dim=-1,
        )
        return output, context

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_context(tokens)[0]


class MambaExpert(nn.Module):
    """Nine residual four-direction blocks grouped into three stages."""

    def __init__(
        self,
        *,
        width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int] = (3, 3, 3),
        private_width: int = 64,
        mixer_factory: Callable[[int], nn.Module] = production_mamba_factory,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 3 or min(stage_depths) <= 0:
            raise ValueError("stage_depths must contain three positive depths")
        self.width = width
        self.grid_size = grid_size
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FourDirectionMambaBlock(width, grid_size, mixer_factory)
                        for _ in range(depth)
                    ]
                )
                for depth in stage_depths
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.private_projection = nn.Linear(width, private_width)

    @classmethod
    def with_tiny_mixer(
        cls,
        *,
        width: int,
        grid_size: tuple[int, int],
        stage_depths: tuple[int, int, int],
        private_width: int,
    ) -> "MambaExpert":
        return cls(
            width=width,
            grid_size=grid_size,
            stage_depths=stage_depths,
            private_width=private_width,
            mixer_factory=TinySequenceMixer,
        )

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            prod(self.grid_size),
            self.width,
        ):
            raise ValueError("Mamba tokens must match the configured N,D")

    def initialize(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_tokens(tokens)
        return tokens, None

    def run_stage(
        self,
        runtime: tuple[torch.Tensor, torch.Tensor | None],
        stage: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage not in (1, 2, 3):
            raise ValueError("Mamba stage must be 1, 2, or 3")
        tokens, direction_context = runtime
        for block in self.stages[stage - 1]:
            tokens, direction_context = block.forward_with_context(tokens)
        if direction_context is None:
            raise RuntimeError("Mamba stage executed no blocks")
        return tokens, direction_context

    def summarize(
        self,
        runtime: tuple[torch.Tensor, torch.Tensor | None],
        stage: int,
    ) -> PackedExpertOutput:
        tokens, direction_context = runtime
        output_tokens = self.output_norm(tokens) if stage == 3 else tokens
        global_embedding = output_tokens.mean(dim=1)
        private_embedding = self.private_projection(global_embedding)
        if direction_context is None:
            raise RuntimeError("Mamba expert executed no blocks")
        return PackedExpertOutput(
            tokens=output_tokens,
            global_embedding=global_embedding,
            private_embedding=private_embedding,
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


__all__ = ["MambaExpert", "FourDirectionMambaBlock", "TinySequenceMixer"]
