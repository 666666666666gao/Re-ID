"""Small differentiable experts for public-seam CPU tests."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ..state import EXPERT_ORDER, PackedExpertOutput


class TinyExpert(nn.Module):
    """A complete, inexpensive image-to-token expert used only in tiny configs."""

    def __init__(self, width: int, token_count: int, private_width: int) -> None:
        super().__init__()
        if min(width, token_count, private_width) <= 0:
            raise ValueError("tiny expert dimensions must be positive")
        self.token_count = token_count
        self.stem = nn.Conv2d(3, width, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((token_count, 1))
        self.activation = nn.SiLU()
        self.private_projection = nn.Linear(width, private_width)

    def forward(self, packed_images: torch.Tensor) -> PackedExpertOutput:
        features = self.activation(self.stem(packed_images))
        tokens = self.pool(features).squeeze(-1).transpose(1, 2)
        global_embedding = tokens.mean(dim=1)
        private_embedding = self.private_projection(global_embedding)
        return PackedExpertOutput(
            tokens=tokens,
            global_embedding=global_embedding,
            private_embedding=private_embedding,
            role_payload={"summary": global_embedding.mean(dim=-1, keepdim=True)},
            stage=3,
        )


def make_tiny_experts(
    *, widths: Mapping[str, int], token_count: int, private_width: int
) -> dict[str, TinyExpert]:
    """Build the three ordered injected experts for a deterministic tiny config."""

    if set(widths) != set(EXPERT_ORDER):
        raise ValueError(f"widths must contain exactly {EXPERT_ORDER}")
    return {
        expert: TinyExpert(widths[expert], token_count, private_width)
        for expert in EXPERT_ORDER
    }


__all__ = ["TinyExpert", "make_tiny_experts"]
