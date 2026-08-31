"""Keep probability-form BCE numerically safe while the frozen runner uses AMP."""

from __future__ import annotations

from functools import wraps

import torch
import torch.nn.functional as F


if not getattr(F.binary_cross_entropy, "_trifusion_amp_safe", False):
    _ORIGINAL_BINARY_CROSS_ENTROPY = F.binary_cross_entropy

    @wraps(_ORIGINAL_BINARY_CROSS_ENTROPY)
    def _amp_safe_binary_cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if input.is_cuda and torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return _ORIGINAL_BINARY_CROSS_ENTROPY(
                    input.float(),
                    target.float(),
                    weight=None if weight is None else weight.float(),
                    size_average=size_average,
                    reduce=reduce,
                    reduction=reduction,
                )
        return _ORIGINAL_BINARY_CROSS_ENTROPY(
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    _amp_safe_binary_cross_entropy._trifusion_amp_safe = True
    F.binary_cross_entropy = _amp_safe_binary_cross_entropy
