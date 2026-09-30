#!/usr/bin/env python3
"""Check matched state and allocation algebra; no production Mamba claim."""

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))
from trifusion.patch_memory_roles import PatchMemoryRoles
from trifusion.slot_competition_roles import allocation_weights
from trifusion.slot_competition_fp32_roles import FP32SlotCompetitionRoles


def main():
    torch.set_num_threads(2)
    reports = []
    for grid in ((16, 8), (8, 16)):
        models = []
        for normalization in ("independent", "competitive"):
            torch.manual_seed(42)
            models.append(FP32SlotCompetitionRoles(
                attention_normalization=normalization, grid=grid,
                mamba_factory=lambda width: nn.Linear(width, width),
            ))
        independent, competitive = models
        assert independent.state_dict().keys() == competitive.state_dict().keys()
        assert all(torch.equal(value, competitive.state_dict()[name])
                   for name, value in independent.state_dict().items())
        patches = torch.randn(2, 3, 128, 128)
        context = torch.randn(2, 3, 128)
        positions = independent._positions(torch.randn(2, 3, 128))
        torch.manual_seed(42)
        original = PatchMemoryRoles(memory_mode="full", grid=grid,
                                    mamba_factory=lambda width: nn.Linear(width, width))
        original.load_state_dict(independent.state_dict(), strict=True)
        parity = (original.sample_context(patches, positions, context, 0) -
                  independent.sample_context(patches, positions, context, 0)).abs().max().item()
        assert parity == 0
        logits = torch.randn(2, 3, 16, 128, requires_grad=True)
        shared_patch_shift = torch.randn(2, 3, 1, 128)
        competing = allocation_weights(logits, "competitive")
        shifted = allocation_weights(logits + shared_patch_shift, "competitive")
        error = (competing - shifted).abs().max().item()
        assert error < 1e-6
        assert torch.allclose(competing.sum(-1), torch.ones(2, 3, 16), atol=1e-6)
        assert (allocation_weights(logits, "independent") -
                allocation_weights(logits + shared_patch_shift, "independent")).abs().max() > 0.01
        (competing * torch.randn_like(competing)).sum().backward()
        assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) > 0
        stages = torch.randn(3, 3, 2, 3, 129, 768)
        for model in models:
            evidence = model(stages, torch.randn(2, 512))
            assert all(getattr(evidence, name).shape == (2, 3, 16, 128)
                       and torch.isfinite(getattr(evidence, name)).all()
                       for name in ("cnn", "transformer", "mamba"))
        reports.append({"grid": grid, "state_equal": True, "original_control_max_difference": parity,
                        "common_patch_shift_invariance_max_difference": error,
                        "allocation_gradient_finite_nonzero": True, "role_shapes_finite": True})
    print(json.dumps({"status": "CPU_ALGEBRA_PASS", "production_mamba_tested": False,
                      "checks": reports}))


if __name__ == "__main__":
    main()
