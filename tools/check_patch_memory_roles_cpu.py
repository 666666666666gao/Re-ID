#!/usr/bin/env python3
"""CPU algebra checks; the linear Mamba substitute is not a production witness."""

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))
from trifusion.patch_memory_roles import PatchMemoryRoles


def main():
    torch.set_num_threads(2)
    results = []
    for grid in ((16, 8), (8, 16)):
        models = []
        for mode in ("local", "full"):
            torch.manual_seed(42)
            models.append(PatchMemoryRoles(memory_mode=mode, grid=grid, mamba_factory=lambda width: nn.Linear(width, width)))
        local, full = models
        assert local.state_dict().keys() == full.state_dict().keys()
        assert all(torch.equal(value, full.state_dict()[name]) for name, value in local.state_dict().items())
        assert torch.equal(local.local_support.sum(-1), torch.full((16,), 9))
        assert sum(p.numel() for p in local.parameters() if p.requires_grad) == sum(p.numel() for p in full.parameters() if p.requires_grad)
        patches = torch.randn(2, 3, 128, 128, requires_grad=True)
        contexts = torch.randn(2, 3, 128)
        positions = local._positions(torch.randn(2, 3, 128))
        selected = local.sample_context(patches, positions, contexts, 0)
        selected[:, :, 0].square().sum().backward()
        outside = ~local.local_support[0]
        assert torch.count_nonzero(patches.grad[:, :, outside]) == 0
        patches.grad.zero_()
        full.sample_context(patches, positions, contexts, 0)[:, :, 0].square().sum().backward()
        assert torch.count_nonzero(patches.grad[:, :, outside]) > 0
        changed = patches.detach().clone()
        changed[:, :, outside] += torch.randn_like(changed[:, :, outside])
        assert torch.equal(selected[:, :, 0], local.sample_context(changed, positions, contexts, 0)[:, :, 0])
        assert not torch.allclose(full.sample_context(patches, positions, contexts, 0)[:, :, 0], full.sample_context(changed, positions, contexts, 0)[:, :, 0])
        stages = torch.randn(3, 3, 2, 3, 129, 768)
        context = torch.randn(2, 512)
        for model in models:
            evidence = model(stages, context)
            assert all(getattr(evidence, name).shape == (2, 3, 16, 128) for name in ("cnn", "transformer", "mamba"))
            assert all(torch.isfinite(getattr(evidence, name)).all() for name in ("cnn", "transformer", "mamba"))
        results.append({"grid": grid, "paired_state_equal": True, "local_candidates": 9,
                        "full_candidates": 128, "outside_patch_gradient_isolation": True,
                        "outside_patch_content_sensitivity": True, "role_shapes_finite": True})
    print(json.dumps({"status": "CPU_ALGEBRA_PASS", "production_mamba_tested": False, "checks": results}))


if __name__ == "__main__":
    main()
