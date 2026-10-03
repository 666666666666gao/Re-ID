#!/usr/bin/env python3
"""Structural fixture only: linear Mamba stub, no production neural/GPU claim."""

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modeling"))
from trifusion.role_global_tokens import GlobalTokenRoles, TOKEN_MODES


def main():
    torch.set_num_threads(2)
    rows = []
    for grid in ((16, 8), (8, 16)):
        models = []
        for mode in TOKEN_MODES:
            torch.manual_seed(42)
            models.append(GlobalTokenRoles(token_mode=mode, grid=grid,
                                           mamba_factory=lambda width: nn.Linear(width, width)))
        reference = models[0].state_dict()
        assert all(set(model.state_dict()) == set(reference) and
                   all(torch.equal(value, model.state_dict()[key]) for key, value in reference.items())
                   for model in models)
        stages = torch.randn(3, 3, 2, 3, 129, 768)
        context = torch.randn(2, 512)
        global_value = torch.randn(2, 1536)
        shapes = []
        handle = models[1].transformer.register_forward_pre_hook(
            lambda _module, inputs: shapes.append(list(inputs[0].shape)))
        outputs = [model(stages, context, global_value) for model in models]
        handle.remove()
        assert shapes == [[6, 17, 128]]
        names = ("cnn", "transformer", "mamba")
        assert all(torch.equal(getattr(outputs[0], name), getattr(output, name))
                   for output in outputs for name in names)
        assert all(getattr(output, name).shape == (2, 3, 16, 128) and
                   bool(torch.isfinite(getattr(output, name)).all()) for output in outputs for name in names)
        loss = (outputs[1].transformer * torch.randn_like(outputs[1].transformer)).sum()
        loss.backward()
        gradient = models[1].global_projection.weight.grad
        assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum() > 0)
        with torch.no_grad():
            torch.manual_seed(43)
            replacement = torch.randn_like(models[0].global_projection.weight) * 0.03
            for model in models:
                model.global_projection.weight.copy_(replacement)
            changed = global_value.roll(1, dims=0)
            original = [model(stages, context, global_value) for model in models]
            altered = [model(stages, context, changed) for model in models]
        differences = {mode: float((a.transformer - b.transformer).abs().max())
                       for mode, a, b in zip(TOKEN_MODES, original, altered)}
        assert differences["static"] == differences["direct"] == 0
        assert differences["token"] > 0
        assert (models[2].projected_global(global_value) -
                models[2].projected_global(changed)).abs().max() > 0
        rows.append({"grid": list(grid), "initial_state_equal": True, "initial_role_outputs_equal": True,
                     "transformer_input_shape": shapes[0], "role_output_shape": [2, 3, 16, 128],
                     "global_projection_gradient_finite_nonzero": True,
                     "changed_global_role_difference": differences, "direct_projection_changes": True})
    print(json.dumps({"status": "CPU_STRUCTURE_PASS", "production_mamba_tested": False,
                      "real_training_data": False, "checks": rows}))


if __name__ == "__main__":
    main()
