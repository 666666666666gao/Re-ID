"""CUDA-only mathematical checks for V21; no dataset or benchmark access."""
import copy

import torch
from torch import nn

from trifusion.sam_training_v21 import training_step


def test_sam_matches_quadratic_gradient_at_perturbed_parameters():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    original = parameter.detach().clone()
    diagonal = torch.tensor([2.0, 8.0], device="cuda")
    gradient = diagonal * original
    perturbed = original + 0.05 * gradient / gradient.norm()
    expected = original - 0.01 * diagonal * perturbed
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    row = training_step([("weight", parameter)], [], optimizer, scaler,
                        lambda: (0.5 * diagonal * parameter.square()).sum(), rho=0.05)
    torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
    assert abs(row["actual_perturbation_norm"] - 0.05) < 1e-6
    assert row["forward_backward_passes"] == 2 and not row["overflow"]


def test_control_is_one_actual_adamw_update_and_sam_scale_cancels():
    original = torch.tensor([1.0, -2.0], device="cuda")
    a, b = nn.Parameter(original.clone()), nn.Parameter(original.clone())
    opt_a = torch.optim.AdamW([a], lr=0.001, weight_decay=0.0001)
    opt_b = torch.optim.AdamW([b], lr=0.001, weight_decay=0.0001)
    scaler_a = torch.amp.GradScaler("cuda", init_scale=256.0)
    scaler_b = torch.amp.GradScaler("cuda", init_scale=256.0)
    row = training_step([("weight", a)], [], opt_a, scaler_a, lambda: a.square().sum(), rho=0.0)
    scaler_b.scale(b.square().sum()).backward()
    scaler_b.step(opt_b)
    scaler_b.update()
    assert torch.equal(a, b) and row["forward_backward_passes"] == 1
    assert row["actual_perturbation_norm"] == 0.0
    sam_parameters = []
    for scale in (128.0, 256.0):
        p = nn.Parameter(original.clone())
        optimizer = torch.optim.SGD([p], lr=0.01)
        scaler = torch.amp.GradScaler("cuda", init_scale=scale)
        training_step([("weight", p)], [], optimizer, scaler, lambda: p.square().sum(), rho=0.05)
        sam_parameters.append(p.detach())
    assert torch.equal(*sam_parameters)


def test_batchnorm_retains_first_pass_statistics_and_frozen_parameter():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.BatchNorm1d(4), nn.Linear(4, 2)).cuda().train()
    reference = copy.deepcopy(model)
    x = torch.arange(24, dtype=torch.float32, device="cuda").reshape(8, 3) / 24
    target = torch.arange(16, dtype=torch.float32, device="cuda").reshape(8, 2) / 16
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        reference(x)
    frozen = nn.Parameter(torch.tensor([3.0], device="cuda"), requires_grad=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)

    def closure():
        with torch.autocast("cuda", dtype=torch.float16):
            return (model(x).float() - target).square().mean() + frozen.sum() * 0

    training_step(list(model.named_parameters()), [model[1]], optimizer, scaler, closure, rho=0.05)
    assert model[1].num_batches_tracked.item() == 1
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        assert torch.equal(getattr(model[1], name), getattr(reference[1], name))
    assert frozen.item() == 3.0 and frozen.grad is None
