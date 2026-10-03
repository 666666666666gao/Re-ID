"""One-batch SAM updates for the fixed TriFusion V21 comparison.

Equations follow Foret et al., ICLR 2021: https://arxiv.org/abs/2010.01412.
This independent PyTorch implementation keeps first-pass BatchNorm statistics.
"""
from __future__ import annotations

import torch


def gradient_state(named_parameters):
    live = set()
    squared = []
    for name, parameter in named_parameters:
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        squared.append(parameter.grad.float().square().sum())
        if bool(parameter.grad.abs().sum() > 0):
            live.add(name)
    norm = torch.stack(squared).sum().sqrt()
    assert bool(torch.isfinite(norm)) and bool(norm > 0)
    return norm, live


def training_step(named_parameters, batchnorms, optimizer, scaler, closure, *, rho):
    """Use one pass for AdamW or two passes for SAM, then one optimizer update."""
    optimizer.zero_grad(set_to_none=True)
    scale = scaler.get_scale()
    loss = closure()
    assert bool(torch.isfinite(loss))
    scaler.scale(loss).backward()
    scaled_norm, first_live = gradient_state(named_parameters)
    first_loss = float(loss.detach())
    gradient_loss = first_loss
    perturbation_norm = 0.0

    if rho > 0:
        originals = [parameter.detach().clone() for _, parameter in named_parameters]
        statistics = [(buffer, buffer.detach().clone()) for module in batchnorms
                      for buffer in (module.running_mean, module.running_var, module.num_batches_tracked)]
        with torch.no_grad():
            for _, parameter in named_parameters:
                parameter.add_(parameter.grad * (rho / scaled_norm))
            perturbation_norm = float(torch.stack([
                (parameter - original).float().square().sum()
                for (_, parameter), original in zip(named_parameters, originals, strict=True)
            ]).sum().sqrt())
        optimizer.zero_grad(set_to_none=True)
        loss = closure()
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward()
        gradient_loss = float(loss.detach())
        with torch.no_grad():
            for (_, parameter), original in zip(named_parameters, originals, strict=True):
                parameter.copy_(original)
                assert torch.equal(parameter, original)
            for buffer, original in statistics:
                buffer.copy_(original)
                assert torch.equal(buffer, original)

    scaler.unscale_(optimizer)
    update_norm, update_live = gradient_state(named_parameters)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss_at_parameters": first_loss,
        "loss_for_update_gradient": gradient_loss,
        "first_gradient_norm": float(scaled_norm / scale),
        "update_gradient_norm": float(update_norm),
        "actual_perturbation_norm": perturbation_norm,
        "forward_backward_passes": 2 if rho > 0 else 1,
        "sam_restore_batches": int(rho > 0),
        "overflow": scaler.get_scale() < scale,
        "first_nonzero_gradient_names": sorted(first_live),
        "update_nonzero_gradient_names": sorted(update_live),
    }
