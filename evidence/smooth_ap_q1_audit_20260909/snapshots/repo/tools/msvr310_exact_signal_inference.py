"""Preserve the measured B0 SIM matmul dispatch during frozen-model inference."""


def exact_signal_forward(model, batch):
    import torch

    assert not model.training
    name = "baseline.signal.SIM.modal_interactive.cross_attn.in_proj_weight"
    weight = model.baseline.signal.SIM.modal_interactive.cross_attn.in_proj_weight
    assert not weight.requires_grad
    # PyTorch 2.5.1 should_fold inspects the small operand's requires_grad even
    # under no_grad. A detached view restores B0's mm dispatch without unfreezing
    # the registered parameter or building an autograd graph.
    with torch.no_grad():
        dispatch_view = weight.detach().requires_grad_(True)
        output = torch.func.functional_call(
            model, {name: dispatch_view}, (batch,), {"return_aux": True},
        )
    assert model.baseline.signal.SIM.modal_interactive.cross_attn.in_proj_weight is weight
    assert not weight.requires_grad and dispatch_view.grad is None
    assert not output.baseline_embedding.requires_grad
    return output
