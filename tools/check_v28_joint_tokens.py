#!/usr/bin/env python3
"""Remote synthetic checks of the actual CUDA Mamba and V28 bank algebra."""
import json

import numpy as np
import torch
from torch.nn import functional as F

from trifusion.joint_tokens_v28 import JointResidualTokens, normalized_joint_bank
from trifusion.state import EXPERT_ORDER


def run():
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        model = JointResidualTokens().cuda()
        x = torch.randn(2, 3, 3, 128, 768, device="cuda", requires_grad=True)
        modal = {name: F.normalize(torch.randn(2, 3, 512, device="cuda"), dim=-1)
                 for name in EXPERT_ORDER}
        calls = []

        def capture(_module, inputs, _output):
            calls.append(list(inputs[0].shape))

        with model.mixer.register_forward_hook(capture):
            correction = model(x)
        assert calls == [[2, 1152, 128], [2, 1152, 128]]
        assert all(torch.count_nonzero(value) == 0 for value in correction.values())
        bank = normalized_joint_bank(modal, correction)
        zero_bank = normalized_joint_bank(modal, {name: torch.zeros_like(modal[name]) for name in EXPERT_ORDER})
        assert torch.equal(bank, zero_bank)
        arrays = []
        for name in EXPERT_ORDER:
            a = modal[name].detach().cpu().numpy().astype(np.float64)
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            a = a.reshape(2, -1)
            arrays.append(a / np.linalg.norm(a, axis=-1, keepdims=True))
        expected = np.concatenate(arrays, axis=1)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        error = float(np.max(np.abs(expected - bank.detach().cpu().numpy())))
        assert error < 2e-6
        target = torch.randn_like(bank)
        (bank * target).sum().backward()
        zero_stage = {}
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            zero_stage[name] = float(p.grad.abs().sum())
            assert (zero_stage[name] > 0) == name.startswith("up."), name
        model.zero_grad(set_to_none=True)
        x.grad = None
        # A synthetic nonzero output projection tests connectivity; no optimizer,
        # dataset, checkpoint, or real training update is used here.
        for up in model.up.values():
            torch.nn.init.normal_(up.weight, std=0.001)
        correction = model(x)
        correction["transformer"].square().sum().backward()
        cross_role = float(x.grad[:, 0].abs().sum())
        assert cross_role > 0 and torch.isfinite(x.grad).all()
        model.zero_grad(set_to_none=True)
        x.grad = None
        correction = model(x)
        (normalized_joint_bank(modal, correction) * target).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in model.parameters())
        assert all(x.grad[:, i].abs().sum() > 0 for i in range(3))
        receipt = {
            "status": "PASS_SYNTHETIC_ACTUAL_CUDA_MAMBA",
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_tensors": len(list(model.parameters())),
            "mamba_call_shapes": calls, "numpy_bank_max_error": error,
            "zero_correction_bank_exact": True,
            "zero_init_gradient_abs_sums": zero_stage,
            "transformer_correction_gradient_into_cnn_tokens": cross_role,
            "all_joint_parameters_connected_after_nonzero_output": True,
            "all_three_input_roles_connected": True, "optimizer_updates": 0,
            "model_or_dataset_files_loaded": 0,
        }
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
