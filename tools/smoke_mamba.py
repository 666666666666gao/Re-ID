#!/usr/bin/env python3
"""Exercise the compiled Mamba CUDA forward and backward paths."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from mamba_ssm import Mamba


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int, default=1555)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Mamba smoke requires an NVIDIA CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
    inputs = torch.randn(2, 64, 64, device=device, requires_grad=True)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs = model(inputs)
    loss = outputs.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters]
    checks = {
        "output_finite": bool(torch.isfinite(outputs).all()),
        "input_grad_finite": bool(torch.isfinite(inputs.grad).all()),
        "all_parameter_grads_present": all(gradient is not None for gradient in gradients),
        "all_parameter_grads_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mamba_ssm": version("mamba-ssm"),
        "causal_conv1d": version("causal-conv1d"),
        "transformers": version("transformers"),
        "input_shape": list(inputs.shape),
        "output_shape": list(outputs.shape),
        "loss": float(loss),
        "trainable_parameter_tensors": len(parameters),
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
        "seed": args.seed,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
