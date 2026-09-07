#!/usr/bin/env python3
"""Real captured derivative regression; zero images and optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from trifusion.joint_tokens_v28 import JointResidualTokens
from trifusion.joint_tokens_v28_fp32 import FP32JointResidualTokens
from trifusion.state import EXPERT_ORDER


def check(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "c569133afd9742f85699b70b1dfd953c8c4f5ac7ff9f9f500afb182db7f6943f"
    fixture = torch.load(path, map_location="cuda", weights_only=True)
    rows = []
    for name, constructor in (("original_amp", JointResidualTokens), ("local_fp32_fix", FP32JointResidualTokens)):
        model = constructor().cuda()
        model.load_state_dict(fixture["state"], strict=True)
        x = fixture["input"].detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(x)
        torch.autograd.backward([outputs[e] for e in EXPERT_ORDER],
                                [fixture["upstream_scaled"][e].to(outputs[e].dtype) for e in EXPERT_ORDER])
        gradient = model.mixer.dt_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        nonzero = int(torch.count_nonzero(gradient))
        assert nonzero == (0 if name == "original_amp" else 2048)
        rows.append({"implementation": name, "nonzero_dt_elements": nonzero,
                     "scaled_dt_abs_max": float(gradient.abs().max()),
                     "scaled_dt_l2": float(gradient.norm()),
                     "all_parameter_grads_finite": all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                     "output_dtype": str(outputs["cnn"].dtype),
                     "output_error_from_original_amp": max(float((outputs[e].float() - fixture["expected_amp_output"][e].float()).abs().max()) for e in EXPERT_ORDER)})
        assert rows[-1]["all_parameter_grads_finite"]
        del model, outputs, x
    return {"status": "PASS_REAL_SOURCE_SMALL_DERIVATIVE_REGRESSION", "cases": rows,
            "fixture_sha256": "c569133afd9742f85699b70b1dfd953c8c4f5ac7ff9f9f500afb182db7f6943f",
            "source_image_reads": 0, "optimizer_updates": 0, "retrieval_evaluation": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.fixture)
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result))
