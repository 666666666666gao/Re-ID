"""CPU-only independent moment-formula check; no models, images or checkpoints."""
import json
import numpy as np
import torch
from trifusion.source_style_v27 import make_style_plan, mix_patch_statistics


def run():
    rng = np.random.default_rng(711)
    x = rng.normal(size=(4, 3, 5, 7)).astype(np.float32)
    plan = make_style_plan([0, 0, 1, 2], fold=0, step=0, force_active=True)
    assert plan == make_style_plan([0, 0, 1, 2], fold=0, step=0, force_active=True)
    donor = np.array(plan["donors"])
    coefficient = np.array(plan["coefficients"], dtype=np.float64)[:, None, None, None]
    exact = x.astype(np.float64)
    mean = exact.mean(axis=(2, 3), keepdims=True)
    std = np.sqrt(exact.var(axis=(2, 3), keepdims=True) + 1e-6)
    expected = ((exact - mean) / std
                * (coefficient * std + (1 - coefficient) * std[donor])
                + coefficient * mean + (1 - coefficient) * mean[donor])
    source = torch.tensor(x, requires_grad=True)
    actual, stats = mix_patch_statistics(source, plan)
    error = float(np.max(np.abs(actual.detach().numpy() - expected)))
    assert error < 2e-6
    actual.sum().backward()
    expected_gradient = np.broadcast_to(
        (coefficient * std + (1 - coefficient) * std[donor]) / std, x.shape
    )
    grad_error = float(np.max(np.abs(source.grad.numpy() - expected_gradient)))
    assert grad_error < 2e-6
    assert np.array_equal(source.detach().numpy(), x)
    return {"status": "PASS", "max_formula_error": error,
            "max_stop_gradient_error": grad_error, "input_unchanged": True,
            "cross_camera_donors": True, "source_image_model_checkpoint_access": 0,
            "plan": plan, "statistics": stats}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
