"""Remote CPU analytical gradient check; no dataset, model, or checkpoint access."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from trifusion.role_modal_responsibility_v26 import EXPERTS, responsibility_loss


def run():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(4, 9, 7)).astype(np.float32)
    x = raw.astype(np.float64)
    radii = np.linalg.norm(x, axis=-1)
    unit = x / radii[:, :, None]
    baseline = F.normalize(torch.tensor(rng.normal(size=(4, 11)), dtype=torch.float32), dim=1)
    tensors = {e: torch.tensor(raw[:, i * 3:(i + 1) * 3], requires_grad=True) for i, e in enumerate(EXPERTS)}
    bank = F.normalize(torch.cat([F.normalize(tensors[e], dim=-1).flatten(1) for e in EXPERTS], dim=1), dim=1)
    fused = torch.cat([baseline, bank], dim=1).detach().requires_grad_()
    base = baseline.clone().requires_grad_()
    output = SimpleNamespace(modal_residual_embeddings=tensors, fused_embedding=fused, baseline_embedding=base)
    labels = torch.tensor([0, 0, 1, 1])
    loss, stats = responsibility_loss(output, labels, temperature=0.1)
    gradients = torch.autograd.grad(loss, [*tensors.values(), fused, base], allow_unused=True)
    assert gradients[-2:] == (None, None), "Responsibility weights must not backpropagate through fused or baseline."
    actual_gradient = torch.cat(gradients[:3], dim=1).numpy()
    f = F.normalize(fused.detach(), dim=1).numpy().astype(np.float64)
    expected_gradient = np.zeros_like(x)
    value, weights = 0.0, []
    triplets = [(q, p, n) for q in range(4) for p in range(4) for n in range(4)
                if q != p and labels[q] == labels[p] and labels[q] != labels[n]]
    assert len(triplets) == stats["responsibility_triplets"] == 8
    assert stats["responsibility_slot_triplets"] == 72
    for q, p, n in triplets:
        mf = np.dot(f[q], f[p]) - np.dot(f[q], f[n])
        for slot in range(9):
            cosine_positive = np.dot(unit[q, slot], unit[p, slot])
            cosine_negative = np.dot(unit[q, slot], unit[n, slot])
            margin = cosine_positive - cosine_negative
            negative_probability = 1 / (1 + np.exp(margin / 0.1))
            w = negative_probability / (1 + np.exp(mf / 0.1))
            weights.append(w)
            value += w * 0.1 * np.logaddexp(0, -margin / 0.1) / 72
            coefficient = -w * negative_probability / 72
            expected_gradient[q, slot] += coefficient * (
                unit[p, slot] - unit[n, slot] - margin * unit[q, slot]) / radii[q, slot]
            expected_gradient[p, slot] += coefficient * (
                unit[q, slot] - cosine_positive * unit[p, slot]) / radii[p, slot]
            expected_gradient[n, slot] -= coefficient * (
                unit[q, slot] - cosine_negative * unit[n, slot]) / radii[n, slot]
    np.testing.assert_allclose(float(loss.detach()), value, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-5, atol=2e-8)
    assert all(np.linalg.norm(expected_gradient[:, slot]) > 0 for slot in range(9))
    assert max(weights) - min(weights) > 0.5
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = SimpleNamespace(modal_residual_embeddings={e: t[permutation] for e, t in tensors.items()},
                               fused_embedding=fused[permutation], baseline_embedding=base[permutation])
    other, _ = responsibility_loss(permuted, labels[permutation], temperature=0.1)
    torch.testing.assert_close(loss, other, rtol=1e-6, atol=1e-8)
    result = {"status": "PASS_ANALYTICAL_VALUE_GRADIENT_DETACHED_WEIGHTS_CLASS_ZERO_PERMUTATION",
                      "triplets": 8, "slots": 9,
                      "loss_error": abs(float(loss.detach()) - value),
                      "gradient_max_error": float(np.max(np.abs(actual_gradient - expected_gradient))),
                      "fused_and_baseline_auxiliary_gradient": None,
                      "model_forwards": 0, "dataset_accesses": 0, "checkpoint_loads": 0}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    run()
