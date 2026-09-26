import torch
from torch.nn import functional as F

from modeling.trifusion.signal_preserving_v8 import ExpertFormationFusion, ExpertFormationRepresentations
from modeling.trifusion.state import EXPERT_ORDER
from tools.diagnose_joint_fusion_scale import differentiable_scale_embeddings


def fixture():
    generator = torch.Generator().manual_seed(42)
    baseline = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    raw = {name: torch.randn(4, 3, generator=generator, dtype=torch.float64,
                            requires_grad=True) for name in EXPERT_ORDER}
    residuals = {name: F.normalize(value, dim=1) for name, value in raw.items()}
    representations = ExpertFormationRepresentations(residuals, residuals)
    fusion = ExpertFormationFusion(baseline_width=6, expert_width=3)
    direction = torch.randn(4, 15, generator=generator, dtype=torch.float64)
    return baseline, raw, representations, fusion, direction


def test_current_stop_gradient_differs_only_radially_for_normalized_fusion():
    baseline, raw, representations, fusion, direction = fixture()
    baseline.requires_grad_()
    current = fusion(baseline, representations)
    reference, branches = differentiable_scale_embeddings(baseline, representations.residual_embeddings)
    assert torch.equal(current.fused_embedding, reference)
    assert all(torch.equal(current.branch_embeddings[name], branches[name]) for name in EXPERT_ORDER)
    left = (F.normalize(current.fused_embedding, dim=1) * direction).sum()
    right = (F.normalize(reference, dim=1) * direction).sum()
    gleft = torch.autograd.grad(left, (baseline, *raw.values()), retain_graph=True)
    gright = torch.autograd.grad(right, (baseline, *raw.values()))
    difference = gleft[0] - gright[0]
    unit = F.normalize(baseline.detach(), dim=1)
    tangent = difference - (difference * unit).sum(dim=1, keepdim=True) * unit
    assert difference.norm() > 0.01
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), atol=1e-12, rtol=0)
    for a, b in zip(gleft[1:], gright[1:]):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
    # Forward normalization is scale-invariant. Its complete derivative has zero radial slope.
    eps = 1e-5
    values = [(F.normalize(fusion(baseline.detach() * scale, representations).fused_embedding,
                           dim=1) * direction).sum() for scale in (1 - eps, 1 + eps)]
    finite_difference = (values[1] - values[0]) / (2 * eps)
    torch.testing.assert_close(finite_difference, torch.zeros_like(finite_difference), atol=1e-9, rtol=0)
    assert abs(float((gright[0] * baseline).sum())) < 1e-12
    assert abs(float((gleft[0] * baseline).sum())) > 0.01


def test_frozen_baseline_has_unchanged_role_derivatives():
    baseline, raw, representations, fusion, direction = fixture()
    current = fusion(baseline, representations).fused_embedding
    reference, _ = differentiable_scale_embeddings(baseline, representations.residual_embeddings)
    left = torch.autograd.grad((F.normalize(current, dim=1) * direction).sum(),
                               tuple(raw.values()), retain_graph=True)
    right = torch.autograd.grad((F.normalize(reference, dim=1) * direction).sum(), tuple(raw.values()))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
