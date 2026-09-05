"""CUDA contract tests for V22 MCNL; no dataset or model is loaded."""
import pytest
import torch
import torch.nn.functional as F

from trifusion.camera_negative_v22 import camera_pair_support, multi_camera_negative_loss


def explicit_reference(x, labels, cameras):
    x = F.normalize(x.double(), dim=1)
    values = []
    for i in range(len(labels)):
        positive = [j for j in range(len(labels)) if j != i and labels[j] == labels[i]]
        same_negative = [j for j in range(len(labels)) if labels[j] != labels[i] and cameras[j] == cameras[i]]
        other_negative = [j for j in range(len(labels)) if labels[j] != labels[i] and cameras[j] != cameras[i]]
        if positive and same_negative and other_negative:
            dp = max(float((x[i] - x[j]).norm()) for j in positive)
            ds = min(float((x[i] - x[j]).norm()) for j in same_negative)
            do = min(float((x[i] - x[j]).norm()) for j in other_negative)
            values.append(max(0.0, 0.1 + dp - do) + max(0.0, 0.1 + do - ds))
    return sum(values) / len(values)


def fixture():
    x = torch.tensor([[1., 0., .2], [.9, .1, .2], [1., .3, .1], [.8, .4, .1],
                      [-1., .1, .2], [-.8, .3, .1], [.2, -.8, .5], [.1, -.9, .3]], device="cuda")
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device="cuda")
    cameras = torch.tensor([0, 1, 0, 0, 1, 1, 2, 2], device="cuda")
    return x, labels, cameras


def test_formula_and_missing_camera_groups():
    x, labels, cameras = fixture()
    loss, stats = multi_camera_negative_loss(x, labels, cameras)
    expected = explicit_reference(x, labels.tolist(), cameras.tolist())
    assert float(loss) == pytest.approx(expected, abs=2e-6)
    assert stats["valid_rows"] == 6
    assert stats["same_negative_missing_rows"] == 2
    assert stats["other_negative_missing_rows"] == 0
    positive, same_negative, other_negative, valid, _ = camera_pair_support(labels, cameras)
    assert bool(positive[0, 1]) and not bool(other_negative[0, 1])
    assert not bool(positive.diagonal().any())
    assert not bool((same_negative & other_negative).any())
    assert valid.tolist() == [True] * 6 + [False] * 2


def test_gradient_and_invariances():
    x, labels, cameras = fixture()
    x.requires_grad_(True)
    loss, _ = multi_camera_negative_loss(x, labels, cameras)
    loss.backward()
    assert bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0
    permutation = torch.tensor([6, 2, 0, 7, 3, 1, 5, 4], device="cuda")
    permuted, _ = multi_camera_negative_loss(x.detach()[permutation], labels[permutation], cameras[permutation])
    scaled, _ = multi_camera_negative_loss(x.detach() * 3, labels, cameras)
    orthogonal = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], device="cuda")
    rotated, _ = multi_camera_negative_loss(x.detach() @ orthogonal, labels, cameras)
    for value in (permuted, scaled, rotated):
        assert torch.allclose(loss.detach(), value, atol=2e-6, rtol=0)
    half = x.detach().half().requires_grad_(True)
    half_loss, _ = multi_camera_negative_loss(half, labels, cameras)
    half_loss.backward()
    assert half_loss.dtype == torch.float32
    assert torch.allclose(loss.detach(), half_loss.detach(), atol=1e-3, rtol=0)
    assert bool(torch.isfinite(half.grad).all()) and float(half.grad.abs().sum()) > 0


def test_empty_support_fails_explicitly():
    x, labels, cameras = fixture()
    with pytest.raises(AssertionError, match="both negative-camera groups"):
        multi_camera_negative_loss(x, labels, torch.zeros_like(cameras))
