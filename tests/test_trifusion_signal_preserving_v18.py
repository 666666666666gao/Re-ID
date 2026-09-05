import torch

from trifusion.signal_preserving_v17 import TriadicCorrectionV17
from trifusion.signal_preserving_v18 import PairedViewProjectionV18, estimate_paired_direction


def test_source_paired_direction_ignores_between_identity_offset():
    features = torch.tensor([[-1., 2., 0.], [1., 2., 0.], [-1., 0., 7.], [1., 0., 7.]])
    direction, receipt = estimate_paired_direction(features, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
    torch.testing.assert_close(direction, torch.tensor([1., 0., 0.]))
    assert receipt["identity_camera_pairs"] == [[0, 0, 1], [1, 0, 1]]
    assert receipt["top_direction_energy_fraction"] == 1.


def test_projection_removes_source_direction_before_and_after_learned_correction():
    torch.manual_seed(42)
    core = TriadicCorrectionV17(residual_width=6, adapter_width=4)
    directions = torch.eye(6)[:3]
    block = PairedViewProjectionV18(core, directions, enabled=True)
    values = {name: torch.randn(5, 6) for name in ("cnn", "transformer", "mamba")}
    output = block(values)
    for i, value in enumerate(output.corrected_residuals.values()):
        torch.testing.assert_close(value @ directions[i], torch.zeros(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(value.norm(dim=1), torch.ones(5))
    target = torch.randn_like(output.fused_residual)
    (output.fused_residual * target).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.parameters())
    block.enabled = False
    torch.testing.assert_close(block(values).fused_residual, core(values).fused_residual, atol=0, rtol=0)
