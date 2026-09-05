"""Mathematical contracts for V20; run in the remote CUDA environment."""
import math

import torch

from trifusion.cross_modal_identity_v20 import cross_modal_identity_loss

EXPERTS = ("cnn", "transformer", "mamba")


def test_all_eight_same_identity_targets_have_the_correct_entropy_floor():
    labels = torch.arange(8, device="cuda").repeat_interleave(8)
    value = torch.eye(8, device="cuda")[labels, None].expand(-1, 3, -1)
    loss = cross_modal_identity_loss({expert: value for expert in EXPERTS}, labels, temperature=0.07)
    expected = math.log(8) + math.log1p(7 * math.exp(-1 / 0.07))
    torch.testing.assert_close(loss, torch.tensor(expected, device="cuda"), rtol=1e-6, atol=1e-6)


def test_identity_relabeling_batch_order_and_modality_order_are_invariant():
    torch.manual_seed(42)
    labels = torch.arange(4, device="cuda").repeat_interleave(2)
    values = {e: torch.randn(8, 3, 12, device="cuda") for e in EXPERTS}
    original = cross_modal_identity_loss(values, labels, temperature=0.07)
    order = torch.tensor([5, 1, 3, 0, 7, 2, 6, 4], device="cuda")
    permuted = {e: value[order][:, [2, 0, 1]] for e, value in values.items()}
    actual = cross_modal_identity_loss(permuted, labels[order] * 11 + 7, temperature=0.07)
    torch.testing.assert_close(original, actual)


def test_loss_reaches_every_expert_and_every_modality():
    torch.manual_seed(42)
    labels = torch.arange(4, device="cuda").repeat_interleave(2)
    values = {e: torch.randn(8, 3, 12, device="cuda", requires_grad=True) for e in EXPERTS}
    loss = cross_modal_identity_loss(values, labels, temperature=0.07)
    loss.backward()
    for value in values.values():
        assert torch.isfinite(value.grad).all()
        assert (value.grad.abs().sum(dim=(0, 2)) > 0).all()
