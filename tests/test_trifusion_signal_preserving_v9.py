from __future__ import annotations

import torch
import torch.nn.functional as F


def _modal_residual() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.randn(4, 3, 3, 8)


def test_v9_each_expert_receives_orthogonal_messages_from_both_peers() -> None:
    from modeling.trifusion.signal_preserving_v9 import OrthogonalTriadicRelay

    relay = OrthogonalTriadicRelay(
        residual_width=8,
        hidden_width=12,
        relay_depth=2,
    ).eval()
    modal_quality = torch.full((4, 3), 1.0 / 3.0)
    original = _modal_residual()
    output = relay(original, modal_quality)

    assert output.enhanced.shape == (4, 3, 3, 12)
    assert len(output.receiver_inputs) == 2
    assert len(output.orthogonal_messages) == 2
    for receivers, messages in zip(
        output.receiver_inputs,
        output.orthogonal_messages,
        strict=True,
    ):
        cosine = F.cosine_similarity(receivers, messages, dim=-1)
        assert float(cosine.abs().max()) <= 1e-5

    for receiver in range(3):
        for peer in range(3):
            if peer == receiver:
                continue
            changed = original.clone()
            changed[:, peer] = changed[:, peer] + 0.7
            changed_output = relay(changed, modal_quality)
            assert not torch.allclose(
                output.enhanced[:, receiver],
                changed_output.enhanced[:, receiver],
            )
