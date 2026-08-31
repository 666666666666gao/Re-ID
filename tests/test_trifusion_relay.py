from __future__ import annotations

import unittest

import torch


def _make_states(modality_mask: torch.Tensor, *, requires_grad: bool = False):
    from modeling.trifusion import ExpertState, ExpertStateMap

    torch.manual_seed(23)
    states = {}
    for expert in ("cnn", "transformer", "mamba"):
        tokens = torch.randn(2, 3, 4, 4) * modality_mask[:, :, None, None]
        tokens.requires_grad_(requires_grad)
        states[expert] = ExpertState(
            tokens=tokens,
            global_embedding=tokens.mean(dim=2),
            private_embedding=torch.randn(2, 3, 2)
            * modality_mask[:, :, None],
            role_payload={"summary": tokens.mean(dim=(2, 3), keepdim=True)},
            modality_mask=modality_mask,
            stage=1,
            expert=expert,
        )
    return ExpertStateMap(states, modality_mask=modality_mask)


class HeterogeneousRelayTests(unittest.TestCase):
    def test_zero_gamma_is_identity_with_masked_normalized_no_self_gates(self) -> None:
        from modeling.trifusion import HeterogeneousRelay, ReliabilityResult

        modality_mask = torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        )
        states = _make_states(modality_mask)
        reliability_values = torch.tensor(
            [
                [[0.8, 0.6, 0.4], [0.5, 0.7, 0.9], [0.3, 0.4, 0.6]],
                [[0.9, 0.0, 0.7], [0.4, 0.0, 0.6], [0.2, 0.0, 0.8]],
            ]
        )
        reliability = ReliabilityResult(
            alpha=torch.full_like(reliability_values, 2.0),
            beta=torch.full_like(reliability_values, 2.0),
            r=reliability_values,
            u=torch.full_like(reliability_values, 0.5),
            modality_mask=modality_mask,
        )
        relay = HeterogeneousRelay(
            expert_widths={"cnn": 4, "transformer": 4, "mamba": 4},
            relay_rank=4,
            token_grid=(2, 2),
            gamma_init=0.0,
        )

        result = relay(states, reliability, stage=1)

        self.assertEqual(result.gates.shape, (2, 3, 3, 3))
        diagonal = result.gates.diagonal(dim1=1, dim2=2)
        self.assertEqual(diagonal.count_nonzero().item(), 0)
        invalid = (~modality_mask)[:, None, None, :].expand_as(result.gates)
        self.assertEqual(result.gates[invalid].count_nonzero().item(), 0)
        source_mass = result.gates.sum(dim=2)
        torch.testing.assert_close(
            source_mass[modality_mask[:, None, :].expand_as(source_mass)],
            torch.ones_like(
                source_mass[modality_mask[:, None, :].expand_as(source_mass)]
            ),
        )
        for expert in states:
            torch.testing.assert_close(
                result.states[expert].tokens, states[expert].tokens
            )
            self.assertGreater(result.private_energy[expert].item(), 0.0)

    def test_nonzero_gamma_changes_receivers_and_preserves_gradient_paths(self) -> None:
        from modeling.trifusion import HeterogeneousRelay, ReliabilityResult

        modality_mask = torch.ones(2, 3, dtype=torch.bool)
        states = _make_states(modality_mask, requires_grad=True)
        posterior_shape = (2, 3, 3)
        reliability = ReliabilityResult(
            alpha=torch.full(posterior_shape, 2.0),
            beta=torch.full(posterior_shape, 2.0),
            r=torch.full(posterior_shape, 0.5),
            u=torch.full(posterior_shape, 0.5),
            modality_mask=modality_mask,
        )
        relay = HeterogeneousRelay(
            expert_widths={"cnn": 4, "transformer": 4, "mamba": 4},
            relay_rank=4,
            token_grid=(2, 2),
            gamma_init=1.0,
        )

        result = relay(states, reliability, stage=1)
        changed_receivers = sum(
            not torch.allclose(result.states[expert].tokens, states[expert].tokens)
            for expert in states
        )
        self.assertGreaterEqual(changed_receivers, 2)

        loss = sum(state.tokens.square().mean() for state in result.states.values())
        loss = loss + sum(result.private_energy.values())
        loss.backward()
        for expert in states:
            self.assertIsNotNone(states[expert].tokens.grad)
            self.assertTrue(torch.isfinite(states[expert].tokens.grad).all())
            self.assertGreater(states[expert].tokens.grad.abs().sum().item(), 0.0)
        relay_gradients = [
            parameter.grad for parameter in relay.parameters() if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in relay_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in relay_gradients))


if __name__ == "__main__":
    unittest.main()
