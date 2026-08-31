from __future__ import annotations

import unittest

import torch


def _states_for_reliability(modality_mask: torch.Tensor):
    from modeling.trifusion import ExpertState, ExpertStateMap

    torch.manual_seed(29)
    states = {}
    for expert in ("cnn", "transformer", "mamba"):
        tokens = torch.randn(2, 3, 4, 4) * modality_mask[:, :, None, None]
        states[expert] = ExpertState(
            tokens=tokens,
            global_embedding=tokens.mean(dim=2),
            private_embedding=tokens.mean(dim=2)[..., :2],
            role_payload={"summary": tokens.mean(dim=-1, keepdim=True)},
            modality_mask=modality_mask,
            stage=1,
            expert=expert,
        )
    return ExpertStateMap(states, modality_mask=modality_mask)


class ReliabilityPosteriorTests(unittest.TestCase):
    def test_joint_beta_posterior_is_bounded_and_hard_masks_missing_entries(self) -> None:
        from modeling.trifusion import ReliabilityPosterior

        modality_mask = torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        )
        states = _states_for_reliability(modality_mask)
        posterior = ReliabilityPosterior(
            expert_widths={"cnn": 4, "transformer": 4, "mamba": 4},
            hidden_width=12,
            heads=3,
            kappa_min=2.0,
        )

        result = posterior(states, modality_mask)

        self.assertEqual(result.alpha.shape, (2, 3, 3))
        self.assertTrue((result.alpha > 1.0).all())
        self.assertTrue((result.beta > 1.0).all())
        self.assertTrue(torch.isfinite(result.r).all())
        self.assertTrue(torch.isfinite(result.u).all())
        valid = modality_mask[:, None, :].expand_as(result.r)
        self.assertTrue(((result.r[valid] > 0.0) & (result.r[valid] < 1.0)).all())
        self.assertTrue(((result.u[valid] > 0.0) & (result.u[valid] < 1.0)).all())
        self.assertEqual(result.r[~valid].count_nonzero().item(), 0)
        self.assertEqual(result.u[~valid].count_nonzero().item(), 0)


class CollaborativeFusionTests(unittest.TestCase):
    def test_valid_weights_normalize_and_missing_contributions_have_zero_mass(self) -> None:
        from modeling.trifusion import (
            CollaborativeFusion,
            ReliabilityResult,
        )

        modality_mask = torch.tensor(
            [[True, True, True], [False, True, False]], dtype=torch.bool
        )
        states = _states_for_reliability(modality_mask)
        r = torch.tensor(
            [
                [[0.9, 0.7, 0.4], [0.5, 0.8, 0.6], [0.3, 0.2, 0.9]],
                [[0.0, 0.8, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0]],
            ]
        )
        reliability = ReliabilityResult(
            alpha=torch.full_like(r, 2.0),
            beta=torch.full_like(r, 2.0),
            r=r,
            u=torch.full_like(r, 0.25)
            * modality_mask[:, None, :],
            modality_mask=modality_mask,
        )
        fusion = CollaborativeFusion(
            expert_widths={"cnn": 4, "transformer": 4, "mamba": 4},
            embedding_width=6,
        )

        result = fusion(states, reliability, modality_mask)

        self.assertEqual(result.fused_embedding.shape, (2, 6))
        self.assertTrue(torch.isfinite(result.fused_embedding).all())
        self.assertEqual(result.contribution_embeddings.shape, (2, 3, 3, 6))
        self.assertEqual(tuple(result.branch_embeddings), ("cnn", "transformer", "mamba"))
        torch.testing.assert_close(result.weights.sum(dim=(1, 2)), torch.ones(2))
        valid = modality_mask[:, None, :].expand_as(result.weights)
        self.assertEqual(result.weights[~valid].count_nonzero().item(), 0)
        invalid_contributions = (~valid)[..., None].expand_as(
            result.contribution_embeddings
        )
        self.assertEqual(
            result.contribution_embeddings[invalid_contributions].count_nonzero().item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
