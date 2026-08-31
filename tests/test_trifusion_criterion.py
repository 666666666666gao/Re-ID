from __future__ import annotations

import unittest

import torch


class TriFusionCriterionTests(unittest.TestCase):
    def test_pre_circ_training_keeps_named_zero_reliability_loss(self) -> None:
        from modeling.trifusion import ReliabilityResult, TriFusionOutput
        from modeling.trifusion.criterion import TriFusionCriterion

        torch.manual_seed(39)
        batch_size = 4
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        modality_mask = torch.ones(batch_size, 3, dtype=torch.bool)
        reliability_values = torch.full((batch_size, 3, 3), 0.5)
        fused_embedding = torch.randn(batch_size, 6, requires_grad=True)
        output = TriFusionOutput(
            fused_embedding=fused_embedding,
            branch_embeddings={
                expert: torch.randn(batch_size, 6, requires_grad=True)
                for expert in ("cnn", "transformer", "mamba")
            },
            contribution_embeddings=torch.randn(batch_size, 3, 3, 6),
            reliability=ReliabilityResult(
                alpha=torch.full_like(reliability_values, 2.0),
                beta=torch.full_like(reliability_values, 2.0),
                r=reliability_values,
                u=torch.full_like(reliability_values, 0.5),
                modality_mask=modality_mask,
            ),
            relay_results=(),
            peer_teaching=None,
            fused_logits=torch.randn(batch_size, 2, requires_grad=True),
            branch_logits={
                expert: torch.randn(batch_size, 2, requires_grad=True)
                for expert in ("cnn", "transformer", "mamba")
            },
            modality_mask=modality_mask,
            diagnostics={"all_finite": True},
        )
        criterion = TriFusionCriterion(target_cache=None)

        losses = criterion(output, labels)

        self.assertEqual(losses["reliability"].item(), 0.0)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(fused_embedding.grad)

    def test_named_losses_use_immutable_circ_lookup_and_backpropagate(self) -> None:
        from modeling.trifusion import ReliabilityResult, TriFusionOutput
        from modeling.trifusion.criterion import TriFusionCriterion
        from modeling.trifusion.intervention_targets import CIRCTargetCache

        torch.manual_seed(41)
        batch_size = 4
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        condition = {"family": "gaussian", "severity": 2, "seed": 17}
        condition_key = '{"family":"gaussian","seed":17,"severity":2}'
        sample_keys = [f"sample-{index}" for index in range(batch_size)]
        cache_rows = []
        for sample_index, sample_key in enumerate(sample_keys[:2]):
            contributions = {
                f"{expert}.{modality}": {
                    "effects": {"total": 0.1, "direct": 0.04, "relay": 0.03},
                    "helpful_target": int((sample_index + expert_index) % 2 == 0),
                    "valid": True,
                }
                for expert_index, expert in enumerate(
                    ("cnn", "transformer", "mamba")
                )
                for modality in ("RGB", "NI", "TI")
            }
            cache_rows.append(
                {
                    "protocol_hash": "ab" * 32,
                    "sample_key": sample_key,
                    "identity": sample_index,
                    "condition_key": condition_key,
                    "generator_training_identities": [99],
                    "cross_camera_support": True,
                    "edge_values_used_as_training_targets": False,
                    "contributions": contributions,
                }
            )
        cache = CIRCTargetCache(cache_rows, receipt={"protocol_hash": "ab" * 32})

        reliability_logits = torch.randn(batch_size, 3, 3, requires_grad=True)
        reliability_values = torch.sigmoid(reliability_logits)
        modality_mask = torch.ones(batch_size, 3, dtype=torch.bool)
        reliability = ReliabilityResult(
            alpha=1.0 + reliability_values * 3.0,
            beta=1.0 + (1.0 - reliability_values) * 3.0,
            r=reliability_values,
            u=torch.full_like(reliability_values, 0.4),
            modality_mask=modality_mask,
        )
        fused_embedding = torch.randn(batch_size, 6, requires_grad=True)
        branch_embeddings = {
            expert: torch.randn(batch_size, 6, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        }
        output = TriFusionOutput(
            fused_embedding=fused_embedding,
            branch_embeddings=branch_embeddings,
            contribution_embeddings=torch.randn(batch_size, 3, 3, 6),
            reliability=reliability,
            relay_results=(),
            peer_teaching=None,
            fused_logits=torch.randn(batch_size, 2, requires_grad=True),
            branch_logits={
                expert: torch.randn(batch_size, 2, requires_grad=True)
                for expert in ("cnn", "transformer", "mamba")
            },
            modality_mask=modality_mask,
            diagnostics={"all_finite": True},
        )
        criterion = TriFusionCriterion(target_cache=cache, triplet_margin=0.3)

        losses = criterion(
            output,
            labels,
            sample_keys=sample_keys,
            conditions=[condition] * batch_size,
        )

        self.assertEqual(
            set(losses),
            {
                "id_fused",
                "triplet_fused",
                "id_cnn",
                "id_transformer",
                "id_mamba",
                "triplet_cnn",
                "triplet_transformer",
                "triplet_mamba",
                "reliability",
                "peer_logits",
                "peer_role",
                "private_diversity",
            },
        )
        self.assertTrue(all(loss.ndim == 0 for loss in losses.values()))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(reliability_logits.grad)
        self.assertGreater(reliability_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(fused_embedding.grad)

        unsupported = criterion(
            output,
            labels,
            sample_keys=[f"unsupported-{index}" for index in range(batch_size)],
            conditions=[condition] * batch_size,
        )
        self.assertEqual(unsupported["reliability"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
