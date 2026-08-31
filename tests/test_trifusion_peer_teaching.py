from __future__ import annotations

import unittest

import torch


class RoleDirectedPeerTeachingTests(unittest.TestCase):
    def test_direction_rejection_and_teacher_stop_gradient(self) -> None:
        from modeling.trifusion import (
            ExpertState,
            ExpertStateMap,
            ReliabilityResult,
            RoleDirectedPeerTeaching,
        )

        torch.manual_seed(43)
        modality_mask = torch.ones(2, 3, dtype=torch.bool)
        globals_by_expert = {
            expert: torch.randn(2, 3, 4, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        }
        payloads = {
            expert: torch.randn(2, 3, 2, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        }
        states = ExpertStateMap(
            {
                expert: ExpertState(
                    tokens=global_embedding[:, :, None, :].expand(-1, -1, 4, -1),
                    global_embedding=global_embedding,
                    private_embedding=torch.randn(2, 3, 2),
                    role_payload={"role": payloads[expert]},
                    modality_mask=modality_mask,
                    stage=3,
                    expert=expert,
                )
                for expert, global_embedding in globals_by_expert.items()
            },
            modality_mask=modality_mask,
        )
        quality = torch.tensor([0.9, 0.5, 0.1]).view(1, 3, 1).expand(2, 3, 3)
        reliability = ReliabilityResult(
            alpha=torch.full_like(quality, 2.0),
            beta=torch.full_like(quality, 2.0),
            r=quality,
            u=torch.full_like(quality, 0.4),
            modality_mask=modality_mask,
        )
        teaching = RoleDirectedPeerTeaching(
            expert_widths={"cnn": 4, "transformer": 4, "mamba": 4},
            num_classes=3,
            role_width=4,
            quality_delta=0.2,
            minimum_teacher_quality=0.6,
        )

        result = teaching(states, reliability, torch.tensor([0, 1]))

        self.assertEqual(result.direction_gates.shape, (2, 3, 3))
        self.assertEqual(
            result.direction_gates.diagonal(dim1=1, dim2=2).count_nonzero().item(),
            0,
        )
        self.assertTrue(result.direction_gates[:, 1, 0].all())
        self.assertTrue(result.direction_gates[:, 2, 0].all())
        self.assertEqual(result.direction_gates[:, 0].count_nonzero().item(), 0)

        (result.logit_kl + result.role_loss).backward()
        teacher_gradient = globals_by_expert["cnn"].grad
        self.assertTrue(
            teacher_gradient is None or teacher_gradient.count_nonzero().item() == 0
        )
        self.assertGreater(
            globals_by_expert["transformer"].grad.abs().sum().item(), 0.0
        )
        self.assertGreater(globals_by_expert["mamba"].grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(result.private_diversity))


if __name__ == "__main__":
    unittest.main()
