from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F


class SharedSemanticResidualExpertTests(unittest.TestCase):
    def test_all_public_experts_start_near_clip_tokens_and_receive_gradients(
        self,
    ) -> None:
        from modeling.trifusion import (
            SemanticCNNExpert,
            SemanticMambaExpert,
            SemanticTransformerExpert,
        )
        from modeling.trifusion.experts.mamba import TinySequenceMixer

        torch.manual_seed(83)
        experts = {
            "cnn": SemanticCNNExpert(
                width=8,
                adapter_width=4,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
            ),
            "transformer": SemanticTransformerExpert(
                width=8,
                adapter_width=4,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
            ),
            "mamba": SemanticMambaExpert(
                width=8,
                adapter_width=4,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
                mixer_factory=TinySequenceMixer,
            ),
        }
        semantic = F.layer_norm(torch.randn(2, 4, 8), (8,))

        for name, expert in experts.items():
            base = semantic.detach().clone().requires_grad_(True)
            runtime = expert.initialize(base)
            runtime = expert.run_stage(runtime, 1)
            output = expert.summarize(runtime, 1)
            delta = (output.tokens - base).abs()

            self.assertGreater(delta.max().item(), 0.0, name)
            self.assertLess(delta.max().item(), 0.05, name)
            loss = (
                output.tokens.square().mean()
                + output.global_embedding.square().mean()
                + output.private_embedding.square().mean()
                + sum(
                    value.square().mean()
                    for value in output.role_payload.values()
                )
            )
            loss.backward()

            self.assertIsNotNone(base.grad, name)
            self.assertGreater(base.grad.abs().sum().item(), 0.0, name)
            stage_gradients = [
                parameter.grad
                for parameter in expert.stages[0].parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(stage_gradients, name)
            self.assertTrue(
                all(gradient is not None for gradient in stage_gradients),
                name,
            )
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in stage_gradients),
                name,
            )
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in stage_gradients),
                0.0,
                name,
            )
            self.assertIsNotNone(expert.private_projection.weight.grad, name)
            self.assertGreater(
                expert.private_projection.weight.grad.abs().sum().item(),
                0.0,
                name,
            )


if __name__ == "__main__":
    unittest.main()
