from __future__ import annotations

import unittest

import torch


def _build_tiny_encoder():
    from modeling.trifusion import (
        HeterogeneousRelay,
        ReliabilityPosterior,
        TriBranchEncoder,
    )
    from modeling.trifusion.experts.cnn import CNNExpert
    from modeling.trifusion.experts.mamba import MambaExpert
    from modeling.trifusion.experts.transformer import TransformerExpert
    from modeling.trifusion.tokenizer import SharedCLIPTokenizer

    widths = {"cnn": 4, "transformer": 8, "mamba": 5}
    tokenizer = SharedCLIPTokenizer(
        patch_projection=torch.nn.Conv2d(3, 8, kernel_size=2, stride=2),
        positional_embedding=torch.nn.Parameter(torch.zeros(5, 8)),
        expert_widths=widths,
    )
    experts = {
        "cnn": CNNExpert(
            width=4,
            grid_size=(2, 2),
            stage_depths=(1, 1, 1),
            private_width=3,
        ),
        "transformer": TransformerExpert.from_scratch(
            width=8,
            grid_size=(2, 2),
            layers=3,
            heads=2,
            private_width=3,
        ),
        "mamba": MambaExpert.with_tiny_mixer(
            width=5,
            grid_size=(2, 2),
            stage_depths=(1, 1, 1),
            private_width=3,
        ),
    }
    return (
        TriBranchEncoder(
            experts,
            tokenizer=tokenizer,
            reliability_gate=ReliabilityPosterior(
                expert_widths=widths, hidden_width=12, heads=3
            ),
            collaborator=HeterogeneousRelay(
                expert_widths=widths,
                relay_rank=4,
                token_grid=(2, 2),
                gamma_init=0.2,
            ),
        ),
        widths,
    )


class DeepCollaborationScheduleTests(unittest.TestCase):
    def test_one_stage1_posterior_is_reused_by_both_deep_relays(self) -> None:
        from modeling.trifusion import (
            HeterogeneousRelay,
            ReliabilityPosterior,
            TriBranchEncoder,
        )
        from modeling.trifusion.experts.cnn import CNNExpert
        from modeling.trifusion.experts.mamba import MambaExpert
        from modeling.trifusion.experts.transformer import TransformerExpert
        from modeling.trifusion.tokenizer import SharedCLIPTokenizer

        torch.manual_seed(31)
        widths = {"cnn": 4, "transformer": 8, "mamba": 5}
        tokenizer = SharedCLIPTokenizer(
            patch_projection=torch.nn.Conv2d(3, 8, kernel_size=2, stride=2),
            positional_embedding=torch.nn.Parameter(torch.zeros(5, 8)),
            expert_widths=widths,
        )
        experts = {
            "cnn": CNNExpert(
                width=4,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
            ),
            "transformer": TransformerExpert.from_scratch(
                width=8,
                grid_size=(2, 2),
                layers=3,
                heads=2,
                private_width=3,
            ),
            "mamba": MambaExpert.with_tiny_mixer(
                width=5,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
            ),
        }
        posterior = ReliabilityPosterior(
            expert_widths=widths, hidden_width=12, heads=3
        )
        relay = HeterogeneousRelay(
            expert_widths=widths,
            relay_rank=4,
            token_grid=(2, 2),
            gamma_init=0.2,
        )
        encoder = TriBranchEncoder(
            experts,
            tokenizer=tokenizer,
            reliability_gate=posterior,
            collaborator=relay,
        )
        images = {
            modality: torch.randn(2, 3, 4, 4)
            for modality in ("RGB", "NI", "TI")
        }
        modality_mask = torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        )

        states = encoder(images, modality_mask)

        self.assertIsNotNone(states.reliability)
        self.assertEqual(len(states.relay_results), 2)
        self.assertEqual(tuple(result.stage for result in states.relay_results), (1, 2))
        self.assertIs(states.relay_results[0].reliability, states.reliability)
        self.assertIs(states.relay_results[1].reliability, states.reliability)
        for state in states.values():
            self.assertEqual(state.stage, 3)
            self.assertTrue(torch.isfinite(state.tokens).all())

    def test_one_forward_returns_fused_and_all_named_branch_embeddings(self) -> None:
        from modeling.trifusion import (
            CollaborativeFusion,
            TriFusionOutput,
            TriFusionReID,
        )

        torch.manual_seed(37)
        encoder, widths = _build_tiny_encoder()
        model = TriFusionReID(
            encoder=encoder,
            fusion=CollaborativeFusion(
                expert_widths=widths, embedding_width=6
            ),
            embedding_width=6,
            num_classes=4,
        ).eval()
        batch = {
            "images": {
                modality: torch.randn(2, 3, 4, 4)
                for modality in ("RGB", "NI", "TI")
            },
            "modality_mask": torch.tensor(
                [[True, False, True], [False, True, False]], dtype=torch.bool
            ),
        }

        with torch.no_grad():
            output = model(batch, return_aux=True)

        self.assertIsInstance(output, TriFusionOutput)
        self.assertEqual(output.fused_embedding.shape, (2, 6))
        self.assertEqual(tuple(output.branch_embeddings), ("cnn", "transformer", "mamba"))
        self.assertTrue(
            all(embedding.shape == (2, 6) for embedding in output.branch_embeddings.values())
        )
        self.assertEqual(output.contribution_embeddings.shape, (2, 3, 3, 6))
        self.assertEqual(output.fused_logits.shape, (2, 4))
        self.assertTrue(
            all(logits.shape == (2, 4) for logits in output.branch_logits.values())
        )
        self.assertEqual(len(output.relay_results), 2)
        self.assertTrue(output.diagnostics["all_finite"])


if __name__ == "__main__":
    unittest.main()
