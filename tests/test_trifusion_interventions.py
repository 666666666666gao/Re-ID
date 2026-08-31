from __future__ import annotations

import unittest

import torch


def _build_tiny_model():
    from modeling.trifusion import (
        CollaborativeFusion,
        HeterogeneousRelay,
        ReliabilityPosterior,
        TriBranchEncoder,
        TriFusionReID,
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
    encoder = TriBranchEncoder(
        experts,
        tokenizer=tokenizer,
        reliability_gate=ReliabilityPosterior(
            expert_widths=widths,
            hidden_width=12,
            heads=3,
        ),
        collaborator=HeterogeneousRelay(
            expert_widths=widths,
            relay_rank=4,
            token_grid=(2, 2),
            gamma_init=0.2,
        ),
    )
    return TriFusionReID(
        encoder=encoder,
        fusion=CollaborativeFusion(expert_widths=widths, embedding_width=6),
        embedding_width=6,
        num_classes=0,
    ).eval()


class FullNetworkInterventionTests(unittest.TestCase):
    def test_direct_relay_and_total_remove_distinct_full_network_paths(self) -> None:
        torch.manual_seed(73)
        model = _build_tiny_model()
        images = {
            modality: torch.randn(2, 3, 4, 4)
            for modality in ("RGB", "NI", "TI")
        }
        original_images = {
            modality: image.clone() for modality, image in images.items()
        }
        modality_mask = torch.ones(2, 3, dtype=torch.bool)
        batch = {"images": images, "modality_mask": modality_mask}

        with torch.no_grad():
            baseline = model(batch, return_aux=True)
            direct = model(
                {
                    **batch,
                    "intervention": {
                        "kind": "direct",
                        "expert": "cnn",
                        "modality": "RGB",
                    },
                },
                return_aux=True,
            )
            relay = model(
                {
                    **batch,
                    "intervention": {
                        "kind": "relay",
                        "expert": "cnn",
                        "modality": "RGB",
                    },
                },
                return_aux=True,
            )
            total = model(
                {
                    **batch,
                    "intervention": {
                        "kind": "total",
                        "expert": "cnn",
                        "modality": "RGB",
                    },
                },
                return_aux=True,
            )

        for modality, original in original_images.items():
            self.assertTrue(torch.equal(images[modality], original))
        self.assertTrue(torch.equal(batch["modality_mask"], modality_mask))

        cnn_index = 0
        rgb_index = 0
        self.assertTrue(
            torch.equal(
                baseline.relay_results[0].gates,
                direct.relay_results[0].gates,
            )
        )
        self.assertGreater(
            float(relay.fusion_weights[:, cnn_index, rgb_index].min()),
            0.0,
        )
        self.assertTrue(
            torch.equal(
                direct.fusion_weights[:, cnn_index, rgb_index],
                torch.zeros(2),
            )
        )
        self.assertTrue(
            torch.equal(
                total.fusion_weights[:, cnn_index, rgb_index],
                torch.zeros(2),
            )
        )
        self.assertTrue(
            torch.equal(
                relay.relay_results[0].gates[:, :, cnn_index, rgb_index],
                torch.zeros(2, 3),
            )
        )
        self.assertTrue(
            torch.equal(
                relay.relay_results[0].gates,
                total.relay_results[0].gates,
            )
        )
        for output in (baseline, direct, relay, total):
            self.assertTrue(
                torch.allclose(
                    output.fusion_weights.sum(dim=(1, 2)),
                    torch.ones(2),
                )
            )
        self.assertFalse(
            torch.allclose(baseline.fused_embedding, direct.fused_embedding)
        )
        self.assertFalse(
            torch.allclose(baseline.fused_embedding, relay.fused_embedding)
        )
        self.assertFalse(torch.allclose(direct.fused_embedding, total.fused_embedding))

    def test_edge_intervention_removes_only_the_named_stage_edge(self) -> None:
        torch.manual_seed(79)
        model = _build_tiny_model()
        batch = {
            "images": {
                modality: torch.randn(2, 3, 4, 4)
                for modality in ("RGB", "NI", "TI")
            },
            "modality_mask": torch.ones(2, 3, dtype=torch.bool),
        }

        with torch.no_grad():
            baseline = model(batch, return_aux=True)
            intervened = model(
                {
                    **batch,
                    "intervention": {
                        "kind": "edge",
                        "stage": 1,
                        "source": "cnn",
                        "target": "transformer",
                        "modality": "RGB",
                    },
                },
                return_aux=True,
            )

        cnn_index = 0
        transformer_index = 1
        rgb_index = 0
        baseline_stage1 = baseline.relay_results[0].gates
        intervened_stage1 = intervened.relay_results[0].gates
        changed = ~torch.isclose(baseline_stage1, intervened_stage1)

        self.assertTrue(
            torch.equal(
                intervened_stage1[:, transformer_index, cnn_index, rgb_index],
                torch.zeros(2),
            )
        )
        self.assertTrue(
            torch.equal(
                changed[:, 0],
                torch.zeros_like(changed[:, 0]),
            )
        )
        self.assertTrue(
            torch.equal(
                changed[:, 2],
                torch.zeros_like(changed[:, 2]),
            )
        )
        self.assertTrue(
            torch.equal(
                baseline.relay_results[1].gates,
                intervened.relay_results[1].gates,
            )
        )
        self.assertTrue(
            torch.equal(baseline.fusion_weights, intervened.fusion_weights)
        )
        self.assertFalse(
            torch.allclose(baseline.fused_embedding, intervened.fused_embedding)
        )


if __name__ == "__main__":
    unittest.main()
