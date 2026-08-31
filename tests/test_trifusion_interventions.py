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
    def test_edge_intervention_rejects_a_missing_modality_row(self) -> None:
        torch.manual_seed(71)
        model = _build_tiny_model()
        batch = {
            "images": {
                modality: torch.randn(2, 3, 4, 4)
                for modality in ("RGB", "NI", "TI")
            },
            "modality_mask": torch.tensor(
                [[True, True, True], [False, True, True]],
                dtype=torch.bool,
            ),
            "intervention": {
                "kind": "edge",
                "stage": 1,
                "source": "cnn",
                "target": "transformer",
                "modality": "RGB",
            },
        }

        with self.assertRaisesRegex(ValueError, "invalid.*rows.*1"):
            model(batch, return_aux=True)

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
        for stage_index in (0, 1):
            self.assertTrue(
                torch.equal(
                    baseline.relay_results[stage_index].gates,
                    direct.relay_results[stage_index].gates,
                )
            )
            self.assertTrue(
                torch.equal(
                    relay.relay_results[stage_index].gates[
                        :, :, cnn_index, rgb_index
                    ],
                    torch.zeros(2, 3),
                )
            )
            self.assertTrue(
                torch.equal(
                    relay.relay_results[stage_index].gates,
                    total.relay_results[stage_index].gates,
                )
            )
        self.assertTrue(
            torch.equal(
                baseline.contribution_embeddings,
                direct.contribution_embeddings,
            )
        )
        self.assertFalse(
            torch.allclose(
                baseline.contribution_embeddings,
                relay.contribution_embeddings,
            )
        )
        self.assertTrue(
            torch.equal(
                relay.contribution_embeddings,
                total.contribution_embeddings,
            )
        )
        self.assertFalse(
            torch.allclose(baseline.fused_embedding, direct.fused_embedding)
        )
        self.assertFalse(
            torch.allclose(baseline.fused_embedding, relay.fused_embedding)
        )
        self.assertFalse(torch.allclose(relay.fused_embedding, total.fused_embedding))
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

        cnn_index = 0
        transformer_index = 1
        rgb_index = 0
        for stage in (1, 2):
            with self.subTest(stage=stage), torch.no_grad():
                intervened = model(
                    {
                        **batch,
                        "intervention": {
                            "kind": "edge",
                            "stage": stage,
                            "source": "cnn",
                            "target": "transformer",
                            "modality": "RGB",
                        },
                    },
                    return_aux=True,
                )
            stage_index = stage - 1
            other_stage_index = 1 - stage_index
            baseline_gates = baseline.relay_results[stage_index].gates
            intervened_gates = intervened.relay_results[stage_index].gates
            changed = ~torch.isclose(baseline_gates, intervened_gates)
            permitted_changes = torch.zeros_like(changed)
            permitted_changes[:, transformer_index, :, rgb_index] = True

            self.assertTrue(
                torch.equal(
                    intervened_gates[
                        :, transformer_index, cnn_index, rgb_index
                    ],
                    torch.zeros(2),
                )
            )
            self.assertFalse(bool((changed & ~permitted_changes).any()))
            self.assertTrue(
                torch.equal(
                    baseline.relay_results[other_stage_index].gates,
                    intervened.relay_results[other_stage_index].gates,
                )
            )
            self.assertFalse(
                torch.allclose(
                    baseline.contribution_embeddings,
                    intervened.contribution_embeddings,
                )
            )
            self.assertFalse(
                torch.allclose(
                    baseline.fused_embedding,
                    intervened.fused_embedding,
                )
            )


if __name__ == "__main__":
    unittest.main()
