from __future__ import annotations

import unittest

import torch


class TriFusionStateTests(unittest.TestCase):
    def test_expert_state_map_is_immutable_and_exposes_only_expert_keys(self) -> None:
        from modeling.trifusion import ExpertState, ExpertStateMap

        modality_mask = torch.ones(2, 3, dtype=torch.bool)
        states = {
            expert: ExpertState(
                tokens=torch.ones(2, 3, 2, 4),
                global_embedding=torch.ones(2, 3, 4),
                private_embedding=torch.ones(2, 3, 2),
                role_payload={"summary": torch.ones(2, 3, 1)},
                modality_mask=modality_mask,
                stage=3,
                expert=expert,
            )
            for expert in ("cnn", "transformer", "mamba")
        }

        result = ExpertStateMap(states, modality_mask=modality_mask)

        self.assertEqual(tuple(result), ("cnn", "transformer", "mamba"))
        self.assertEqual(tuple(result.keys()), ("cnn", "transformer", "mamba"))
        with self.assertRaises(KeyError):
            _ = result["reliability"]
        with self.assertRaises(TypeError):
            result["cnn"] = states["mamba"]  # type: ignore[index]


class TriBranchEncoderTests(unittest.TestCase):
    def test_all_missing_rows_report_their_batch_indices(self) -> None:
        from modeling.trifusion import TriBranchEncoder

        encoder = TriBranchEncoder(
            {
                "cnn": torch.nn.Identity(),
                "transformer": torch.nn.Identity(),
                "mamba": torch.nn.Identity(),
            }
        )
        images = {
            modality: torch.randn(2, 3, 8, 4)
            for modality in ("RGB", "NI", "TI")
        }
        modality_mask = torch.tensor(
            [[True, True, True], [False, False, False]], dtype=torch.bool
        )

        with self.assertRaisesRegex(ValueError, r"all-missing.*\[1\]"):
            encoder(images, modality_mask)

    def test_every_expert_receives_valid_slots_and_invalid_slots_stay_zero(self) -> None:
        from modeling.trifusion import TriBranchEncoder
        from modeling.trifusion.experts.tiny import make_tiny_experts

        torch.manual_seed(7)
        widths = {"cnn": 4, "transformer": 6, "mamba": 5}
        encoder = TriBranchEncoder(
            make_tiny_experts(widths=widths, token_count=2, private_width=3)
        )
        images = {
            modality: torch.full((2, 3, 8, 4), float(index + 1))
            for index, modality in enumerate(("RGB", "NI", "TI"))
        }
        modality_mask = torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        )

        states = encoder(images, modality_mask)

        self.assertEqual(tuple(states), ("cnn", "transformer", "mamba"))
        for expert, width in widths.items():
            state = states[expert]
            self.assertEqual(state.tokens.shape, (2, 3, 2, width))
            self.assertEqual(state.global_embedding.shape, (2, 3, width))
            self.assertEqual(state.private_embedding.shape, (2, 3, 3))
            self.assertTrue(torch.isfinite(state.tokens).all())
            self.assertGreater(state.tokens[modality_mask].abs().sum().item(), 0.0)
            self.assertEqual(state.tokens[~modality_mask].count_nonzero().item(), 0)
            self.assertEqual(
                state.global_embedding[~modality_mask].count_nonzero().item(), 0
            )
            self.assertEqual(
                state.private_embedding[~modality_mask].count_nonzero().item(), 0
            )


class SharedTokenizerTests(unittest.TestCase):
    def test_one_shared_patch_projection_feeds_three_expert_token_spaces(self) -> None:
        from modeling.trifusion.tokenizer import SharedCLIPTokenizer

        torch.manual_seed(11)
        patch_projection = torch.nn.Conv2d(
            3, 8, kernel_size=2, stride=2, bias=False
        )
        positional_embedding = torch.nn.Parameter(torch.zeros(5, 8))
        tokenizer = SharedCLIPTokenizer(
            patch_projection=patch_projection,
            positional_embedding=positional_embedding,
            expert_widths={"cnn": 4, "transformer": 8, "mamba": 5},
        )
        packed_images = torch.ones(2, 3, 4, 4)
        packed_modalities = torch.tensor([0, 1], dtype=torch.long)

        outputs = tokenizer(packed_images, packed_modalities)

        self.assertIs(tokenizer.patch_projection, patch_projection)
        self.assertEqual(tuple(outputs), ("cnn", "transformer", "mamba"))
        self.assertEqual(outputs["cnn"].shape, (2, 4, 4))
        self.assertEqual(outputs["transformer"].shape, (2, 4, 8))
        self.assertEqual(outputs["mamba"].shape, (2, 4, 5))
        self.assertFalse(torch.equal(outputs["transformer"][0], outputs["transformer"][1]))


class StandaloneExpertTests(unittest.TestCase):
    def test_three_distinct_complete_experts_emit_role_states_with_gradients(self) -> None:
        from modeling.trifusion.experts.cnn import CNNExpert
        from modeling.trifusion.experts.mamba import MambaExpert
        from modeling.trifusion.experts.transformer import TransformerExpert

        torch.manual_seed(17)
        experts = {
            "cnn": CNNExpert(
                width=8,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
                expansion=2,
            ),
            "transformer": TransformerExpert.from_scratch(
                width=8,
                grid_size=(2, 2),
                layers=3,
                heads=2,
                private_width=3,
            ),
            "mamba": MambaExpert.with_tiny_mixer(
                width=8,
                grid_size=(2, 2),
                stage_depths=(1, 1, 1),
                private_width=3,
            ),
        }
        expected_payloads = {
            "cnn": {"local_neighbors", "horizontal_parts"},
            "transformer": {"global_to_patch"},
            "mamba": {"direction_context"},
        }

        for expert_name, expert in experts.items():
            tokens = torch.randn(2, 4, 8, requires_grad=True)
            output = expert(tokens)
            self.assertEqual(output.tokens.shape, (2, 4, 8))
            self.assertEqual(output.global_embedding.shape, (2, 8))
            self.assertEqual(output.private_embedding.shape, (2, 3))
            self.assertEqual(set(output.role_payload), expected_payloads[expert_name])
            loss = (
                output.tokens.square().mean()
                + output.global_embedding.square().mean()
                + output.private_embedding.square().mean()
                + sum(value.square().mean() for value in output.role_payload.values())
            )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in expert.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients), 0.0
            )

    def test_shared_tokenizer_and_complete_experts_form_one_encoder_graph(self) -> None:
        from modeling.trifusion import TriBranchEncoder
        from modeling.trifusion.experts.cnn import CNNExpert
        from modeling.trifusion.experts.mamba import MambaExpert
        from modeling.trifusion.experts.transformer import TransformerExpert
        from modeling.trifusion.tokenizer import SharedCLIPTokenizer

        torch.manual_seed(19)
        patch_projection = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2)
        tokenizer = SharedCLIPTokenizer(
            patch_projection=patch_projection,
            positional_embedding=torch.nn.Parameter(torch.zeros(5, 8)),
            expert_widths={"cnn": 4, "transformer": 8, "mamba": 5},
        )
        encoder = TriBranchEncoder(
            {
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
            },
            tokenizer=tokenizer,
        )
        images = {
            modality: torch.randn(2, 3, 4, 4)
            for modality in ("RGB", "NI", "TI")
        }
        modality_mask = torch.tensor(
            [[True, True, False], [False, True, True]], dtype=torch.bool
        )

        states = encoder(images, modality_mask)
        loss = sum(state.global_embedding.square().mean() for state in states.values())
        loss.backward()

        self.assertEqual(states["cnn"].tokens.shape, (2, 3, 4, 4))
        self.assertEqual(states["transformer"].tokens.shape, (2, 3, 4, 8))
        self.assertEqual(states["mamba"].tokens.shape, (2, 3, 4, 5))
        self.assertIsNotNone(patch_projection.weight.grad)
        self.assertGreater(patch_projection.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
