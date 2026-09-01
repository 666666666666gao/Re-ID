from __future__ import annotations

import unittest
import os
from pathlib import Path

import torch


class AnchorPreservingSemanticDecompositionTests(unittest.TestCase):
    def test_centered_patch_field_has_exact_clip_cls_mean(self) -> None:
        from modeling.trifusion.cascade_v2 import AnchorPreservingSemanticTokenizer

        projection = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2) / 48)
        tokenizer = AnchorPreservingSemanticTokenizer(
            patch_projection=projection,
            positional_embedding=torch.nn.Parameter(torch.zeros(5, 4)),
            class_embedding=torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0, -4.0])),
            pre_norm=torch.nn.Identity(),
            post_norm=torch.nn.Identity(),
            shared_blocks=[torch.nn.Identity()],
            gradient_checkpointing=False,
        )
        images = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4) / 96
        modalities = torch.tensor([0, 2], dtype=torch.long)

        fields = tokenizer(images, modalities)

        expected_cls = torch.tensor([1.0, -2.0, 3.0, -4.0]).expand(2, -1)
        for expert in ("cnn", "transformer", "mamba"):
            self.assertTrue(
                torch.allclose(fields[expert].mean(dim=1), expected_cls, atol=1e-6)
            )
        self.assertTrue(torch.equal(fields["cnn"], fields["transformer"]))
        self.assertTrue(torch.equal(fields["cnn"], fields["mamba"]))


class StageUpdatedReliabilityExchangeTests(unittest.TestCase):
    def test_each_exchange_and_final_fusion_receive_current_stage_quality(self) -> None:
        from modeling.trifusion import HeterogeneousRelay, ReliabilityPosterior
        from modeling.trifusion.cascade_v2 import StageUpdatedTriBranchEncoder
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
            "cnn": CNNExpert(width=4, grid_size=(2, 2), stage_depths=(1, 1, 1), private_width=3),
            "transformer": TransformerExpert.from_scratch(
                width=8, grid_size=(2, 2), layers=3, heads=2, private_width=3
            ),
            "mamba": MambaExpert.with_tiny_mixer(
                width=5, grid_size=(2, 2), stage_depths=(1, 1, 1), private_width=3
            ),
        }

        class CountingGate(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = ReliabilityPosterior(
                    expert_widths=widths, hidden_width=12, heads=3
                )
                self.results = []

            def forward(self, states, modality_mask):
                result = self.gate(states, modality_mask)
                self.results.append(result)
                return result

        gate = CountingGate()
        encoder = StageUpdatedTriBranchEncoder(
            experts,
            tokenizer=tokenizer,
            reliability_gate=gate,
            collaborator=HeterogeneousRelay(
                expert_widths=widths,
                relay_rank=4,
                token_grid=(2, 2),
                gamma_init=0.2,
            ),
        )
        images = {
            modality: torch.randn(2, 3, 4, 4)
            for modality in ("RGB", "NI", "TI")
        }
        mask = torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        )

        states = encoder(images, mask)

        self.assertEqual(len(gate.results), 3)
        self.assertIs(states.relay_results[0].reliability, gate.results[0])
        self.assertIs(states.relay_results[1].reliability, gate.results[1])
        self.assertIs(states.reliability, gate.results[2])


class QualityGatedInformationPreservingFusionTests(unittest.TestCase):
    def test_uniform_quality_preserves_every_valid_expert_modality_block(self) -> None:
        from modeling.trifusion.cascade_v2 import InformationPreservingFusion
        from modeling.trifusion.state import (
            EXPERT_ORDER,
            ExpertState,
            ExpertStateMap,
            ReliabilityResult,
        )

        mask = torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        )
        states = {}
        for expert_index, expert in enumerate(EXPERT_ORDER):
            values = torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[7.0, 8.0], [0.0, 0.0], [9.0, 10.0]],
                ]
            ) + expert_index * 10.0
            values = values * mask[..., None]
            states[expert] = ExpertState(
                tokens=values.unsqueeze(2),
                global_embedding=values,
                private_embedding=values,
                role_payload={"worked": values[..., :1]},
                modality_mask=mask,
                stage=3,
                expert=expert,
            )
        state_map = ExpertStateMap(states, modality_mask=mask)
        valid = mask[:, None, :].expand(2, 3, 3)
        reliability = ReliabilityResult(
            alpha=valid.float() * 2.0,
            beta=valid.float() * 2.0,
            r=valid.float() * 0.5,
            u=valid.float() * 0.5,
            modality_mask=mask,
        )
        fusion = InformationPreservingFusion(
            expert_widths={expert: 2 for expert in EXPERT_ORDER},
            embedding_width=2,
        )
        with torch.no_grad():
            fusion.semantic_projection.weight.copy_(torch.eye(2))
            for projection in fusion.residual_projections.values():
                projection.weight.zero_()

        result = fusion(state_map, reliability, mask)

        self.assertEqual(result.fused_embedding.shape, (2, 20))
        self.assertTrue(
            all(value.shape == (2, 8) for value in result.branch_embeddings.values())
        )
        self.assertTrue(
            torch.allclose(
                result.fused_embedding[0, :18],
                result.contribution_embeddings[0].reshape(-1),
                atol=1e-6,
            )
        )
        for expert_index in range(3):
            start = (expert_index * 3 + 1) * 2
            self.assertTrue(torch.equal(result.fused_embedding[1, start : start + 2], torch.zeros(2)))
        self.assertTrue(torch.allclose(result.weights.sum(dim=(1, 2)), torch.ones(2)))

    def test_reid_public_forward_uses_wide_retrieval_features_and_neck_only_for_logits(self) -> None:
        from modeling.trifusion import HeterogeneousRelay, TriBranchEncoder
        from modeling.trifusion.cascade_v2 import (
            CascadeV2ReID,
            InformationPreservingFusion,
        )
        from modeling.trifusion.experts.cnn import CNNExpert
        from modeling.trifusion.experts.mamba import MambaExpert
        from modeling.trifusion.experts.transformer import TransformerExpert
        from modeling.trifusion.reliability import UniformReliabilityGate
        from modeling.trifusion.tokenizer import SharedCLIPTokenizer

        widths = {"cnn": 4, "transformer": 4, "mamba": 4}
        encoder = TriBranchEncoder(
            {
                "cnn": CNNExpert(
                    width=4, grid_size=(2, 2), stage_depths=(1, 1, 1), private_width=2
                ),
                "transformer": TransformerExpert.from_scratch(
                    width=4, grid_size=(2, 2), layers=3, heads=2, private_width=2
                ),
                "mamba": MambaExpert.with_tiny_mixer(
                    width=4, grid_size=(2, 2), stage_depths=(1, 1, 1), private_width=2
                ),
            },
            tokenizer=SharedCLIPTokenizer(
                patch_projection=torch.nn.Conv2d(3, 4, kernel_size=2, stride=2),
                positional_embedding=torch.nn.Parameter(torch.zeros(5, 4)),
                expert_widths=widths,
            ),
            reliability_gate=UniformReliabilityGate(),
            collaborator=HeterogeneousRelay(
                expert_widths=widths, relay_rank=2, token_grid=(2, 2), gamma_init=0.1
            ),
        )
        fusion = InformationPreservingFusion(
            expert_widths=widths, embedding_width=2
        )
        model = CascadeV2ReID(
            encoder=encoder,
            fusion=fusion,
            num_classes=3,
        ).eval()
        batch = {
            "images": {
                modality: torch.randn(2, 3, 4, 4)
                for modality in ("RGB", "NI", "TI")
            },
            "modality_mask": torch.ones(2, 3, dtype=torch.bool),
        }

        with torch.no_grad():
            output = model(batch, return_aux=True)

        self.assertEqual(model.fused_neck.num_features, 20)
        self.assertTrue(
            all(neck.num_features == 8 for neck in model.branch_necks.values())
        )
        self.assertEqual(output.fused_embedding.shape, (2, 20))
        self.assertTrue(
            all(value.shape == (2, 8) for value in output.branch_embeddings.values())
        )
        self.assertEqual(output.fused_logits.shape, (2, 3))
        self.assertTrue(
            all(value.shape == (2, 3) for value in output.branch_logits.values())
        )


class CascadeV2BuilderContractTests(unittest.TestCase):
    def test_real_clip_build_exposes_all_three_v2_innovations(self) -> None:
        checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("TRIFUSION_CLIP_CHECKPOINT is not configured")
        from modeling.trifusion.cascade_v2_builder import (
            build_trifusion_cascade_v2_from_clip,
        )
        from modeling.trifusion.experts.mamba import TinySequenceMixer
        from modeling.trifusion.cascade_v2 import InformationPreservingFusion

        result = build_trifusion_cascade_v2_from_clip(
            Path(checkpoint_value),
            num_classes=141,
            architecture="shared_semantic_cascade_v2",
            reliability_mode="uniform",
            adapter_width=192,
            gradient_checkpointing=True,
            mamba_mixer_factory=TinySequenceMixer,
        )
        model = result.model

        self.assertEqual(result.provenance["architecture"], "shared_semantic_cascade_v2")
        self.assertEqual(result.provenance["semantic_decomposition"], "centered_patch_exact_cls")
        self.assertEqual(result.provenance["reliability_refresh_stages"], [1, 2, 3])
        self.assertEqual(result.provenance["fusion"], "quality_gated_blockwise_5120")
        self.assertTrue(model.encoder.tokenizer.center_patch_residuals)
        self.assertTrue(model.encoder.refresh_reliability_each_stage)
        self.assertIsInstance(model.fusion, InformationPreservingFusion)
        self.assertTrue(model.retrieval_before_neck)
        self.assertEqual(model.fused_embedding_width, 5120)
        self.assertEqual(model.branch_embedding_width, 2048)
        self.assertTrue(
            all(
                torch.count_nonzero(projection.weight).item() == 0
                for projection in model.fusion.residual_projections.values()
            )
        )
        self.assertLessEqual(result.provenance["total_parameters"], 120_000_000)


class SignedEffectReliabilityRankingTests(unittest.TestCase):
    def test_constant_router_is_penalized_more_than_correct_effect_order(self) -> None:
        from modeling.trifusion import ReliabilityResult, TriFusionOutput
        from modeling.trifusion.cascade_v2 import CascadeV2Criterion
        from modeling.trifusion.intervention_targets import CIRCTargetBatch

        effects = torch.linspace(-0.4, 0.4, 9).reshape(1, 3, 3).repeat(2, 1, 1)
        valid = torch.ones_like(effects, dtype=torch.bool)

        class FixedTargetCache:
            def lookup(self, *_args, device=None, **_kwargs):
                return CIRCTargetBatch(
                    helpful_targets=(effects > 0).float().to(device),
                    valid_mask=valid.to(device),
                    signed_total_effects=effects.to(device),
                    provenance_keys=("sample-0", "sample-1"),
                )

        def output_for(reliability_values: torch.Tensor) -> TriFusionOutput:
            modality_mask = torch.ones(2, 3, dtype=torch.bool)
            embedding = torch.zeros(2, 6)
            return TriFusionOutput(
                fused_embedding=embedding,
                branch_embeddings={
                    expert: embedding.clone()
                    for expert in ("cnn", "transformer", "mamba")
                },
                contribution_embeddings=torch.zeros(2, 3, 3, 6),
                reliability=ReliabilityResult(
                    alpha=1.0 + reliability_values,
                    beta=2.0 - reliability_values,
                    r=reliability_values,
                    u=torch.zeros_like(reliability_values),
                    modality_mask=modality_mask,
                ),
                relay_results=(),
                peer_teaching=None,
                fused_logits=torch.zeros(2, 2),
                branch_logits={
                    expert: torch.zeros(2, 2)
                    for expert in ("cnn", "transformer", "mamba")
                },
                modality_mask=modality_mask,
                diagnostics={"all_finite": True},
            )

        cache = FixedTargetCache()
        base = CascadeV2Criterion(
            target_cache=cache,
            brier_weight=0.0,
            evidence_weight=0.0,
            effect_rank_weight=0.0,
        )
        ranked = CascadeV2Criterion(
            target_cache=cache,
            brier_weight=0.0,
            evidence_weight=0.0,
            effect_rank_weight=1.0,
            effect_rank_margin=0.05,
        )
        labels = torch.tensor([0, 1])
        metadata = {
            "sample_keys": ["sample-0", "sample-1"],
            "conditions": [{"family": "clean"}] * 2,
        }
        constant = output_for(torch.full((2, 3, 3), 0.5))
        correctly_ordered = output_for(
            torch.linspace(0.1, 0.9, 9).reshape(1, 3, 3).repeat(2, 1, 1)
        )

        constant_rank = (
            ranked(constant, labels, **metadata)["reliability"]
            - base(constant, labels, **metadata)["reliability"]
        )
        ordered_rank = (
            ranked(correctly_ordered, labels, **metadata)["reliability"]
            - base(correctly_ordered, labels, **metadata)["reliability"]
        )

        self.assertGreater(constant_rank.item(), 0.0)
        self.assertLess(ordered_rank.item(), constant_rank.item())


class CascadeV2GeneralizationObjectiveTests(unittest.TestCase):
    def test_identity_losses_apply_configured_label_smoothing(self) -> None:
        import torch.nn.functional as F

        from modeling.trifusion import ReliabilityResult, TriFusionOutput
        from modeling.trifusion.cascade_v2 import CascadeV2Criterion

        labels = torch.tensor([0, 1])
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        modality_mask = torch.ones(2, 3, dtype=torch.bool)
        reliability_values = torch.full((2, 3, 3), 0.5)
        embedding = torch.zeros(2, 6)
        output = TriFusionOutput(
            fused_embedding=embedding,
            branch_embeddings={
                expert: embedding.clone()
                for expert in ("cnn", "transformer", "mamba")
            },
            contribution_embeddings=torch.zeros(2, 3, 3, 6),
            reliability=ReliabilityResult(
                alpha=torch.full_like(reliability_values, 2.0),
                beta=torch.full_like(reliability_values, 2.0),
                r=reliability_values,
                u=torch.full_like(reliability_values, 0.5),
                modality_mask=modality_mask,
            ),
            relay_results=(),
            peer_teaching=None,
            fused_logits=logits,
            branch_logits={
                expert: logits.clone()
                for expert in ("cnn", "transformer", "mamba")
            },
            modality_mask=modality_mask,
            diagnostics={"all_finite": True},
        )
        criterion = CascadeV2Criterion(
            target_cache=None,
            label_smoothing=0.1,
        )

        losses = criterion(output, labels)

        expected = F.cross_entropy(logits, labels, label_smoothing=0.1)
        self.assertTrue(torch.allclose(losses["id_fused"], expected))
        for expert in ("cnn", "transformer", "mamba"):
            self.assertTrue(torch.allclose(losses[f"id_{expert}"], expected))


if __name__ == "__main__":
    unittest.main()
