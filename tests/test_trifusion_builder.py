from __future__ import annotations

import os
import unittest
from pathlib import Path


class ProductionBuilderTests(unittest.TestCase):
    def test_real_clip_builder_can_construct_hfer_uniform_target_generator(self) -> None:
        checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("TRIFUSION_CLIP_CHECKPOINT is not configured")

        from modeling.trifusion.builder import build_trifusion_from_clip
        from modeling.trifusion.experts.mamba import TinySequenceMixer
        from modeling.trifusion.reliability import UniformReliabilityGate

        result = build_trifusion_from_clip(
            Path(checkpoint_value),
            num_classes=94,
            reliability_mode="uniform",
            mamba_mixer_factory=TinySequenceMixer,
        )

        self.assertIsInstance(
            result.model.encoder.reliability_gate,
            UniformReliabilityGate,
        )
        self.assertEqual(result.provenance["reliability_mode"], "uniform")
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in result.model.encoder.reliability_gate.parameters()
            ),
            0,
        )

    def test_real_clip_checkpoint_builds_full_shared_tokenizer_architecture(self) -> None:
        checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("TRIFUSION_CLIP_CHECKPOINT is not configured")
        checkpoint = Path(checkpoint_value)
        self.assertTrue(checkpoint.is_file())

        from modeling.trifusion.builder import build_trifusion_from_clip
        from modeling.trifusion.experts.mamba import TinySequenceMixer

        result = build_trifusion_from_clip(
            checkpoint,
            num_classes=171,
            mamba_mixer_factory=TinySequenceMixer,
        )
        encoder = result.model.encoder

        self.assertEqual(len(encoder.experts["transformer"].blocks), 12)
        self.assertEqual(
            sum(len(stage) for stage in encoder.experts["cnn"].stages), 9
        )
        self.assertEqual(
            sum(len(stage) for stage in encoder.experts["mamba"].stages), 9
        )
        self.assertEqual(
            encoder.tokenizer.patch_projection.weight.shape,
            (768, 3, 16, 16),
        )
        self.assertEqual(encoder.tokenizer.positional_embedding.shape, (129, 768))
        self.assertEqual(result.provenance["clip_checkpoint_sha256"], result.checkpoint_sha256)
        self.assertLessEqual(
            sum(parameter.numel() for parameter in result.model.parameters()),
            120_000_000,
        )

    def test_shared_semantic_residual_builder_gives_every_expert_the_full_clip_space(
        self,
    ) -> None:
        checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("TRIFUSION_CLIP_CHECKPOINT is not configured")

        from modeling.trifusion.builder import build_trifusion_from_clip
        from modeling.trifusion.experts.mamba import TinySequenceMixer

        result = build_trifusion_from_clip(
            Path(checkpoint_value),
            num_classes=171,
            architecture="shared_semantic_residual",
            adapter_width=192,
            gradient_checkpointing=True,
            mamba_mixer_factory=TinySequenceMixer,
        )
        encoder = result.model.encoder

        self.assertEqual(
            result.provenance["architecture"],
            "shared_semantic_residual",
        )
        self.assertEqual(result.provenance["shared_clip_layers"], 12)
        self.assertEqual(
            result.provenance["expert_widths"],
            {"cnn": 768, "transformer": 768, "mamba": 768},
        )
        self.assertTrue(result.provenance["gradient_checkpointing"])
        self.assertTrue(result.provenance["final_reliability_refresh"])
        self.assertTrue(encoder.refresh_final_reliability)
        self.assertEqual(len(encoder.tokenizer.shared_blocks), 12)
        self.assertEqual(
            encoder.tokenizer.patch_projection.weight.shape,
            (768, 3, 16, 16),
        )
        projections = result.model.fusion.contribution_projections
        self.assertTrue(
            all(
                projections[expert].weight.equal(projections["transformer"].weight)
                for expert in ("cnn", "mamba")
            )
        )
        self.assertLessEqual(result.provenance["total_parameters"], 120_000_000)

    def test_real_cnn_standalone_builder_binds_the_cli_variant_contract(self) -> None:
        checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest("TRIFUSION_CLIP_CHECKPOINT is not configured")

        from modeling.trifusion.builder import build_single_branch_from_clip
        from modeling.trifusion.variants import resolve_variant, variant_sha256

        contract = resolve_variant("cnn_standalone")
        result = build_single_branch_from_clip(
            Path(checkpoint_value),
            expert_name="cnn",
            num_classes=141,
        )

        self.assertEqual(result.provenance["variant"], "cnn_standalone")
        self.assertEqual(result.provenance["active_experts"], ["cnn"])
        self.assertEqual(result.provenance["dormant_experts"], [])
        self.assertEqual(
            result.provenance["variant_contract_sha256"],
            variant_sha256(contract),
        )


if __name__ == "__main__":
    unittest.main()
