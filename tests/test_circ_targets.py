from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CIRCTargetBuilderTests(unittest.TestCase):
    def test_fold_assignment_canonicalizes_zero_padded_numeric_identity(self) -> None:
        from modeling.trifusion.intervention_targets import assign_identity_fold

        self.assertEqual(
            assign_identity_fold(
                "000009",
                fold_salt="TriFusion-CIRC-fold-v1",
                fold_count=3,
            ),
            assign_identity_fold(
                9,
                fold_salt="TriFusion-CIRC-fold-v1",
                fold_count=3,
            ),
        )

    def test_fold_assignment_rejects_signed_or_non_decimal_identity(self) -> None:
        from modeling.trifusion.intervention_targets import assign_identity_fold

        for identity in ("+9", "-9", "person-9", ""):
            with self.subTest(identity=identity), self.assertRaisesRegex(
                ValueError,
                "unsigned decimal",
            ):
                assign_identity_fold(
                    identity,
                    fold_salt="TriFusion-CIRC-fold-v1",
                    fold_count=3,
                )

    def test_hash_selected_edge_is_deterministic_with_two_per_row_budget(self) -> None:
        from modeling.trifusion.intervention_targets import select_audit_edge

        valid_edges = [
            f"{source}->{target}:{modality}"
            for source in ("cnn", "transformer", "mamba")
            for target in ("cnn", "transformer", "mamba")
            if source != target
            for modality in ("RGB", "NI", "TI")
        ]
        condition = {"family": "gaussian", "severity": 2, "seed": 17}

        first = [
            select_audit_edge(
                valid_edges,
                protocol_hash="ab" * 32,
                sample_key="person-001/cam-2/frame-3",
                condition=condition,
                stage=stage,
            )
            for stage in (1, 2)
        ]
        repeated = [
            select_audit_edge(
                list(reversed(valid_edges)),
                protocol_hash="ab" * 32,
                sample_key="person-001/cam-2/frame-3",
                condition=condition,
                stage=stage,
            )
            for stage in (1, 2)
        ]

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(item.edge in valid_edges for item in first))
        self.assertTrue(all(len(item.digest_sha256) == 64 for item in first))

    def test_development_cli_emits_identity_disjoint_immutable_target_cache(self) -> None:
        from tools.build_circ_targets import main

        experts = ("cnn", "transformer", "mamba")
        modalities = ("RGB", "NI", "TI")
        interventions = {}
        for expert_index, expert in enumerate(experts):
            for modality_index, modality in enumerate(modalities):
                valid = modality != "TI"
                scale = float((expert_index + 1) * (modality_index + 1))
                interventions[f"{expert}.{modality}"] = {
                    "total": 0.1 * scale if valid else 0.0,
                    "direct": 0.04 * scale if valid else 0.0,
                    "relay": 0.03 * scale if valid else 0.0,
                }
        valid_edges = [
            f"{source}->{target}:{modality}"
            for source in experts
            for target in experts
            if source != target
            for modality in ("RGB", "NI")
        ]
        config = {
            "schema_version": "circ-source-v1",
            "protocol_hash": "ab" * 32,
            "fold_salt": "fixture-fold-v1",
            "fold_count": 3,
            "epsilon": 0.02,
            "configuration_frozen": False,
            "official_test_access_count": 0,
            "development_forbidden_identities": [999],
            "samples": [
                {
                    "sample_key": "id101-cam1-frame1",
                    "identity": 101,
                    "camera": 1,
                    "dataset_split": "train",
                    "modality_mask": [True, True, False],
                    "cross_camera_positive_cameras": [2],
                    "condition": {
                        "family": "gaussian",
                        "severity": 2,
                        "seed": 17,
                    },
                    "generator_training_identities": [102, 103, 104],
                    "generator_checkpoint_sha256": "cd" * 32,
                    "reference_bank_sha256": "ef" * 32,
                    "intervention_seeds": [1017, 2017, 3017],
                    "interventions": interventions,
                    "edge_effects": {
                        "1": {edge: 0.01 for edge in valid_edges},
                        "2": {edge: -0.01 for edge in valid_edges},
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            first_output = root / "first"
            second_output = root / "second"

            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "development",
                        "--output",
                        str(first_output),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "development",
                        "--output",
                        str(second_output),
                    ]
                ),
                0,
            )

            target_bytes = (first_output / "targets.jsonl").read_bytes()
            self.assertEqual(
                target_bytes, (second_output / "targets.jsonl").read_bytes()
            )
            row = json.loads(target_bytes)
            receipt = json.loads(
                (first_output / "receipt.json").read_text(encoding="utf-8")
            )
            symmetry_receipt = json.loads(
                (first_output / "symmetry_receipt.json").read_text(encoding="utf-8")
            )
            transfer_receipt = json.loads(
                (first_output / "target_transfer_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn(row["identity"], row["generator_training_identities"])
            self.assertEqual(len(row["contributions"]), 9)
            self.assertEqual(len(row["edge_audit"]), 2)
            self.assertTrue(row["cross_camera_support"])
            self.assertTrue(receipt["zero_identity_overlap"])
            self.assertEqual(receipt["edge_audit_runs"], 2)
            self.assertFalse(symmetry_receipt["claim_eligible"])
            self.assertFalse(transfer_receipt["claim_eligible"])
            self.assertEqual(
                receipt["targets_sha256"], hashlib.sha256(target_bytes).hexdigest()
            )

    def test_cli_orchestrates_three_fixed_endpoint_generators_without_target_eval(
        self,
    ) -> None:
        from modeling.trifusion.protocol import trifusion_source_hashes
        from modeling.trifusion.variants import resolve_variant, variant_sha256
        from tools.build_circ_targets import main

        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            protocol_path = project / "protocols/circ_target_v1.json"
            generator_config = (
                project / "configs/RGBNT201/TriFusion-circ-generator-low-vram.yml"
            )
            contract_sha256 = variant_sha256(
                resolve_variant("hfer_uniform_generator")
            )
            selector_root = root / "selector"
            selector_root.mkdir()
            selector_checkpoint = selector_root / "best_dev_model.pth"
            selector_checkpoint.write_bytes(b"frozen-selector")
            selector_identity = {
                "variant": "hfer_uniform_generator",
                "variant_contract_sha256": contract_sha256,
                "config_sha256": hashlib.sha256(
                    generator_config.read_bytes()
                ).hexdigest(),
                "circ_protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "source_sha256": trifusion_source_hashes(),
                "official_test_access_during_development": False,
                "optimization": {"max_epochs": 60},
            }
            selector_identity_path = selector_root / "run_identity.json"
            selector_identity_path.write_text(
                json.dumps(selector_identity, sort_keys=True),
                encoding="utf-8",
            )
            selector_identity_sha256 = hashlib.sha256(
                selector_identity_path.read_bytes()
            ).hexdigest()
            selector_resume = selector_root / ".resume"
            selector_resume.mkdir()
            selector_state = selector_resume / "generation-0060-complete.pt"
            selector_state.write_bytes(b"selector-full-state")
            selector_manifest = selector_resume / "latest.json"
            selector_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "epoch": 60,
                        "phase": "complete",
                        "run_identity_sha256": selector_identity_sha256,
                        "current": {
                            "path": ".resume/generation-0060-complete.pt",
                            "sha256": hashlib.sha256(
                                selector_state.read_bytes()
                            ).hexdigest(),
                        },
                        "previous": None,
                        "completion_evidence": {},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            metrics = {
                name: {
                    "mAP": 51.0 if name == "fused" else 40.0,
                    "Rank-1": 52.0,
                    "Rank-5": 60.0,
                    "Rank-10": 65.0,
                }
                for name in ("fused", "cnn", "transformer", "mamba")
            }
            endpoint_receipt = selector_root / "best_dev_receipt.json"
            endpoint_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "trifusion-dev-selection-v1",
                        "variant": "hfer_uniform_generator",
                        "epoch": 9,
                        "selection_output": "fused",
                        "dev_selection_mAP": 51.0,
                        "metrics_percent": metrics,
                        "variant_contract_sha256": contract_sha256,
                        "phase": "complete",
                        "schedule_horizon_epochs": 60,
                        "dev_evaluation_count": 60,
                        "model_constructed": True,
                        "training_started": True,
                        "fatal_or_nonfinite_detected": False,
                        "config_sha256": hashlib.sha256(
                            generator_config.read_bytes()
                        ).hexdigest(),
                        "circ_protocol_sha256": hashlib.sha256(
                            protocol_path.read_bytes()
                        ).hexdigest(),
                        "checkpoint": str(selector_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            selector_checkpoint.read_bytes()
                        ).hexdigest(),
                        "run_identity": str(selector_identity_path),
                        "run_identity_sha256": selector_identity_sha256,
                        "recovery_manifest": str(selector_manifest),
                        "recovery_manifest_sha256": hashlib.sha256(
                            selector_manifest.read_bytes()
                        ).hexdigest(),
                        "official_test_access_count": 0,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            fake_worker = root / "fake_fold_worker.py"
            fake_worker.write_text(
                "#!/usr/bin/env python3\n"
                "import hashlib, json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "fold = int(args[args.index('--_worker-fold') + 1])\n"
                "out = pathlib.Path(args[args.index('--output') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "identity_path = out / 'run_identity.json'\n"
                "identity = json.loads(identity_path.read_text())\n"
                "identity_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()\n"
                "resume = out / '.resume'; resume.mkdir(exist_ok=True)\n"
                "state = resume / 'generation-0009-complete.pt'\n"
                "state.write_bytes(f'full-state-{fold}'.encode())\n"
                "manifest = resume / 'latest.json'\n"
                "manifest.write_text(json.dumps({'schema_version': '1.0', 'epoch': 9, "
                "'phase': 'complete', 'run_identity_sha256': identity_sha, "
                "'current': {'path': '.resume/generation-0009-complete.pt', "
                "'sha256': hashlib.sha256(state.read_bytes()).hexdigest()}, "
                "'previous': None, 'completion_evidence': {}}), encoding='utf-8')\n"
                "checkpoint = out / 'generator.pth'\n"
                "checkpoint.write_bytes(f'fold-{fold}'.encode())\n"
                "receipt = {'status': 'COMPLETE', 'phase': 'complete', 'epoch': 9, "
                "'target_fold': fold, "
                "'fixed_endpoint': 9, 'schedule_horizon_epochs': 60, "
                "'generator_target_identity_overlap': 0, "
                "'dev_evaluation_count': 0, 'target_loader_iteration_count': 0, "
                "'variant': identity['variant'], "
                "'variant_contract_sha256': identity['variant_contract_sha256'], "
                "'circ_protocol_sha256': identity['circ_protocol_sha256'], "
                "'config_sha256': identity['generator_config_sha256'], "
                "'source_sha256': identity['source_sha256'], "
                "'run_identity_sha256': identity_sha, "
                "'model_constructed': True, 'training_started': True, "
                "'fatal_or_nonfinite_detected': False, 'parameter_budget_pass': True, "
                "'contract_testing': True, 'scientific_evidence_eligible': False, "
                "'data_provenance': {'target_fold': fold, "
                "'generator_target_identity_overlap': 0, "
                "'target_forbidden_dev_identity_overlap': 0, 'official_test_records': 0}, "
                "'train_history': {str(epoch): {'total': 1.0} for epoch in range(1, 10)}, "
                "'recovery_manifest': str(manifest), "
                "'recovery_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), "
                "'official_test_access_count': 0, 'checkpoint': str(checkpoint), "
                "'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}\n"
                "(out / 'generator_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')\n",
                encoding="utf-8",
            )
            fake_worker.chmod(0o755)
            config_path = root / "orchestration.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "circ-generator-orchestration-v1",
                        "operation": "train-oof-generators",
                        "circ_protocol": str(protocol_path),
                        "generator_config": str(generator_config),
                        "endpoint_receipt": str(endpoint_receipt),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "generators"

            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "1",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": str(fake_worker),
                },
            ):
                self.assertEqual(
                    main(
                        [
                            "--config",
                            str(config_path),
                            "--mode",
                            "development",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )

            receipt = json.loads(
                (output / "generators_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(receipt["completed_folds"], [0, 1, 2])
            self.assertEqual(receipt["fixed_endpoint"], 9)
            self.assertEqual(receipt["dev_evaluation_count"], 0)
            self.assertEqual(receipt["target_loader_iteration_count"], 0)
            self.assertEqual(receipt["official_test_access_count"], 0)
            self.assertTrue(receipt["zero_identity_overlap"])
            self.assertTrue(receipt["contract_testing"])
            self.assertTrue(receipt["test_override_used"])
            self.assertFalse(receipt["scientific_evidence_eligible"])

            original_selector_manifest = selector_manifest.read_bytes()
            empty_manifest = json.loads(original_selector_manifest)
            empty_manifest.pop("current")
            selector_manifest.write_text(
                json.dumps(empty_manifest, sort_keys=True),
                encoding="utf-8",
            )
            endpoint_without_current = json.loads(
                endpoint_receipt.read_text(encoding="utf-8")
            )
            endpoint_without_current["recovery_manifest_sha256"] = hashlib.sha256(
                selector_manifest.read_bytes()
            ).hexdigest()
            endpoint_receipt.write_text(
                json.dumps(endpoint_without_current, sort_keys=True),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "1",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": str(fake_worker),
                },
            ), self.assertRaisesRegex(
                ValueError,
                "current recovery generation",
            ):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "development",
                        "--output",
                        str(root / "empty-selector-recovery"),
                    ]
                )
            selector_manifest.write_bytes(original_selector_manifest)
            endpoint_without_current["recovery_manifest_sha256"] = hashlib.sha256(
                selector_manifest.read_bytes()
            ).hexdigest()
            endpoint_receipt.write_text(
                json.dumps(endpoint_without_current, sort_keys=True),
                encoding="utf-8",
            )

            fold_one_receipt = output / "fold-1/generator_receipt.json"
            failed = json.loads(fold_one_receipt.read_text(encoding="utf-8"))
            failed["status"] = "FAILED"
            fold_one_receipt.write_text(json.dumps(failed), encoding="utf-8")
            sparse_attempt = output / "fold-1/worker-attempt-0004.log"
            sparse_attempt.write_text("preserve-me", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "1",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": str(fake_worker),
                },
            ):
                self.assertEqual(
                    main(
                        [
                            "--config",
                            str(config_path),
                            "--mode",
                            "development",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            self.assertTrue((output / "fold-1/worker-attempt-0005.log").is_file())
            self.assertEqual(sparse_attempt.read_text(encoding="utf-8"), "preserve-me")

            fold_zero_receipt = output / "fold-0/generator_receipt.json"
            forged = json.loads(fold_zero_receipt.read_text(encoding="utf-8"))
            forged["run_identity_sha256"] = "0" * 64
            fold_zero_receipt.write_text(json.dumps(forged), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "1",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": str(fake_worker),
                },
            ), self.assertRaisesRegex(ValueError, "receipt contract failed"):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "development",
                        "--output",
                        str(output),
                    ]
                )

    def test_target_cache_lookup_is_keyed_and_never_gradient_bearing(self) -> None:
        from modeling.trifusion.intervention_targets import CIRCTargetCache

        condition = {"family": "gaussian", "severity": 2, "seed": 17}
        condition_key = json.dumps(
            condition, sort_keys=True, separators=(",", ":")
        )
        contributions = {
            f"{expert}.{modality}": {
                "effects": {
                    "total": 0.1,
                    "direct": 0.04,
                    "relay": 0.03,
                    "interaction": 0.03,
                },
                "labels": {
                    "total": "helpful",
                    "direct": "helpful",
                    "relay": "helpful",
                },
                "helpful_target": 1,
                "valid": True,
            }
            for expert in ("cnn", "transformer", "mamba")
            for modality in ("RGB", "NI", "TI")
        }
        row = {
            "schema_version": "circ-target-v1",
            "mode": "development",
            "protocol_hash": "ab" * 32,
            "sample_key": "sample-1",
            "identity": 101,
            "condition": condition,
            "condition_key": condition_key,
            "generator_training_identities": [102, 103],
            "cross_camera_support": True,
            "contributions": contributions,
            "edge_values_used_as_training_targets": False,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            targets_bytes = (json.dumps(row, sort_keys=True) + "\n").encode()
            (root / "targets.jsonl").write_bytes(targets_bytes)
            (root / "receipt.json").write_text(
                json.dumps(
                    {
                        "protocol_hash": "ab" * 32,
                        "targets_sha256": hashlib.sha256(targets_bytes).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            cache = CIRCTargetCache.from_directory(root)
            batch = cache.lookup(["sample-1"], [condition])

            self.assertEqual(batch.helpful_targets.shape, (1, 3, 3))
            self.assertEqual(batch.valid_mask.shape, (1, 3, 3))
            self.assertIsNone(batch.helpful_targets.grad_fn)
            self.assertFalse(batch.helpful_targets.requires_grad)
            self.assertTrue(batch.valid_mask.all())
            self.assertNotIn(101, cache.rows[0]["generator_training_identities"])

    def test_cli_verifies_scientific_checkpoint_recovery_binding(self) -> None:
        import torch

        from modeling.trifusion.protocol import trifusion_source_hashes
        from modeling.trifusion.variants import resolve_variant, variant_sha256
        from tools.build_circ_targets import main

        project = Path(__file__).resolve().parents[1]
        protocol_path = project / "protocols/circ_target_v1.json"
        generator_config = (
            project / "configs/RGBNT201/TriFusion-circ-generator-low-vram.yml"
        )
        protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        generator_config_sha256 = hashlib.sha256(
            generator_config.read_bytes()
        ).hexdigest()
        contract_sha256 = variant_sha256(
            resolve_variant("hfer_uniform_generator")
        )
        source_sha256 = trifusion_source_hashes()
        state_hashes = (
            "ed2562fdf682d2131f1a7882f4451a30dafba8ee916c8d8488348b86f47d1f7f",
            "c10074aef181c4a987d35379185a7c4cbd3febbc71bc8f4dd527294846d9d60b",
            "eab9636a2e52ebe3c7e0fa28ebe733416a2bd0a0ee887aefa04a1e40750a2418",
        )

        def canonical_sha256(value: object) -> str:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selector_root = root / "selector"
            selector_root.mkdir()
            selector_checkpoint = selector_root / "best_dev_model.pth"
            torch.save({"weight": torch.tensor([7.0])}, selector_checkpoint)
            metrics = {
                name: {
                    "mAP": 51.0 if name == "fused" else 40.0,
                    "Rank-1": 52.0,
                    "Rank-5": 60.0,
                    "Rank-10": 65.0,
                }
                for name in ("fused", "cnn", "transformer", "mamba")
            }
            selector_identity = {
                "variant": "hfer_uniform_generator",
                "variant_contract_sha256": contract_sha256,
                "config_sha256": generator_config_sha256,
                "circ_protocol_sha256": protocol_sha256,
                "source_sha256": source_sha256,
                "official_test_access_during_development": False,
                "optimization": {"max_epochs": 60},
                "contract_testing": False,
                "scientific_evidence_eligible": True,
            }
            selector_identity_path = selector_root / "run_identity.json"
            selector_identity_path.write_text(
                json.dumps(selector_identity, sort_keys=True),
                encoding="utf-8",
            )
            selector_identity_sha256 = hashlib.sha256(
                selector_identity_path.read_bytes()
            ).hexdigest()
            selector_resume = selector_root / ".resume"
            selector_resume.mkdir()
            selector_generation = selector_resume / "generation-0060-complete.pt"
            selector_generation.write_bytes(b"opaque-full-selector-recovery")
            selector_manifest = selector_resume / "latest.json"
            selector_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "epoch": 60,
                        "phase": "complete",
                        "run_identity_sha256": selector_identity_sha256,
                        "current": {
                            "path": ".resume/generation-0060-complete.pt",
                            "sha256": hashlib.sha256(
                                selector_generation.read_bytes()
                            ).hexdigest(),
                        },
                        "previous": None,
                        "completion_evidence": {
                            "kind": "selector",
                            "epoch": 60,
                            "phase": "complete",
                            "run_identity_sha256": selector_identity_sha256,
                            "best_epoch": 9,
                            "best_map": 51.0,
                            "best_metrics": metrics,
                            "best_checkpoint_sha256": hashlib.sha256(
                                selector_checkpoint.read_bytes()
                            ).hexdigest(),
                            "contract_testing": False,
                            "scientific_evidence_eligible": True,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            endpoint_receipt = selector_root / "best_dev_receipt.json"
            endpoint_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "trifusion-dev-selection-v1",
                        "variant": "hfer_uniform_generator",
                        "epoch": 9,
                        "selection_output": "fused",
                        "dev_selection_mAP": 51.0,
                        "metrics_percent": metrics,
                        "variant_contract_sha256": contract_sha256,
                        "phase": "complete",
                        "schedule_horizon_epochs": 60,
                        "dev_evaluation_count": 60,
                        "model_constructed": True,
                        "training_started": True,
                        "fatal_or_nonfinite_detected": False,
                        "config_sha256": generator_config_sha256,
                        "circ_protocol_sha256": protocol_sha256,
                        "checkpoint": str(selector_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            selector_checkpoint.read_bytes()
                        ).hexdigest(),
                        "run_identity": str(selector_identity_path),
                        "run_identity_sha256": selector_identity_sha256,
                        "recovery_manifest": str(selector_manifest),
                        "recovery_manifest_sha256": hashlib.sha256(
                            selector_manifest.read_bytes()
                        ).hexdigest(),
                        "official_test_access_count": 0,
                        "contract_testing": False,
                        "scientific_evidence_eligible": True,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            orchestration_config = {
                "schema_version": "circ-generator-orchestration-v1",
                "operation": "train-oof-generators",
                "circ_protocol": str(protocol_path),
                "generator_config": str(generator_config),
                "endpoint_receipt": str(endpoint_receipt),
            }
            orchestration_path = root / "orchestration.json"
            orchestration_path.write_text(
                json.dumps(orchestration_config),
                encoding="utf-8",
            )
            orchestration_sha256 = hashlib.sha256(
                orchestration_path.read_bytes()
            ).hexdigest()
            endpoint_sha256 = hashlib.sha256(
                endpoint_receipt.read_bytes()
            ).hexdigest()
            output = root / "generators"
            output.mkdir()

            for fold, model_state_sha256 in enumerate(state_hashes):
                fold_output = output / f"fold-{fold}"
                fold_output.mkdir()
                identity = {
                    "schema_version": "circ-generator-run-v1",
                    "operation": "train-oof-generator",
                    "orchestration_config_sha256": orchestration_sha256,
                    "circ_protocol_sha256": protocol_sha256,
                    "generator_config_sha256": generator_config_sha256,
                    "endpoint_receipt_sha256": endpoint_sha256,
                    "selector_checkpoint_sha256": hashlib.sha256(
                        selector_checkpoint.read_bytes()
                    ).hexdigest(),
                    "source_sha256": source_sha256,
                    "variant": "hfer_uniform_generator",
                    "variant_contract_sha256": contract_sha256,
                    "target_fold": fold,
                    "fixed_endpoint": 9,
                    "schedule_horizon_epochs": 60,
                    "official_test_access_count": 0,
                    "contract_testing": False,
                    "test_override_used": False,
                    "scientific_evidence_eligible": True,
                }
                identity_path = fold_output / "run_identity.json"
                identity_path.write_text(
                    json.dumps(identity, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                identity_sha256 = hashlib.sha256(
                    identity_path.read_bytes()
                ).hexdigest()
                checkpoint = fold_output / "generator.pth"
                torch.save(
                    {"weight": torch.tensor([float(fold)], dtype=torch.float32)},
                    checkpoint,
                )
                checkpoint_sha256 = hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest()
                train_history = {
                    str(epoch): {"total": float(epoch)} for epoch in range(1, 10)
                }
                provenance = {
                    "target_fold": fold,
                    "generator_target_identity_overlap": 0,
                    "target_forbidden_dev_identity_overlap": 0,
                    "official_test_records": 0,
                }
                resume = fold_output / ".resume"
                resume.mkdir()
                generation = resume / "generation-0009-complete.pt"
                generation.write_bytes(f"opaque-fold-{fold}-recovery".encode())
                manifest = resume / "latest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "epoch": 9,
                            "phase": "complete",
                            "run_identity_sha256": identity_sha256,
                            "current": {
                                "path": ".resume/generation-0009-complete.pt",
                                "sha256": hashlib.sha256(
                                    generation.read_bytes()
                                ).hexdigest(),
                            },
                            "previous": None,
                            "completion_evidence": {
                                "kind": "oof-generator",
                                "epoch": 9,
                                "phase": "complete",
                                "run_identity_sha256": identity_sha256,
                                "train_history_sha256": canonical_sha256(
                                    train_history
                                ),
                                "data_provenance_sha256": canonical_sha256(
                                    provenance
                                ),
                                "contract_testing": False,
                                "scientific_evidence_eligible": True,
                                "target_fold": fold,
                                "generator_checkpoint_sha256": checkpoint_sha256,
                                "generator_model_state_sha256": model_state_sha256,
                            },
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                receipt = {
                    "status": "COMPLETE",
                    "phase": "complete",
                    "epoch": 9,
                    "target_fold": fold,
                    "fixed_endpoint": 9,
                    "schedule_horizon_epochs": 60,
                    "generator_target_identity_overlap": 0,
                    "dev_evaluation_count": 0,
                    "target_loader_iteration_count": 0,
                    "official_test_access_count": 0,
                    "variant": "hfer_uniform_generator",
                    "variant_contract_sha256": contract_sha256,
                    "circ_protocol_sha256": protocol_sha256,
                    "config_sha256": generator_config_sha256,
                    "source_sha256": source_sha256,
                    "run_identity_sha256": identity_sha256,
                    "model_constructed": True,
                    "training_started": True,
                    "fatal_or_nonfinite_detected": False,
                    "parameter_budget_pass": True,
                    "contract_testing": False,
                    "scientific_evidence_eligible": True,
                    "data_provenance": provenance,
                    "train_history": train_history,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_state_sha256": model_state_sha256,
                    "recovery_manifest": str(manifest),
                    "recovery_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                }
                (fold_output / "generator_receipt.json").write_text(
                    json.dumps(receipt),
                    encoding="utf-8",
                )

            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "0",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": "",
                },
            ):
                self.assertEqual(
                    main(
                        [
                            "--config",
                            str(orchestration_path),
                            "--mode",
                            "development",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            aggregate = json.loads(
                (output / "generators_receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(aggregate["scientific_evidence_eligible"])
            self.assertFalse(aggregate["contract_testing"])

            fold_zero_receipt_path = output / "fold-0/generator_receipt.json"
            fold_zero_receipt = json.loads(
                fold_zero_receipt_path.read_text(encoding="utf-8")
            )
            replaced_checkpoint = output / "fold-0/generator.pth"
            torch.save({"weight": torch.tensor([99.0])}, replaced_checkpoint)
            fold_zero_receipt["checkpoint_sha256"] = hashlib.sha256(
                replaced_checkpoint.read_bytes()
            ).hexdigest()
            fold_zero_receipt["model_state_sha256"] = (
                "992a6f9b109d2cb38dc4bbe9ff9da300667882805adb9b995b5ac4c2efb078e7"
            )
            fold_zero_receipt_path.write_text(
                json.dumps(fold_zero_receipt),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "TRIFUSION_CONTRACT_TESTING": "0",
                    "TRIFUSION_CIRC_TEST_EXECUTABLE": "",
                },
            ), self.assertRaisesRegex(ValueError, "receipt contract failed"):
                main(
                    [
                        "--config",
                        str(orchestration_path),
                        "--mode",
                        "development",
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
