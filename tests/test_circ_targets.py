from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
