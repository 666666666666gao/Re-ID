from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


class RunDemoResumableTests(unittest.TestCase):
    def test_long_run_requires_a_hash_bound_parity_reference(self) -> None:
        from tools.run_demo_resumable import validate_parity_gate_options

        with self.assertRaisesRegex(ValueError, "frozen parity epoch is 10"):
            validate_parity_gate_options(
                max_epochs=50,
                parity_epoch=51,
                reference_path=None,
                expected_sha256=None,
            )

        with self.assertRaisesRegex(RuntimeError, "parity reference is required"):
            validate_parity_gate_options(
                max_epochs=50,
                parity_epoch=10,
                reference_path=None,
                expected_sha256=None,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            reference_path = Path(temporary_directory) / "DeMo_10.pth"
            torch.save({"weight": torch.tensor([1.0])}, reference_path)
            expected_sha256 = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                RuntimeError, "frozen parity reference SHA-256"
            ):
                validate_parity_gate_options(
                    max_epochs=50,
                    parity_epoch=10,
                    reference_path=reference_path,
                    expected_sha256=expected_sha256,
                )
            validated_path, validated_sha256 = validate_parity_gate_options(
                max_epochs=50,
                parity_epoch=10,
                reference_path=reference_path,
                expected_sha256=expected_sha256,
                required_reference_sha256=expected_sha256,
            )
            self.assertEqual(validated_path, reference_path.resolve())
            self.assertEqual(validated_sha256, expected_sha256)

    def test_tensor_parity_gate_rejects_one_changed_element(self) -> None:
        from tools.demo_resumable_training import (
            assert_checkpoint_tensor_parity,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.pth"
            candidate_path = root / "candidate.pth"
            torch.save(
                {
                    "weight": torch.tensor([[1.0, 2.0]]),
                    "counter": torch.tensor(3),
                },
                reference_path,
            )
            torch.save(
                {
                    "weight": torch.tensor([[1.0, 2.001]]),
                    "counter": torch.tensor(3),
                },
                candidate_path,
            )
            reference_sha256 = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "tensor mismatch.*weight"):
                assert_checkpoint_tensor_parity(
                    candidate_path=candidate_path,
                    reference_path=reference_path,
                    expected_reference_sha256=reference_sha256,
                )

    def test_runtime_descriptor_has_a_canonical_stable_digest(self) -> None:
        from tools.demo_resumable_training import runtime_descriptor_sha256

        descriptor = {
            "driver": "581.80",
            "env": {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        }
        self.assertEqual(
            runtime_descriptor_sha256(descriptor),
            "0e6407f5536ac5e991e74c7b86671a5f4b1985ff24902015af4eda529c8e1977",
        )

    def test_reproducibility_environment_must_match_the_fixed_seed(self) -> None:
        from tools.run_demo_resumable import (
            validate_reproducibility_environment,
        )

        with mock.patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "41",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED=42"):
                validate_reproducibility_environment(seed=42)

        with mock.patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        ):
            self.assertEqual(
                validate_reproducibility_environment(seed=42),
                {
                    "PYTHONHASHSEED": "42",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )


if __name__ == "__main__":
    unittest.main()
