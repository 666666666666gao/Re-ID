from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


class AMPSafeSitecustomizeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_probability_bce_is_safe_inside_cuda_autocast(self) -> None:
        project = Path(__file__).resolve().parents[1]
        patch_path = project / "tools/runtime_amp_safe/sitecustomize.py"
        original = F.binary_cross_entropy
        try:
            spec = importlib.util.spec_from_file_location(
                "trifusion_amp_safe_sitecustomize",
                patch_path,
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            predicted = torch.tensor([0.25, 0.75], device="cuda")
            target = torch.tensor([0.0, 1.0], device="cuda")
            with torch.cuda.amp.autocast(enabled=True):
                loss = F.binary_cross_entropy(predicted, target)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(
                getattr(F.binary_cross_entropy, "_trifusion_amp_safe", False)
            )
        finally:
            F.binary_cross_entropy = original


if __name__ == "__main__":
    unittest.main()
