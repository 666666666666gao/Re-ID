from __future__ import annotations

import copy
import json
import unittest


class PeftBoaSourceAuditTests(unittest.TestCase):
    @staticmethod
    def _valid_receipt() -> dict[str, object]:
        return {
            "source_config": {
                "checkpoint_period": 60,
                "dataset": "RGBNT201",
                "eval_period": 1,
                "frozen": True,
                "max_epochs": 120,
                "num_instance": 4,
                "reranking": "no",
                "seed": 1111,
                "size_test": [256, 128],
                "size_train": [256, 128],
                "test_batch": 64,
                "train_batch": 64,
            },
            "effective_smoke_batch": 32,
            "smoke_seed": 42,
            "num_query": 836,
            "num_classes": 171,
            "num_cameras": 4,
            "num_views": 1,
            "rgb_shape": [32, 3, 256, 128],
            "nir_shape": [32, 3, 256, 128],
            "tir_shape": [32, 3, 256, 128],
            "pid_shape": [32],
            "pid_unique": 8,
            "pretrain_exists": True,
        }

    def test_parse_loader_receipt_is_fail_closed(self) -> None:
        from tools.audit_peft_boa_source import parse_loader_receipt

        receipt = self._valid_receipt()
        stdout = (
            "upstream diagnostic\n"
            "PEFT_BOA_LOADER_RECEIPT="
            + json.dumps(receipt, sort_keys=True)
            + "\n"
        )
        self.assertEqual(parse_loader_receipt(stdout), receipt)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_loader_receipt("no receipt\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_loader_receipt(stdout + stdout)
        with self.assertRaisesRegex(ValueError, "valid JSON object"):
            parse_loader_receipt("PEFT_BOA_LOADER_RECEIPT=[]\n")

    def test_probe_contract_rejects_protocol_or_shape_drift(self) -> None:
        from tools.audit_peft_boa_source import validate_probe_receipt

        receipt = self._valid_receipt()
        self.assertEqual(validate_probe_receipt(receipt), [])

        drifted = copy.deepcopy(receipt)
        drifted["source_config"]["max_epochs"] = 50
        drifted["source_config"]["reranking"] = "yes"
        drifted["smoke_seed"] = 7
        drifted["rgb_shape"] = [16, 3, 256, 128]
        drifted["pretrain_exists"] = False
        errors = validate_probe_receipt(drifted)
        self.assertTrue(any("max_epochs" in error for error in errors))
        self.assertTrue(any("reranking" in error for error in errors))
        self.assertTrue(any("smoke_seed" in error for error in errors))
        self.assertTrue(any("rgb_shape" in error for error in errors))
        self.assertTrue(any("pretrain_exists" in error for error in errors))

    def test_upstream_test_selection_risk_requires_both_code_signals(self) -> None:
        from tools.audit_peft_boa_source import detect_protocol_risks

        processor = """
        if epoch % eval_period == 0:
            cmc, mAP, *_ = evaluator.compute()
            if mAP >= best_index['mAP']:
                torch.save(model.state_dict(), 'BoAbest.pth')
        """
        risks = detect_protocol_risks(processor, eval_period=1)
        self.assertTrue(risks["official_test_each_epoch"])
        self.assertTrue(risks["test_selected_best_checkpoint"])

        missing_best = detect_protocol_risks(
            "if epoch % eval_period == 0: evaluator.compute()",
            eval_period=1,
        )
        self.assertTrue(missing_best["official_test_each_epoch"])
        self.assertFalse(missing_best["test_selected_best_checkpoint"])

    def test_source_and_pretrain_provenance_are_fail_closed(self) -> None:
        from tools.audit_peft_boa_source import (
            EXPECTED_PRETRAIN_BYTES,
            EXPECTED_PRETRAIN_SHA256,
            EXPECTED_REMOTE,
            validate_pretrain_identity,
            validate_source_provenance,
        )

        commit = "d2b198be634ac4f9f5744eebf6e0a6604e490deb"
        self.assertEqual(
            validate_source_provenance(
                actual_commit=commit,
                expected_commit=commit,
                remote=EXPECTED_REMOTE,
                status_before="",
                status_after="",
            ),
            [],
        )
        source_errors = validate_source_provenance(
            actual_commit="0" * 40,
            expected_commit=commit,
            remote="https://example.invalid/fork.git",
            status_before="?? cache.bin",
            status_after=" M README.md",
        )
        self.assertEqual(len(source_errors), 4)
        self.assertTrue(any("source_commit" in error for error in source_errors))
        self.assertTrue(any("source_remote" in error for error in source_errors))

        self.assertEqual(
            validate_pretrain_identity(
                actual_bytes=EXPECTED_PRETRAIN_BYTES,
                actual_sha256=EXPECTED_PRETRAIN_SHA256,
            ),
            [],
        )
        pretrain_errors = validate_pretrain_identity(
            actual_bytes=1,
            actual_sha256="f" * 64,
        )
        self.assertEqual(len(pretrain_errors), 2)


if __name__ == "__main__":
    unittest.main()
