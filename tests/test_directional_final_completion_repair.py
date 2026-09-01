from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


MISSING_LOADER_ERROR = (
    "NameError: name 'build_rgbnt201_record_eval_loader' is not defined"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class DirectionalFinalCompletionRepairTests(unittest.TestCase):
    def _failed_state(self, root: Path) -> tuple[Path, Path, Path, Path]:
        output = root / "output"
        failed_entry = root / "ledger" / "launch-0001"
        runner = root / "run_trifusion_experiment.py"
        config = root / "config.yml"
        runner.write_bytes(b"frozen final runner\n")
        config.write_bytes(b"EXPERIMENT:\n  SEED: 42\n")

        checkpoint = output / "fixed_final_model.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"epoch-60 checkpoint")
        metrics = {
            "schema_version": "trifusion-official-fixed-v1",
            "fixed_epoch": 60,
            "official_test_access_count": 1,
            "official_test_evaluation_count": 1,
            "further_model_selection": False,
            "checkpoint_sha256": _sha256(checkpoint),
            "run_identity_sha256": "pending",
            "metrics_percent": {
                "fused": {"mAP": 59.1478, "Rank-1": 63.2775},
                "cnn": {"mAP": 59.1561, "Rank-1": 63.7560},
                "transformer": {"mAP": 59.1219, "Rank-1": 62.6794},
                "mamba": {"mAP": 58.8748, "Rank-1": 62.4402},
            },
        }
        identity = {
            "data_mode": "postfreeze-final",
            "variant": "trifusion_circ_urgc",
            "runner_sha256": _sha256(runner),
            "config_sha256": _sha256(config),
            "scientific_evidence_eligible": True,
            "contract_testing": False,
            "official_test_access_during_development": False,
        }
        _write_json(output / "run_identity.json", identity)
        identity_sha = _sha256(output / "run_identity.json")
        metrics["run_identity_sha256"] = identity_sha
        _write_json(output / "official_test_metrics.json", metrics)
        metrics_sha = _sha256(output / "official_test_metrics.json")
        _write_json(
            output / "official_test_access_guard.json",
            {
                "schema_version": "trifusion-official-access-guard-v1",
                "status": "COMPLETE",
                "fixed_epoch": 60,
                "official_test_access_count": 1,
                "checkpoint_sha256": _sha256(checkpoint),
                "metrics_sha256": metrics_sha,
                "run_identity_sha256": identity_sha,
            },
        )
        guard_sha = _sha256(output / "official_test_access_guard.json")
        generation = output / ".resume/generation-0060-complete.pt"
        generation.parent.mkdir(parents=True)
        torch.save(
            {
                "epoch": 60,
                "phase": "complete",
                "best_epoch": 60,
                "best_map": metrics["metrics_percent"]["fused"]["mAP"],
                "best_metrics": metrics["metrics_percent"],
                "best_checkpoint_sha256": _sha256(checkpoint),
                "dev_evaluation_count": 0,
                "eval_loader_iteration_count": 1,
                "run_identity_sha256": identity_sha,
            },
            generation,
        )
        _write_json(
            output / ".resume/latest.json",
            {
                "epoch": 60,
                "phase": "complete",
                "run_identity_sha256": identity_sha,
                "current": {
                    "path": ".resume/generation-0060-complete.pt",
                    "sha256": _sha256(generation),
                },
                "completion_evidence": {
                    "kind": "postfreeze-final-fixed",
                    "epoch": 60,
                    "phase": "complete",
                    "official_test_evaluation_count": 1,
                    "further_model_selection": False,
                    "fixed_metrics": metrics["metrics_percent"],
                    "fixed_checkpoint_sha256": _sha256(checkpoint),
                    "official_metrics_receipt_sha256": metrics_sha,
                    "official_access_guard_sha256": guard_sha,
                    "scientific_evidence_eligible": True,
                },
            },
        )
        _write_json(
            output / "final_worker_result.json",
            {
                "status": "FAILED",
                "error": MISSING_LOADER_ERROR,
                "official_test_access_count": 1,
            },
        )
        (output / "final_worker.log").write_text("missing import\n", encoding="utf-8")
        _write_json(
            output / "run_summary.json",
            {
                "status": "FAILED",
                "blockers": ["dev_worker_failed"],
                "worker_returncode": 4,
                "official_test_access_count": 0,
            },
        )
        _write_json(
            failed_entry / "prelaunch_receipt.json",
            {"status": "AUTHORIZED", "official_test_access_count": 0},
        )
        (failed_entry / "launcher.log").write_text("runner failed\n", encoding="utf-8")
        _write_json(
            failed_entry / "failure_receipt.json",
            {
                "status": "FAIL",
                "error": "frozen final runner failed with return code 2",
                "official_test_access_count": 1,
                "official_test_access_ambiguous": False,
                "official_test_metrics": {
                    "exists": True,
                    "sha256": metrics_sha,
                },
                "official_test_guard": {
                    "exists": True,
                    "sha256": guard_sha,
                },
                "final_worker_result": {
                    "exists": True,
                    "sha256": _sha256(output / "final_worker_result.json"),
                },
            },
        )
        return output, failed_entry, runner, config

    def test_accepts_exact_post_official_missing_import_failure(self) -> None:
        from tools.repair_trifusion_directional_final_completion import (
            validate_failed_state,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output, failed_entry, runner, config = self._failed_state(
                Path(temporary_directory)
            )
            snapshot = validate_failed_state(
                output_dir=output,
                failed_ledger_entry=failed_entry,
                runner_path=runner,
                config_path=config,
            )
            self.assertEqual(snapshot["recovery_epoch"], 60)
            self.assertEqual(snapshot["official_test_access_count"], 1)
            self.assertEqual(snapshot["official_test_evaluation_count"], 1)
            self.assertFalse(snapshot["training_reexecution_allowed"])
            self.assertFalse(snapshot["official_test_reexecution_allowed"])

    def test_rejects_metric_or_access_drift(self) -> None:
        from tools.repair_trifusion_directional_final_completion import (
            validate_failed_state,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output, failed_entry, runner, config = self._failed_state(
                Path(temporary_directory)
            )
            metrics_path = output / "official_test_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["official_test_evaluation_count"] = 2
            _write_json(metrics_path, metrics)
            with self.assertRaisesRegex(ValueError, "official metrics"):
                validate_failed_state(
                    output_dir=output,
                    failed_ledger_entry=failed_entry,
                    runner_path=runner,
                    config_path=config,
                )

    def test_rejects_noncomplete_recovery_or_different_failure(self) -> None:
        from tools.repair_trifusion_directional_final_completion import (
            validate_failed_state,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output, failed_entry, runner, config = self._failed_state(
                Path(temporary_directory)
            )
            worker_path = output / "final_worker_result.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["error"] = "RuntimeError: unrelated failure"
            _write_json(worker_path, worker)
            with self.assertRaisesRegex(ValueError, "missing-import failure"):
                validate_failed_state(
                    output_dir=output,
                    failed_ledger_entry=failed_entry,
                    runner_path=runner,
                    config_path=config,
                )

    def test_official_evaluation_guard_always_raises(self) -> None:
        from tools.repair_trifusion_directional_final_completion import (
            _forbidden_official_evaluation,
        )

        with self.assertRaisesRegex(RuntimeError, "forbids official test re-evaluation"):
            _forbidden_official_evaluation()

    def test_summary_context_reuses_directional_authorization(self) -> None:
        from tools.repair_trifusion_directional_final_completion import (
            _directional_summary_authorization,
        )

        class FakeRunner:
            @staticmethod
            def _preflight(*_args: object, **_kwargs: object) -> dict:
                return {"status": "BLOCKED"}

            @staticmethod
            def _run_identity(preflight: dict, variant: str) -> dict:
                return {"status": preflight["status"], "variant": variant}

        runner = FakeRunner()
        original_preflight = runner._preflight
        original_identity = runner._run_identity
        authorized = {"status": "READY", "launch_allowed": True}
        evidence = {"authorization_sha256": "a" * 64}
        with _directional_summary_authorization(runner, authorized, evidence):
            self.assertEqual(runner._preflight(), authorized)
            identity = runner._run_identity(authorized, "trifusion_circ_urgc")
            self.assertEqual(identity["directional_authorization"], evidence)
            self.assertEqual(
                identity["scientific_evidence_scope"],
                "calibrated_directional_training_input",
            )
            self.assertFalse(identity["query_gallery_symmetry_claim_eligible"])
            self.assertTrue(identity["calibration_claim_eligible"])
        self.assertIs(runner._preflight, original_preflight)
        self.assertIs(runner._run_identity, original_identity)

    def test_verifier_rejects_failure_receipt_in_repair_entry(self) -> None:
        from tools.repair_trifusion_directional_final_completion import verify_repair

        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = Path(temporary_directory) / "repair-0001"
            _write_json(entry / "failure_receipt.json", {"status": "FAIL"})
            with self.assertRaisesRegex(ValueError, "failure receipt"):
                verify_repair(entry)


if __name__ == "__main__":
    unittest.main()
