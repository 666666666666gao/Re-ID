from __future__ import annotations

import hashlib
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import signal
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


class CompletionRepairTests(unittest.TestCase):
    def _failed_state(self, root: Path) -> tuple[Path, Path, Path, Path]:
        output = root / "output"
        failed_ledger = root / "ledger" / "launch-0002"
        runner = root / "run_trifusion_experiment.py"
        config = root / "config.yml"
        runner.write_bytes(b"frozen runner\n")
        config.write_bytes(b"EXPERIMENT:\n  SEED: 42\n")
        checkpoint = output / "best_dev_model.pth"
        generation = output / ".resume/generation-0060-complete.pt"
        checkpoint.parent.mkdir(parents=True)
        generation.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"best checkpoint")
        identity = {
            "data_mode": "development",
            "variant": "trifusion_circ_urgc",
            "runner_sha256": _sha256(runner),
            "config_sha256": _sha256(config),
            "scientific_evidence_eligible": True,
            "contract_testing": False,
            "official_test_access_during_development": False,
        }
        _write_json(output / "run_identity.json", identity)
        torch.save(
            {
                "epoch": 60,
                "phase": "complete",
                "dev_evaluation_count": 60,
                "best_epoch": 36,
                "best_map": 47.4,
                "best_checkpoint_sha256": _sha256(checkpoint),
                "run_identity_sha256": _sha256(output / "run_identity.json"),
            },
            generation,
        )
        _write_json(
            output / ".resume/latest.json",
            {
                "epoch": 60,
                "phase": "complete",
                "run_identity_sha256": _sha256(output / "run_identity.json"),
                "current": {
                    "path": ".resume/generation-0060-complete.pt",
                    "sha256": _sha256(generation),
                },
                "completion_evidence": {
                    "epoch": 60,
                    "phase": "complete",
                    "best_epoch": 36,
                    "best_map": 47.4,
                    "best_checkpoint_sha256": _sha256(checkpoint),
                    "scientific_evidence_eligible": True,
                },
            },
        )
        _write_json(
            output / "best_dev_receipt.json",
            {
                "epoch": 36,
                "dev_selection_mAP": 47.4,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "official_test_access_count": 0,
                "scientific_evidence_eligible": True,
            },
        )
        _write_json(
            output / "dev_worker_result.json",
            {
                "status": "FAILED",
                "error": MISSING_LOADER_ERROR,
                "official_test_access_count": 0,
            },
        )
        (output / "dev_worker.log").write_text(
            "missing import failure\n", encoding="utf-8"
        )
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
            failed_ledger / "completion_receipt.json",
            {
                "status": "FAIL",
                "returncode": 2,
                "official_test_access_count": 0,
            },
        )
        _write_json(
            failed_ledger / "prelaunch_receipt.json",
            {
                "status": "AUTHORIZED",
                "variant": "trifusion_circ_urgc",
                "official_test_access_count": 0,
            },
        )
        (failed_ledger / "runner.log").write_text(
            "frozen runner reported failure\n", encoding="utf-8"
        )
        return output, failed_ledger, runner, config

    def test_accepts_only_exact_post_training_missing_import_failure(self) -> None:
        from tools.repair_trifusion_completion import validate_failed_state

        with tempfile.TemporaryDirectory() as temporary_directory:
            output, failed_ledger, runner, config = self._failed_state(
                Path(temporary_directory)
            )
            snapshot = validate_failed_state(
                output_dir=output,
                failed_ledger_entry=failed_ledger,
                runner_path=runner,
                config_path=config,
            )
            self.assertEqual(snapshot["failed_symbol"], "build_rgbnt201_record_eval_loader")
            self.assertEqual(snapshot["recovery_epoch"], 60)
            self.assertEqual(snapshot["best_epoch"], 36)
            self.assertFalse(snapshot["training_reexecution_allowed"])

    def test_rejects_incomplete_recovery_and_any_official_test_access(self) -> None:
        from tools.repair_trifusion_completion import validate_failed_state

        with tempfile.TemporaryDirectory() as temporary_directory:
            output, failed_ledger, runner, config = self._failed_state(
                Path(temporary_directory)
            )
            latest_path = output / ".resume/latest.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            latest["phase"] = "post_eval"
            _write_json(latest_path, latest)
            with self.assertRaisesRegex(ValueError, "complete recovery endpoint"):
                validate_failed_state(
                    output_dir=output,
                    failed_ledger_entry=failed_ledger,
                    runner_path=runner,
                    config_path=config,
                )

            latest["phase"] = "complete"
            _write_json(latest_path, latest)
            generation_path = output / latest["current"]["path"]
            generation = torch.load(
                generation_path, map_location="cpu", weights_only=False
            )
            generation["phase"] = "post_eval"
            torch.save(generation, generation_path)
            latest["current"]["sha256"] = _sha256(generation_path)
            _write_json(latest_path, latest)
            with self.assertRaisesRegex(ValueError, "generation-internal"):
                validate_failed_state(
                    output_dir=output,
                    failed_ledger_entry=failed_ledger,
                    runner_path=runner,
                    config_path=config,
                )

            generation["phase"] = "complete"
            torch.save(generation, generation_path)
            latest["current"]["sha256"] = _sha256(generation_path)
            _write_json(latest_path, latest)
            worker_path = output / "dev_worker_result.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["official_test_access_count"] = 1
            _write_json(worker_path, worker)
            with self.assertRaisesRegex(ValueError, "official test"):
                validate_failed_state(
                    output_dir=output,
                    failed_ledger_entry=failed_ledger,
                    runner_path=runner,
                    config_path=config,
                )

    def test_canonical_binding_rejects_a_self_reported_alternate_root(self) -> None:
        from tools.repair_trifusion_completion import _validate_canonical_bindings

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "noncanonical paths"):
                _validate_canonical_bindings(
                    output_dir=root / "output",
                    failed_ledger_entry=root / "ledger/launch-0002",
                    ledger_dir=root / "ledger",
                    runner_path=root / "runner.py",
                    config_path=root / "config.yml",
                )

    def test_rollback_restores_failed_files_and_moves_generated_audit(self) -> None:
        from tools.repair_trifusion_completion import _restore_failed_output

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            entry = root / "repair-0001"
            snapshots = entry / "failed_snapshots"
            snapshots.mkdir(parents=True)
            destinations = {
                "dev_worker_result.json": output / "dev_worker_result.json",
                "dev_worker.log": output / "dev_worker.log",
                "run_summary.json": output / "run_summary.json",
                "best_dev_receipt.json": output / "best_dev_receipt.json",
                "latest.json": output / ".resume/latest.json",
            }
            for name, destination in destinations.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                (snapshots / name).write_bytes(f"original:{name}".encode())
                destination.write_bytes(f"mutated:{name}".encode())
            generated = output / "router_calibration_receipt.json"
            generated.write_bytes(b"generated audit")
            _restore_failed_output(output, entry)
            for name, destination in destinations.items():
                self.assertEqual(destination.read_bytes(), (snapshots / name).read_bytes())
            self.assertFalse(generated.exists())
            self.assertEqual(
                (entry / "rolled_back_generated/router_calibration_receipt.json").read_bytes(),
                b"generated audit",
            )

    def test_final_verifier_rejects_any_failure_receipt(self) -> None:
        from tools.repair_trifusion_completion import verify_repair

        with tempfile.TemporaryDirectory() as temporary_directory:
            entry = Path(temporary_directory) / "repair-0001"
            _write_json(entry / "failure_receipt.json", {"status": "FAIL"})
            with self.assertRaisesRegex(ValueError, "failure receipt"):
                verify_repair(entry)

    def test_repair_installs_sigint_and_sigterm_rollback_handlers(self) -> None:
        from tools.repair_trifusion_completion import (
            _install_repair_signal_handlers,
            _restore_signal_handlers,
        )

        previous = _install_repair_signal_handlers()
        try:
            self.assertEqual(set(previous), {signal.SIGINT, signal.SIGTERM})
        finally:
            _restore_signal_handlers(previous)

    def test_frozen_runner_import_accepts_text_warning_output(self) -> None:
        from tools.repair_trifusion_completion import (
            EXPECTED_RUNNER,
            MISSING_LOADER_SYMBOL,
            _load_frozen_runner,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "repair.log"
            with log_path.open("x", encoding="utf-8") as handle, redirect_stdout(
                handle
            ), redirect_stderr(handle):
                module = _load_frozen_runner(EXPECTED_RUNNER)
            self.assertTrue(hasattr(module, MISSING_LOADER_SYMBOL))

    def test_worker_failure_message_preserves_the_pre_rollback_error(self) -> None:
        from tools.repair_trifusion_completion import _worker_failure_message

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "dev_worker_result.json"
            _write_json(
                result_path,
                {"status": "FAILED", "error": "ValueError: exact audit defect"},
            )
            self.assertEqual(
                _worker_failure_message(result_path, 4),
                "completion repair audit worker returned 4: "
                "ValueError: exact audit defect",
            )

    def test_audit_path_adapter_preserves_complete_sample_keys(self) -> None:
        from tools.repair_trifusion_completion import _adapt_audit_loader_paths

        class FakeLoader:
            @staticmethod
            def collate_fn(_batch: object) -> tuple[object, ...]:
                return (
                    "images",
                    "pids",
                    "camids",
                    "camids_batch",
                    "viewids",
                    ("000159_cam3_1_00.jpg", "000160_cam1_1_00.jpg"),
                )

        loader = _adapt_audit_loader_paths(FakeLoader())
        collated = loader.collate_fn(None)
        self.assertEqual(
            collated[-1],
            (("000159_cam3_1_00.jpg",), ("000160_cam1_1_00.jpg",)),
        )


if __name__ == "__main__":
    unittest.main()
