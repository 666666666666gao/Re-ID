from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class CIRCMeanOnlyOverlayTests(unittest.TestCase):
    def test_overlay_preserves_failed_calibration_and_authorizes_mean_only_training(
        self,
    ) -> None:
        from tools.authorize_circ_mean_only import (
            build_overlay,
            verify_overlay,
            verify_training_config,
        )
        from tools.run_trifusion_mean_only import (
            _completion_payload,
            validate_launch_request,
            verify_completion,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            parent = root / "parent"
            cache = parent / "cache"
            cache.mkdir(parents=True)
            targets = b'{"sample_key":"query-1"}\n'
            (cache / "targets.jsonl").write_bytes(targets)
            failed_audit = {
                "schema_version": "circ-calibration-audit-v1",
                "status": "COMPLETE",
                "empirical_concentration_coverage": {
                    "clean": {
                        "claim_eligible": False,
                        "empirical_coverage": 0.75,
                    }
                },
            }
            cache_receipt = {
                "protocol_hash": "ab" * 32,
                "row_count": 1,
                "cross_camera_primary_rows": 1,
                "same_camera_only_rows": 0,
                "invalid_support_rows": 0,
                "zero_identity_overlap": True,
                "official_test_access_count": 0,
                "targets_sha256": hashlib.sha256(targets).hexdigest(),
                "calibration_audit": failed_audit,
            }
            _write_json(cache / "receipt.json", cache_receipt)
            _write_json(
                cache / "calibration_receipt.json",
                {**failed_audit, "protocol_hash": "ab" * 32},
            )
            _write_json(cache / "symmetry_receipt.json", {"status": "PASS"})
            _write_json(
                cache / "target_transfer_receipt.json",
                {"status": "PENDING_DEPLOYED_MODEL"},
            )
            _write_json(
                parent / "run_identity.json",
                {"official_test_access_count": 0},
            )
            (parent / "scored_source.json").write_bytes(b"immutable-source\n")
            scoring = {
                "status": "COMPLETE",
                "contract_testing": False,
                "official_test_access_count": 0,
                "scientific_evidence_eligible": False,
                "calibration_claim_eligible": False,
                "targets_sha256": cache_receipt["targets_sha256"],
                "calibration_audit": failed_audit,
                "symmetry_audit": {"status": "PASS", "claim_eligible": True},
            }
            _write_json(parent / "scoring_receipt.json", scoring)
            artifact_names = (
                "scoring_receipt.json",
                "run_identity.json",
                "scored_source.json",
                "cache/receipt.json",
                "cache/calibration_receipt.json",
                "cache/targets.jsonl",
                "cache/symmetry_receipt.json",
                "cache/target_transfer_receipt.json",
            )
            protocol = root / "mean-only-protocol.json"
            _write_json(
                protocol,
                {
                    "schema_version": "circ-mean-only-overlay-protocol-v1",
                    "official_test_access_count": 0,
                    "parent_circ_protocol_sha256": "ab" * 32,
                    "parent_artifacts_sha256": {
                        name: _sha256(parent / name) for name in artifact_names
                    },
                    "policy": {
                        "posterior_mean_control_only": True,
                        "uncertainty_control_allowed": False,
                        "concentration_claim_allowed": False,
                        "parent_calibration_failure_must_be_preserved": True,
                    },
                },
            )
            output = root / "overlay"
            project = Path(__file__).resolve().parents[1]
            launcher = project / "tools/run_trifusion_mean_only.py"
            runner = project / "tools/run_trifusion_experiment.py"
            amp_safe = project / "tools/runtime_amp_safe/sitecustomize.py"
            dev_output = root / "dev-output"
            preflight_output = root / "preflight-output"
            ledger = root / "launch-ledger"

            config = root / "train.json"
            config_payload = {
                "EXPERIMENT": {"SEED": 42},
                "OPTIMIZATION": {"AMP": True},
                "LOSS": {"EVIDENCE_WEIGHT": 0.0},
                "CIRC": {
                    "TARGET_CACHE": str(output / "cache"),
                    "SCORING_RECEIPT": str(output / "scoring_receipt.json"),
                },
                "PROTOCOL": {
                    "CIRC_MEAN_ONLY_OVERLAY": str(protocol),
                    "CIRC_MEAN_ONLY_OVERLAY_SHA256": _sha256(protocol),
                    "CIRC_SCIENTIFIC_EVIDENCE_SCOPE": "mean_only_training_input",
                    "CIRC_UNCERTAINTY_CONTROL_ALLOWED": False,
                    "OFFICIAL_TEST_DURING_DEVELOPMENT": False,
                },
                "EXECUTION": {
                    "MEAN_ONLY_LAUNCHER": str(launcher),
                    "MEAN_ONLY_LAUNCHER_SHA256": _sha256(launcher),
                    "FROZEN_RUNNER": str(runner),
                    "FROZEN_RUNNER_SHA256": _sha256(runner),
                    "AMP_SAFE_SITECUSTOMIZE": str(amp_safe),
                    "AMP_SAFE_SITECUSTOMIZE_SHA256": _sha256(amp_safe),
                    "MEAN_ONLY_DEV_OUTPUT_DIR": str(dev_output),
                    "MEAN_ONLY_PREFLIGHT_OUTPUT_DIR": str(preflight_output),
                    "MEAN_ONLY_LEDGER_DIR": str(ledger),
                    "REQUIRE_COMPLETION_RECEIPT": True,
                    "RESULT_QUALIFICATION_GATE": "mean_only_completion_receipt",
                },
            }
            _write_json(config, config_payload)

            build_overlay(
                parent_root=parent,
                protocol_path=protocol,
                output_root=output,
                config_path=config,
            )
            receipt = verify_overlay(
                parent_root=parent,
                protocol_path=protocol,
                output_root=output,
                config_path=config,
            )

            config_evidence = verify_training_config(
                config_path=config,
                protocol_path=protocol,
                output_root=output,
            )
            self.assertEqual(config_evidence["seed"], 42)
            self.assertEqual(config_evidence["evidence_weight"], 0.0)
            launch = validate_launch_request(
                mode="dev",
                config_path=config,
                parent_root=parent,
                protocol_path=protocol,
                overlay_root=output,
                output_dir=dev_output,
                ledger_dir=ledger,
            )
            self.assertEqual(launch["variant"], "trifusion_circ_urgc")
            self.assertEqual(launch["runner_sha256"], _sha256(runner))
            self.assertEqual(
                launch["amp_safe_sitecustomize_sha256"],
                _sha256(amp_safe),
            )
            self.assertEqual(
                launch["result_qualification_gate"],
                "mean_only_completion_receipt",
            )
            with self.assertRaisesRegex(ValueError, "launch validation failed"):
                validate_launch_request(
                    mode="dev",
                    config_path=config,
                    parent_root=parent,
                    protocol_path=protocol,
                    overlay_root=output,
                    output_dir=root / "unregistered-output",
                    ledger_dir=ledger,
                )

            dev_output.mkdir()
            identity_path = dev_output / "run_identity.json"
            identity_payload = {
                "data_mode": "development",
                "variant": "trifusion_circ_urgc",
                "runner_sha256": _sha256(runner),
                "config_sha256": launch["config_sha256"],
                "scientific_evidence_eligible": True,
                "contract_testing": False,
                "official_test_access_during_development": False,
            }
            _write_json(
                identity_path,
                identity_payload,
            )
            current = dev_output / ".resume/current.pth"
            current.parent.mkdir()
            current.write_bytes(b"checkpoint")
            latest_path = dev_output / ".resume/latest.json"
            latest_payload = {
                "epoch": 60,
                "phase": "complete",
                "run_identity_sha256": _sha256(identity_path),
                "current": {
                    "path": ".resume/current.pth",
                    "sha256": _sha256(current),
                },
            }
            _write_json(latest_path, latest_payload)
            _write_json(
                dev_output / "run_summary.json",
                {
                    "status": "PASS",
                    "phase": "complete",
                    "scientific_evidence_eligible": True,
                    "official_test_access_count": 0,
                },
            )
            entry = ledger / "launch-0001"
            entry.mkdir(parents=True)
            _write_json(
                entry / "prelaunch_receipt.json",
                {
                    **launch,
                    "schema_version": "trifusion-mean-only-prelaunch-v1",
                    "status": "AUTHORIZED",
                    "started_at_utc": "2026-09-01T00:00:00+00:00",
                },
            )
            (entry / "runner.log").write_bytes(b"complete\n")
            valid_completion = _completion_payload(
                validation=launch,
                entry=entry,
                returncode=0,
            )
            _write_json(entry / "completion_receipt.json", valid_completion)
            self.assertTrue(verify_completion(entry)["scientific_evidence_eligible"])

            for field, bad_value in (
                ("artifacts", {}),
                ("mode", "preflight"),
                ("returncode", 7),
            ):
                tampered_completion = dict(valid_completion)
                tampered_completion[field] = bad_value
                _write_json(entry / "completion_receipt.json", tampered_completion)
                with self.assertRaisesRegex(ValueError, "completion verification failed"):
                    verify_completion(entry)
            _write_json(entry / "completion_receipt.json", valid_completion)

            prelaunch = json.loads((entry / "prelaunch_receipt.json").read_text())
            prelaunch["status"] = "FORGED"
            _write_json(entry / "prelaunch_receipt.json", prelaunch)
            with self.assertRaisesRegex(ValueError, "completion verification failed"):
                verify_completion(entry)
            prelaunch["status"] = "AUTHORIZED"
            _write_json(entry / "prelaunch_receipt.json", prelaunch)
            valid_completion["prelaunch_receipt_sha256"] = _sha256(
                entry / "prelaunch_receipt.json"
            )
            _write_json(entry / "completion_receipt.json", valid_completion)

            latest_path.unlink()
            with self.assertRaisesRegex(ValueError, "completion verification failed"):
                verify_completion(entry)
            _write_json(latest_path, latest_payload)

            for field, bad_value in (
                ("config_sha256", "00" * 32),
                ("runner_sha256", "11" * 32),
                ("variant", "forged_variant"),
            ):
                forged_identity = dict(identity_payload)
                forged_identity[field] = bad_value
                _write_json(identity_path, forged_identity)
                forged_latest = dict(latest_payload)
                forged_latest["run_identity_sha256"] = _sha256(identity_path)
                _write_json(latest_path, forged_latest)
                forged_completion = _completion_payload(
                    validation=launch,
                    entry=entry,
                    returncode=0,
                )
                _write_json(entry / "completion_receipt.json", forged_completion)
                with self.assertRaisesRegex(ValueError, "completion verification failed"):
                    verify_completion(entry)
            _write_json(identity_path, identity_payload)
            _write_json(latest_path, latest_payload)

            for name in artifact_names[3:]:
                self.assertEqual((parent / name).read_bytes(), (output / name).read_bytes())
            promoted = json.loads((output / "scoring_receipt.json").read_text())
            self.assertFalse(promoted["calibration_claim_eligible"])
            self.assertFalse(promoted["concentration_claim_eligible"])
            self.assertFalse(promoted["uncertainty_control_allowed"])
            self.assertTrue(promoted["mean_only_training_eligible"])
            self.assertTrue(promoted["scientific_evidence_eligible"])
            self.assertEqual(promoted["calibration_audit"], failed_audit)
            self.assertEqual(receipt["official_test_access_count"], 0)

            promoted["concentration_claim_eligible"] = True
            _write_json(output / "scoring_receipt.json", promoted)
            with self.assertRaisesRegex(ValueError, "verification failed"):
                verify_overlay(
                    parent_root=parent,
                    protocol_path=protocol,
                    output_root=output,
                    config_path=config,
                )

            config_payload["LOSS"]["EVIDENCE_WEIGHT"] = 0.1
            _write_json(config, config_payload)
            with self.assertRaisesRegex(ValueError, "configuration verification failed"):
                verify_training_config(
                    config_path=config,
                    protocol_path=protocol,
                    output_root=output,
                )


if __name__ == "__main__":
    unittest.main()
