from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest


def _blocked_preflight() -> dict:
    return {
        "status": "BLOCKED",
        "launch_allowed": False,
        "blockers": ["invalid_circ_target_cache_evidence"],
        "model_constructed": False,
        "training_started": False,
        "official_test_access_count": 0,
        "sota_claim_supported": False,
    }


def _ready_counterfactual() -> dict:
    receipt = _blocked_preflight()
    receipt.update({"status": "READY", "launch_allowed": True, "blockers": []})
    return receipt


def _failed_symmetry() -> dict:
    return {
        "status": "FAIL",
        "claim_eligible": False,
        "sample_rows": 128,
        "sign_agreement": 0.671875,
        "minimum_sign_agreement": 0.7,
        "spearman": 0.7433562992125984,
        "minimum_spearman": 0.5,
    }


def _valid_calibration() -> dict:
    return {
        "status": "COMPLETE",
        "empirical_concentration_coverage": {
            "clean": {"claim_eligible": True},
            "occlusion": {"claim_eligible": True},
        },
    }


def test_directional_authorization_preserves_failed_symmetry_claim() -> None:
    from tools.run_trifusion_directional_final import (
        SCIENTIFIC_EVIDENCE_SCOPE,
        _authorize_preflight,
    )

    authorized = _authorize_preflight(
        actual=_blocked_preflight(),
        counterfactual=_ready_counterfactual(),
        symmetry=_failed_symmetry(),
        calibration=_valid_calibration(),
        authorization_sha256="ab" * 32,
    )

    assert authorized["status"] == "READY"
    assert authorized["launch_allowed"] is True
    assert authorized["blockers"] == []
    assert authorized["scientific_evidence_scope"] == SCIENTIFIC_EVIDENCE_SCOPE
    assert authorized["query_gallery_symmetry_claim_eligible"] is False
    assert authorized["calibration_claim_eligible"] is True
    assert authorized["official_test_access_count"] == 0
    assert authorized["sota_claim_supported"] is False
    assert authorized["directional_authorization_sha256"] == "ab" * 32
    assert authorized["observed_failed_symmetry"] == _failed_symmetry()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda actual, _counter, _symmetry, _calibration: actual["blockers"].append("other"), "sole preflight blocker"),
        (lambda _actual, counter, _symmetry, _calibration: counter.update({"status": "BLOCKED", "launch_allowed": False}), "counterfactual preflight"),
        (lambda _actual, _counter, symmetry, _calibration: symmetry.update({"status": "PASS", "claim_eligible": True}), "registered failure"),
        (lambda _actual, _counter, _symmetry, calibration: calibration["empirical_concentration_coverage"]["clean"].update({"claim_eligible": False}), "calibration"),
    ],
)
def test_directional_authorization_is_fail_closed(mutation, message: str) -> None:
    from tools.run_trifusion_directional_final import _authorize_preflight

    actual = copy.deepcopy(_blocked_preflight())
    counterfactual = copy.deepcopy(_ready_counterfactual())
    symmetry = copy.deepcopy(_failed_symmetry())
    calibration = copy.deepcopy(_valid_calibration())
    mutation(actual, counterfactual, symmetry, calibration)

    with pytest.raises(ValueError, match=message):
        _authorize_preflight(
            actual=actual,
            counterfactual=counterfactual,
            symmetry=symmetry,
            calibration=calibration,
            authorization_sha256="ab" * 32,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _metrics() -> dict:
    return {
        name: {
            "mAP": 86.0 + index / 10,
            "Rank-1": 88.0 + index / 10,
            "Rank-5": 93.0 + index / 10,
            "Rank-10": 96.0 + index / 10,
        }
        for index, name in enumerate(("fused", "cnn", "transformer", "mamba"))
    }


def _build_valid_completion_chain(tmp_path: Path) -> tuple[Path, Path, dict]:
    from tools.run_trifusion_directional_final import (
        AMP_SAFE_SITECUSTOMIZE,
        FROZEN_RUNNER,
        FROZEN_RUNNER_SHA256,
        REQUIRED_PARENT_ARTIFACTS,
        SCIENTIFIC_EVIDENCE_SCOPE,
        _completion_payload,
    )
    from tools import run_trifusion_directional_final as launcher

    entry = tmp_path / "ledger" / "launch-0001"
    output = tmp_path / "output"
    entry.mkdir(parents=True)
    output.mkdir()
    (entry / "launcher.log").write_text("worker completed\n", encoding="utf-8")

    config_path = tmp_path / "config.yml"
    config_path.write_text("schema: test\n", encoding="utf-8")
    config_sha = _sha256(config_path)
    parent_root = tmp_path / "parent-root"
    parent_root.mkdir()
    parent_hashes = {}
    for name in REQUIRED_PARENT_ARTIFACTS:
        path = parent_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"parent:{name}\n".encode())
        parent_hashes[name] = _sha256(path)
    launcher_path = Path(launcher.__file__).resolve()
    authorization_path = tmp_path / "authorization-root.json"
    authorization = {
        "schema_version": "circ-directional-final-authorization-v1",
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "official_test_access_count": 0,
        "parent_root": str(parent_root.resolve()),
        "parent_artifacts_sha256": parent_hashes,
        "policy": {
            "calibrated_uncertainty_control_allowed": True,
            "query_gallery_symmetry_claim_allowed": False,
            "parent_symmetry_failure_must_be_preserved": True,
            "target_cache_must_remain_byte_identical": True,
            "further_model_selection": False,
            "official_test_exactly_once_after_fixed_endpoint": True,
        },
        "execution": {
            "config": str(config_path.resolve()),
            "config_sha256": config_sha,
            "runner": str(FROZEN_RUNNER.resolve()),
            "runner_sha256": FROZEN_RUNNER_SHA256,
            "launcher": str(launcher_path),
            "launcher_sha256": _sha256(launcher_path),
            "amp_safe_sitecustomize": str(AMP_SAFE_SITECUSTOMIZE.resolve()),
            "amp_safe_sitecustomize_sha256": _sha256(AMP_SAFE_SITECUSTOMIZE),
            "output_dir": str(output.resolve()),
            "ledger_dir": str(entry.parent.resolve()),
        },
    }
    _write_json(authorization_path, authorization)
    authorization_sha = _sha256(authorization_path)
    directional_evidence = {
        "authorization_path": str(authorization_path.resolve()),
        "authorization_sha256": authorization_sha,
        "parent_root": str(parent_root.resolve()),
        "parent_artifacts_sha256": parent_hashes,
        "parent_scoring_receipt_sha256": parent_hashes["scoring_receipt.json"],
        "parent_targets_sha256": parent_hashes["cache/targets.jsonl"],
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "official_test_access_count": 0,
    }
    prelaunch = {
        "schema_version": "trifusion-directional-final-prelaunch-v1",
        "status": "AUTHORIZED",
        "variant": "trifusion_circ_urgc",
        "config": str(config_path.resolve()),
        "config_sha256": config_sha,
        "runner": str(FROZEN_RUNNER.resolve()),
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "launcher": str(launcher_path),
        "launcher_sha256": _sha256(launcher_path),
        "amp_safe_sitecustomize": str(AMP_SAFE_SITECUSTOMIZE.resolve()),
        "amp_safe_sitecustomize_sha256": _sha256(AMP_SAFE_SITECUSTOMIZE),
        "output_dir": str(output.resolve()),
        "ledger_dir": str(entry.parent.resolve()),
        "recovery_before": {"valid": True, "kind": "fresh"},
        "directional_authorization": directional_evidence,
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "official_test_access_count": 0,
    }
    _write_json(entry / "prelaunch_receipt.json", prelaunch)
    identity = {
        "data_mode": "postfreeze-final",
        "variant": "trifusion_circ_urgc",
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "config_sha256": config_sha,
        "optimization": {"max_epochs": 60},
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "all_171_training_identities": True,
        "former_dev_identities_training_only": True,
        "further_model_selection": False,
        "official_test_evaluations_before_fixed_endpoint": 0,
        "official_test_evaluations_after_fixed_endpoint": 1,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
        "directional_authorization": directional_evidence,
    }
    _write_json(output / "run_identity.json", identity)
    identity_sha = _sha256(output / "run_identity.json")

    checkpoint = output / "fixed_final_model.pth"
    checkpoint.write_bytes(b"fixed model bytes")
    checkpoint_sha = _sha256(checkpoint)
    router = output / "router_calibration_receipt.json"
    _write_json(
        router,
        {
            "model_checkpoint": str(checkpoint),
            "model_checkpoint_sha256": checkpoint_sha,
            "official_test_access_count": 0,
        },
    )
    router_sha = _sha256(router)

    metrics = _metrics()
    official = {
        "schema_version": "trifusion-official-fixed-v1",
        "fixed_epoch": 60,
        "metrics_percent": metrics,
        "query_records": 836,
        "gallery_records": 836,
        "official_test_evaluation_count": 1,
        "official_test_access_count": 1,
        "further_model_selection": False,
        "checkpoint_sha256": checkpoint_sha,
        "run_identity_sha256": identity_sha,
    }
    _write_json(output / "official_test_metrics.json", official)
    official_sha = _sha256(output / "official_test_metrics.json")
    guard = {
        "schema_version": "trifusion-official-access-guard-v1",
        "fixed_epoch": 60,
        "checkpoint_sha256": checkpoint_sha,
        "run_identity_sha256": identity_sha,
        "official_test_access_count": 1,
        "status": "COMPLETE",
        "metrics_sha256": official_sha,
    }
    _write_json(output / "official_test_access_guard.json", guard)
    guard_sha = _sha256(output / "official_test_access_guard.json")

    generation = output / ".resume" / "generation-0060-complete.pt"
    generation.parent.mkdir()
    generation.write_bytes(b"recovery generation")
    generation_sha = _sha256(generation)
    manifest = {
        "schema_version": "1.0",
        "epoch": 60,
        "phase": "complete",
        "run_identity_sha256": identity_sha,
        "current": {
            "path": str(generation.relative_to(output)),
            "sha256": generation_sha,
        },
        "completion_evidence": {
            "kind": "postfreeze-final-fixed",
            "epoch": 60,
            "phase": "complete",
            "fixed_epoch": 60,
            "official_test_evaluation_count": 1,
            "further_model_selection": False,
            "fixed_metrics": metrics,
            "fixed_checkpoint_sha256": checkpoint_sha,
            "official_metrics_receipt_sha256": official_sha,
            "official_access_guard_sha256": guard_sha,
            "run_identity_sha256": identity_sha,
            "contract_testing": False,
            "scientific_evidence_eligible": True,
        },
    }
    _write_json(output / ".resume" / "latest.json", manifest)
    manifest_sha = _sha256(output / ".resume" / "latest.json")

    fixed = {
        "schema_version": "trifusion-postfreeze-final-v1",
        "mode": "postfreeze-final",
        "epoch": 60,
        "phase": "complete",
        "metrics_percent": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "router_calibration_receipt": str(router),
        "router_calibration_receipt_sha256": router_sha,
        "training_split": "RGBNT201/train_171 all identities",
        "evaluation_split": "RGBNT201 official test",
        "further_model_selection": False,
        "official_test_evaluation_count": 1,
        "official_test_access_count": 1,
        "run_identity": str(output / "run_identity.json"),
        "run_identity_sha256": identity_sha,
        "recovery_manifest": str(output / ".resume" / "latest.json"),
        "recovery_manifest_sha256": manifest_sha,
        "model_constructed": True,
        "training_started": True,
        "fatal_or_nonfinite_detected": False,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
    }
    _write_json(output / "fixed_final_receipt.json", fixed)

    worker = {
        "status": "COMPLETE",
        "mode": "postfreeze-final",
        "epoch": 60,
        "phase": "complete",
        "metrics_percent": metrics,
        "last_metrics_percent": metrics,
        "dev_evaluation_count": 0,
        "official_test_evaluation_count": 1,
        "official_test_access_count": 1,
        "further_model_selection": False,
        "query_records": 836,
        "gallery_records": 836,
        "train_records": 3951,
        "fixed_checkpoint": str(checkpoint),
        "fixed_checkpoint_sha256": checkpoint_sha,
        "router_calibration_receipt": str(router),
        "router_calibration_receipt_sha256": router_sha,
        "fatal_or_nonfinite_detected": False,
        "model_constructed": True,
        "training_started": True,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
    }
    _write_json(output / "final_worker_result.json", worker)
    worker_sha = _sha256(output / "final_worker_result.json")
    summary = {
        **worker,
        "status": "PASS",
        "metric_result": metrics,
        "worker_result_sha256": worker_sha,
        "run_identity_sha256": identity_sha,
        "directional_authorization": directional_evidence,
        "scientific_evidence_scope": SCIENTIFIC_EVIDENCE_SCOPE,
        "query_gallery_symmetry_claim_eligible": False,
        "calibration_claim_eligible": True,
        "sota_claim_supported": False,
        "single_seed_target_exceeded": True,
    }
    _write_json(output / "run_summary.json", summary)
    completion = _completion_payload(entry=entry, output_dir=output, returncode=0)
    _write_json(entry / "completion_receipt.json", completion)
    return entry, output, completion


def test_completion_cannot_pass_with_missing_required_artifacts(tmp_path: Path) -> None:
    from tools.run_trifusion_directional_final import _completion_payload

    entry = tmp_path / "entry"
    output = tmp_path / "output"
    entry.mkdir()
    output.mkdir()
    _write_json(entry / "prelaunch_receipt.json", {"status": "AUTHORIZED"})
    (entry / "launcher.log").write_text("done\n", encoding="utf-8")
    _write_json(
        output / "run_summary.json",
        {
            "status": "PASS",
            "mode": "postfreeze-final",
            "official_test_access_count": 1,
            "official_test_evaluation_count": 1,
            "further_model_selection": False,
            "query_gallery_symmetry_claim_eligible": False,
            "scientific_evidence_scope": "calibrated_directional_training_input",
        },
    )

    receipt = _completion_payload(entry=entry, output_dir=output, returncode=0)

    assert receipt["status"] == "FAIL"
    assert any(not item["exists"] for item in receipt["artifacts"].values())


def test_public_completion_verifier_accepts_full_bound_chain(tmp_path: Path) -> None:
    from tools.run_trifusion_directional_final import verify_completion

    entry, _output, _receipt = _build_valid_completion_chain(tmp_path)
    verified = verify_completion(entry)

    assert verified["status"] == "PASS"
    assert verified["verified"] is True
    assert verified["official_test_access_count"] == 1


@pytest.mark.parametrize(
    "artifact_name",
    [
        "run_summary",
        "run_identity",
        "recovery_manifest",
        "recovery_checkpoint",
        "fixed_final_receipt",
        "fixed_checkpoint",
        "official_test_metrics",
        "official_test_guard",
        "final_worker_result",
        "router_calibration_receipt",
    ],
)
def test_public_completion_verifier_rejects_any_artifact_tamper(
    tmp_path: Path, artifact_name: str
) -> None:
    from tools.run_trifusion_directional_final import verify_completion

    entry, _output, receipt = _build_valid_completion_chain(tmp_path)
    artifact = Path(receipt["artifacts"][artifact_name]["path"])
    with artifact.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="artifact"):
        verify_completion(entry)


def test_directional_identity_evidence_excludes_live_gpu_state() -> None:
    from tools.run_trifusion_directional_final import _stable_authorization_evidence

    common = {
        "authorization_path": Path("/tmp/auth.json"),
        "authorization_sha256": "ab" * 32,
        "parent_root": Path("/tmp/parent"),
        "parent_artifacts_sha256": {"receipt": "cd" * 32},
        "parent_scoring_receipt_sha256": "ef" * 32,
        "parent_targets_sha256": "12" * 32,
    }
    first = _stable_authorization_evidence(**common)
    second = _stable_authorization_evidence(**common)

    assert first == second
    assert all("preflight" not in key and "gpu" not in key for key in first)


def test_temporary_environment_restores_both_python_variables(monkeypatch) -> None:
    from tools.run_trifusion_directional_final import _temporary_worker_environment

    monkeypatch.setenv("PYTHONPATH", "before-path")
    monkeypatch.setenv("PYTHONNOUSERSITE", "before-no-user-site")
    with pytest.raises(RuntimeError, match="stop"):
        with _temporary_worker_environment(Path("/amp-safe")):
            assert os.environ["PYTHONPATH"] == "/amp-safe"
            assert os.environ["PYTHONNOUSERSITE"] == "1"
            raise RuntimeError("stop")

    assert os.environ["PYTHONPATH"] == "before-path"
    assert os.environ["PYTHONNOUSERSITE"] == "before-no-user-site"


def test_failure_payload_conservatively_counts_reserved_official_access(
    tmp_path: Path,
) -> None:
    from tools.run_trifusion_directional_final import _failure_payload

    entry = tmp_path / "entry"
    output = tmp_path / "output"
    entry.mkdir()
    output.mkdir()
    _write_json(entry / "prelaunch_receipt.json", {"status": "AUTHORIZED"})
    (entry / "launcher.log").write_text("interrupted\n", encoding="utf-8")
    _write_json(
        output / "official_test_access_guard.json",
        {"status": "STARTED", "official_test_access_count": 1},
    )

    payload = _failure_payload(
        entry=entry,
        output_dir=output,
        error=RuntimeError("boom"),
    )

    assert payload["status"] == "FAIL"
    assert payload["official_test_access_count"] == 1
    assert payload["official_test_access_ambiguous"] is True
    assert payload["prelaunch_receipt_sha256"] == _sha256(
        entry / "prelaunch_receipt.json"
    )
    assert payload["runner_log"]["sha256"] == _sha256(entry / "launcher.log")


def test_launch_failure_restores_environment_and_writes_append_only_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    from tools import run_trifusion_directional_final as launcher

    authorization = tmp_path / "authorization.json"
    config = tmp_path / "config.yml"
    output = tmp_path / "output"
    ledger = tmp_path / "ledger"
    _write_json(authorization, {})
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "_validate_authorization",
        lambda **_kwargs: (
            {"status": "READY", "launch_allowed": True},
            {"authorization_sha256": "ab" * 32},
        ),
    )
    monkeypatch.setattr(
        launcher.frozen_runner,
        "_validate_recovery",
        lambda _output: {"valid": True, "kind": "fresh", "epoch": 0, "phase": "initial"},
    )
    monkeypatch.setattr(
        launcher.frozen_runner,
        "_dev",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker boom")),
    )
    monkeypatch.setenv("PYTHONPATH", "original-path")
    monkeypatch.setenv("PYTHONNOUSERSITE", "original-no-user-site")

    with pytest.raises(RuntimeError, match="worker boom"):
        launcher.launch(
            authorization_path=authorization,
            config_path=config,
            output_dir=output,
            ledger_dir=ledger,
        )

    receipt_path = ledger / "launch-0001" / "failure_receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["official_test_access_count"] == 0
    assert os.environ["PYTHONPATH"] == "original-path"
    assert os.environ["PYTHONNOUSERSITE"] == "original-no-user-site"


def test_completion_verifier_reanchors_authorization_file(tmp_path: Path) -> None:
    from tools.run_trifusion_directional_final import verify_completion

    entry, output, _receipt = _build_valid_completion_chain(tmp_path)
    identity = json.loads((output / "run_identity.json").read_text(encoding="utf-8"))
    authorization_path = Path(
        identity["directional_authorization"]["authorization_path"]
    )
    _write_json(authorization_path, {"root": "tampered"})

    with pytest.raises(ValueError, match="authorization"):
        verify_completion(entry)


def test_completion_verifier_reanchors_parent_artifacts(tmp_path: Path) -> None:
    from tools.run_trifusion_directional_final import verify_completion

    entry, output, _receipt = _build_valid_completion_chain(tmp_path)
    identity = json.loads((output / "run_identity.json").read_text(encoding="utf-8"))
    evidence = identity["directional_authorization"]
    parent_root = Path(evidence["parent_root"])
    name = sorted(evidence["parent_artifacts_sha256"])[0]
    _write_json(parent_root / name, {"parent": "tampered"})

    with pytest.raises(ValueError, match="parent artifact"):
        verify_completion(entry)


def test_completion_verifier_rejects_conflicting_failure_receipt(tmp_path: Path) -> None:
    from tools.run_trifusion_directional_final import verify_completion

    entry, _output, _receipt = _build_valid_completion_chain(tmp_path)
    _write_json(entry / "failure_receipt.json", {"status": "FAIL"})

    with pytest.raises(ValueError, match="failure receipt"):
        verify_completion(entry)


def test_failure_publication_ignores_second_termination_signal() -> None:
    import signal

    from tools.run_trifusion_directional_final import _ignore_termination_signals

    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with _ignore_termination_signals():
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert {signum: signal.getsignal(signum) for signum in before} == before


def test_successful_launch_verifies_candidate_before_atomic_formal_publication(
    tmp_path: Path, monkeypatch
) -> None:
    from tools import run_trifusion_directional_final as launcher

    authorization = tmp_path / "authorization.json"
    config = tmp_path / "config.yml"
    output = tmp_path / "output"
    ledger = tmp_path / "ledger"
    _write_json(authorization, {})
    config.write_text("{}\n", encoding="utf-8")
    evidence = {"authorization_sha256": "ab" * 32}
    authorized = {"status": "READY", "launch_allowed": True}
    monkeypatch.setattr(
        launcher,
        "_validate_authorization",
        lambda **_kwargs: (authorized, evidence),
    )
    monkeypatch.setattr(
        launcher.frozen_runner,
        "_validate_recovery",
        lambda _output: {"valid": True, "kind": "fresh", "epoch": 0, "phase": "initial"},
    )
    summary = {
        "status": "PASS",
        "official_test_access_count": 1,
        "official_test_evaluation_count": 1,
    }
    monkeypatch.setattr(
        launcher.frozen_runner,
        "_dev",
        lambda *_args, **_kwargs: (summary, 0),
    )
    completion = {
        "status": "PASS",
        "official_test_access_count": 1,
        "official_test_evaluation_count": 1,
    }
    monkeypatch.setattr(
        launcher,
        "_completion_payload",
        lambda **_kwargs: completion,
    )
    verification_order = []

    def fake_verify(entry: Path, *, receipt_name: str = "completion_receipt.json") -> dict:
        verification_order.append(receipt_name)
        assert (entry / receipt_name).is_file()
        if receipt_name == "completion_candidate.json":
            assert not (entry / "completion_receipt.json").exists()
        return {**completion, "verified": True}

    monkeypatch.setattr(launcher, "verify_completion", fake_verify)

    assert launcher.launch(
        authorization_path=authorization,
        config_path=config,
        output_dir=output,
        ledger_dir=ledger,
    ) == 0

    entry = ledger / "launch-0001"
    assert verification_order == [
        "completion_candidate.json",
        "completion_receipt.json",
    ]
    assert (entry / "completion_receipt.json").is_file()
    assert not (entry / "completion_candidate.json").exists()
    assert not (entry / "failure_receipt.json").exists()


def test_launch_rejects_invalid_recovery_before_ledger_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    from tools import run_trifusion_directional_final as launcher

    authorization = tmp_path / "authorization.json"
    config = tmp_path / "config.yml"
    output = tmp_path / "output"
    ledger = tmp_path / "ledger"
    _write_json(authorization, {})
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "_validate_authorization",
        lambda **_kwargs: (
            {"status": "READY", "launch_allowed": True},
            {"authorization_sha256": "ab" * 32},
        ),
    )
    monkeypatch.setattr(
        launcher.frozen_runner,
        "_validate_recovery",
        lambda _output: {"valid": False, "error": "foreign recovery"},
    )

    with pytest.raises(ValueError, match="recovery is invalid"):
        launcher.launch(
            authorization_path=authorization,
            config_path=config,
            output_dir=output,
            ledger_dir=ledger,
        )

    assert not ledger.exists()
