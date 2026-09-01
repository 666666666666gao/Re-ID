from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_trifusion_experiment.py"


def test_circ_training_begins_with_router_only_immutable_target_phase() -> None:
    from modeling.trifusion.training_phases import (
        active_loss_weights,
        parameter_trainable_in_phase,
        registered_optimizer_parameter_names,
        resolve_training_phase,
    )

    warm = resolve_training_phase(
        epoch=1,
        circ_enabled=True,
        router_warm_epochs=7,
        schedule_horizon_epochs=60,
    )
    joint = resolve_training_phase(
        epoch=8,
        circ_enabled=True,
        router_warm_epochs=7,
        schedule_horizon_epochs=60,
    )
    weights = {"id_fused": 1.0, "triplet_fused": 1.0, "reliability": 1.0}

    assert warm.name == "router_only"
    assert active_loss_weights(weights, warm) == {"reliability": 1.0}
    assert parameter_trainable_in_phase("encoder.reliability_gate.head.weight", warm)
    assert not parameter_trainable_in_phase("encoder.experts.cnn.stem.weight", warm)
    assert joint.name == "joint_hfer_urgc"
    assert active_loss_weights(weights, joint) == weights
    assert parameter_trainable_in_phase("encoder.experts.cnn.stem.weight", joint)
    assert not parameter_trainable_in_phase("encoder.private_projection.weight", joint)

    parameters = {
        "encoder.reliability_gate.head.weight": SimpleNamespace(requires_grad=True),
        "encoder.experts.cnn.stem.weight": SimpleNamespace(requires_grad=True),
        "encoder.private_projection.weight": SimpleNamespace(requires_grad=False),
    }
    model = SimpleNamespace(named_parameters=lambda: parameters.items())
    assert registered_optimizer_parameter_names(model) == frozenset(
        {
            "encoder.reliability_gate.head.weight",
            "encoder.experts.cnn.stem.weight",
        }
    )


def test_shared_semantic_optimizer_keeps_the_full_clip_trunk_at_pretrained_lr() -> None:
    import tools.run_trifusion_experiment as runner

    parameters = {
        "encoder.tokenizer.patch_projection.weight": SimpleNamespace(
            requires_grad=True, label="patch"
        ),
        "encoder.tokenizer.positional_embedding": SimpleNamespace(
            requires_grad=True, label="position"
        ),
        "encoder.tokenizer.class_embedding": SimpleNamespace(
            requires_grad=True, label="class"
        ),
        "encoder.tokenizer.pre_norm.weight": SimpleNamespace(
            requires_grad=True, label="pre_norm"
        ),
        "encoder.tokenizer.shared_blocks.0.block.attn.in_proj_weight": SimpleNamespace(
            requires_grad=True, label="shared_block"
        ),
        "encoder.tokenizer.post_norm.weight": SimpleNamespace(
            requires_grad=True, label="post_norm"
        ),
        "encoder.tokenizer.modality_embedding.weight": SimpleNamespace(
            requires_grad=True, label="modality_adapter"
        ),
        "encoder.experts.cnn.stages.0.0.down.weight": SimpleNamespace(
            requires_grad=True, label="cnn_adapter"
        ),
        "fusion.contribution_projections.cnn.weight": SimpleNamespace(
            requires_grad=True, label="clip_initialized_fusion"
        ),
    }
    model = SimpleNamespace(named_parameters=lambda: parameters.items())

    pretrained, new = runner._partition_trainable_parameters(
        model,
        family="collaborative",
        architecture="shared_semantic_residual",
    )

    assert [parameter.label for parameter in pretrained] == [
        "patch", "position", "class", "pre_norm", "shared_block", "post_norm",
        "clip_initialized_fusion",
    ]
    assert [parameter.label for parameter in new] == ["modality_adapter", "cnn_adapter"]


def test_cascade_v2_optimizer_keeps_clip_anchor_projection_at_pretrained_lr() -> None:
    import tools.run_trifusion_cascade_v2 as runner

    parameters = {
        "encoder.tokenizer.shared_blocks.0.block.attn.in_proj_weight": SimpleNamespace(
            requires_grad=True, label="shared_block"
        ),
        "fusion.semantic_projection.weight": SimpleNamespace(
            requires_grad=True, label="clip_anchor_projection"
        ),
        "fusion.residual_projections.cnn.weight": SimpleNamespace(
            requires_grad=True, label="cnn_residual"
        ),
    }
    model = SimpleNamespace(named_parameters=lambda: parameters.items())

    pretrained, new = runner._partition_trainable_parameters(
        model,
        family="collaborative",
        architecture="shared_semantic_cascade_v2",
    )

    assert [parameter.label for parameter in pretrained] == [
        "shared_block",
        "clip_anchor_projection",
    ]
    assert [parameter.label for parameter in new] == ["cnn_residual"]


def test_protocol_validator_git_status_detects_uncommitted_drift(monkeypatch) -> None:
    import tools.run_trifusion_experiment as runner

    def fake_run(command, **_kwargs):
        if command[:3] == ["git", "diff", "--quiet"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command[:4] == ["git", "diff", "--cached", "--quiet"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:2] == ["git", "ls-files"]:
            return SimpleNamespace(returncode=0, stdout="modeling/trifusion/protocol.py\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    status = runner._git_tracked_file_status(
        PROJECT / "modeling/trifusion/protocol.py"
    )

    assert status["tracked"] is True
    assert status["worktree_clean"] is False
    assert status["index_clean"] is True
    assert status["clean"] is False


def test_postfreeze_parent_accepts_only_one_fixed_official_evaluation(
    tmp_path: Path, monkeypatch,
) -> None:
    import tools.run_trifusion_experiment as runner

    config = tmp_path / "final.yml"
    config.write_text("SCHEMA_VERSION: 1\n", encoding="utf-8")
    output = tmp_path / "final"
    output.mkdir()
    contract = {
        "evaluation_outputs": ["fused", "cnn", "transformer", "mamba"],
        "active_experts": ["cnn", "transformer", "mamba"],
    }
    preflight = {
        "launch_allowed": True,
        "status": "READY",
        "blockers": [],
        "variant_contract": contract,
        "data_mode": "postfreeze-final",
    }
    recovery_calls = iter(
        (
            {"valid": True, "kind": "fresh", "epoch": 0, "phase": "initial"},
            {"valid": True, "kind": "resume", "epoch": 60, "phase": "complete"},
        )
    )
    monkeypatch.setattr(runner, "_preflight", lambda *args, **kwargs: dict(preflight))
    monkeypatch.setattr(runner, "_run_identity", lambda *args, **kwargs: {"run": "final"})
    monkeypatch.setattr(runner, "_validate_recovery", lambda _path: next(recovery_calls))

    metrics = {
        name: {"mAP": 86.0, "Rank-1": 88.2, "Rank-5": 95.0, "Rank-10": 97.0}
        for name in contract["evaluation_outputs"]
    }

    def fake_run(command, **_kwargs):
        (output / "final_worker_result.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "mode": "postfreeze-final",
                    "epoch": 60,
                    "phase": "complete",
                    "official_test_access_count": 1,
                    "official_test_evaluation_count": 1,
                    "dev_evaluation_count": 0,
                    "query_records": 836,
                    "gallery_records": 836,
                    "train_records": 3951,
                    "further_model_selection": False,
                    "model_constructed": True,
                    "training_started": True,
                    "metrics_percent": metrics,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="fixed final\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("TRIFUSION_CONTRACT_TESTING", "1")
    monkeypatch.setenv("TRIFUSION_TEST_EXECUTABLE", "/tmp/fake-final-worker")

    summary, returncode = runner._dev(
        config,
        "trifusion_circ_urgc",
        output,
        data_mode="postfreeze-final",
    )

    assert returncode == 0
    assert summary["status"] == "PASS"
    assert summary["official_test_evaluation_count"] == 1
    assert summary["further_model_selection"] is False
    assert summary["single_seed_target_exceeded"] is True
    assert summary["sota_claim_supported"] is False


def test_official_fixed_endpoint_is_evaluated_once_and_reused_after_resume(
    tmp_path: Path,
) -> None:
    import tools.run_trifusion_experiment as runner

    output = tmp_path / "fixed"
    output.mkdir()
    identity = output / "run_identity.json"
    identity.write_text('{"run":"fixed"}\n', encoding="utf-8")
    checkpoint = output / "fixed_final_model.pth"
    checkpoint.write_bytes(b"fixed-model")
    calls = 0
    expected = {
        "fused": {"mAP": 86.0, "Rank-1": 88.2, "Rank-5": 95.0, "Rank-10": 97.0}
    }

    def evaluate():
        nonlocal calls
        calls += 1
        return expected

    first = runner._evaluate_official_fixed_endpoint_once(
        output_dir=output,
        checkpoint_path=checkpoint,
        fixed_epoch=60,
        query_records=836,
        gallery_records=836,
        run_identity_sha256=hashlib.sha256(identity.read_bytes()).hexdigest(),
        evaluate=evaluate,
    )
    resumed = runner._evaluate_official_fixed_endpoint_once(
        output_dir=output,
        checkpoint_path=checkpoint,
        fixed_epoch=60,
        query_records=836,
        gallery_records=836,
        run_identity_sha256=hashlib.sha256(identity.read_bytes()).hexdigest(),
        evaluate=evaluate,
    )

    assert first == expected
    assert resumed == expected
    assert calls == 1
    guard = json.loads(
        (output / "official_test_access_guard.json").read_text(encoding="utf-8")
    )
    assert guard["status"] == "COMPLETE"
    assert guard["official_test_access_count"] == 1


def test_complete_recovery_repairs_a_missing_parent_summary(
    tmp_path: Path, monkeypatch,
) -> None:
    import tools.run_trifusion_experiment as runner

    config = tmp_path / "final.yml"
    config.write_text("SCHEMA_VERSION: 1\n", encoding="utf-8")
    output = tmp_path / "final"
    output.mkdir()
    identity_payload = {"run": "final"}
    (output / "run_identity.json").write_text(
        json.dumps(identity_payload) + "\n", encoding="utf-8"
    )
    contract = {
        "evaluation_outputs": ["fused", "cnn", "transformer", "mamba"],
        "active_experts": ["cnn", "transformer", "mamba"],
    }
    preflight = {
        "launch_allowed": True,
        "status": "READY",
        "blockers": [],
        "variant_contract": contract,
        "data_mode": "postfreeze-final",
    }
    recovery = {"valid": True, "kind": "resume", "epoch": 60, "phase": "complete"}
    monkeypatch.setattr(runner, "_preflight", lambda *args, **kwargs: dict(preflight))
    monkeypatch.setattr(runner, "_run_identity", lambda *args, **kwargs: identity_payload)
    monkeypatch.setattr(runner, "_validate_recovery", lambda _path: dict(recovery))
    metrics = {
        name: {"mAP": 86.0, "Rank-1": 88.2, "Rank-5": 95.0, "Rank-10": 97.0}
        for name in contract["evaluation_outputs"]
    }
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        (output / "final_worker_result.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "mode": "postfreeze-final",
                    "epoch": 60,
                    "phase": "complete",
                    "official_test_access_count": 1,
                    "official_test_evaluation_count": 1,
                    "dev_evaluation_count": 0,
                    "query_records": 836,
                    "gallery_records": 836,
                    "train_records": 3951,
                    "further_model_selection": False,
                    "model_constructed": True,
                    "training_started": True,
                    "metrics_percent": metrics,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="recovered tail\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("TRIFUSION_CONTRACT_TESTING", "1")
    monkeypatch.setenv("TRIFUSION_TEST_EXECUTABLE", "/tmp/fake-final-worker")

    summary, returncode = runner._dev(
        config,
        "trifusion_circ_urgc",
        output,
        data_mode="postfreeze-final",
    )

    assert returncode == 0
    assert summary["status"] == "PASS"
    assert summary["single_seed_target_exceeded"] is True
    assert calls == 1


def _run_preflight(tmp_path: Path, memory_used_mib: int) -> dict:
    fake_bin = tmp_path / f"gpu-{memory_used_mib}"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        f"printf 'NVIDIA GeForce RTX 4060 Laptop GPU, {memory_used_mib}, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / f"preflight-{memory_used_mib}"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "core_pre_circ",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))


def test_trifusion_preflight_binds_train_only_dev_protocol_and_gpu_boundary(
    tmp_path: Path,
) -> None:
    blocked = _run_preflight(tmp_path, 500)
    ready = _run_preflight(tmp_path, 499)

    assert blocked["status"] == "BLOCKED"
    assert blocked["launch_allowed"] is False
    assert blocked["gpu"]["memory_used_mib"] == 500
    assert blocked["model_constructed"] is False

    assert ready["status"] == "READY"
    assert ready["launch_allowed"] is True
    assert ready["variant"] == "core_pre_circ"
    assert ready["data_protocol"] == {
        "fit_identities": 141,
        "fit_records": 3126,
        "dev_identities": 30,
        "query_records": 825,
        "gallery_records": 825,
        "identity_overlap": 0,
        "official_test_records": 0,
        "uses_test_labels": False,
    }
    assert ready["optimization"]["train_batch_size"] == 16
    assert ready["optimization"]["num_instances"] == 4
    assert ready["optimization"]["gradient_accumulation"] == 1
    assert ready["model_constructed"] is False
    assert ready["training_started"] is False
    assert ready["metric_result"] is None


def test_trifusion_low_vram_profile_keeps_full_core_with_valid_pk_batch(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "low-vram-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 1061, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "low-vram-preflight"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "core_pre_circ",
            "--config",
            str(PROJECT / "configs/RGBNT201/TriFusion-low-vram.yml"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "READY"
    assert receipt["resource_profile"] == "low_vram_b8k4"
    assert receipt["gpu_gate"] == {
        "policy": "minimum_free_memory",
        "required_free_mib": 6144,
        "observed_free_mib": 7127,
        "passed": True,
    }
    assert receipt["optimization"] == {
        "train_batch_size": 8,
        "num_instances": 4,
        "eval_batch_size": 32,
        "num_workers": 0,
        "gradient_accumulation": 1,
        "amp": True,
        "amp_init_scale": 1024.0,
        "max_epochs": 60,
    }
    assert receipt["variant_contract"]["active_experts"] == [
        "cnn",
        "transformer",
        "mamba",
    ]
    assert receipt["variant_contract"]["collaborator"] == "hfer"
    assert receipt["variant_contract"]["fusion"] == "reliability_weighted"
    assert receipt["model_constructed"] is False
    assert receipt["training_started"] is False


def test_trifusion_rtx3090_profile_selects_shared_semantic_b32k4(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "rtx3090-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 3090, 120, 24576\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "rtx3090-preflight"
    config = (
        PROJECT
        / "configs/RGBNT201/TriFusion-shared-semantic-rtx3090.yml"
    )
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "core_pre_circ",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(
        (output_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "READY"
    assert receipt["resource_profile"] == "rtx3090_b32k4"
    assert receipt["model_architecture"] == "shared_semantic_residual"
    assert receipt["gradient_checkpointing"] is True
    assert receipt["optimization"] == {
        "train_batch_size": 32,
        "num_instances": 4,
        "eval_batch_size": 64,
        "num_workers": 4,
        "gradient_accumulation": 1,
        "amp": True,
        "amp_init_scale": 512.0,
        "max_epochs": 60,
    }
    assert receipt["gpu_gate"] == {
        "policy": "minimum_free_memory",
        "required_free_mib": 22000,
        "observed_free_mib": 24456,
        "passed": True,
    }


def test_shared_semantic_rtx3090_configs_cover_the_complete_main_pipeline() -> None:
    config_root = PROJECT / "configs/RGBNT201"
    generator_path = (
        config_root / "TriFusion-circ-generator-shared-semantic-rtx3090.yml"
    )
    development_path = (
        config_root / "TriFusion-circ-urgc-shared-semantic-rtx3090.yml"
    )
    final_path = (
        config_root
        / "TriFusion-circ-urgc-postfreeze-final-shared-semantic-rtx3090.yml"
    )
    generator = yaml.safe_load(generator_path.read_text(encoding="utf-8"))
    development = yaml.safe_load(development_path.read_text(encoding="utf-8"))
    final = yaml.safe_load(final_path.read_text(encoding="utf-8"))

    assert generator["EXPERIMENT"]["VARIANT"] == "hfer_uniform_generator"
    assert development["EXPERIMENT"]["VARIANT"] == "trifusion_circ_urgc"
    assert final["EXPERIMENT"]["VARIANT"] == "trifusion_circ_urgc"
    for config in (generator, development, final):
        assert config["EXPERIMENT"]["RESOURCE_PROFILE"] == "rtx3090_b32k4"
        assert config["DATA"]["TRAIN_BATCH_SIZE"] == 32
        assert config["DATA"]["NUM_INSTANCES"] == 4
        assert config["MODEL"]["ARCHITECTURE"] == "shared_semantic_residual"
        assert config["MODEL"]["ADAPTER_WIDTH"] == 192
        assert config["MODEL"]["GRADIENT_CHECKPOINTING"] is True
        assert config["OPTIMIZATION"]["AMP_INIT_SCALE"] == 512.0

    selector_root = (
        "/root/autodl-tmp/trifusion-v2/artifacts/"
        "trifusion_shared_semantic_hfer_uniform_selector_seed42"
    )
    assert development["CIRC"]["WARM_START_CHECKPOINT"] == (
        selector_root + "/best_dev_model.pth"
    )
    assert final["CIRC"]["WARM_START_RECEIPT"] == (
        selector_root + "/best_dev_receipt.json"
    )
    assert "development" in development["CIRC"]["TARGET_CACHE"]
    assert "postfreeze_final" in final["CIRC"]["TARGET_CACHE"]

    generator_orchestration = json.loads(
        (
            config_root
            / "CIRC-generators-development-shared-semantic-rtx3090.json"
        ).read_text(encoding="utf-8")
    )
    scoring_orchestration = json.loads(
        (
            config_root
            / "CIRC-score-development-shared-semantic-rtx3090.json"
        ).read_text(encoding="utf-8")
    )
    transfer_orchestration = json.loads(
        (
            config_root
            / "CIRC-transfer-development-shared-semantic-rtx3090.json"
        ).read_text(encoding="utf-8")
    )
    assert generator_orchestration["generator_config"] == generator_path.name
    assert scoring_orchestration["generator_orchestration_config"] == (
        "CIRC-generators-development-shared-semantic-rtx3090.json"
    )
    assert transfer_orchestration["model_config"] == development_path.name


def test_cascade_v2_uniform_selector_config_is_train_only_and_generalization_aware() -> None:
    config_path = (
        PROJECT
        / "configs/RGBNT201/TriFusion-cascade-v2-hfer-uniform-rtx3090.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["EXPERIMENT"]["VARIANT"] == "hfer_uniform_generator"
    assert config["EXPERIMENT"]["SEED"] == 42
    assert config["MODEL"]["ARCHITECTURE"] == "shared_semantic_cascade_v2"
    assert config["MODEL"]["PARAMETER_BUDGET"] == 120_000_000
    assert config["DATA"]["TRAIN_BATCH_SIZE"] == 32
    assert config["DATA"]["NUM_INSTANCES"] == 4
    assert config["LOSS"]["LABEL_SMOOTHING"] == 0.1
    assert config["LOSS"]["EFFECT_RANK_WEIGHT"] == 0.0
    assert config["LOSS"]["RELIABILITY"] == 0.0
    assert config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"] is False
    assert config["PROTOCOL"]["MODEL_SELECTION"] == "dev_mAP"


def test_trifusion_capacity_runs_eight_train_only_steps_after_gate(tmp_path: Path) -> None:
    fake_bin = tmp_path / "capacity-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "fake_trifusion_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'worker_invocation.json').write_text(json.dumps({"
        "'argv': args, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}), encoding='utf-8')\n"
        "result = {'status': 'PASS', 'steps': 8, 'batch_size': 16, 'num_instances': 4, "
        "'finite_losses': True, 'finite_gradients': False, "
        "'gradient_safety_pass': True, 'model_parameters_finite': True, "
        "'amp_overflow_events': 1, 'amp_overflow_recovered': True, "
        "'last_step_gradients_finite': True, 'gradient_parameter_coverage': 1.0, "
        "'model_constructed': True, 'training_started': True, "
        "'official_test_access_count': 0, 'dev_loader_iterations': 0, "
        "'parameter_budget_pass': True, 'total_parameters': 95874282, "
        "'peak_allocated_mib': 6100.0, 'peak_reserved_mib': 6900.0}\n"
        "(out / 'worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "capacity"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_TEST_EXECUTABLE"] = str(fake_worker)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "capacity",
            "--variant",
            "core_pre_circ",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "capacity.json").read_text(encoding="utf-8"))
    invocation = json.loads((output_dir / "worker_invocation.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["worker_executed"] is True
    assert receipt["steps"] == 8
    assert receipt["batch_size"] == 16
    assert receipt["num_instances"] == 4
    assert receipt["finite_gradients"] is False
    assert receipt["gradient_safety_pass"] is True
    assert receipt["amp_overflow_recovered"] is True
    assert receipt["official_test_access_count"] == 0
    assert receipt["dev_loader_iterations"] == 0
    assert receipt["finite_losses"] is True
    assert receipt["gradient_parameter_coverage"] == 1.0
    assert receipt["parameter_budget_pass"] is True
    assert receipt["model_constructed"] is True
    assert receipt["training_started"] is True
    assert invocation["CUDA_VISIBLE_DEVICES"] == "0"


def test_trifusion_low_vram_capacity_enforces_the_profile_batch_contract(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "low-vram-capacity-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "fake_low_vram_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "result = {'status': 'PASS', 'steps': 8, 'batch_size': 8, 'num_instances': 4, "
        "'finite_losses': True, 'finite_gradients': True, "
        "'gradient_safety_pass': True, 'model_parameters_finite': True, "
        "'amp_overflow_events': 0, 'amp_overflow_recovered': False, "
        "'last_step_gradients_finite': True, 'gradient_parameter_coverage': 1.0, "
        "'model_constructed': True, 'training_started': True, "
        "'official_test_access_count': 0, 'dev_loader_iterations': 0, "
        "'parameter_budget_pass': True, 'total_parameters': 95874282, "
        "'peak_allocated_mib': 4800.0, 'peak_reserved_mib': 5600.0}\n"
        "(out / 'worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "low-vram-capacity"
    low_vram_config = PROJECT / "configs/RGBNT201/TriFusion-low-vram.yml"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_TEST_EXECUTABLE"] = str(fake_worker)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "capacity",
            "--variant",
            "core_pre_circ",
            "--config",
            str(low_vram_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "capacity.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["resource_profile"] == "low_vram_b8k4"
    assert receipt["batch_size"] == 8
    assert receipt["num_instances"] == 4
    assert receipt["optimization"]["eval_batch_size"] == 32
    assert receipt["worker_command"][receipt["worker_command"].index("--config") + 1] == str(
        low_vram_config
    )


def test_trifusion_dev_seals_branch_and_fused_metrics_without_official_test(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "dev-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "fake_dev_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "resume = out / '.resume'; resume.mkdir(parents=True, exist_ok=True)\n"
        "state = resume / 'generation-0060-complete.pt'; state.write_bytes(b'dev-full-state')\n"
        "sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "identity = out / 'run_identity.json'\n"
        "manifest = {'schema_version': '1.0', 'epoch': 60, 'phase': 'complete', "
        "'run_identity_sha256': sha(identity), 'current': {"
        "'path': '.resume/generation-0060-complete.pt', 'sha256': sha(state)}, 'previous': None}\n"
        "(resume / 'latest.json').write_text(json.dumps(manifest), encoding='utf-8')\n"
        "metrics = {name: {'mAP': 50.0 + i, 'Rank-1': 60.0 + i, 'Rank-5': 70.0 + i, "
        "'Rank-10': 75.0 + i} for i, name in enumerate(['fused','cnn','transformer','mamba'])}\n"
        "result = {'status': 'COMPLETE', 'epoch': 60, 'phase': 'complete', 'best_epoch': 60, "
        "'official_test_access_count': 0, 'dev_evaluation_count': 60, 'metrics_percent': metrics, "
        "'query_records': 825, 'gallery_records': 825, 'train_records': 3126, "
        "'model_constructed': True, 'training_started': True}\n"
        "(out / 'dev_worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "dev"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_TEST_EXECUTABLE"] = str(fake_worker)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "dev",
            "--variant",
            "core_pre_circ",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    identity = output_dir / "run_identity.json"
    latest = json.loads((output_dir / ".resume/latest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["official_test_access_count"] == 0
    assert summary["dev_evaluation_count"] == 60
    assert set(summary["metrics_percent"]) == {"fused", "cnn", "transformer", "mamba"}
    assert summary["query_records"] == 825
    assert summary["gallery_records"] == 825
    assert summary["claim_scope"] == "train-only development result"
    assert summary["model_constructed"] is True
    assert summary["training_started"] is True
    assert summary["metric_result"] == summary["metrics_percent"]
    assert summary["claim_boundary"] == (
        "train-only development metrics; no official-test metric and no SOTA claim"
    )
    assert summary["sota_claim_supported"] is False
    assert latest["run_identity_sha256"] == hashlib.sha256(identity.read_bytes()).hexdigest()


def test_trifusion_overfit_requires_large_loss_drop_on_one_fixed_batch(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "overfit-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "fake_overfit_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "result = {'status': 'PASS', 'steps': 100, 'batch_size': 16, 'num_instances': 4, "
        "'fixed_batch_sha256': 'ab' * 32, 'initial_loss': 9.0, 'final_loss': 1.2, "
        "'loss_ratio': 1.2 / 9.0, 'finite_losses': True, 'finite_gradients': True, "
        "'gradient_safety_pass': True, 'model_parameters_finite': True, "
        "'official_test_access_count': 0, 'dev_loader_iterations': 0}\n"
        "(out / 'worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "overfit"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_TEST_EXECUTABLE"] = str(fake_worker)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "overfit",
            "--variant",
            "core_pre_circ",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "overfit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["steps"] == 100
    assert receipt["fixed_batch_sha256"] == "ab" * 32
    assert receipt["loss_ratio"] <= 0.2
    assert receipt["finite_losses"] is True
    assert receipt["finite_gradients"] is True
    assert receipt["official_test_access_count"] == 0
    assert receipt["dev_loader_iterations"] == 0


def test_trifusion_low_vram_overfit_uses_the_same_b8k4_profile(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "low-vram-overfit-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "fake_low_vram_overfit_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "result = {'status': 'PASS', 'steps': 100, 'batch_size': 8, 'num_instances': 4, "
        "'fixed_batch_sha256': 'cd' * 32, 'initial_loss': 9.0, 'final_loss': 1.0, "
        "'loss_ratio': 1.0 / 9.0, 'finite_losses': True, 'finite_gradients': True, "
        "'gradient_safety_pass': True, 'model_parameters_finite': True, "
        "'official_test_access_count': 0, 'dev_loader_iterations': 0}\n"
        "(out / 'worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "low-vram-overfit"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_TEST_EXECUTABLE"] = str(fake_worker)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "overfit",
            "--variant",
            "core_pre_circ",
            "--config",
            str(PROJECT / "configs/RGBNT201/TriFusion-low-vram.yml"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "overfit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["resource_profile"] == "low_vram_b8k4"
    assert receipt["batch_size"] == 8
    assert receipt["num_instances"] == 4
    assert receipt["fixed_batch_sha256"] == "cd" * 32
