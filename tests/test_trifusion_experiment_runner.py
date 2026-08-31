from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_trifusion_experiment.py"


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
        "'finite_losses': True, 'finite_gradients': True, 'gradient_parameter_coverage': 1.0, "
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
    assert receipt["official_test_access_count"] == 0
    assert receipt["dev_loader_iterations"] == 0
    assert receipt["finite_losses"] is True
    assert receipt["finite_gradients"] is True
    assert receipt["gradient_parameter_coverage"] == 1.0
    assert receipt["parameter_budget_pass"] is True
    assert invocation["CUDA_VISIBLE_DEVICES"] == "0"


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
        "'query_records': 825, 'gallery_records': 825, 'train_records': 3126}\n"
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
