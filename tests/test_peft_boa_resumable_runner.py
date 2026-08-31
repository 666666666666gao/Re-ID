from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_peft_boa_resumable.py"


def _fake_gpu(tmp_path: Path, memory_used_mib: int) -> Path:
    fake_bin = tmp_path / f"gpu-{memory_used_mib}"
    fake_bin.mkdir()
    executable = fake_bin / "nvidia-smi"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf 'NVIDIA GeForce RTX 4060 Laptop GPU, {memory_used_mib}, 8188\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return fake_bin


def test_peft_capacity_blocks_500_and_runs_eight_steps_at_499(tmp_path: Path) -> None:
    fake_worker = tmp_path / "fake_peft_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'worker_invocation.json').write_text(json.dumps({"
        "'argv': args, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'PYTHONNOUSERSITE': os.environ.get('PYTHONNOUSERSITE')}), encoding='utf-8')\n"
        "result = {'status': 'PASS', 'steps': 8, 'batch_size': 64, "
        "'num_instances': 4, 'official_test_iteration_count': 0, "
        "'finite_losses': True, 'finite_gradients': True, "
        "'trainable_parameter_gradient_coverage': 1.0, "
        "'peak_allocated_mib': 6200.0, 'peak_reserved_mib': 7000.0}\n"
        "(out / 'worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)

    blocked_output = tmp_path / "blocked"
    blocked_env = dict(os.environ)
    blocked_env["PATH"] = f"{_fake_gpu(tmp_path, 500)}:{blocked_env['PATH']}"
    blocked_env["TRIFUSION_CONTRACT_TESTING"] = "1"
    blocked_env["TRIFUSION_PEFT_TEST_EXECUTABLE"] = str(fake_worker)
    blocked = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "capacity", "--output-dir", str(blocked_output)],
        cwd=PROJECT,
        env=blocked_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    blocked_receipt = json.loads((blocked_output / "capacity.json").read_text(encoding="utf-8"))
    assert blocked_receipt["status"] == "BLOCKED"
    assert blocked_receipt["launch_allowed"] is False
    assert blocked_receipt["worker_executed"] is False
    assert not (blocked_output / "worker_invocation.json").exists()

    ready_output = tmp_path / "ready"
    ready_env = dict(os.environ)
    ready_env["PATH"] = f"{_fake_gpu(tmp_path, 499)}:{ready_env['PATH']}"
    ready_env["TRIFUSION_CONTRACT_TESTING"] = "1"
    ready_env["TRIFUSION_PEFT_TEST_EXECUTABLE"] = str(fake_worker)
    ready = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "capacity", "--output-dir", str(ready_output)],
        cwd=PROJECT,
        env=ready_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ready.returncode == 0, ready.stdout + ready.stderr
    ready_receipt = json.loads((ready_output / "capacity.json").read_text(encoding="utf-8"))
    invocation = json.loads((ready_output / "worker_invocation.json").read_text(encoding="utf-8"))
    assert ready_receipt["status"] == "PASS"
    assert ready_receipt["launch_allowed"] is True
    assert ready_receipt["worker_executed"] is True
    assert ready_receipt["steps"] == 8
    assert ready_receipt["batch_size"] == 64
    assert ready_receipt["num_instances"] == 4
    assert ready_receipt["official_test_iteration_count"] == 0
    assert ready_receipt["finite_losses"] is True
    assert ready_receipt["finite_gradients"] is True
    assert ready_receipt["trainable_parameter_gradient_coverage"] == 1.0
    assert invocation["CUDA_VISIBLE_DEVICES"] == "0"
    assert invocation["PYTHONNOUSERSITE"] == "1"


def test_peft_fixed120_rejects_nonempty_foreign_output_before_worker(tmp_path: Path) -> None:
    output_dir = tmp_path / "foreign"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("not a recovery generation", encoding="utf-8")
    marker = tmp_path / "worker-ran"
    fake_worker = tmp_path / "must_not_run.sh"
    fake_worker.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    fake_worker.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{_fake_gpu(tmp_path, 499)}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_PEFT_TEST_EXECUTABLE"] = str(fake_worker)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "fixed120", "--output-dir", str(output_dir)],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "RECOVERY_REJECTED"
    assert summary["worker_executed"] is False
    assert "nonempty_output_without_valid_recovery_manifest" in summary["blockers"]
    assert not marker.exists()


def test_peft_fixed120_seals_once_and_complete_resume_is_idempotent(tmp_path: Path) -> None:
    fake_worker = tmp_path / "fake_fixed_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "count_path = out / 'worker_count.txt'\n"
        "count = int(count_path.read_text()) + 1 if count_path.exists() else 1\n"
        "count_path.write_text(str(count))\n"
        "resume = out / '.resume'; resume.mkdir(parents=True, exist_ok=True)\n"
        "state = resume / 'generation-0120-complete.pt'; state.write_bytes(b'full-state')\n"
        "sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "identity = out / 'run_identity.json'\n"
        "manifest = {'schema_version': '1.0', 'epoch': 120, 'phase': 'complete', "
        "'run_identity_sha256': sha(identity), 'current': {"
        "'path': '.resume/generation-0120-complete.pt', 'sha256': sha(state)}, "
        "'previous': None}\n"
        "(resume / 'latest.json').write_text(json.dumps(manifest), encoding='utf-8')\n"
        "e80 = out / 'BoA_80_preregistered.pth'; e80.write_bytes(b'epoch80')\n"
        "e120 = out / 'BoA_120_fixed.pth'; e120.write_bytes(b'epoch120')\n"
        "result = {'status': 'COMPLETE', 'epoch': 120, 'phase': 'complete', "
        "'official_test_iteration_count': 1, 'official_test_loader_iterations_before_fixed_checkpoint': 0, "
        "'exports_saved_before_official_test': True, 'epoch80_checkpoint': str(e80), "
        "'epoch80_checkpoint_sha256': sha(e80), 'fixed_checkpoint': str(e120), "
        "'fixed_checkpoint_sha256': sha(e120), 'metrics_percent': {"
        "'mAP': 82.2, 'Rank-1': 85.8, 'Rank-5': 91.5, 'Rank-10': 93.5}}\n"
        "(out / 'fixed_worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "fixed120"
    env = dict(os.environ)
    env["PATH"] = f"{_fake_gpu(tmp_path, 499)}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_PEFT_TEST_EXECUTABLE"] = str(fake_worker)
    command = [
        sys.executable,
        str(RUNNER),
        "--mode",
        "fixed120",
        "--output-dir",
        str(output_dir),
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    latest = json.loads((output_dir / ".resume/latest.json").read_text(encoding="utf-8"))
    identity = output_dir / "run_identity.json"
    assert summary["status"] == "PASS"
    assert summary["phase"] == "complete"
    assert summary["official_test_iteration_count"] == 1
    assert summary["official_test_loader_iterations_before_fixed_checkpoint"] == 0
    assert summary["exports_saved_before_official_test"] is True
    assert summary["primary_label"] == "fixed/e120"
    assert summary["epoch80_label"] == "released-test-selected/e80 calibration only"
    assert summary["sota_claim_supported"] is False
    assert latest["run_identity_sha256"] == hashlib.sha256(identity.read_bytes()).hexdigest()
    assert (output_dir / "worker_count.txt").read_text(encoding="utf-8") == "1"

    second = subprocess.run(
        command,
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert (output_dir / "worker_count.txt").read_text(encoding="utf-8") == "1"
    summary_after = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary_after["complete_resume_no_work"] is True
