from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools" / "run_mfrnet_checkpoint_eval.py"


def _run_preflight(tmp_path: Path, memory_used_mib: int) -> dict:
    fake_bin = tmp_path / f"bin-{memory_used_mib}"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/bin/sh\n"
        f"printf 'NVIDIA GeForce RTX 4060 Laptop GPU, {memory_used_mib}, 8188\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    output_dir = tmp_path / f"preflight-{memory_used_mib}"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
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
    receipt_path = output_dir / "receipt.json"
    assert receipt_path.is_file()
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def test_mfrnet_preflight_uses_strict_500_mib_boundary(tmp_path: Path) -> None:
    blocked = _run_preflight(tmp_path, memory_used_mib=500)
    ready = _run_preflight(tmp_path, memory_used_mib=499)

    assert blocked["mode"] == "preflight"
    assert blocked["status"] == "BLOCKED"
    assert blocked["launch_allowed"] is False
    assert blocked["gpu"]["memory_used_mib"] == 500
    assert blocked["upstream_command_executed"] is False

    assert ready["status"] == "READY"
    assert ready["launch_allowed"] is True
    assert ready["gpu"]["memory_used_mib"] == 499
    assert ready["scientific_protocol"] == {
        "dataset": "RGBNT201",
        "feature_norm": "yes",
        "missing_modality": "nothing",
        "neck_feature": "before",
        "query_items": 836,
        "gallery_items": 836,
        "reranking": "no",
        "return_pattern": 3,
        "test_batch_size": 128,
        "test_size": [256, 128],
    }
    assert ready["upstream_command_executed"] is False


def test_mfrnet_official128_preserves_command_environment_and_logs(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin-official"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_upstream = tmp_path / "fake_mfrnet.py"
    fake_upstream.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('OUTPUT_DIR') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'invocation.json').write_text(json.dumps({"
        "'argv': args, 'PYTHONNOUSERSITE': os.environ.get('PYTHONNOUSERSITE'), "
        "'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}), encoding='utf-8')\n"
        "log = '14/14\\nmAP: 80.7%\\nCMC curve, Rank-1  :83.6%\\n' "
        "+ 'CMC curve, Rank-5  :90.1%\\nCMC curve, Rank-10 :92.4%\\n'\n"
        "(out / 'test_log.txt').write_text(log, encoding='utf-8')\n"
        "print(log, end='')\n",
        encoding="utf-8",
    )
    fake_upstream.chmod(0o755)
    output_dir = tmp_path / "official128"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_MFRNET_TEST_EXECUTABLE"] = str(fake_upstream)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "official128", "--output-dir", str(output_dir)],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    invocation = json.loads((output_dir / "upstream/invocation.json").read_text(encoding="utf-8"))
    combined = output_dir / "combined.log"
    upstream_log = output_dir / "upstream/test_log.txt"
    assert receipt["status"] == "PASS"
    assert receipt["mode"] == "official128"
    assert receipt["upstream_command_executed"] is True
    assert receipt["official_test_evaluation_count"] == 1
    assert receipt["metrics_percent"] == {
        "mAP": 80.7,
        "Rank-1": 83.6,
        "Rank-5": 90.1,
        "Rank-10": 92.4,
    }
    assert receipt["batch_count"] == 14
    assert receipt["query_items"] == 836
    assert receipt["gallery_items"] == 836
    assert invocation["PYTHONNOUSERSITE"] == "1"
    assert invocation["CUDA_VISIBLE_DEVICES"] == "0"
    assert invocation["argv"] == receipt["resolved_command"][1:]
    assert receipt["combined_log_sha256"] == hashlib.sha256(combined.read_bytes()).hexdigest()
    assert receipt["upstream_log_sha256"] == hashlib.sha256(upstream_log.read_bytes()).hexdigest()


def test_mfrnet_preflight_rejects_config_hash_drift_without_launch(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin-drift"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    drifted_config = tmp_path / "MFRNet.yml"
    official_config = Path(
        "/root/mmreid-trifusion/baselines/MFRNet/configs/RGBNT201/MFRNet.yml"
    )
    drifted_config.write_bytes(official_config.read_bytes() + b"\n# drift\n")
    output_dir = tmp_path / "drift"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TRIFUSION_CONTRACT_TESTING"] = "1"
    env["TRIFUSION_MFRNET_TEST_CONFIG"] = str(drifted_config)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "preflight", "--output-dir", str(output_dir)],
        cwd=PROJECT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "BLOCKED"
    assert receipt["launch_allowed"] is False
    assert "hash_drift:config" in receipt["blockers"]
    assert receipt["file_checks"]["config"]["path"] == str(drifted_config)
    assert receipt["upstream_command_executed"] is False
