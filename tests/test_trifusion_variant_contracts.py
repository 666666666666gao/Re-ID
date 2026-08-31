from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_trifusion_experiment.py"


def test_hfer_uniform_generator_preflight_binds_target_generation_topology(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "uniform-generator-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 3090, 1, 24576\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "uniform-generator-preflight"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "hfer_uniform_generator",
            "--config",
            str(PROJECT / "configs/RGBNT201/TriFusion-circ-generator-shared-semantic-rtx3090.yml"),
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
    assert receipt["variant_contract"]["active_experts"] == [
        "cnn",
        "transformer",
        "mamba",
    ]
    assert receipt["variant_contract"]["collaborator"] == "hfer"
    assert receipt["variant_contract"]["reliability"] == "uniform"
    assert receipt["variant_contract"]["fusion"] == "uniform"
    assert receipt["variant_contract"]["claim_role"] == "CIRC target generator"
    assert receipt["variant_contract"]["circ_targets_required"] is False
    assert receipt["official_test_access_count"] == 0


def test_full_circ_urgc_preflight_fails_closed_until_scientific_assets_exist(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "full-circ-gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 1061, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "full-circ-preflight"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "trifusion_circ_urgc",
            "--config",
            str(PROJECT / "configs/RGBNT201/TriFusion-circ-urgc-low-vram.yml"),
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
    assert receipt["status"] == "BLOCKED"
    assert "missing_circ_training_assets" in receipt["blockers"]
    assert receipt["variant_contract"] == {
        "active_experts": ["cnn", "transformer", "mamba"],
        "circ_targets_required": True,
        "claim_role": "full HFER+CIRC+URGC main method",
        "collaborator": "hfer",
        "evaluation_outputs": ["fused", "cnn", "transformer", "mamba"],
        "family": "collaborative",
        "fusion": "reliability_weighted",
        "peer_mode": "none",
        "reliability": "joint_beta_circ",
        "variant": "trifusion_circ_urgc",
    }
    assert receipt["launch_allowed"] is False
    assert receipt["model_constructed"] is False


def test_cnn_standalone_preflight_exposes_a_real_single_expert_topology(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "gpu"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "preflight"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "cnn_standalone",
            "--config",
            str(PROJECT / "configs/RGBNT201/variants/cnn_standalone.yml"),
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
    assert receipt["variant_contract"] == {
        "active_experts": ["cnn"],
        "circ_targets_required": False,
        "claim_role": "R020",
        "collaborator": "none",
        "evaluation_outputs": ["cnn"],
        "family": "standalone",
        "fusion": "single_expert",
        "peer_mode": "none",
        "reliability": "none",
        "variant": "cnn_standalone",
    }
    assert len(receipt["variant_contract_sha256"]) == 64
    assert receipt["model_constructed"] is False
    assert receipt["metric_result"] is None


def test_transformer_standalone_preflight_has_no_collaborative_components(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "gpu-transformer"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "preflight-transformer"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "transformer_standalone",
            "--config",
            str(PROJECT / "configs/RGBNT201/variants/transformer_standalone.yml"),
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
    assert receipt["variant_contract"]["active_experts"] == ["transformer"]
    assert receipt["variant_contract"]["evaluation_outputs"] == ["transformer"]
    assert receipt["variant_contract"]["collaborator"] == "none"
    assert receipt["variant_contract"]["reliability"] == "none"
    assert receipt["variant_contract"]["fusion"] == "single_expert"
    assert receipt["variant_contract"]["claim_role"] == "R021"


def test_mamba_standalone_preflight_has_no_collaborative_components(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "gpu-mamba"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    output_dir = tmp_path / "preflight-mamba"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "preflight",
            "--variant",
            "mamba_standalone",
            "--config",
            str(PROJECT / "configs/RGBNT201/variants/mamba_standalone.yml"),
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
    assert receipt["variant_contract"]["active_experts"] == ["mamba"]
    assert receipt["variant_contract"]["evaluation_outputs"] == ["mamba"]
    assert receipt["variant_contract"]["collaborator"] == "none"
    assert receipt["variant_contract"]["reliability"] == "none"
    assert receipt["variant_contract"]["fusion"] == "single_expert"
    assert receipt["variant_contract"]["claim_role"] == "R022"


def test_cnn_standalone_dev_accepts_only_its_named_metric_and_zero_test_access(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "gpu-dev"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4060 Laptop GPU, 499, 8188\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    fake_worker = tmp_path / "standalone_dev_worker.py"
    fake_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "resume = out / '.resume'; resume.mkdir(parents=True, exist_ok=True)\n"
        "state = resume / 'generation-0060-complete.pt'; state.write_bytes(b'single-full-state')\n"
        "sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "identity = out / 'run_identity.json'\n"
        "manifest = {'schema_version': '1.0', 'epoch': 60, 'phase': 'complete', "
        "'run_identity_sha256': sha(identity), 'current': {"
        "'path': '.resume/generation-0060-complete.pt', 'sha256': sha(state)}, 'previous': None}\n"
        "(resume / 'latest.json').write_text(json.dumps(manifest), encoding='utf-8')\n"
        "metrics = {'cnn': {'mAP': 51.0, 'Rank-1': 61.0, 'Rank-5': 71.0, 'Rank-10': 76.0}}\n"
        "result = {'status': 'COMPLETE', 'epoch': 60, 'phase': 'complete', 'best_epoch': 60, "
        "'selection_output': 'cnn', 'official_test_access_count': 0, 'dev_evaluation_count': 60, "
        "'metrics_percent': metrics, 'query_records': 825, 'gallery_records': 825, "
        "'train_records': 3126, 'model_constructed': True, "
        "'training_started': True}\n"
        "(out / 'dev_worker_result.json').write_text(json.dumps(result), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    output_dir = tmp_path / "standalone-dev"
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
            "cnn_standalone",
            "--config",
            str(PROJECT / "configs/RGBNT201/variants/cnn_standalone.yml"),
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
    assert summary["status"] == "PASS"
    assert summary["selection_output"] == "cnn"
    assert set(summary["metrics_percent"]) == {"cnn"}
    assert summary["official_test_access_count"] == 0
    assert summary["sota_claim_supported"] is False
