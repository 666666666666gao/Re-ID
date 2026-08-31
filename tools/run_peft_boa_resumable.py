#!/usr/bin/env python3
"""Crash-safe, fixed-endpoint PEFT-BoA RGBNT201 reproduction runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path("/root/mmreid-trifusion/baselines/PEFT-BoA")
PEFT_PYTHON = Path("/root/miniconda3/envs/peft_boa/bin/python")
CONFIG = SOURCE / "configs/RGBNT201/PEFT-BoA.yml"
PROCESSOR = SOURCE / "engine/processor.py"
TRAIN_ENTRYPOINT = SOURCE / "train_net.py"
CLIP = Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt")
DATASET = Path("/root/mmreid-trifusion/data/RGBNT201")
DATASET_RECEIPT = PROJECT / "evidence/rgbnt201_audit_20260831.json"
REQUIREMENTS_LOCK = PROJECT / "environment/peft_boa_requirements-lock.txt"

EXPECTED = {
    "source_commit": "d2b198be634ac4f9f5744eebf6e0a6604e490deb",
    "config_sha256": "c7c2ea5914f388f60f9316560ba865e5e47074f70a316f909315cb9dff358e6b",
    "processor_sha256": "d6c1ec1600bbc144d5971fecc1ad77436c5021517226fce42383640be67ea29f",
    "train_entrypoint_sha256": "25a2c48037dd4586eeb14d280b7cddb396424eb0ccc93273a030598d895e5ff5",
    "clip_sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "dataset_receipt_sha256": "ec36309921a3dd7c12d46bb60a83406440ba316f171e419a67ad2cc83bf24318",
    "requirements_lock_sha256": "e8c489e7518e758e76faca1f9aa90118df462be6616a28805b842c2c8d5ff26a",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_packages() -> dict[str, str]:
    required: dict[str, str] = {}
    for raw_line in REQUIREMENTS_LOCK.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")) or "==" not in line:
            continue
        name, version = line.split("==", 1)
        required[_normalise_distribution(name)] = version
    return required


def _package_check() -> dict[str, Any]:
    completed = _run([str(PEFT_PYTHON), "-m", "pip", "list", "--format=json"])
    if completed.returncode != 0:
        return {"passed": False, "error": completed.stderr.strip()}
    installed = {
        _normalise_distribution(item["name"]): item["version"]
        for item in json.loads(completed.stdout)
    }
    required = _required_packages()
    mismatches = {
        name: {"required": version, "installed": installed.get(name)}
        for name, version in required.items()
        if installed.get(name) != version
    }
    pip_check = _run([str(PEFT_PYTHON), "-m", "pip", "check"])
    return {
        "passed": not mismatches and pip_check.returncode == 0,
        "required_count": len(required),
        "mismatches": mismatches,
        "pip_check": pip_check.stdout.strip(),
        "python_no_user_site": True,
    }


def _gpu_state() -> dict[str, Any]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        return {"query_passed": False, "error": completed.stderr.strip()}
    fields = [field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 3:
        return {"query_passed": False, "error": "unexpected nvidia-smi output"}
    return {
        "query_passed": True,
        "name": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
    }


def _protocol() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset_name = str(config["DATASETS"]["NAMES"]).strip("()'\" ")
    return {
        "dataset": dataset_name,
        "seed": 1111,
        "max_epochs": int(config["SOLVER"]["MAX_EPOCHS"]),
        "optimizer": str(config["SOLVER"]["OPTIMIZER_NAME"]),
        "batch_size": int(config["SOLVER"]["IMS_PER_BATCH"]),
        "num_instances": int(config["DATALOADER"]["NUM_INSTANCE"]),
        "train_size": [int(value) for value in config["INPUT"]["SIZE_TRAIN"]],
        "test_size": [int(value) for value in config["INPUT"]["SIZE_TEST"]],
        "reranking": str(config["TEST"]["RE_RANKING"]),
        "feature_norm": str(config["TEST"]["FEAT_NORM"]),
        "frozen_clip": bool(config["MODEL"]["FROZEN"]),
        "train_items": len(list((DATASET / "train_171/RGB").glob("*.jpg"))),
        "query_items": len(list((DATASET / "test/RGB").glob("*.jpg"))),
        "gallery_items": len(list((DATASET / "test/RGB").glob("*.jpg"))),
    }


def _preflight(mode: str) -> dict[str, Any]:
    blockers: list[str] = []
    file_checks: dict[str, Any] = {}
    immutable = {
        "config": (CONFIG, EXPECTED["config_sha256"]),
        "processor": (PROCESSOR, EXPECTED["processor_sha256"]),
        "train_entrypoint": (TRAIN_ENTRYPOINT, EXPECTED["train_entrypoint_sha256"]),
        "clip": (CLIP, EXPECTED["clip_sha256"]),
        "dataset_receipt": (DATASET_RECEIPT, EXPECTED["dataset_receipt_sha256"]),
        "requirements_lock": (REQUIREMENTS_LOCK, EXPECTED["requirements_lock_sha256"]),
    }
    for label, (path, expected_hash) in immutable.items():
        if not path.is_file():
            blockers.append(f"missing:{label}")
            file_checks[label] = {"path": str(path), "exists": False}
            continue
        actual_hash = _sha256(path)
        file_checks[label] = {
            "path": str(path),
            "exists": True,
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "hash_match": actual_hash == expected_hash,
        }
        if actual_hash != expected_hash:
            blockers.append(f"hash_drift:{label}")
    commit = _run(["git", "rev-parse", "HEAD"], cwd=SOURCE)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SOURCE)
    source_check = {
        "commit": commit.stdout.strip(),
        "expected_commit": EXPECTED["source_commit"],
        "tracked_worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }
    if commit.returncode or commit.stdout.strip() != EXPECTED["source_commit"]:
        blockers.append("source_commit_drift")
    if not source_check["tracked_worktree_clean"]:
        blockers.append("source_tracked_worktree_dirty")
    package_check = (
        _package_check() if PEFT_PYTHON.is_file() else {"passed": False, "error": "missing python"}
    )
    if not package_check["passed"]:
        blockers.append("environment_lock_mismatch")
    try:
        protocol = _protocol()
    except Exception as error:
        protocol = {"error": f"{type(error).__name__}: {error}"}
        blockers.append("protocol_parse_failed")
    expected_protocol = {
        "dataset": "RGBNT201",
        "seed": 1111,
        "max_epochs": 120,
        "optimizer": "AdamW",
        "batch_size": 64,
        "num_instances": 4,
        "train_size": [256, 128],
        "test_size": [256, 128],
        "reranking": "no",
        "feature_norm": "yes",
        "frozen_clip": True,
        "train_items": 3951,
        "query_items": 836,
        "gallery_items": 836,
    }
    if protocol != expected_protocol:
        blockers.append("scientific_protocol_drift")
    gpu = _gpu_state()
    gpu_eligible = bool(gpu.get("query_passed")) and int(gpu["memory_used_mib"]) < 500
    if not gpu.get("query_passed"):
        blockers.append("gpu_query_failed")
    elif not gpu_eligible:
        blockers.append("gpu_memory_gate")
    return {
        "schema_version": "1.0",
        "mode": mode,
        "status": "READY" if not blockers else "BLOCKED",
        "launch_allowed": not blockers,
        "required_memory_used_strictly_below_mib": 500,
        "blockers": blockers,
        "gpu": gpu,
        "source_check": source_check,
        "file_checks": file_checks,
        "package_check": package_check,
        "scientific_protocol": protocol,
        "runner_sha256": _sha256(Path(__file__)),
        "worker_executed": False,
        "official_test_iteration_count": 0,
        "sota_claim_supported": False,
    }


def _validate_recovery(output_dir: Path) -> dict[str, Any]:
    children = list(output_dir.iterdir())
    if not children:
        return {"valid": True, "kind": "fresh", "epoch": 0, "phase": "initial"}
    latest_path = output_dir / ".resume/latest.json"
    identity_path = output_dir / "run_identity.json"
    if not latest_path.is_file() or not identity_path.is_file():
        return {
            "valid": False,
            "kind": "foreign_or_partial",
            "error": "nonempty_output_without_valid_recovery_manifest",
        }
    try:
        manifest = json.loads(latest_path.read_text(encoding="utf-8"))
        identity_hash = _sha256(identity_path)
        if manifest.get("run_identity_sha256") != identity_hash:
            raise ValueError("run identity hash mismatch")
        epoch = int(manifest["epoch"])
        phase = str(manifest["phase"])
        if epoch < 0 or epoch > 120:
            raise ValueError("epoch outside [0,120]")
        if phase not in {"epoch_boundary", "post_train", "complete"}:
            raise ValueError("invalid phase")
        current = manifest["current"]
        current_path = output_dir / str(current["path"])
        if not current_path.is_file() or _sha256(current_path) != current["sha256"]:
            raise ValueError("current generation missing or corrupt")
        previous = manifest.get("previous")
        if previous is not None:
            previous_path = output_dir / str(previous["path"])
            if not previous_path.is_file() or _sha256(previous_path) != previous["sha256"]:
                raise ValueError("previous generation missing or corrupt")
        return {
            "valid": True,
            "kind": "resume",
            "epoch": epoch,
            "phase": phase,
            "manifest_sha256": _sha256(latest_path),
            "run_identity_sha256": identity_hash,
        }
    except Exception as error:
        return {
            "valid": False,
            "kind": "foreign_or_partial",
            "error": f"invalid_recovery_manifest:{type(error).__name__}:{error}",
        }


def _capacity(output_dir: Path) -> tuple[dict[str, Any], int]:
    receipt = _preflight("capacity")
    if not receipt["launch_allowed"]:
        receipt["claim_boundary"] = "capacity blocked before CUDA model import; no metric"
        return receipt, 0
    test_executable = os.environ.get("TRIFUSION_PEFT_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("test_executable_without_contract_testing")
        return receipt, 2
    command = (
        [test_executable, "--_worker", "capacity", "--output-dir", str(output_dir)]
        if test_executable
        else [
            str(PEFT_PYTHON),
            str(Path(__file__)),
            "--_worker",
            "capacity",
            "--output-dir",
            str(output_dir),
        ]
    )
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command,
        cwd=SOURCE,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    combined_log = output_dir / "capacity_worker.log"
    combined_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    worker_result_path = output_dir / "worker_result.json"
    receipt.update(
        {
            "worker_executed": True,
            "worker_command": command,
            "worker_returncode": completed.returncode,
            "worker_log_sha256": _sha256(combined_log),
            "test_override_used": bool(test_executable),
        }
    )
    if completed.returncode or not worker_result_path.is_file():
        receipt["status"] = "FAILED"
        receipt["blockers"].append("capacity_worker_failed")
        return receipt, 2
    worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "steps": 8,
        "batch_size": 64,
        "num_instances": 4,
        "official_test_iteration_count": 0,
        "finite_losses": True,
        "finite_gradients": True,
        "trainable_parameter_gradient_coverage": 1.0,
    }
    invalid = {key: {"expected": value, "actual": worker_result.get(key)} for key, value in required.items() if worker_result.get(key) != value}
    receipt.update(worker_result)
    receipt["worker_result_sha256"] = _sha256(worker_result_path)
    if invalid:
        receipt["status"] = "FAILED"
        receipt["blockers"].append("capacity_contract_failed")
        receipt["capacity_contract_mismatches"] = invalid
        return receipt, 2
    receipt["status"] = "PASS"
    receipt["claim_boundary"] = "eight-step capacity only; no official-test access or metric"
    return receipt, 0


def _worker_capacity(output_dir: Path) -> int:
    result_path = output_dir / "worker_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    second_gpu_check = _gpu_state()
    if not second_gpu_check.get("query_passed") or int(
        second_gpu_check.get("memory_used_mib", 500)
    ) >= 500:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "stage": "worker_gpu_recheck",
                "gpu": second_gpu_check,
                "steps": 0,
                "official_test_iteration_count": 0,
            },
        )
        return 3

    try:
        os.chdir(SOURCE)
        sys.path.insert(0, str(SOURCE))
        import importlib
        import random

        import numpy as np
        import torch
        import torchvision.transforms as transforms
        from config import cfg
        from layers.make_loss import make_loss
        from modeling import make_model
        from solver.make_optimizer import make_optimizer
        from solver.scheduler_factory import create_scheduler

        loader_module = importlib.import_module("data.datasets.make_dataloader")
        rgbnt201_module = importlib.import_module("data.datasets.RGBNT201")

        seed = 1111
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        cfg.merge_from_file(str(CONFIG))
        cfg.defrost()
        cfg.MODEL.PRETRAIN_PATH_T = str(CLIP)
        cfg.DATASETS.ROOT_DIR = "/root/mmreid-trifusion/data"
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.SOLVER.SEED = seed
        cfg.SOLVER.IMS_PER_BATCH = 64
        cfg.DATALOADER.NUM_INSTANCE = 4
        cfg.OUTPUT_DIR = str(output_dir)
        cfg.freeze()

        train_dir = DATASET / "train_171"
        dataset_probe = rgbnt201_module.RGBNT201.__new__(rgbnt201_module.RGBNT201)
        train_records = dataset_probe._process_dir(str(train_dir), relabel=True)
        train_transform = transforms.Compose(
            [
                transforms.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                transforms.Pad(cfg.INPUT.PADDING),
                transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                loader_module.RandomErasing(
                    probability=cfg.INPUT.RE_PROB,
                    mode="pixel",
                    max_count=1,
                    device="cpu",
                ),
            ]
        )
        train_set = loader_module.ImageDataset(train_records, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=64,
            sampler=loader_module.RandomIdentitySampler(train_records, 64, 4),
            num_workers=0,
            collate_fn=loader_module.train_collate_fn,
        )
        num_classes = len({int(record[1]) for record in train_records})
        camera_num = len({int(record[2]) for record in train_records})
        view_num = len({int(record[3]) for record in train_records})

        model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
        ).cuda()
        loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = torch.cuda.amp.GradScaler()
        scheduler.step(1)
        model.train()
        torch.cuda.reset_peak_memory_stats()

        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        received_finite_gradient: set[str] = set()
        losses: list[float] = []
        all_losses_finite = True
        all_gradients_finite = True
        iterator = iter(train_loader)
        for _ in range(8):
            images, identities, camera_ids, view_ids, _paths = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            optimizer_center.zero_grad(set_to_none=True)
            images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
            identities = identities.cuda(non_blocking=False)
            camera_ids = camera_ids.cuda(non_blocking=False)
            view_ids = view_ids.cuda(non_blocking=False)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(
                    images,
                    label=identities,
                    cam_label=camera_ids,
                    view_label=view_ids,
                )
                loss = torch.zeros((), device="cuda")
                for index in range(0, len(outputs), 2):
                    loss = loss + loss_fn(
                        score=outputs[index],
                        feat=outputs[index + 1],
                        target=identities,
                        target_cam=camera_ids,
                    )
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            all_losses_finite = all_losses_finite and bool(torch.isfinite(loss).item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                finite = bool(torch.isfinite(parameter.grad).all().item())
                all_gradients_finite = all_gradients_finite and finite
                if finite:
                    received_finite_gradient.add(name)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()

        coverage = (
            len(received_finite_gradient) / len(trainable) if trainable else 0.0
        )
        status = (
            "PASS"
            if all_losses_finite and all_gradients_finite and coverage == 1.0
            else "FAILED"
        )
        result = {
            "status": status,
            "steps": 8,
            "batch_size": 64,
            "num_instances": 4,
            "seed": seed,
            "train_records": len(train_records),
            "num_train_ids": num_classes,
            "num_train_cameras": camera_num,
            "official_test_loader_constructed": False,
            "official_test_iteration_count": 0,
            "finite_losses": all_losses_finite,
            "finite_gradients": all_gradients_finite,
            "losses": losses,
            "trainable_parameter_tensors": len(trainable),
            "finite_gradient_parameter_tensors": len(received_finite_gradient),
            "trainable_parameter_gradient_coverage": coverage,
            "missing_gradient_parameters": sorted(trainable - received_finite_gradient),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "torch": torch.__version__,
            "second_gpu_gate": second_gpu_check,
        }
        _atomic_json(result_path, result)
        return 0 if status == "PASS" else 4
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        status = "OOM" if "out of memory" in message.lower() else "FAILED"
        _atomic_json(
            result_path,
            {
                "status": status,
                "steps": 0,
                "official_test_iteration_count": 0,
                "error": message,
                "traceback": traceback.format_exc(),
                "second_gpu_gate": second_gpu_check,
            },
        )
        return 4


def _run_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    runtime = _run(
        [
            str(PEFT_PYTHON),
            "-c",
            "import json,platform,torch; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__}))",
        ]
    )
    runtime_identity = (
        json.loads(runtime.stdout) if runtime.returncode == 0 else {"error": runtime.stderr.strip()}
    )
    return {
        "schema_version": "1.0",
        "run_type": "PEFT-BoA/RGBNT201/fixed120",
        "source_commit": EXPECTED["source_commit"],
        "config_sha256": EXPECTED["config_sha256"],
        "processor_sha256": EXPECTED["processor_sha256"],
        "train_entrypoint_sha256": EXPECTED["train_entrypoint_sha256"],
        "clip_sha256": EXPECTED["clip_sha256"],
        "dataset_receipt_sha256": EXPECTED["dataset_receipt_sha256"],
        "requirements_lock_sha256": EXPECTED["requirements_lock_sha256"],
        "runner_sha256": _sha256(Path(__file__)),
        "runtime": runtime_identity,
        "protocol": preflight["scientific_protocol"],
        "overrides": {
            "MODEL.PRETRAIN_PATH_T": str(CLIP),
            "DATASETS.ROOT_DIR": "/root/mmreid-trifusion/data",
            "DATALOADER.NUM_WORKERS": 0,
            "MODEL.DEVICE_ID": "0",
        },
        "selection": "fixed epoch 120; official test exactly once after durable checkpoint",
        "wandb": False,
    }


def _fixed120(output_dir: Path) -> tuple[dict[str, Any], int]:
    recovery = _validate_recovery(output_dir)
    if not recovery["valid"]:
        receipt = _preflight("fixed120")
        receipt["status"] = "RECOVERY_REJECTED"
        receipt["launch_allowed"] = False
        receipt["recovery"] = recovery
        receipt["blockers"].append(
            "nonempty_output_without_valid_recovery_manifest"
            if recovery["error"] == "nonempty_output_without_valid_recovery_manifest"
            else "invalid_recovery_manifest"
        )
        return receipt, 2

    preflight = _preflight("fixed120")
    expected_identity = _run_identity(preflight)
    identity_path = output_dir / "run_identity.json"
    if recovery["kind"] == "fresh":
        if not preflight["launch_allowed"]:
            preflight["recovery"] = recovery
            preflight["claim_boundary"] = "fixed120 blocked before CUDA model import"
            return preflight, 0
        _atomic_json(identity_path, expected_identity)
    else:
        try:
            actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except Exception as error:
            preflight["status"] = "RECOVERY_REJECTED"
            preflight["launch_allowed"] = False
            preflight["blockers"].append("run_identity_unreadable")
            preflight["recovery_error"] = f"{type(error).__name__}: {error}"
            return preflight, 2
        if actual_identity != expected_identity:
            preflight["status"] = "RECOVERY_REJECTED"
            preflight["launch_allowed"] = False
            preflight["blockers"].append("foreign_run_identity")
            preflight["recovery"] = recovery
            return preflight, 2
        if recovery["phase"] == "complete":
            summary_path = output_dir / "run_summary.json"
            if not summary_path.is_file():
                preflight["status"] = "RECOVERY_REJECTED"
                preflight["launch_allowed"] = False
                preflight["blockers"].append("complete_state_without_run_summary")
                return preflight, 2
            complete_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if complete_summary.get("status") != "PASS":
                preflight["status"] = "RECOVERY_REJECTED"
                preflight["launch_allowed"] = False
                preflight["blockers"].append("complete_state_without_pass_summary")
                return preflight, 2
            complete_summary["complete_resume_no_work"] = True
            complete_summary["worker_executed"] = False
            return complete_summary, 0
        if not preflight["launch_allowed"]:
            preflight["recovery"] = recovery
            preflight["claim_boundary"] = "resume blocked before CUDA model import"
            return preflight, 0

    test_executable = os.environ.get("TRIFUSION_PEFT_TEST_EXECUTABLE")
    contract_testing = os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1"
    if test_executable and not contract_testing:
        preflight["status"] = "FAILED"
        preflight["launch_allowed"] = False
        preflight["blockers"].append("test_executable_without_contract_testing")
        return preflight, 2
    command = (
        [test_executable, "--_worker", "fixed120", "--output-dir", str(output_dir)]
        if test_executable
        else [
            str(PEFT_PYTHON),
            str(Path(__file__)),
            "--_worker",
            "fixed120",
            "--output-dir",
            str(output_dir),
        ]
    )
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        command,
        cwd=SOURCE,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    worker_log = output_dir / "fixed120_worker.log"
    worker_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    result_path = output_dir / "fixed_worker_result.json"
    preflight.update(
        {
            "worker_executed": True,
            "worker_command": command,
            "worker_returncode": completed.returncode,
            "worker_log_sha256": _sha256(worker_log),
            "test_override_used": bool(test_executable),
            "run_identity_sha256": _sha256(identity_path),
        }
    )
    if completed.returncode or not result_path.is_file():
        preflight["status"] = "FAILED"
        preflight["blockers"].append("fixed120_worker_failed")
        return preflight, 2
    worker_result = json.loads(result_path.read_text(encoding="utf-8"))
    recovery_after = _validate_recovery(output_dir)
    required = {
        "status": "COMPLETE",
        "epoch": 120,
        "phase": "complete",
        "official_test_iteration_count": 1,
        "official_test_loader_iterations_before_fixed_checkpoint": 0,
        "exports_saved_before_official_test": True,
    }
    mismatches = {
        key: {"expected": value, "actual": worker_result.get(key)}
        for key, value in required.items()
        if worker_result.get(key) != value
    }
    for path_key, hash_key in (
        ("epoch80_checkpoint", "epoch80_checkpoint_sha256"),
        ("fixed_checkpoint", "fixed_checkpoint_sha256"),
    ):
        path = Path(str(worker_result.get(path_key, "")))
        if not path.is_file() or _sha256(path) != worker_result.get(hash_key):
            mismatches[path_key] = {"expected": "existing hash-bound export", "actual": str(path)}
    if not recovery_after["valid"] or recovery_after.get("phase") != "complete":
        mismatches["recovery"] = {"expected": "valid complete", "actual": recovery_after}
    preflight.update(worker_result)
    preflight["worker_result_sha256"] = _sha256(result_path)
    preflight["recovery"] = recovery_after
    preflight["primary_label"] = "fixed/e120"
    preflight["epoch80_label"] = "released-test-selected/e80 calibration only"
    preflight["sota_claim_supported"] = False
    preflight["complete_resume_no_work"] = False
    if mismatches:
        preflight["status"] = "FAILED"
        preflight["blockers"].append("fixed120_contract_failed")
        preflight["fixed120_contract_mismatches"] = mismatches
        return preflight, 2
    preflight["status"] = "PASS"
    preflight["claim_boundary"] = "local fixed/e120 baseline only; no test-selected promotion and no SOTA claim"
    return preflight, 0


def _atomic_torch_save(path: Path, payload: dict[str, Any], torch_module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch_module.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _capture_rng(random_module: Any, numpy_module: Any, torch_module: Any) -> dict[str, Any]:
    return {
        "python": random_module.getstate(),
        "numpy": numpy_module.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": torch_module.cuda.get_rng_state_all(),
    }


def _restore_rng(
    state: dict[str, Any], random_module: Any, numpy_module: Any, torch_module: Any
) -> None:
    random_module.setstate(state["python"])
    numpy_module.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def _save_generation(
    output_dir: Path,
    *,
    epoch: int,
    phase: str,
    state: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    resume_dir = output_dir / ".resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    identity_hash = _sha256(output_dir / "run_identity.json")
    latest_path = resume_dir / "latest.json"
    old_manifest = (
        json.loads(latest_path.read_text(encoding="utf-8"))
        if latest_path.is_file()
        else None
    )
    filename = f"generation-{epoch:04d}-{phase}.pt"
    generation_path = resume_dir / filename
    _atomic_torch_save(generation_path, state, torch_module)
    current = {
        "path": str(generation_path.relative_to(output_dir)),
        "sha256": _sha256(generation_path),
        "bytes": generation_path.stat().st_size,
    }
    previous = old_manifest.get("current") if old_manifest else None
    if previous and previous.get("path") == current["path"]:
        previous = old_manifest.get("previous")
    manifest = {
        "schema_version": "1.0",
        "epoch": epoch,
        "phase": phase,
        "run_identity_sha256": identity_hash,
        "current": current,
        "previous": previous,
    }
    _atomic_json(latest_path, manifest)
    keep = {current["path"]}
    if previous:
        keep.add(previous["path"])
    resolved_resume = resume_dir.resolve()
    for candidate in resume_dir.glob("generation-*.pt"):
        relative = str(candidate.relative_to(output_dir))
        if relative in keep:
            continue
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != resolved_resume:
            raise RuntimeError(f"unsafe recovery cleanup target: {resolved_candidate}")
        candidate.unlink()
    return manifest


def _worker_fixed120(output_dir: Path) -> int:
    result_path = output_dir / "fixed_worker_result.json"
    second_gpu_check = _gpu_state()
    if not second_gpu_check.get("query_passed") or int(
        second_gpu_check.get("memory_used_mib", 500)
    ) >= 500:
        _atomic_json(
            result_path,
            {
                "status": "BLOCKED",
                "epoch": 0,
                "phase": "worker_gpu_recheck",
                "official_test_iteration_count": 0,
                "gpu": second_gpu_check,
            },
        )
        return 3
    try:
        os.chdir(SOURCE)
        sys.path.insert(0, str(SOURCE))
        import random

        import numpy as np
        import torch
        from config import cfg
        from data import make_dataloader
        from layers.make_loss import make_loss
        from modeling import make_model
        from solver.make_optimizer import make_optimizer
        from solver.scheduler_factory import create_scheduler
        from utils.metrics import R1_mAP_eval

        seed = 1111
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        cfg.merge_from_file(str(CONFIG))
        cfg.defrost()
        cfg.MODEL.PRETRAIN_PATH_T = str(CLIP)
        cfg.DATASETS.ROOT_DIR = "/root/mmreid-trifusion/data"
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.SOLVER.SEED = seed
        cfg.SOLVER.IMS_PER_BATCH = 64
        cfg.DATALOADER.NUM_INSTANCE = 4
        cfg.OUTPUT_DIR = str(output_dir)
        cfg.freeze()

        (
            train_loader,
            _train_loader_normal,
            val_loader,
            num_query,
            num_classes,
            camera_num,
            view_num,
        ) = make_dataloader(cfg)
        model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
        ).cuda()
        loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = torch.cuda.amp.GradScaler()
        identity_hash = _sha256(output_dir / "run_identity.json")
        latest_path = output_dir / ".resume/latest.json"
        official_test_evaluation_count = 0
        resume_history: list[dict[str, Any]] = []
        start_epoch = 1
        phase = "initial"

        if latest_path.is_file():
            manifest = json.loads(latest_path.read_text(encoding="utf-8"))
            if manifest["run_identity_sha256"] != identity_hash:
                raise RuntimeError("worker run identity mismatch")
            generation_path = output_dir / manifest["current"]["path"]
            if _sha256(generation_path) != manifest["current"]["sha256"]:
                raise RuntimeError("worker current recovery generation hash mismatch")
            saved = torch.load(generation_path, map_location="cpu")
            required_state = {
                "model",
                "optimizer",
                "optimizer_center",
                "scheduler",
                "scaler",
                "center_criterion",
                "rng",
                "epoch",
                "phase",
                "run_identity_sha256",
                "official_test_evaluation_count",
            }
            missing = sorted(required_state - set(saved))
            if missing:
                raise RuntimeError(f"incomplete recovery generation: {missing}")
            if saved["run_identity_sha256"] != identity_hash:
                raise RuntimeError("foreign recovery generation")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            optimizer_center.load_state_dict(saved["optimizer_center"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            center_criterion.load_state_dict(saved["center_criterion"], strict=True)
            _restore_rng(saved["rng"], random, np, torch)
            phase = str(saved["phase"])
            saved_epoch = int(saved["epoch"])
            official_test_evaluation_count = int(saved["official_test_evaluation_count"])
            start_epoch = saved_epoch + 1 if phase == "epoch_boundary" else 121
            resume_history = list(saved.get("resume_history", []))
            resume_history.append(
                {
                    "epoch": saved_epoch,
                    "phase": phase,
                    "generation_sha256": manifest["current"]["sha256"],
                }
            )
        else:
            initial_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_center": optimizer_center.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "center_criterion": center_criterion.state_dict(),
                "rng": _capture_rng(random, np, torch),
                "epoch": 0,
                "phase": "epoch_boundary",
                "run_identity_sha256": identity_hash,
                "official_test_evaluation_count": 0,
                "resume_history": [],
            }
            _save_generation(
                output_dir,
                epoch=0,
                phase="epoch_boundary",
                state=initial_state,
                torch_module=torch,
            )
            phase = "epoch_boundary"

        epoch80_path = output_dir / "BoA_80_preregistered.pth"
        fixed_path = output_dir / "BoA_120_fixed.pth"
        fixed_receipt_path = output_dir / "fixed_checkpoint_receipt.json"
        train_epoch_losses: dict[str, float] = {}
        torch.cuda.reset_peak_memory_stats()

        if phase == "epoch_boundary":
            for epoch in range(start_epoch, 121):
                scheduler.step(epoch)
                model.train()
                loss_sum = 0.0
                sample_count = 0
                for images, identities, camera_ids, view_ids, _paths in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_center.zero_grad(set_to_none=True)
                    images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
                    identities = identities.cuda(non_blocking=False)
                    camera_ids = camera_ids.cuda(non_blocking=False)
                    view_ids = view_ids.cuda(non_blocking=False)
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(
                            images,
                            label=identities,
                            cam_label=camera_ids,
                            view_label=view_ids,
                        )
                        loss = torch.zeros((), device="cuda")
                        for index in range(0, len(outputs), 2):
                            loss = loss + loss_fn(
                                score=outputs[index],
                                feat=outputs[index + 1],
                                target=identities,
                                target_cam=camera_ids,
                            )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                        for parameter in center_criterion.parameters():
                            if parameter.grad is not None:
                                parameter.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                        scaler.step(optimizer_center)
                        scaler.update()
                    batch_size = int(identities.shape[0])
                    loss_sum += float(loss.detach().cpu()) * batch_size
                    sample_count += batch_size
                torch.cuda.synchronize()
                train_epoch_losses[str(epoch)] = loss_sum / sample_count

                if epoch == 80:
                    _atomic_torch_save(epoch80_path, model.state_dict(), torch)
                if epoch == 120:
                    _atomic_torch_save(fixed_path, model.state_dict(), torch)
                    _atomic_json(
                        fixed_receipt_path,
                        {
                            "schema_version": "1.0",
                            "epoch": 120,
                            "path": str(fixed_path),
                            "bytes": fixed_path.stat().st_size,
                            "sha256": _sha256(fixed_path),
                            "run_identity_sha256": identity_hash,
                            "official_test_evaluation_count_at_save": 0,
                            "saved_before_official_test": True,
                            "label": "fixed/e120",
                        },
                    )
                boundary_phase = "post_train" if epoch == 120 else "epoch_boundary"
                full_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_center": optimizer_center.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "center_criterion": center_criterion.state_dict(),
                    "rng": _capture_rng(random, np, torch),
                    "epoch": epoch,
                    "phase": boundary_phase,
                    "run_identity_sha256": identity_hash,
                    "official_test_evaluation_count": 0,
                    "resume_history": resume_history,
                    "train_epoch_losses": train_epoch_losses,
                }
                _save_generation(
                    output_dir,
                    epoch=epoch,
                    phase=boundary_phase,
                    state=full_state,
                    torch_module=torch,
                )
            phase = "post_train"

        if phase != "post_train":
            raise RuntimeError(f"invalid phase before fixed evaluation: {phase}")
        if official_test_evaluation_count != 0:
            raise RuntimeError("official test already accessed before post_train evaluation")
        if not fixed_path.is_file() or not fixed_receipt_path.is_file():
            raise RuntimeError("durable fixed checkpoint/receipt missing")
        fixed_receipt = json.loads(fixed_receipt_path.read_text(encoding="utf-8"))
        if _sha256(fixed_path) != fixed_receipt["sha256"]:
            raise RuntimeError("fixed checkpoint hash mismatch before evaluation")

        model.load_state_dict(torch.load(fixed_path, map_location="cpu"), strict=True)
        model.eval()
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        official_test_batches = 0
        for images, identities, camera_ids, camera_ids_batch, view_ids, _paths in val_loader:
            official_test_batches += 1
            with torch.no_grad():
                images = {name: tensor.cuda(non_blocking=False) for name, tensor in images.items()}
                camera_ids_batch = camera_ids_batch.cuda(non_blocking=False)
                view_ids = view_ids.cuda(non_blocking=False)
                features = model(
                    images,
                    cam_label=camera_ids_batch,
                    view_label=view_ids,
                )
                evaluator.update((features, identities, camera_ids))
        cmc, mean_ap, _dist, _pids, _camids, _qf, _gf = evaluator.compute()
        official_test_evaluation_count = 1
        metrics = {
            "mAP": float(mean_ap * 100.0),
            "Rank-1": float(cmc[0] * 100.0),
            "Rank-5": float(cmc[4] * 100.0),
            "Rank-10": float(cmc[9] * 100.0),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite fixed evaluation metrics")
        fixed_eval_path = output_dir / "fixed_eval.json"
        _atomic_json(
            fixed_eval_path,
            {
                "schema_version": "1.0",
                "label": "fixed/e120",
                "metrics_percent": metrics,
                "query_items": int(num_query),
                "gallery_items": len(val_loader.dataset) - int(num_query),
                "official_test_batches": official_test_batches,
                "official_test_evaluation_count": official_test_evaluation_count,
                "reranking": False,
                "checkpoint_sha256": _sha256(fixed_path),
                "evaluator_source_sha256": _sha256(SOURCE / "utils/metrics.py"),
            },
        )
        complete_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_center": optimizer_center.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "center_criterion": center_criterion.state_dict(),
            "rng": _capture_rng(random, np, torch),
            "epoch": 120,
            "phase": "complete",
            "run_identity_sha256": identity_hash,
            "official_test_evaluation_count": 1,
            "resume_history": resume_history,
            "metrics_percent": metrics,
        }
        _save_generation(
            output_dir,
            epoch=120,
            phase="complete",
            state=complete_state,
            torch_module=torch,
        )
        result = {
            "status": "COMPLETE",
            "epoch": 120,
            "phase": "complete",
            "official_test_iteration_count": 1,
            "official_test_loader_iterations_before_fixed_checkpoint": 0,
            "official_test_batches": official_test_batches,
            "exports_saved_before_official_test": True,
            "epoch80_checkpoint": str(epoch80_path),
            "epoch80_checkpoint_sha256": _sha256(epoch80_path),
            "fixed_checkpoint": str(fixed_path),
            "fixed_checkpoint_sha256": _sha256(fixed_path),
            "fixed_checkpoint_receipt_sha256": _sha256(fixed_receipt_path),
            "fixed_eval_sha256": _sha256(fixed_eval_path),
            "metrics_percent": metrics,
            "resume_history": resume_history,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "fatal_or_nonfinite_detected": False,
            "second_gpu_gate": second_gpu_check,
        }
        _atomic_json(result_path, result)
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        status = "OOM" if "out of memory" in message.lower() else "FAILED"
        _atomic_json(
            result_path,
            {
                "status": status,
                "official_test_iteration_count": 0,
                "error": message,
                "traceback": traceback.format_exc(),
                "second_gpu_gate": second_gpu_check,
            },
        )
        return 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capacity", "fixed120"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--_worker", choices=("capacity", "fixed120"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args._worker:
        if args._worker == "capacity":
            return _worker_capacity(args.output_dir)
        return _worker_fixed120(args.output_dir)
    if args.mode is None:
        print("--mode is required", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "capacity":
        receipt, returncode = _capacity(args.output_dir)
        receipt_path = args.output_dir / "capacity.json"
    else:
        receipt, returncode = _fixed120(args.output_dir)
        receipt_path = args.output_dir / "run_summary.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
