#!/usr/bin/env python3
"""Audit a pinned PEFT-BoA checkout without constructing its CUDA model.

The audit proves source provenance, the published RGBNT201 configuration, and
one real CPU loader batch.  It deliberately does not claim training parity or
runtime compatibility for the model itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


RECEIPT_PREFIX = "PEFT_BOA_LOADER_RECEIPT="
EXPECTED_CONFIG = {
    "checkpoint_period": 60,
    "dataset": "RGBNT201",
    "eval_period": 1,
    "frozen": True,
    "max_epochs": 120,
    "num_instance": 4,
    "reranking": "no",
    "seed": 1111,
    "size_test": [256, 128],
    "size_train": [256, 128],
    "test_batch": 64,
    "train_batch": 64,
}
EXPECTED_DATASET = {
    "num_query": 836,
    "num_classes": 171,
    "num_cameras": 4,
    "num_views": 1,
}
EXPECTED_REMOTE = "https://github.com/fffunly/PEFT-BoA.git"
EXPECTED_PRETRAIN_BYTES = 350_837_078
EXPECTED_PRETRAIN_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)
SMOKE_SEED = 42
AUDITED_FILES = (
    "requirements.txt",
    "configs/RGBNT201/PEFT-BoA.yml",
    "data/datasets/RGBNT201.py",
    "data/datasets/make_dataloader.py",
    "engine/processor.py",
    "modeling/make_model.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def parse_loader_receipt(stdout: str) -> dict[str, Any]:
    payloads = [
        line[len(RECEIPT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    if len(payloads) != 1:
        raise ValueError(
            f"expected exactly one {RECEIPT_PREFIX!r} line, found {len(payloads)}"
        )
    try:
        decoded = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise ValueError("loader receipt is not valid JSON object") from error
    if not isinstance(decoded, dict):
        raise ValueError("loader receipt is not valid JSON object")
    return decoded


def validate_probe_receipt(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_config = receipt.get("source_config")
    if not isinstance(source_config, dict):
        return ["source_config must be an object"]

    for key, expected in EXPECTED_CONFIG.items():
        actual = source_config.get(key)
        if actual != expected:
            errors.append(f"source_config.{key}: expected {expected!r}, got {actual!r}")

    for key, expected in EXPECTED_DATASET.items():
        actual = receipt.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    if receipt.get("smoke_seed") != SMOKE_SEED:
        errors.append(
            f"smoke_seed: expected {SMOKE_SEED!r}, got {receipt.get('smoke_seed')!r}"
        )

    smoke_batch = receipt.get("effective_smoke_batch")
    if not isinstance(smoke_batch, int) or smoke_batch <= 0:
        errors.append(f"effective_smoke_batch must be positive int, got {smoke_batch!r}")
        expected_image_shape: list[int] | None = None
    else:
        expected_image_shape = [smoke_batch, 3, 256, 128]

    if expected_image_shape is not None:
        for modality_key in ("rgb_shape", "nir_shape", "tir_shape"):
            actual = receipt.get(modality_key)
            if actual != expected_image_shape:
                errors.append(
                    f"{modality_key}: expected {expected_image_shape!r}, got {actual!r}"
                )
        expected_pid_shape = [smoke_batch]
        if receipt.get("pid_shape") != expected_pid_shape:
            errors.append(
                f"pid_shape: expected {expected_pid_shape!r}, got {receipt.get('pid_shape')!r}"
            )
        num_instance = source_config.get("num_instance")
        if isinstance(num_instance, int) and smoke_batch % num_instance == 0:
            expected_unique = smoke_batch // num_instance
            if receipt.get("pid_unique") != expected_unique:
                errors.append(
                    f"pid_unique: expected {expected_unique!r}, got {receipt.get('pid_unique')!r}"
                )

    if receipt.get("pretrain_exists") is not True:
        errors.append(
            f"pretrain_exists: expected True, got {receipt.get('pretrain_exists')!r}"
        )
    return errors


def detect_protocol_risks(processor_text: str, *, eval_period: int) -> dict[str, bool]:
    evaluates_periodically = (
        "epoch % eval_period == 0" in processor_text
        and "evaluator.compute()" in processor_text
    )
    selects_by_test_map = bool(
        re.search(r"mAP\s*>=\s*best_index\[['\"]mAP['\"]\]", processor_text)
    )
    saves_best = "best.pth" in processor_text
    saves_model_state_only = "torch.save(model.state_dict()" in processor_text
    saves_optimizer = "optimizer.state_dict()" in processor_text
    return {
        "official_test_each_epoch": eval_period == 1 and evaluates_periodically,
        "test_selected_best_checkpoint": selects_by_test_map and saves_best,
        "model_only_periodic_checkpoint": saves_model_state_only and not saves_optimizer,
    }


def validate_source_provenance(
    *,
    actual_commit: str,
    expected_commit: str,
    remote: str,
    status_before: str,
    status_after: str,
) -> list[str]:
    errors: list[str] = []
    if actual_commit != expected_commit:
        errors.append(
            f"source_commit: expected {expected_commit!r}, got {actual_commit!r}"
        )
    if remote != EXPECTED_REMOTE:
        errors.append(f"source_remote: expected {EXPECTED_REMOTE!r}, got {remote!r}")
    if status_before:
        errors.append(f"source tree dirty before audit: {status_before!r}")
    if status_after:
        errors.append(f"source tree dirty after audit: {status_after!r}")
    return errors


def validate_pretrain_identity(
    *, actual_bytes: int, actual_sha256: str
) -> list[str]:
    errors: list[str] = []
    if actual_bytes != EXPECTED_PRETRAIN_BYTES:
        errors.append(
            f"pretrain_bytes: expected {EXPECTED_PRETRAIN_BYTES}, got {actual_bytes}"
        )
    if actual_sha256 != EXPECTED_PRETRAIN_SHA256:
        errors.append(
            f"pretrain_sha256: expected {EXPECTED_PRETRAIN_SHA256!r}, "
            f"got {actual_sha256!r}"
        )
    return errors


def build_loader_probe(
    *,
    config_path: Path,
    data_root: Path,
    pretrain_path: Path,
    smoke_batch: int,
) -> str:
    values = {
        "config_path": str(config_path),
        "data_root": str(data_root),
        "pretrain_path": str(pretrain_path),
        "smoke_batch": smoke_batch,
        "receipt_prefix": RECEIPT_PREFIX,
        "smoke_seed": SMOKE_SEED,
    }
    encoded = json.dumps(values, sort_keys=True)
    return f"""
import json
from pathlib import Path

arguments = json.loads({encoded!r})
from config import cfg

cfg.merge_from_file(arguments["config_path"])
source_config = {{
    "checkpoint_period": int(cfg.SOLVER.CHECKPOINT_PERIOD),
    "dataset": cfg.DATASETS.NAMES,
    "eval_period": int(cfg.SOLVER.EVAL_PERIOD),
    "frozen": bool(cfg.MODEL.FROZEN),
    "max_epochs": int(cfg.SOLVER.MAX_EPOCHS),
    "num_instance": int(cfg.DATALOADER.NUM_INSTANCE),
    "reranking": cfg.TEST.RE_RANKING,
    "seed": int(cfg.SOLVER.SEED),
    "size_test": list(cfg.INPUT.SIZE_TEST),
    "size_train": list(cfg.INPUT.SIZE_TRAIN),
    "test_batch": int(cfg.TEST.IMS_PER_BATCH),
    "train_batch": int(cfg.SOLVER.IMS_PER_BATCH),
}}
cfg.merge_from_list([
    "DATASETS.ROOT_DIR", arguments["data_root"],
    "DATALOADER.NUM_WORKERS", "0",
    "SOLVER.IMS_PER_BATCH", str(arguments["smoke_batch"]),
    "TEST.IMS_PER_BATCH", "64",
    "MODEL.PRETRAIN_PATH_T", arguments["pretrain_path"],
])
cfg.freeze()

import torch
import torchvision
import timm
import random
import numpy as np

random.seed(arguments["smoke_seed"])
np.random.seed(arguments["smoke_seed"])
torch.manual_seed(arguments["smoke_seed"])
from data import make_dataloader

train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
images, pids, cameras, views, paths = next(iter(train_loader))
receipt = {{
    "source_config": source_config,
    "effective_smoke_batch": int(arguments["smoke_batch"]),
    "smoke_seed": int(arguments["smoke_seed"]),
    "num_query": int(num_query),
    "num_classes": int(num_classes),
    "num_cameras": int(camera_num),
    "num_views": int(view_num),
    "train_batches": int(len(train_loader)),
    "val_batches": int(len(val_loader)),
    "rgb_shape": list(images["RGB"].shape),
    "nir_shape": list(images["NI"].shape),
    "tir_shape": list(images["TI"].shape),
    "pid_shape": list(pids.shape),
    "pid_unique": int(pids.unique().numel()),
    "camera_batch_min": int(cameras.min()),
    "camera_batch_max": int(cameras.max()),
    "pretrain_exists": Path(cfg.MODEL.PRETRAIN_PATH_T).is_file(),
    "runtime": {{
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
    }},
}}
print(arguments["receipt_prefix"] + json.dumps(receipt, sort_keys=True))
"""


def audit_source(
    *,
    source: Path,
    expected_commit: str,
    data_root: Path,
    pretrain_path: Path,
    smoke_batch: int,
) -> dict[str, Any]:
    source = source.resolve()
    data_root = data_root.resolve()
    pretrain_path = pretrain_path.resolve()
    missing_files = [relative for relative in AUDITED_FILES if not (source / relative).is_file()]
    if missing_files:
        raise FileNotFoundError(f"missing PEFT-BoA source files: {missing_files}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing data root: {data_root}")
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"missing CLIP pretrain: {pretrain_path}")

    actual_commit = run_checked(["git", "rev-parse", "HEAD"], cwd=source)
    source_status_before = run_checked(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=source
    )
    remote = run_checked(["git", "remote", "get-url", "origin"], cwd=source)

    config_path = source / "configs/RGBNT201/PEFT-BoA.yml"
    probe = build_loader_probe(
        config_path=config_path,
        data_root=data_root,
        pretrain_path=pretrain_path,
        smoke_batch=smoke_batch,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe_result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe_result.returncode != 0:
        raise RuntimeError(
            f"PEFT-BoA loader probe failed ({probe_result.returncode})\n"
            f"stdout:\n{probe_result.stdout}\nstderr:\n{probe_result.stderr}"
        )
    receipt = parse_loader_receipt(probe_result.stdout)

    source_status_after = run_checked(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=source
    )
    processor_text = (source / "engine/processor.py").read_text(encoding="utf-8")
    model_text = (source / "modeling/make_model.py").read_text(encoding="utf-8")
    eval_period = int(receipt["source_config"]["eval_period"])
    protocol_risks = detect_protocol_risks(
        processor_text,
        eval_period=eval_period,
    )

    errors = validate_probe_receipt(receipt)
    errors.extend(
        validate_source_provenance(
            actual_commit=actual_commit,
            expected_commit=expected_commit,
            remote=remote,
            status_before=source_status_before,
            status_after=source_status_after,
        )
    )
    pretrain_bytes = pretrain_path.stat().st_size
    pretrain_sha256 = sha256_file(pretrain_path)
    errors.extend(
        validate_pretrain_identity(
            actual_bytes=pretrain_bytes,
            actual_sha256=pretrain_sha256,
        )
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "baseline": "PEFT-BoA",
        "audit_scope": "pinned source plus real CPU RGBNT201 loader batch; no model construction",
        "source": {
            "path": str(source),
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "remote": remote,
            "status_before": source_status_before,
            "status_after": source_status_after,
            "files_sha256": {
                relative: sha256_file(source / relative) for relative in AUDITED_FILES
            },
        },
        "paths": {
            "data_root": str(data_root),
            "pretrain": str(pretrain_path),
            "pretrain_bytes": pretrain_bytes,
            "pretrain_sha256": pretrain_sha256,
        },
        "loader_probe": receipt,
        "protocol_risks": protocol_risks,
        "compatibility_observations": {
            "requirements_torch": "2.1.1+cu118",
            "audited_runtime_torch": receipt.get("runtime", {}).get("torch"),
            "model_constructor_forces_cuda": 'vit_model.to("cuda")' in model_text,
            "model_forward_backward_tested": False,
            "published_checkpoint_available": False,
        },
        "training_readiness": {
            "ready": False,
            "blockers": [
                "upstream evaluates official test every epoch and saves a test-mAP-selected best checkpoint",
                "upstream periodic checkpoints contain model weights only, not optimizer/scheduler/scaler/RNG state",
                "model forward/backward compatibility and 8 GiB capacity have not passed the GPU idle gate",
            ],
        },
        "errors": errors,
        "valid": not errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/mmreid-trifusion/baselines/PEFT-BoA"),
    )
    parser.add_argument(
        "--expected-commit",
        default="d2b198be634ac4f9f5744eebf6e0a6604e490deb",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/data"),
    )
    parser.add_argument(
        "--pretrain",
        type=Path,
        default=Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt"),
    )
    parser.add_argument("--smoke-batch", type=int, default=32)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.smoke_batch <= 0 or args.smoke_batch % EXPECTED_CONFIG["num_instance"] != 0:
        parser.error("--smoke-batch must be positive and divisible by NUM_INSTANCE=4")

    report = audit_source(
        source=args.source,
        expected_commit=args.expected_commit,
        data_root=args.data_root,
        pretrain_path=args.pretrain,
        smoke_batch=args.smoke_batch,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        output = args.json_out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
