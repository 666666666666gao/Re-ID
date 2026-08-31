#!/usr/bin/env python3
"""DeMo's original epoch computation wrapped in durable epoch boundaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.cuda import amp

from tools.demo_resume_state import RunIdentity, run_resumable_epochs


def _resolved_config_sha256(cfg: object) -> str:
    return hashlib.sha256(cfg.dump().encode("utf-8")).hexdigest()


def runtime_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """Hash a canonical runtime descriptor for fail-closed resumption."""

    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_checkpoint_tensor_parity(
    *,
    candidate_path: Path,
    reference_path: Path,
    expected_reference_sha256: str,
) -> dict[str, Any]:
    """Require exact tensor equality with a hash-bound reference checkpoint."""

    candidate_path = Path(candidate_path)
    reference_path = Path(reference_path)
    actual_reference_sha256 = _sha256_file(reference_path)
    if actual_reference_sha256 != expected_reference_sha256:
        raise RuntimeError(
            "parity reference SHA-256 mismatch: "
            f"expected {expected_reference_sha256}, got {actual_reference_sha256}"
        )
    candidate = torch.load(
        candidate_path, map_location="cpu", weights_only=True
    )
    reference = torch.load(
        reference_path, map_location="cpu", weights_only=True
    )
    if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
        raise RuntimeError("parity checkpoints must be state-dict mappings")
    if list(candidate) != list(reference):
        missing = sorted(set(reference) - set(candidate))
        unexpected = sorted(set(candidate) - set(reference))
        raise RuntimeError(
            "checkpoint tensor keys differ: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    for tensor_name in reference:
        candidate_tensor = candidate[tensor_name]
        reference_tensor = reference[tensor_name]
        if not torch.is_tensor(candidate_tensor) or not torch.is_tensor(
            reference_tensor
        ):
            raise RuntimeError(
                f"checkpoint entry is not a tensor: {tensor_name}"
            )
        if (
            candidate_tensor.dtype != reference_tensor.dtype
            or candidate_tensor.shape != reference_tensor.shape
            or not torch.equal(candidate_tensor, reference_tensor)
        ):
            detail = (
                f"candidate(dtype={candidate_tensor.dtype}, "
                f"shape={tuple(candidate_tensor.shape)}) "
                f"reference(dtype={reference_tensor.dtype}, "
                f"shape={tuple(reference_tensor.shape)})"
            )
            if (
                candidate_tensor.shape == reference_tensor.shape
                and candidate_tensor.is_floating_point()
                and reference_tensor.is_floating_point()
            ):
                maximum_error = (
                    candidate_tensor.to(torch.float64)
                    - reference_tensor.to(torch.float64)
                ).abs().max().item()
                detail += f" max_abs_error={maximum_error:.17g}"
            raise RuntimeError(
                f"checkpoint tensor mismatch at {tensor_name}: {detail}"
            )

    return {
        "valid": True,
        "comparison": "exact tensor equality",
        "tensor_count": len(reference),
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": _sha256_file(candidate_path),
        "reference": str(reference_path.resolve()),
        "reference_sha256": actual_reference_sha256,
    }


def _installed_versions() -> dict[str, str]:
    versions = {}
    for package_name in (
        "numpy",
        "torchvision",
        "Pillow",
        "timm",
        "yacs",
        "scipy",
        "scikit-learn",
    ):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "missing"
    return versions


def collect_runtime_descriptor(local_rank: int) -> dict[str, Any]:
    """Capture trajectory-relevant software, driver, flags, and environment."""

    driver_version = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if not driver_version or len(set(driver_version)) != 1:
        raise RuntimeError(
            f"could not identify one NVIDIA driver version: {driver_version!r}"
        )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "driver": driver_version[0],
        "torch": {
            "version": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cuda_device": {
            "name": torch.cuda.get_device_name(local_rank),
            "capability": list(torch.cuda.get_device_capability(local_rank)),
        },
        "backend_flags": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "packages": _installed_versions(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def do_train_resumable(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
    *,
    recovery_checkpoint: Path,
    baseline_commit: str,
    clip_sha256: str,
    recovery_code_sha256: str,
    parity_epoch: int,
    parity_reference: Path | None,
    parity_reference_sha256: str | None,
):
    """Run frozen DeMo semantics with complete post-train/post-eval state."""

    # Imported only after the pinned baseline has been put first on sys.path.
    from engine.processor import training_neat_eval
    from utils.meter import AverageMeter
    from utils.metrics import R1_mAP, R1_mAP_eval

    if cfg.MODEL.DIST_TRAIN:
        raise RuntimeError("resumable DeMo currently supports one GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("resumable DeMo requires CUDA")

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    if checkpoint_period < 1 or eval_period < 1 or epochs < 1:
        raise ValueError("epoch, evaluation, and checkpoint periods must be positive")
    if epochs >= parity_epoch:
        if parity_reference is None or parity_reference_sha256 is None:
            raise RuntimeError("long DeMo run is missing its tensor-parity gate")
        if parity_epoch % checkpoint_period:
            raise ValueError(
                "parity_epoch must coincide with a model checkpoint epoch"
            )

    device = "cuda"
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info("start resumable training")
    model.to(local_rank)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(
            num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
        )
    else:
        evaluator = R1_mAP_eval(
            num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
        )
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM
    initial_best_index = {
        "mAP": 0.0,
        "Rank-1": 0.0,
        "Rank-5": 0.0,
        "Rank-10": 0.0,
    }
    runtime_descriptor = collect_runtime_descriptor(local_rank)
    runtime_sha256 = runtime_descriptor_sha256(runtime_descriptor)
    logger.info(
        "Recovery runtime descriptor: %s",
        json.dumps(runtime_descriptor, sort_keys=True, ensure_ascii=True),
    )
    logger.info("Recovery runtime SHA-256: %s", runtime_sha256)
    identity = RunIdentity(
        baseline_commit=baseline_commit,
        config_sha256=_resolved_config_sha256(cfg),
        clip_sha256=clip_sha256,
        recovery_code_sha256=recovery_code_sha256,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        device_name=torch.cuda.get_device_name(local_rank),
        runtime_sha256=runtime_sha256,
        parity_epoch=parity_epoch,
        parity_reference_sha256=(
            parity_reference_sha256
            if parity_reference_sha256 is not None
            else "not-applicable"
        ),
    )

    if Path(recovery_checkpoint).exists():
        logger.info(
            "Recovery checkpoint found; validation precedes all resumed work: %s",
            recovery_checkpoint,
        )
    else:
        logger.info(
            "No recovery checkpoint found; recording epoch-0 boundary: %s",
            recovery_checkpoint,
        )

    def train_epoch(epoch: int) -> None:
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        n_iter = -1
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {
                "RGB": img["RGB"].to(device),
                "NI": img["NI"].to(device),
                "TI": img["TI"].to(device),
            }
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(
                    img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view,
                )
                loss = 0
                if len(output) % 2 == 1:
                    index = len(output) - 1
                    for output_index in range(0, index, 2):
                        loss_tmp = loss_fn(
                            score=output[output_index],
                            feat=output[output_index + 1],
                            target=target,
                            target_cam=target_cam,
                        )
                        loss = loss + loss_tmp
                    loss = loss + output[-1]
                else:
                    for output_index in range(0, len(output), 2):
                        loss_tmp = loss_fn(
                            score=output[output_index],
                            feat=output[output_index + 1],
                            target=target,
                            target_cam=target_cam,
                        )
                        loss = loss + loss_tmp
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for parameter in center_criterion.parameters():
                    parameter.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img["RGB"].shape[0])
            acc_meter.update(acc, 1)
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[%s] Iteration[%s/%s] Loss: %.3f, Acc: %.3f, Base Lr: %.2e",
                    epoch,
                    n_iter + 1,
                    len(train_loader),
                    loss_meter.avg,
                    acc_meter.avg,
                    scheduler._get_lr(epoch)[0],
                )

        if n_iter < 0:
            raise RuntimeError("DeMo training loader yielded no batches")
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info(
            "Epoch %s done. Time per batch: %.3f[s] Speed: %.1f[samples/s]",
            epoch,
            time_per_batch,
            train_loader.batch_size / time_per_batch,
        )

        model_checkpoint = Path(cfg.OUTPUT_DIR) / (
            f"{cfg.MODEL.NAME}_{epoch}.pth"
        )
        if epoch % checkpoint_period == 0:
            torch.save(
                model.state_dict(),
                model_checkpoint,
            )
        if parity_reference is not None and epoch == parity_epoch:
            parity_receipt = assert_checkpoint_tensor_parity(
                candidate_path=model_checkpoint,
                reference_path=parity_reference,
                expected_reference_sha256=parity_reference_sha256,
            )
            parity_receipt.update(
                {
                    "epoch": epoch,
                    "baseline": "DeMo",
                    "baseline_commit": baseline_commit,
                    "config_sha256": identity.config_sha256,
                    "recovery_code_sha256": recovery_code_sha256,
                    "runtime_sha256": runtime_sha256,
                }
            )
            receipt_path = (
                Path(cfg.OUTPUT_DIR)
                / ".resume"
                / f"parity_epoch{epoch}.json"
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_receipt = receipt_path.with_suffix(".json.tmp")
            temporary_receipt.write_text(
                json.dumps(
                    parity_receipt,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_receipt, receipt_path)
            logger.info(
                "Epoch-%d exact tensor parity PASS: %d tensors; receipt=%s",
                epoch,
                parity_receipt["tensor_count"],
                receipt_path,
            )

    def evaluate_epoch(epoch: int, best_index: dict[str, float]):
        if test_sign:
            training_neat_eval(
                cfg,
                model,
                val_loader,
                device,
                evaluator,
                epoch,
                logger,
                return_pattern=1,
            )
            training_neat_eval(
                cfg,
                model,
                val_loader,
                device,
                evaluator,
                epoch,
                logger,
                return_pattern=2,
            )
        mAP, cmc = training_neat_eval(
            cfg,
            model,
            val_loader,
            device,
            evaluator,
            epoch,
            logger,
            return_pattern=3,
        )
        updated_best = dict(best_index)
        if mAP >= updated_best["mAP"]:
            updated_best["mAP"] = float(mAP)
            updated_best["Rank-1"] = float(cmc[0])
            updated_best["Rank-5"] = float(cmc[4])
            updated_best["Rank-10"] = float(cmc[9])
            torch.save(
                model.state_dict(),
                os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}best.pth"),
            )
        logger.info("%s", "~" * 50)
        logger.info("Best mAP: %.1f%%", 100 * updated_best["mAP"])
        logger.info("Best Rank-1: %.1f%%", 100 * updated_best["Rank-1"])
        logger.info("Best Rank-5: %.1f%%", 100 * updated_best["Rank-5"])
        logger.info("Best Rank-10: %.1f%%", 100 * updated_best["Rank-10"])
        logger.info("%s", "~" * 50)
        return updated_best

    cursor = run_resumable_epochs(
        checkpoint_path=Path(recovery_checkpoint),
        identity=identity,
        max_epochs=epochs,
        eval_period=eval_period,
        initial_best_index=initial_best_index,
        train_epoch=train_epoch,
        evaluate_epoch=evaluate_epoch,
        model=model,
        optimizer=optimizer,
        center_criterion=center_criterion,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        scaler=scaler,
    )
    logger.info(
        "Resumable DeMo endpoint complete: epoch=%d phase=%s",
        cursor.epoch,
        cursor.phase,
    )
    return cursor
