#!/usr/bin/env python3
"""Run one real RGBNT201 DeMo optimization step with the official loss stack."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/baselines/DeMo"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/data"),
    )
    parser.add_argument(
        "--clip-weight",
        type=Path,
        default=Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "/root/mmreid-trifusion/artifacts/demo_real_step_probe_20260831.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size < 2 or args.batch_size % args.num_instances:
        raise ValueError("batch-size must be divisible by num-instances")
    config_path = args.baseline_root / "configs/RGBNT201/DeMo.yml"
    for required in (
        args.baseline_root,
        args.data_root / "RGBNT201",
        args.clip_weight,
        config_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    baseline_root = args.baseline_root.resolve()
    sys.path.insert(0, str(baseline_root))
    os.chdir(baseline_root)
    from config import cfg  # pylint: disable=import-error,import-outside-toplevel
    from data import make_dataloader  # pylint: disable=import-error,import-outside-toplevel
    from layers.make_loss import make_loss  # pylint: disable=import-error,import-outside-toplevel
    import modeling.meta_arch as meta_arch  # pylint: disable=import-error,import-outside-toplevel
    from modeling import make_model  # pylint: disable=import-error,import-outside-toplevel
    from modeling.clip import clip  # pylint: disable=import-error,import-outside-toplevel
    from solver.make_optimizer import make_optimizer  # pylint: disable=import-error,import-outside-toplevel

    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(
        [
            "MODEL.PRETRAIN_PATH_T",
            str(args.clip_weight.resolve()),
            "DATASETS.ROOT_DIR",
            str(args.data_root.resolve()),
            "SOLVER.IMS_PER_BATCH",
            str(args.batch_size),
            "DATALOADER.NUM_INSTANCE",
            str(args.num_instances),
            "DATALOADER.NUM_WORKERS",
            str(args.workers),
            "SOLVER.SEED",
            str(args.seed),
            "OUTPUT_DIR",
            str(args.json_out.resolve().parent / "demo_probe"),
        ]
    )
    cfg.MODEL.DEVICE_ID = str(args.device)
    cfg.freeze()

    def configured_clip_loader(config, _name, height, width, stride):
        archive = torch.jit.load(
            config.MODEL.PRETRAIN_PATH_T, map_location="cpu"
        ).eval()
        return clip.build_model(config, archive.state_dict(), height, width, stride)

    meta_arch.load_clip_to_cpu = configured_clip_loader
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    (
        train_loader,
        _train_loader_normal,
        _val_loader,
        _num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    ).to(device)
    model.train()
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scaler = torch.amp.GradScaler("cuda")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.cuda.reset_peak_memory_stats(device)
    step_records = []
    batch_iterator = iter(train_loader)
    first_shapes = None
    first_multiplicities = None
    total_started = time.perf_counter()
    for step_index in range(args.steps):
        images, identities, cameras, views, _paths = next(batch_iterator)
        identity_counts = collections.Counter(identities.tolist())
        sampler_valid = (
            len(identity_counts) == args.batch_size // args.num_instances
            and set(identity_counts.values()) == {args.num_instances}
        )
        images = {name: tensor.to(device) for name, tensor in images.items()}
        if first_shapes is None:
            first_shapes = {name: list(tensor.shape) for name, tensor in images.items()}
            first_multiplicities = sorted(identity_counts.values())
        identities = identities.to(device)
        cameras = cameras.to(device)
        views = views.to(device)
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                images,
                label=identities,
                cam_label=cameras,
                view_label=views,
            )
            loss = torch.zeros((), device=device)
            paired_output_count = len(output) - (len(output) % 2)
            for index in range(0, paired_output_count, 2):
                loss = loss + loss_func(
                    score=output[index],
                    feat=output[index + 1],
                    target=identities,
                    target_cam=cameras,
                )
            if len(output) % 2:
                loss = loss + output[-1]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in parameters]
        gradients_present = all(gradient is not None for gradient in gradients)
        gradients_finite = all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        scale_after = float(scaler.get_scale())
        step_records.append(
            {
                "step": step_index + 1,
                "loss": float(loss.detach()),
                "loss_finite": bool(torch.isfinite(loss)),
                "gradients_present": gradients_present,
                "gradients_finite": gradients_finite,
                "identity_sampler_valid": sampler_valid,
                "loss_scale_before": scale_before,
                "loss_scale_after": scale_after,
                "optimizer_step_applied": gradients_finite and scale_after >= scale_before,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    elapsed = time.perf_counter() - total_started
    stable_tail = step_records[-min(2, len(step_records)) :]
    checks = {
        "all_losses_finite": all(record["loss_finite"] for record in step_records),
        "all_gradients_present": all(
            record["gradients_present"] for record in step_records
        ),
        "all_identity_samples_valid": all(
            record["identity_sampler_valid"] for record in step_records
        ),
        "optimizer_step_applied": any(
            record["optimizer_step_applied"] for record in step_records
        ),
        "stable_tail_gradients_finite": all(
            record["gradients_finite"] for record in stable_tail
        ),
    }
    checks["parameters_finite_after_step"] = all(
        bool(torch.isfinite(parameter).all()) for parameter in parameters
    )

    commit = subprocess.run(
        ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "baseline": "DeMo",
        "baseline_commit": commit,
        "dataset": "RGBNT201/train_171",
        "num_classes": num_classes,
        "batch_size": args.batch_size,
        "num_instances": args.num_instances,
        "steps": args.steps,
        "identities_per_batch": args.batch_size // args.num_instances,
        "identity_multiplicities": first_multiplicities,
        "input_shapes": first_shapes,
        "step_records": step_records,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.batch_size * args.steps / elapsed,
        "trainable_parameter_tensors": len(parameters),
        "gradient_tensors_present_last_step": sum(
            gradient is not None for gradient in gradients
        ),
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "amp": "fp16",
        "compatibility_override": {
            "reason": "official DeMo loader hard-codes an unavailable CLIP path",
            "upstream_modified": False,
            "configured_weight": str(args.clip_weight.resolve()),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
