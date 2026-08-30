#!/usr/bin/env python3
"""Launch unmodified DeMo training with explicit, reproducible local paths."""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import torch


DEMO_COMMIT = "b4f323a430b32e3a1637c3e7acb25868cb52e9cd"


def assert_gpu_idle(device: int, max_used_mib: int) -> int:
    if max_used_mib < 0:
        raise ValueError("max-idle-gpu-memory-mib must be non-negative")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for the GPU occupancy preflight")
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        used_mib = int(output)
    except ValueError as error:
        raise RuntimeError(
            f"could not parse GPU {device} memory usage: {output!r}"
        ) from error
    if used_mib > max_used_mib:
        raise RuntimeError(
            f"GPU {device} is busy: {used_mib} MiB used "
            f"(limit {max_used_mib} MiB)"
        )
    return used_mib


def validate_baseline_checkout(baseline_root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(baseline_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != DEMO_COMMIT:
        raise RuntimeError(
            f"DeMo checkout is {commit}; expected frozen commit {DEMO_COMMIT}"
        )
    if tracked_status:
        raise RuntimeError(
            "DeMo checkout has tracked modifications; refuse non-reproducible launch:\n"
            f"{tracked_status}"
        )
    return commit


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
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--eval-period", type=int, default=1)
    parser.add_argument("--checkpoint-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-idle-gpu-memory-mib", type=int, default=512)
    args = parser.parse_args()

    if args.batch_size % args.num_instances:
        raise ValueError("batch-size must be divisible by num-instances")
    baseline_root = args.baseline_root.resolve()
    config_path = baseline_root / "configs/RGBNT201/DeMo.yml"
    train_entry = baseline_root / "tools/train.py"
    for required in (
        baseline_root,
        args.data_root / "RGBNT201",
        args.clip_weight,
        config_path,
        train_entry,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    baseline_commit = validate_baseline_checkout(baseline_root)
    gpu_memory_used_mib = assert_gpu_idle(
        args.device, args.max_idle_gpu_memory_mib
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = args.output_dir.resolve()
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validated DeMo baseline commit: {baseline_commit}", flush=True)
    print(
        f"GPU {args.device} occupancy preflight: {gpu_memory_used_mib} MiB used",
        flush=True,
    )

    sys.path.insert(0, str(baseline_root))
    os.chdir(baseline_root)
    import modeling.meta_arch as meta_arch  # pylint: disable=import-error,import-outside-toplevel
    from modeling.clip import clip  # pylint: disable=import-error,import-outside-toplevel

    def configured_clip_loader(config, _name, height, width, stride):
        archive = torch.jit.load(
            config.MODEL.PRETRAIN_PATH_T, map_location="cpu"
        ).eval()
        return clip.build_model(config, archive.state_dict(), height, width, stride)

    meta_arch.load_clip_to_cpu = configured_clip_loader
    sys.argv = [
        str(train_entry),
        "--config_file",
        str(config_path),
        "MODEL.PRETRAIN_PATH_T",
        str(args.clip_weight.resolve()),
        "MODEL.DEVICE_ID",
        repr(str(args.device)),
        "DATASETS.ROOT_DIR",
        str(args.data_root.resolve()),
        "DATALOADER.NUM_WORKERS",
        str(args.workers),
        "DATALOADER.NUM_INSTANCE",
        str(args.num_instances),
        "SOLVER.IMS_PER_BATCH",
        str(args.batch_size),
        "TEST.IMS_PER_BATCH",
        str(args.test_batch_size),
        "SOLVER.MAX_EPOCHS",
        str(args.max_epochs),
        "SOLVER.EVAL_PERIOD",
        str(args.eval_period),
        "SOLVER.CHECKPOINT_PERIOD",
        str(args.checkpoint_period),
        "SOLVER.SEED",
        str(args.seed),
        "OUTPUT_DIR",
        str(output_dir),
    ]
    runpy.run_path(str(train_entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
