#!/usr/bin/env python3
"""Launch frozen DeMo with exact epoch-boundary crash recovery."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import runpy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.demo_resumable_training import do_train_resumable
from tools.run_demo_baseline import (
    assert_gpu_idle,
    validate_baseline_checkout,
)

DEMO_PARITY_EPOCH = 10
DEMO_PARITY_REFERENCE_SHA256 = (
    "b2ab79f056d73d6b827c52fd27ec0607aeae1a10cd756db5c0cc62f3ab4631c0"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recovery_code_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in (
        Path("tools/demo_resume_state.py"),
        Path("tools/demo_resumable_training.py"),
        Path("tools/run_demo_resumable.py"),
        Path("tools/run_demo_baseline.py"),
        Path("scripts/run_demo_rgbnt201_seed42_resumable.sh"),
    ):
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((project_root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_reproducibility_environment(*, seed: int) -> dict[str, str]:
    """Reject launch settings that cannot reproduce the frozen RNG trajectory."""

    expected = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    mismatches = [
        f"{name}={value}"
        for name, value in expected.items()
        if os.environ.get(name) != value
    ]
    if mismatches:
        raise RuntimeError(
            "reproducibility environment must be set before Python starts: "
            + ", ".join(mismatches)
        )
    return expected


def validate_parity_gate_options(
    *,
    max_epochs: int,
    parity_epoch: int,
    reference_path: Path | None,
    expected_sha256: str | None,
    required_reference_sha256: str = DEMO_PARITY_REFERENCE_SHA256,
) -> tuple[Path | None, str | None]:
    """Require a byte-identified reference before a run can cross its gate."""

    if parity_epoch != DEMO_PARITY_EPOCH:
        raise ValueError(
            f"frozen parity epoch is {DEMO_PARITY_EPOCH}, got {parity_epoch}"
        )
    if max_epochs < parity_epoch:
        return None, None
    if reference_path is None or expected_sha256 is None:
        raise RuntimeError(
            f"parity reference is required for runs reaching epoch {parity_epoch}"
        )
    resolved_reference = Path(reference_path).resolve()
    if not resolved_reference.is_file():
        raise FileNotFoundError(resolved_reference)
    normalized_sha256 = expected_sha256.lower()
    if (
        len(normalized_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in normalized_sha256
        )
    ):
        raise ValueError("parity reference SHA-256 is invalid")
    if normalized_sha256 != required_reference_sha256:
        raise RuntimeError(
            "frozen parity reference SHA-256 must be "
            f"{required_reference_sha256}, got {normalized_sha256}"
        )
    actual_sha256 = _sha256_file(resolved_reference)
    if actual_sha256 != normalized_sha256:
        raise RuntimeError(
            "parity reference SHA-256 mismatch: "
            f"expected {normalized_sha256}, got {actual_sha256}"
        )
    return resolved_reference, normalized_sha256


def _configure_resumable_logger(*, append: bool) -> None:
    from utils import logger as baseline_logger

    def setup_logger(name, save_dir, if_train):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s"
        )
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            log_name = "train_log.txt" if if_train else "test_log.txt"
            file_handler = logging.FileHandler(
                Path(save_dir) / log_name,
                mode="a" if append else "w",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    baseline_logger.setup_logger = setup_logger


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--eval-period", type=int, default=1)
    parser.add_argument("--checkpoint-period", type=int, default=10)
    parser.add_argument("--parity-reference", type=Path)
    parser.add_argument("--parity-reference-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-idle-gpu-memory-mib", type=int, default=499)
    args = parser.parse_args()
    validate_reproducibility_environment(seed=args.seed)

    if (
        args.batch_size < 1
        or args.test_batch_size < 1
        or args.num_instances < 1
        or args.workers < 0
        or args.max_epochs < 1
        or args.eval_period < 1
        or args.checkpoint_period < 1
        or args.seed < 0
        or args.device < 0
    ):
        raise ValueError("batch, worker, epoch, and period options are invalid")
    if args.batch_size % args.num_instances:
        raise ValueError("batch-size must be divisible by num-instances")

    project_root = PROJECT_ROOT
    baseline_root = args.baseline_root.resolve()
    data_root = args.data_root.resolve()
    clip_weight = args.clip_weight.resolve()
    output_dir = args.output_dir.resolve()
    recovery_checkpoint = output_dir / ".resume" / "latest.json"
    config_path = baseline_root / "configs/RGBNT201/DeMo.yml"
    train_entry = baseline_root / "tools/train.py"
    for required in (
        baseline_root,
        data_root / "RGBNT201",
        clip_weight,
        config_path,
        train_entry,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    baseline_commit = validate_baseline_checkout(baseline_root)
    parity_reference, parity_reference_sha256 = validate_parity_gate_options(
        max_epochs=args.max_epochs,
        parity_epoch=DEMO_PARITY_EPOCH,
        reference_path=args.parity_reference,
        expected_sha256=args.parity_reference_sha256,
    )

    has_output = output_dir.is_dir() and any(output_dir.iterdir())
    if has_output and not recovery_checkpoint.is_file():
        raise FileExistsError(
            "refusing a non-empty output without a complete recovery checkpoint: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    gpu_memory_used_mib = assert_gpu_idle(
        args.device, args.max_idle_gpu_memory_mib
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    clip_sha256 = _sha256_file(clip_weight)
    recovery_code_sha256 = _recovery_code_sha256(project_root)
    print(f"Validated DeMo baseline commit: {baseline_commit}", flush=True)
    print(
        f"GPU {args.device} occupancy preflight: {gpu_memory_used_mib} MiB used",
        flush=True,
    )
    print(f"CLIP weight SHA-256: {clip_sha256}", flush=True)
    print(f"Recovery code SHA-256: {recovery_code_sha256}", flush=True)
    if parity_reference is not None:
        print(
            f"Epoch-{DEMO_PARITY_EPOCH} parity reference: "
            f"{parity_reference} ({parity_reference_sha256})",
            flush=True,
        )

    sys.path.insert(0, str(baseline_root))
    os.chdir(baseline_root)
    import engine.processor as baseline_processor
    import modeling.meta_arch as meta_arch
    from modeling.clip import clip

    def configured_clip_loader(config, _name, height, width, stride):
        archive = torch.jit.load(
            config.MODEL.PRETRAIN_PATH_T, map_location="cpu"
        ).eval()
        return clip.build_model(
            config, archive.state_dict(), height, width, stride
        )

    meta_arch.load_clip_to_cpu = configured_clip_loader
    _configure_resumable_logger(append=recovery_checkpoint.is_file())

    def patched_do_train(
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
    ):
        return do_train_resumable(
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
            recovery_checkpoint=recovery_checkpoint,
            baseline_commit=baseline_commit,
            clip_sha256=clip_sha256,
            recovery_code_sha256=recovery_code_sha256,
            parity_epoch=DEMO_PARITY_EPOCH,
            parity_reference=parity_reference,
            parity_reference_sha256=parity_reference_sha256,
        )

    baseline_processor.do_train = patched_do_train
    sys.argv = [
        str(train_entry),
        "--config_file",
        str(config_path),
        "MODEL.PRETRAIN_PATH_T",
        str(clip_weight),
        "MODEL.DEVICE_ID",
        repr(str(args.device)),
        "DATASETS.ROOT_DIR",
        str(data_root),
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
