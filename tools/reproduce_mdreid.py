#!/usr/bin/env python3
"""Reproduce the official MDReID RGBNT201 checkpoint without its broken test entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


MDREID_COMMIT = "3525ac2da1a2a90a5a160c930fac674b4f226f6c"
EXPECTED_AUDIT_IDENTITY_CHECKS = {
    "train_171": True,
    "train_141": True,
    "test": True,
}
EXPECTED_AUDIT_TRIPLET_CHECKS = {
    "train_171": True,
    "train_141": True,
    "test": True,
}
EXPECTED_AUDIT_CAMERA_CHECKS = {
    "train_171": True,
    "train_141": True,
    "test": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_baseline_checkout(checkout: Path) -> str:
    commit = git_commit(checkout)
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != MDREID_COMMIT:
        raise RuntimeError(
            f"MDReID checkout is {commit}; expected frozen commit {MDREID_COMMIT}"
        )
    if tracked_status:
        raise RuntimeError(
            "MDReID checkout has tracked modifications; refuse invalid parity run:\n"
            f"{tracked_status}"
        )
    return commit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/baselines/MDReID"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/RGBNT201/MDReID.yml"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/data"),
    )
    parser.add_argument(
        "--clip-weight",
        type=Path,
        default=Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/root/mmreid-trifusion/checkpoints/MDReID/RGBNT201_MDReIDbest.pth"
        ),
    )
    parser.add_argument(
        "--dataset-audit",
        type=Path,
        default=Path("/root/mmreid-trifusion/artifacts/rgbnt201_audit_20260831.json"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "/root/mmreid-trifusion/artifacts/mdreid_rgbnt201_eval_20260831.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-map", type=float, default=0.821)
    parser.add_argument("--expected-rank1", type=float, default=0.852)
    parser.add_argument("--parity-tolerance", type=float, default=0.0005)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = baseline_root / config_path
    required = [
        baseline_root,
        config_path,
        args.dataset_root,
        args.clip_weight,
        args.checkpoint,
        args.dataset_audit,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing reproduction inputs: {missing}")
    baseline_commit = validate_baseline_checkout(baseline_root)
    if not torch.cuda.is_available():
        raise RuntimeError("MDReID checkpoint reproduction requires CUDA")

    sys.path.insert(0, str(baseline_root))
    os.chdir(baseline_root)
    from config import cfg  # pylint: disable=import-error,import-outside-toplevel
    from data import make_dataloader  # pylint: disable=import-error,import-outside-toplevel
    from modeling import make_model  # pylint: disable=import-error,import-outside-toplevel
    from utils.metrics import R1_mAP_eval  # pylint: disable=import-error,import-outside-toplevel

    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(
        [
            "DATASETS.ROOT_DIR",
            str(args.dataset_root.resolve()),
            "MODEL.PRETRAIN_PATH_T",
            str(args.clip_weight.resolve()),
            "DATALOADER.NUM_WORKERS",
            str(args.workers),
            "TEST.IMS_PER_BATCH",
            str(args.batch_size),
            "OUTPUT_DIR",
            str(args.json_out.resolve().parent / "mdreid"),
        ]
    )
    cfg.MODEL.DEVICE_ID = str(args.device)
    cfg.freeze()

    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    (
        _train_loader,
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
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    del state
    device = torch.device(f"cuda:{args.device}")
    model.to(device).eval()
    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=str(cfg.TEST.FEAT_NORM).lower() == "yes",
        reranking=False,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, pids, camids_cpu, camids, views, image_paths in val_loader:
            images = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in images.items()
            }
            features = model(
                images,
                cam_label=camids.to(device, non_blocking=True),
                view_label=views.to(device, non_blocking=True),
                return_pattern=3,
                img_path=image_paths,
            )
            evaluator.update((features, pids, camids_cpu, image_paths))
    torch.cuda.synchronize(device)
    cmc, mean_ap, _distmat, _pids, _camids, query_features, gallery_features = (
        evaluator.compute()
    )
    elapsed = time.perf_counter() - started
    parity = {
        "map_matches_reported_rounding": abs(float(mean_ap) - args.expected_map)
        <= args.parity_tolerance,
        "rank1_matches_reported_rounding": abs(float(cmc[0]) - args.expected_rank1)
        <= args.parity_tolerance,
    }
    dataset_audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    audit_identity_checks = dataset_audit.get("expected_identity_counts_match")
    audit_triplet_checks = dataset_audit.get("expected_triplet_counts_match")
    audit_camera_checks = dataset_audit.get("expected_camera_sets_match")
    dataset_audit_valid = (
        dataset_audit.get("valid") is True
        and audit_identity_checks == EXPECTED_AUDIT_IDENTITY_CHECKS
        and audit_triplet_checks == EXPECTED_AUDIT_TRIPLET_CHECKS
        and audit_camera_checks == EXPECTED_AUDIT_CAMERA_CHECKS
    )
    report = {
        "valid": not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and dataset_audit_valid
        and all(parity.values()),
        "baseline": "MDReID",
        "baseline_commit": baseline_commit,
        "dataset": "RGBNT201",
        "protocol": {
            "query": "full test split",
            "gallery": "full test split",
            "exclusion": "same identity and same camera",
            "distance": "Euclidean after L2 normalization",
            "reranking": False,
            "num_query": num_query,
            "num_gallery": len(val_loader.dataset) - num_query,
        },
        "metrics": {
            "mAP": float(mean_ap),
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "rank10": float(cmc[9]),
        },
        "reported_anchor": {
            "mAP": args.expected_map,
            "rank1": args.expected_rank1,
            "absolute_tolerance": args.parity_tolerance,
            "checks": parity,
        },
        "strict_checkpoint_load": {
            "missing_keys": incompatible.missing_keys,
            "unexpected_keys": incompatible.unexpected_keys,
        },
        "dataset_audit_gate": {
            "valid": dataset_audit_valid,
            "expected_identity_counts_match": audit_identity_checks,
            "expected_triplet_counts_match": audit_triplet_checks,
            "expected_camera_sets_match": audit_camera_checks,
        },
        "features": {
            "dimension": int(query_features.shape[1]),
            "query_shape": list(query_features.shape),
            "gallery_shape": list(gallery_features.shape),
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "clip_weight": str(args.clip_weight.resolve()),
            "clip_weight_sha256": sha256(args.clip_weight),
            "dataset_audit": str(args.dataset_audit.resolve()),
            "dataset_audit_sha256": sha256(args.dataset_audit),
        },
        "runtime": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
            "glibc": platform.libc_ver(),
            "seed": cfg.SOLVER.SEED,
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
