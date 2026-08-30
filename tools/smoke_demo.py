#!/usr/bin/env python3
"""Smoke the unmodified DeMo training graph while bypassing its hard-coded CLIP path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/mmreid-trifusion/baselines/DeMo"),
    )
    parser.add_argument(
        "--clip-weight",
        type=Path,
        default=Path("/root/mmreid-trifusion/pretrained/ViT-B-16.pt"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("/root/mmreid-trifusion/artifacts/demo_train_smoke_20260831.json"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1555)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    config_path = baseline_root / "configs/RGBNT201/DeMo.yml"
    for required in (baseline_root, config_path, args.clip_weight):
        if not required.exists():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("DeMo training smoke requires CUDA")

    sys.path.insert(0, str(baseline_root))
    os.chdir(baseline_root)
    from config import cfg  # pylint: disable=import-error,import-outside-toplevel
    import modeling.meta_arch as meta_arch  # pylint: disable=import-error,import-outside-toplevel
    from modeling import make_model  # pylint: disable=import-error,import-outside-toplevel
    from modeling.clip import clip  # pylint: disable=import-error,import-outside-toplevel

    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(
        [
            "MODEL.PRETRAIN_PATH_T",
            str(args.clip_weight.resolve()),
            "DATALOADER.NUM_WORKERS",
            "0",
            "OUTPUT_DIR",
            str(args.json_out.resolve().parent / "demo"),
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}")
    model = make_model(cfg, num_class=171, camera_num=4, view_num=1).to(device).train()
    images = {
        modality: torch.randn(args.batch_size, 3, 256, 128, device=device)
        for modality in ("RGB", "NI", "TI")
    }
    labels = torch.arange(args.batch_size, device=device) % 171
    cameras = torch.arange(args.batch_size, device=device) % 4
    views = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    output = model(
        images,
        label=labels,
        cam_label=cameras,
        view_label=views,
    )
    regularizer = output[4] if torch.is_tensor(output[4]) else 0.0
    loss = (
        functional.cross_entropy(output[0], labels)
        + functional.cross_entropy(output[2], labels)
        + regularizer
    )
    loss.backward()
    torch.cuda.synchronize(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters]
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "all_gradients_present": all(gradient is not None for gradient in gradients),
        "all_gradients_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }
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
        "compatibility_override": {
            "reason": "official load_clip_to_cpu hard-codes /13994058190/WYH/PTH/ViT-B-16.pt",
            "upstream_modified": False,
            "configured_weight": str(args.clip_weight.resolve()),
        },
        "batch_size": args.batch_size,
        "output_shapes": [
            list(item.shape) if torch.is_tensor(item) else type(item).__name__
            for item in output
        ],
        "loss": float(loss),
        "trainable_parameter_tensors": len(parameters),
        "gradient_tensors_present": sum(
            gradient is not None for gradient in gradients
        ),
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
