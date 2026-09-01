#!/usr/bin/env python3
"""Train the pinned Signal baseline on the frozen RGBNT201 141/30 dev split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time


EXPECTED_SIGNAL_COMMIT = "cd1b0a672d1fe642e7608731cb4899a19dda7d51"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_record(rgb_path: Path, label: int) -> tuple[list[str], int, int, int]:
    fields = rgb_path.name.split("_")
    camera_id = int(fields[1][3]) - 1
    paths = [
        rgb_path,
        rgb_path.parents[1] / "NI" / rgb_path.name,
        rgb_path.parents[1] / "TI" / rgb_path.name,
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"unpaired RGBNT201 triplet: {rgb_path.name}")
    return [str(path) for path in paths], label, camera_id, -1


def _records_for_ids(
    train_root: Path,
    identities: set[int],
    *,
    relabel: bool,
) -> tuple[tuple[list[str], int, int, int], ...]:
    labels = {identity: index for index, identity in enumerate(sorted(identities))}
    records = []
    for rgb_path in sorted((train_root / "RGB").glob("*.jpg")):
        identity = int(rgb_path.name[:6])
        if identity in identities:
            records.append(
                _parse_record(rgb_path, labels[identity] if relabel else identity)
            )
    return tuple(records)


def _configure_signal_source(source: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_SIGNAL_COMMIT:
        raise ValueError(f"Signal source commit is {commit}, expected {EXPECTED_SIGNAL_COMMIT}")
    sys.path.insert(0, str(source))
    return commit


def _build_loaders(
    *,
    dataset_root: Path,
    protocol_path: Path,
    batch_size: int,
    num_instances: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
):
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import (
        RandomErasing,
        train_collate_fn,
        val_collate_fn,
    )
    from data.datasets.sampler import RandomIdentitySampler

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["selection"]["uses_test_labels"] is not False:
        raise ValueError("development protocol must be test-label blind")
    train_ids = {int(value) for value in protocol["train_ids"]}
    dev_ids = {int(value) for value in protocol["dev_ids"]}
    if len(train_ids) != 141 or len(dev_ids) != 30 or train_ids & dev_ids:
        raise ValueError("frozen 141-fit/30-dev identity registry is invalid")

    train_root = dataset_root / "train_171"
    train_records = _records_for_ids(train_root, train_ids, relabel=True)
    dev_records = _records_for_ids(train_root, dev_ids, relabel=False)
    if len(train_records) != int(protocol["counts"]["train_triplets"]):
        raise ValueError("Signal fit records differ from the frozen protocol")
    if len(dev_records) != int(protocol["counts"]["dev_triplets"]):
        raise ValueError("Signal dev records differ from the frozen protocol")

    train_transform = transforms.Compose(
        [
            transforms.Resize([256, 128], interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop([256, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            RandomErasing(
                probability=0.5,
                mode="pixel",
                max_count=1,
                device="cpu",
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize([256, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    train_loader = DataLoader(
        ImageDataset(train_records, train_transform),
        batch_size=batch_size,
        sampler=RandomIdentitySampler(
            train_records,
            batch_size,
            num_instances,
            seed,
        ),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
    )
    eval_loader = DataLoader(
        ImageDataset(dev_records + dev_records, eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    return train_loader, eval_loader, train_records, dev_records


def _evaluate(model, loader, num_query: int, stage: str) -> tuple[dict[str, float], int]:
    import numpy as np
    import torch
    from utils.metrics import R1_mAP_eval

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")
    model.eval()
    feature_width = 0
    for images, identities, camera_ids, camera_ids_batch, view_ids, paths in loader:
        with torch.no_grad():
            images = {name: tensor.cuda() for name, tensor in images.items()}
            features = model(
                images,
                cam_label=camera_ids_batch.cuda(),
                view_label=view_ids.cuda(),
                training=False,
                sge=stage,
            )
        feature_width = int(features.shape[1])
        evaluator.update((features, identities, camera_ids, paths))
    if feature_width != 3072:
        raise ValueError(f"Signal retrieval feature width is {feature_width}, expected 3072")
    cmc, mean_ap, *_ = evaluator.compute()
    metrics = {
        "mAP": float(mean_ap * 100.0),
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[4] * 100.0),
        "Rank-10": float(cmc[9] * 100.0),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite Signal baseline metric")
    return metrics, feature_width


def run(args: argparse.Namespace) -> dict[str, object]:
    source = args.signal_source.resolve()
    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    clip_weight = args.clip_weight.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    source_commit = _configure_signal_source(source)
    import numpy as np
    import torch
    from config import cfg
    from engine.processor import do_train
    from layers.make_loss import make_loss
    from modeling import make_frame
    from solver.make_optimizer import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(str(args.config.resolve()))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH_T = str(clip_weight)
    cfg.SOLVER.SEED = args.seed
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_INSTANCE = args.num_instances
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.TEST.IMS_PER_BATCH = args.eval_batch_size
    cfg.OUTPUT_DIR = str(output_dir.parent)
    cfg.ckpt_save_path = output_dir.name
    cfg.freeze()

    train_loader, eval_loader, train_records, dev_records = _build_loaders(
        dataset_root=dataset_root,
        protocol_path=protocol_path,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    camera_num = len({record[2] for record in train_records})
    view_num = len({record[3] for record in train_records})
    logger = setup_logger("Signal", str(output_dir), if_train=True)
    logger.info("Pinned Signal commit: %s", source_commit)
    logger.info("Frozen dev protocol: %s", protocol_path)
    logger.info("Fit/dev triplets: %d/%d", len(train_records), len(dev_records))

    model = make_frame(
        cfg,
        num_class=len({record[1] for record in train_records}),
        camera_num=camera_num,
        view_num=view_num,
    )
    loss_fn, center_criterion = make_loss(
        cfg,
        num_classes=len({record[1] for record in train_records}),
    )
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    identity = {
        "schema_version": "signal-baseline-dev-v1",
        "signal_source_commit": source_commit,
        "signal_source_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(source), "diff", "--binary"]
            )
        ).hexdigest(),
        "protocol_sha256": _sha256(protocol_path),
        "clip_sha256": _sha256(clip_weight),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_instances": args.num_instances,
        "fit_triplets": len(train_records),
        "dev_query_triplets": len(dev_records),
        "dev_gallery_triplets": len(dev_records),
        "official_test_access_count": 0,
    }
    (output_dir / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        eval_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        len(dev_records),
        0,
        cfg.MODEL.stageName,
    )
    best_checkpoint = output_dir / "Signalbest.pth"
    model.load_state_dict(torch.load(best_checkpoint, map_location="cpu"), strict=True)
    model.cuda()
    metrics, feature_width = _evaluate(
        model,
        eval_loader,
        len(dev_records),
        cfg.MODEL.stageName,
    )
    result = {
        "schema_version": "signal-baseline-dev-result-v1",
        "status": "PASS",
        "metrics_percent": metrics,
        "retrieval_feature": "concat(direct_3x512,SIM_3x512)",
        "retrieval_feature_width": feature_width,
        "camera_sie": bool(cfg.MODEL.SIE_CAMERA),
        "signal_source_commit": source_commit,
        "checkpoint": str(best_checkpoint),
        "checkpoint_sha256": _sha256(best_checkpoint),
        "parameters": parameters,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
        "fit_triplets": len(train_records),
        "dev_query_triplets": len(dev_records),
        "dev_gallery_triplets": len(dev_records),
        "official_test_access_count": 0,
        "run_identity_sha256": _sha256(output_dir / "run_identity.json"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--clip-weight", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
