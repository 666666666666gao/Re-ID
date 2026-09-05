#!/usr/bin/env python3
"""Fixed-source Signal baselines for the registered MSVR310 identity folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure(config):
    from tools.run_signal_baseline_dev import _configure_signal_source

    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    source = Path(config["signal_source"])
    commit = _configure_signal_source(source)
    source_diff = hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary"])).hexdigest()
    assert commit == config["signal_commit"]
    assert source_diff == config["signal_diff_sha256"]
    assert sha256(config["clip_weight"]) == config["clip_weight_sha256"]
    for name, expected in config["signal_source_file_sha256"].items():
        assert sha256(source / name) == expected, name
    from config import cfg
    cfg.merge_from_file(str(source / "configs/MSVR310/Signal.yml"))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH_T = config["clip_weight"]
    cfg.SOLVER.SEED = 42
    cfg.SOLVER.MAX_EPOCHS = 50
    cfg.SOLVER.IMS_PER_BATCH = 64
    cfg.DATALOADER.NUM_INSTANCE = 8
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.IMS_PER_BATCH = 64
    cfg.freeze()
    assert cfg.DATASETS.NAMES == "MSVR310"
    assert list(cfg.INPUT.SIZE_TRAIN) == list(cfg.INPUT.SIZE_TEST) == [128, 256]
    assert cfg.MODEL.SIE_CAMERA and not cfg.MODEL.SIE_VIEW
    assert cfg.MODEL.DIRECT == 0 and cfg.MODEL.USE_A and cfg.MODEL.USE_B
    assert not cfg.MODEL.FROZEN and cfg.MODEL.METRIC_LOSS_TYPE == "triplet"
    assert cfg.SOLVER.BASE_LR == 5e-6 and cfg.SOLVER.OPTIMIZER_NAME == "Adam"
    assert list(cfg.SOLVER.STEPS) == [20, 40] and cfg.SOLVER.WARMUP_ITERS == 0
    assert not cfg.SOLVER.LARGE_FC_LR
    sys.path.append(str(ROOT / "modeling"))
    return cfg, {"signal_commit": commit, "signal_diff_sha256": source_diff}


def records_for(config, protocol, fold, source):
    indices = fold["source_record_indices"] if source else fold["gallery_record_indices"]
    records = []
    for index in indices:
        row = protocol["records"][index]
        label = fold["source_label_map"][str(row["identity"])] if source else row["identity"]
        paths = [str(Path(config["dataset_root"]) / p) for p in row["paths"]]
        assert all(Path(p).is_file() for p in paths)
        records.append((paths, int(label), row["camera"], row["scene"]))
    return records


def clean_triplet(images):
    from torchvision.transforms import functional as tf
    from torchvision.transforms import InterpolationMode

    return [tf.normalize(tf.to_tensor(tf.resize(image, [128, 256],
            interpolation=InterpolationMode.BILINEAR)), [0.5] * 3, [0.5] * 3)
            for image in images]


def loader_for(records, training):
    from torch.utils.data import DataLoader
    from data.datasets.make_dataloader import train_collate_fn
    from data.datasets.sampler import RandomIdentitySampler
    from trifusion.aligned_data import AlignedTripletImageDataset, SharedGeometryTripletTransform

    transform = SharedGeometryTripletTransform(size=(128, 256)) if training else clean_triplet
    dataset = AlignedTripletImageDataset(records, transform=transform)
    sampler = RandomIdentitySampler(records, 64, 8, 42) if training else None
    return DataLoader(dataset, batch_size=64, sampler=sampler, shuffle=False,
                      num_workers=4, collate_fn=train_collate_fn, pin_memory=True)


def new_model(cfg, fold):
    from tools.build_v12_complete_path_oof_targets import _build_signal_teacher

    assert fold["source_camera_values"] == list(range(8))
    return _build_signal_teacher(cfg, num_classes=len(fold["source_ids"]),
                                 camera_num=8, view_num=0)


def train_source(model, loader, cfg, record_indices, records, *, preflight):
    import numpy as np
    import torch
    from layers.make_loss import make_loss
    from solver.make_optimizer import make_optimizer
    from solver.lr_scheduler310 import WarmupMultiStepLR
    from tools.build_v12_complete_path_oof_targets import _signal_training_loss
    from tools.run_signal_preserving_v5 import _module_state_sha256

    loss_fn, center = make_loss(cfg, num_classes=model.num_classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                 cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS,
                                 cfg.SOLVER.WARMUP_METHOD)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    names = {id(p): name for name, p in model.named_parameters()}
    optimizer_groups = [{"name": names[id(group["params"][0])], "lr": group["lr"],
                         "weight_decay": group["weight_decay"]}
                        for group in optimizer.param_groups]
    name_to_index = {Path(row[0][0]).name: index
                     for row, index in zip(records, record_indices, strict=True)}
    assert len(name_to_index) == len(records)
    initial_state = _module_state_sha256(model)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    gradient_names = set()
    history, steps = [], []
    for epoch in range(1, (1 if preflight else 50) + 1):
        started = time.perf_counter()
        scheduler.step(epoch)
        model.train()
        epoch_losses = []
        for images, labels, cameras, scenes, paths in loader:
            assert labels.numel() == 64
            assert all(tuple(value.shape) == (64, 3, 128, 256) for value in images.values())
            assert sorted(torch.unique(labels, return_counts=True)[1].tolist()) == [8] * 8
            optimizer.zero_grad(set_to_none=True)
            images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
            labels, cameras, scenes = (x.cuda(non_blocking=True) for x in (labels, cameras, scenes))
            components = []

            def recorded_loss(**kwargs):
                component = loss_fn(**kwargs)
                components.append(float(component.detach()))
                return component

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images, label=labels, cam_label=cameras, view_label=scenes,
                               training=True, sge=cfg.MODEL.stageName)
                assert output[0] == 3 and len(output) == 11
                loss = _signal_training_loss(
                    output, loss_fn=recorded_loss, labels=labels, cameras=cameras,
                    stage=cfg.MODEL.stageName, gram_weight=cfg.MODEL.Gram_Loss_weight,
                    patch_weight=cfg.MODEL.PAT_Loss_weight)
            assert torch.isfinite(loss).item()
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient_names.add(name)
                    if preflight:
                        assert torch.isfinite(parameter.grad).all().item(), name
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() >= scale_before, "AMP overflow; fixed run stops"
            row = {"step": len(steps) + 1, "epoch": epoch, "loss": float(loss.detach()),
                   "id_triplet_head_losses": components, "gram_loss": float(output[-2].detach()),
                   "patch_loss": float(output[-1].detach()),
                   "sampled_record_indices": [name_to_index[p] for p in paths],
                   "amp_scale_before": scale_before, "amp_scale_after": scaler.get_scale()}
            steps.append(row)
            epoch_losses.append(row["loss"])
            if preflight and len(steps) == 8:
                break
        history.append({"epoch": epoch, "optimizer_steps": len(epoch_losses),
                        "mean_loss": float(np.mean(epoch_losses)),
                        "learning_rates": sorted({group["lr"] for group in optimizer.param_groups}),
                        "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"event": "signal_source_epoch", **history[-1]}), flush=True)
    return {"epochs": len(history), "optimizer_steps": len(steps), "overflow_events": 0,
            "initial_state_sha256": initial_state, "final_state_sha256": _module_state_sha256(model),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_tensors": len(trainable_names), "gradient_tensors": len(gradient_names),
            "trainable_without_gradient": sorted(trainable_names - gradient_names),
            "optimizer_groups": optimizer_groups, "history": history, "steps": steps}


def extract(model, records, cfg):
    import torch

    model.eval()
    parts = []
    with torch.no_grad():
        for images, _, cameras, scenes, _ in loader_for(records, False):
            images = {key: value.cuda() for key, value in images.items()}
            features = model(images, cam_label=cameras.cuda(), view_label=scenes.cuda(),
                             training=False, sge=cfg.MODEL.stageName)
            assert features.ndim == 2 and features.shape[1] == 3072
            assert torch.isfinite(features).all().item()
            parts.append(features.float().cpu())
    return torch.cat(parts)


def scene_scores(distances, query_ids, gallery_ids, query_scenes, gallery_scenes):
    import numpy as np

    ap, first = [], []
    for index, order in enumerate(np.argsort(distances, axis=1)):
        keep = ~((gallery_ids[order] == query_ids[index]) &
                 (gallery_scenes[order] == query_scenes[index]))
        matches = gallery_ids[order][keep] == query_ids[index]
        positions = np.flatnonzero(matches)
        assert positions.size
        ap.append(float(np.mean(np.cumsum(matches)[positions] / (positions + 1))))
        first.append(int(positions[0] + 1))
    metrics = {"mAP": float(np.mean(ap) * 100)}
    metrics.update({f"Rank-{rank}": float(np.mean(np.asarray(first) <= rank) * 100)
                    for rank in (1, 5, 10)})
    return {"metrics": metrics, "average_precision": ap, "first_match_rank": first}


def evaluate_gallery(features, protocol, fold, directory):
    import numpy as np
    import torch
    from utils.metrics import eval_func_msrv

    rows = [protocol["records"][i] for i in fold["gallery_record_indices"]]
    positions = [row["gallery_position"] for row in fold["query_rows"]]
    for query, position in zip(fold["query_rows"], positions, strict=True):
        assert rows[position]["index"] == query["record_index"]
    ids = np.asarray([r["identity"] for r in rows])
    cameras = np.asarray([r["camera"] for r in rows])
    scenes = np.asarray([r["scene"] for r in rows])
    normalized = torch.nn.functional.normalize(features.float(), dim=1)
    qf, gf = normalized[positions], normalized
    distances = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None]
    distances.addmm_(qf, gf.T, beta=1, alpha=-2)
    scores = scene_scores(distances.numpy(), ids[positions], ids, scenes[positions], scenes)
    # The upstream evaluator writes re.txt. Its working directory is this new fold artifact directory.
    os.chdir(directory)
    cmc, mean_ap = eval_func_msrv(distances.numpy(), ids[positions], ids, cameras[positions],
                                cameras, scenes[positions], scenes)
    differences = {"mAP": abs(scores["metrics"]["mAP"] - float(mean_ap * 100))}
    differences.update({f"Rank-{rank}": abs(scores["metrics"][f"Rank-{rank}"] -
                                           float(cmc[rank - 1] * 100)) for rank in (1, 5, 10)})
    assert differences["mAP"] < 1e-10 and max(differences.values()) < 1e-5
    torch.save({"features": features, "distances": distances,
                "gallery_record_indices": fold["gallery_record_indices"],
                "query_gallery_positions": positions}, directory / "retrieval_arrays.pt")
    return {**scores, "upstream_metric_difference_pp": differences,
            "gallery_manifest": rows, "query_rows": fold["query_rows"],
            "feature_width": 3072, "retrieval_arrays_sha256": sha256(directory / "retrieval_arrays.pt")}


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed

    assert torch.cuda.is_available()
    config_path, output_dir = args.config.resolve(), args.output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / config["protocol"]
    assert sha256(protocol_path) == config["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["epochs"] == 50
    assert (config["train_batch_size"], config["instances_per_identity"],
            config["num_workers"], config["preflight_steps_per_fold"]) == (64, 8, 4, 8)
    assert not output_dir.exists()
    cfg, binding = configure(config)
    output_dir.mkdir(parents=True)
    preflight = args.mode == "preflight"
    if not preflight:
        receipt_path = args.preflight_receipt.resolve()
        preflight_report = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert preflight_report["status"] == "PASS_ENGINEERING_ONLY"
        assert preflight_report["config_sha256"] == sha256(config_path)
        assert preflight_report["runner_sha256"] == sha256(__file__)
    started = time.perf_counter()
    summary = {"schema": "msvr310-signal-source-oof-v1", "mode": args.mode, "status": "RUNNING",
               **binding, "config_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path),
               "runner_sha256": sha256(__file__), "seed": 42, "folds": [],
               "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "project_source_file_sha256": config["project_source_file_sha256"],
               "evaluation_type": "source_only_engineering" if preflight else "train_internal_identity_oof_baseline",
               "official_test_image_access": 0, "fixed_rgbnt201_dev_image_access": 0,
               "expert_training": 0, "epochs_selected_by_heldout": False,
               "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__)}
    (output_dir / "signal_effective_config.yml").write_text(cfg.dump(), encoding="utf-8")
    for fold in protocol["folds"]:
        fold_dir = output_dir / f"fold_{fold['fold']}"
        fold_dir.mkdir()
        _set_seed(42)
        source_records = records_for(config, protocol, fold, True)
        assert not set(fold["source_ids"]) & set(fold["heldout_ids"])
        loader = loader_for(source_records, True)
        model = new_model(cfg, fold)
        torch.cuda.reset_peak_memory_stats()
        training = train_source(model, loader, cfg, fold["source_record_indices"], source_records,
                                preflight=preflight)
        assert training["epochs"] == (1 if preflight else 50)
        assert not training["trainable_without_gradient"]
        assert training["initial_state_sha256"] != training["final_state_sha256"]
        checkpoint = fold_dir / ("signal_m0.pth" if preflight else "signal_epoch50.pth")
        torch.save({"model_state_dict": model.state_dict(), "fold": fold["fold"],
                    "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
                    "config_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path)},
                   checkpoint)
        before_reload = extract(model, source_records[:8], cfg) if preflight else None
        del model, loader
        torch.cuda.empty_cache()
        model = new_model(cfg, fold)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=True)
        assert _module_state_sha256(model) == training["final_state_sha256"]
        row = {"fold": fold["fold"], "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
               "counts": fold["counts"], "training": training,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
        if preflight:
            after_reload = extract(model, source_records[:8], cfg)
            assert training["optimizer_steps"] == 8 and torch.equal(before_reload, after_reload)
            row.update({"strict_reload_exact_feature_parity": True, "clean_source_feature_forwards": 16,
                        "feature_width": 3072, "heldout_image_forwards": 0})
        else:
            gallery_records = records_for(config, protocol, fold, False)
            features = extract(model, gallery_records, cfg)
            row["retrieval"] = evaluate_gallery(features, protocol, fold, fold_dir)
            row["heldout_image_forwards"] = len(gallery_records)
        assert _module_state_sha256(model) == training["final_state_sha256"]
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        write_json(fold_dir / "receipt.json", row)
        summary["folds"].append(row)
        write_json(output_dir / "summary.json", summary)
        del model, state
        torch.cuda.empty_cache()
    if not preflight:
        import numpy as np
        ap = [x for row in summary["folds"] for x in row["retrieval"]["average_precision"]]
        first = [x for row in summary["folds"] for x in row["retrieval"]["first_match_rank"]]
        assert len(ap) == 600
        summary["aggregate"] = {"mAP": float(np.mean(ap) * 100),
                                **{f"Rank-{k}": float(np.mean(np.asarray(first) <= k) * 100)
                                   for k in (1, 5, 10)}}
        summary["preflight_receipt_sha256"] = sha256(receipt_path)
    summary.update({"status": "PASS_ENGINEERING_ONLY" if preflight else "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION",
                    "elapsed_seconds": time.perf_counter() - started,
                    "optimizer_steps": sum(row["training"]["optimizer_steps"] for row in summary["folds"]),
                    "training_source_only": True, "checkpoint_selection": "fixed_epoch_50" if not preflight else "discarded_m0",
                    "heldout_image_forwards": sum(row["heldout_image_forwards"] for row in summary["folds"])})
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", "status": summary["status"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    arguments = parser.parse_args()
    assert arguments.mode == "preflight" or arguments.preflight_receipt is not None
    run(arguments)
