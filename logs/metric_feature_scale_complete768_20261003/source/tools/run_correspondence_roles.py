#!/usr/bin/env python3
"""Train and evaluate the new correspondence-role model on official splits."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modeling"))

from tools.official_three_dataset_data import loader_for, records_for
from tools.official_three_dataset_model import build_model, sha256
from tools.run_official_three_dataset_roles import distance_matrix, read_protocol
from tools.run_signal_preserving_v5 import (
    _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier,
)
from tools.train_signal_preserving_v18 import image_batch
from trifusion.criterion import _batch_hard_triplet
from trifusion.correspondence_roles import CorrespondenceTriFusion
from trifusion.experts.mamba import production_mamba_factory


def build(args, protocol):
    old, cfg, config, binding = build_model(
        protocol, args.signal_source, args.clip_weight, args.baseline_checkpoint,
        args.baseline_sha256, seed=args.seed, plain_baseline=True,
    )
    signal = old.baseline.signal
    del old
    _set_seed(args.seed)
    grid = (16, 8) if args.dataset == "RGBNT201" else (8, 16)
    model = CorrespondenceTriFusion(
        signal, num_classes=len(protocol["train_label_map"]), grid=grid,
        width=args.width, m1=args.m1, m2=args.m2, m3=args.m3,
        mamba_factory=production_mamba_factory,
    ).cuda()
    binding.pop("initial_role_state_sha256")
    binding.update({"architecture": "cross_layer_anchor_correspondence_roles_v1",
                    "m1": args.m1, "m2": args.m2, "m3": args.m3,
                    "width": args.width, "fused_width": 1536,
                    "prediction_weight": args.pred_weight,
                    "learning_rate": config["OPTIMIZATION"]["NEW_MODULE_LR"],
                    "weight_decay": config["OPTIMIZATION"]["WEIGHT_DECAY"],
                    "entry_sha256": sha256(Path(__file__)),
                    "model_source_sha256": sha256(ROOT / "modeling/trifusion/correspondence_roles.py"),
                    "initial_model_state_sha256": _module_state_sha256(model),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model, cfg, config, binding


def _eval_batch(raw, dataset):
    if dataset == "RGBNT201":
        return image_batch(raw[0], raw[3])
    return _training_batch(raw)[0]


def extract(model, protocol, split):
    model.eval()
    rows = []
    records = records_for(protocol, split)
    with torch.inference_mode():
        for raw in loader_for(protocol, records, training=False, method="PLAIN_V8"):
            batch = _eval_batch(raw, protocol["dataset"])
            rows.append(model(batch).float().cpu())
    result = torch.cat(rows)
    assert result.shape == (protocol["counts"][split], 1536)
    return result


def official_metrics(model, protocol, source, *, save_distances=None):
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores
    from utils import metrics as author_metrics

    assert Path(author_metrics.__file__).resolve() == (source / "utils" / "metrics.py").resolve()
    query = extract(model, protocol, "query")
    gallery = extract(model, protocol, "gallery")
    distances = distance_matrix(query, gallery).numpy()
    qrows = protocol["records"]["query"]
    grows = protocol["records"]["gallery"]
    qids, gids = (np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows))
    qcameras, gcameras = (np.asarray([row["camera"] for row in rows]) for rows in (qrows, grows))
    qscenes, gscenes = (np.asarray([row["scene"] for row in rows]) for rows in (qrows, grows))
    if protocol["dataset"] == "MSVR310":
        result = scene_scores(distances, qids, gids, qscenes, gscenes)
        cmc, mean_ap = author_metrics.eval_func_msrv(
            distances, qids, gids, qcameras, gcameras, qscenes, gscenes)
    else:
        result = camera_scores(distances, qids, gids, qcameras, gcameras)
        cmc, mean_ap = author_metrics.eval_func(distances, qids, gids, qcameras, gcameras)
    metrics = {"mAP": 100 * float(mean_ap), "Rank-1": 100 * float(cmc[0]),
               "Rank-5": 100 * float(cmc[4]), "Rank-10": 100 * float(cmc[9])}
    assert all(abs(metrics[name] - result["metrics"][name]) < 1e-5 for name in metrics)
    if save_distances is not None:
        torch.save({"fused": torch.from_numpy(distances), "query_ids": qids,
                    "gallery_ids": gids, "query_cameras": qcameras,
                    "gallery_cameras": gcameras, "query_scenes": qscenes,
                    "gallery_scenes": gscenes}, save_distances)
    return metrics


def checkpoint_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith("backbone.signal.") and not name.startswith("teacher.")}


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-correspondence-roles-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "metrics": metrics, "state": checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-correspondence-roles-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert set(payload["state"]) == set(checkpoint_state(model))
    state = model.state_dict()
    state.update(payload["state"])
    model.load_state_dict(state, strict=True)
    return payload


def train(args, protocol):
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    os.chdir(args.output_dir)
    model, _cfg, config, binding = build(args, protocol)
    frozen_signal_sha256 = _module_state_sha256(model.backbone.signal)
    model.train()
    named_parameters = {name: parameter for name, parameter in model.named_parameters()
                        if parameter.requires_grad}
    parameters = list(named_parameters.values())
    optimizer = torch.optim.AdamW(parameters, lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    records = records_for(protocol, "train")
    loader = loader_for(protocol, records, training=True, method="PLAIN_V8", seed=args.seed)
    receipt = {"schema": "trifusion-correspondence-roles-training-v1",
               "status": "RUNNING", "dataset": args.dataset, "seed": args.seed,
               "epochs": args.epochs, "checkpoint_policy": "best_official_map",
               "protocol_sha256": sha256(args.protocol), "initializer": binding,
               "started_at": datetime.now().astimezone().isoformat(),
               "history": []}
    (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n")
    best = {"mAP": -1.0, "epoch": None}
    live_parameters = set()
    audit_batch = None
    epochs = 1 if args.mode == "m0" else args.epochs
    with (args.output_dir / "training_steps.jsonl").open("x") as step_log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            multiplier = learning_rate_multiplier(epoch, max_epochs=args.epochs, warmup_epochs=5)
            for group in optimizer.param_groups:
                group["lr"] = config["OPTIMIZATION"]["NEW_MODULE_LR"] * multiplier
            model.train()
            losses = []
            for batch_index, raw in enumerate(loader):
                if args.mode == "m0" and batch_index == 8:
                    break
                batch, labels = _training_batch(raw)
                if audit_batch is None:
                    audit_batch = {"images": {name: tensor[:2].clone() for name, tensor in batch["images"].items()},
                                   "camera_ids": batch["camera_ids"][:2].clone()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    id_loss = F.cross_entropy(output["logits"], labels, label_smoothing=0.1)
                    triplet = _batch_hard_triplet(output["fused"], labels, margin=0.3)
                    prediction = output.get("prediction", {})
                    prediction_loss = sum(prediction.values()) if prediction else output["fused"].sum() * 0
                    loss = id_loss + triplet + args.pred_weight * prediction_loss
                assert bool(torch.isfinite(loss))
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
                live_parameters.update(name for name, parameter in named_parameters.items()
                                       if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0))
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() >= scale
                model.update_teacher()
                value = float(loss.detach())
                losses.append(value)
                step_log.write(json.dumps({"epoch": epoch, "batch": batch_index,
                                            "loss": value, "id": float(id_loss.detach()),
                                            "triplet": float(triplet.detach()),
                                            "prediction": {name: float(item.detach()) for name, item in prediction.items()}}) + "\n")
            step_log.flush()
            row = {"epoch": epoch, "steps": len(losses), "mean_loss": float(np.mean(losses)),
                   "seconds": time.perf_counter() - started}
            if args.mode == "train":
                metrics = official_metrics(model, protocol, args.signal_source)
                row["official_fused"] = metrics
                if metrics["mAP"] >= best["mAP"]:
                    best = {"mAP": metrics["mAP"], "epoch": epoch}
                    save_checkpoint(args.output_dir / "best_map.pth", model, args, protocol, epoch, metrics)
            receipt["history"].append(row)
            (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"event": "epoch", "dataset": args.dataset, **row,
                              "best_epoch": best["epoch"]}), flush=True)
    assert frozen_signal_sha256 == _module_state_sha256(model.backbone.signal)
    if args.mode == "m0":
        assert live_parameters == set(named_parameters)
        assert audit_batch is not None
        probe = args.output_dir / "m0_reload_probe.pth"
        save_checkpoint(probe, model, args, protocol, 0, {})
        fresh, _cfg, _config, new_binding = build(args, protocol)
        assert new_binding == binding
        load_checkpoint(probe, fresh, args, protocol)
        model.eval()
        fresh.eval()
        with torch.inference_mode():
            original = model(audit_batch)
            reloaded = fresh(audit_batch)
        assert torch.allclose(original, reloaded, atol=1e-5, rtol=1e-5)
        receipt["m0"] = {"nonzero_gradient_parameters": len(live_parameters),
                         "trainable_parameters": len(named_parameters),
                         "frozen_signal_unchanged": True,
                         "reload_max_abs_difference": float((original - reloaded).abs().max()),
                         "reload_probe_sha256": sha256(probe)}
    receipt["status"] = "M0_PASS" if args.mode == "m0" else "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    receipt["best_epoch"] = best["epoch"]
    receipt["checkpoint"] = str(args.output_dir / "best_map.pth") if args.mode == "train" else None
    receipt["completed_at"] = datetime.now().astimezone().isoformat()
    (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n")


def evaluate(args, protocol):
    os.chdir(args.output_dir)
    training = json.loads((args.output_dir / "training.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert len(training["history"]) == 50
    model, _cfg, _config, binding = build(args, protocol)
    assert binding == training["initializer"]
    checkpoint = args.output_dir / "best_map.pth"
    payload = load_checkpoint(checkpoint, model, args, protocol)
    distances = args.output_dir / "official_distances.pt"
    metrics = official_metrics(model, protocol, args.signal_source, save_distances=distances)
    assert all(abs(metrics[name] - payload["metrics"][name]) < 1e-5 for name in metrics)
    result = {"schema": "trifusion-correspondence-roles-retrieval-v1",
              "status": "COMPLETE", "dataset": args.dataset, "seed": args.seed,
              "selected_epoch": payload["epoch"], "training_epochs": 50,
              "protocol_sha256": sha256(args.protocol),
              "baseline_sha256": args.baseline_sha256,
              "checkpoint_sha256": sha256(checkpoint),
              "distance_sha256": sha256(distances),
              "metrics": metrics, "reranking": False,
              "independent_upstream_metrics_equal": True,
              "completed_at": datetime.now().astimezone().isoformat()}
    (args.output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--mode", choices=("m0", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--signal-source", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, choices=(50,), default=50)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--pred-weight", type=float, default=0.1)
    parser.add_argument("--m1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m3", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in ("protocol", "signal_source", "clip_weight", "baseline_checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    protocol = read_protocol(args.protocol, args.dataset)
    if args.mode == "evaluate":
        evaluate(args, protocol)
    else:
        train(args, protocol)


if __name__ == "__main__":
    main()
