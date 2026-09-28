#!/usr/bin/env python3
"""Train the independent M1 global-only control under the same 50-epoch rule."""

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

from tools import run_correspondence_roles as roles_runner
from tools.official_three_dataset_data import loader_for, records_for
from tools.official_three_dataset_model import sha256
from tools.run_official_three_dataset_roles import read_protocol
from tools.run_signal_preserving_v5 import (
    _module_state_sha256, _training_batch, learning_rate_multiplier,
)
from trifusion.correspondence_global_only import CorrespondenceGlobalOnly
from trifusion.criterion import _batch_hard_triplet


def build(args, protocol):
    values = vars(args).copy()
    values.update(width=128, m1=True, m2=False, m3=False, pred_weight=0.0)
    initialized, cfg, config, binding = roles_runner.build(argparse.Namespace(**values), protocol)
    common_hashes = {name: _module_state_sha256(getattr(initialized, name))
                     for name in ("backbone", "neck", "classifier")}
    model = CorrespondenceGlobalOnly(initialized)
    assert common_hashes == {name: _module_state_sha256(getattr(model, name))
                             for name in common_hashes}
    assert set(dict(model.named_children())) == {"backbone", "neck", "classifier"}
    del initialized
    binding.update(architecture="correspondence_m1_global_only_v1", role_path=False,
                   common_initializer_variant="100", common_initializer_sha256=common_hashes,
                   entry_sha256=sha256(Path(__file__)),
                   global_model_source_sha256=sha256(ROOT / "modeling/trifusion/correspondence_global_only.py"),
                   initializer_entry_sha256=sha256(ROOT / "tools/run_correspondence_roles.py"),
                   initial_model_state_sha256=_module_state_sha256(model),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, cfg, config, binding


def save_checkpoint(path, model, args, epoch, metrics):
    torch.save({"schema": "trifusion-correspondence-global-only-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "metrics": metrics, "state": roles_runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-correspondence-global-only-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert set(payload["state"]) == set(roles_runner.checkpoint_state(model))
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
    named_parameters = {name: parameter for name, parameter in model.named_parameters()
                        if parameter.requires_grad}
    parameters = list(named_parameters.values())
    optimizer = torch.optim.AdamW(parameters, lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    loader = loader_for(protocol, records_for(protocol, "train"), training=True,
                        method="PLAIN_V8", seed=args.seed)
    receipt = {"schema": "trifusion-correspondence-global-only-training-v1",
               "status": "RUNNING", "dataset": args.dataset, "seed": args.seed,
               "epochs": 50, "checkpoint_policy": "best_official_map",
               "protocol_sha256": sha256(args.protocol), "initializer": binding,
               "started_at": datetime.now().astimezone().isoformat(), "history": []}
    receipt_path = args.output_dir / "training.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    best = {"mAP": -1.0, "epoch": None}
    live_parameters = set()
    audit_batch = None
    epochs = 1 if args.mode == "m0" else 50
    with (args.output_dir / "training_steps.jsonl").open("x") as step_log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            multiplier = learning_rate_multiplier(epoch, max_epochs=50, warmup_epochs=5)
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
                    loss = id_loss + triplet
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
                value = float(loss.detach())
                losses.append(value)
                step_log.write(json.dumps({"epoch": epoch, "batch": batch_index, "loss": value,
                                            "id": float(id_loss.detach()), "triplet": float(triplet.detach()),
                                            "prediction": {}}) + "\n")
            step_log.flush()
            row = {"epoch": epoch, "steps": len(losses), "mean_loss": float(np.mean(losses)),
                   "seconds": time.perf_counter() - started}
            if args.mode == "train":
                metrics = roles_runner.official_metrics(model, protocol, args.signal_source)
                row["official_fused"] = metrics
                if metrics["mAP"] >= best["mAP"]:
                    best = {"mAP": metrics["mAP"], "epoch": epoch}
                    save_checkpoint(args.output_dir / "best_map.pth", model, args, epoch, metrics)
            receipt["history"].append(row)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"event": "epoch", "dataset": args.dataset, **row,
                              "best_epoch": best["epoch"]}), flush=True)
    assert frozen_signal_sha256 == _module_state_sha256(model.backbone.signal)
    if args.mode == "m0":
        assert live_parameters == set(named_parameters)
        probe = args.output_dir / "m0_reload_probe.pth"
        save_checkpoint(probe, model, args, 0, {})
        fresh, _cfg, _config, new_binding = build(args, protocol)
        assert new_binding == binding
        load_checkpoint(probe, fresh, args)
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
    receipt.update(status="M0_PASS" if args.mode == "m0" else "BEST_OFFICIAL_MAP_TRAINING_COMPLETE",
                   best_epoch=best["epoch"],
                   checkpoint=str(args.output_dir / "best_map.pth") if args.mode == "train" else None,
                   completed_at=datetime.now().astimezone().isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def evaluate(args, protocol):
    os.chdir(args.output_dir)
    training = json.loads((args.output_dir / "training.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    model, _cfg, _config, binding = build(args, protocol)
    assert binding == training["initializer"]
    checkpoint = args.output_dir / "best_map.pth"
    payload = load_checkpoint(checkpoint, model, args)
    selected = max(training["history"], key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    assert selected["epoch"] == payload["epoch"] == training["best_epoch"]
    assert selected["official_fused"] == payload["metrics"]
    distances = args.output_dir / "official_distances.pt"
    metrics = roles_runner.official_metrics(model, protocol, args.signal_source, save_distances=distances)
    assert all(abs(metrics[name] - payload["metrics"][name]) < 1e-5 for name in metrics)
    result = {"schema": "trifusion-correspondence-global-only-retrieval-v1",
              "status": "COMPLETE", "dataset": args.dataset, "seed": args.seed,
              "selected_epoch": payload["epoch"], "training_epochs": 50,
              "protocol_sha256": sha256(args.protocol), "baseline_sha256": args.baseline_sha256,
              "checkpoint_sha256": sha256(checkpoint), "distance_sha256": sha256(distances),
              "metrics": metrics, "reranking": False, "independent_upstream_metrics_equal": True,
              "completed_at": datetime.now().astimezone().isoformat()}
    (args.output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--mode", choices=("m0", "train", "evaluate"), required=True)
    for name in ("protocol", "signal-source", "clip-weight", "baseline-checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, choices=(50,), default=50)
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
