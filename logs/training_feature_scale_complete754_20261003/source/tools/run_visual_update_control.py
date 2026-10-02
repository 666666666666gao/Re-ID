#!/usr/bin/env python3
"""Matched frozen/low-LR visual update and role/global-only training controls."""

import argparse
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modeling"))
from tools import run_correspondence_roles as runner
from trifusion.role_global_tokens import GlobalTokenTriFusion

VISUAL_LR = 5e-6
SCHEMA = "trifusion-visual-update-control-v1"
VISUAL_PREFIX = "clip_vision_encoder.base."


class SharedGlobalOnly(nn.Module):
    """Reuse the exact initialized shared backbone/head; omit the role computation."""

    def __init__(self, initialized):
        super().__init__()
        self.backbone = initialized.backbone
        self.neck = initialized.neck
        self.classifier = initialized.classifier

    def forward(self, batch, *, return_aux=False):
        _, shared = self.backbone(batch["images"], batch["camera_ids"])
        fused = F.normalize(shared.float(), dim=1)
        if not return_aux:
            return fused
        return {"fused": fused, "logits": self.classifier(self.neck(fused))}


def condition(args):
    return {"visual_update": args.visual_update, "readout": args.readout,
            "visual_lr": VISUAL_LR, "visual_parameter_dtype": "float32"}


def frozen_signal_digest(model, update):
    """Retain the existing baseline immutability check for the frozen state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.backbone.signal.state_dict().items()):
        if update == "low_lr" and name.startswith(VISUAL_PREFIX):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build(args, protocol):
    runner.CorrespondenceTriFusion = partial(
        GlobalTokenTriFusion, token_mode="static", query_mode="context", auxiliary_target="none",
    )
    values = vars(args).copy()
    values.update(width=128, m1=True, m2=True, m3=False, pred_weight=0.0)
    initialized, cfg, config, binding = runner.build(argparse.Namespace(**values), protocol)
    # Both fresh controls use the same FP32 parameter storage, including frozen controls.
    visual = initialized.backbone.signal.clip_vision_encoder.base
    visual.float()
    assert len(list(visual.parameters())) == 152
    assert all(p.dtype == torch.float32 for p in visual.parameters())
    common = {name: runner._module_state_sha256(getattr(initialized, name))
              for name in ("backbone", "neck", "classifier")}
    model = SharedGlobalOnly(initialized) if args.readout == "global_only" else initialized
    assert common == {name: runner._module_state_sha256(getattr(model, name)) for name in common}
    visual.requires_grad_(args.visual_update == "low_lr")
    binding.update(architecture="visual_update_control_v1", condition=condition(args),
                   common_initializer_sha256=common,
                   initial_model_state_sha256=runner._module_state_sha256(model),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   entry_sha256=runner.sha256(Path(__file__)),
                   role_source_sha256=runner.sha256(ROOT / "modeling/trifusion/role_global_tokens.py"),
                   visual_update_semantics="base visual parameters only; camera and nonvisual Signal state frozen")
    return model, cfg, config, binding


def save_checkpoint(path, model, args, epoch, metrics):
    # The previous role checkpoints omit Signal. Full state is necessary for visual updates.
    torch.save({"schema": SCHEMA, "dataset": args.dataset, "seed": args.seed,
                "epoch": epoch, "protocol_sha256": runner.sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256, "condition": condition(args),
                "metrics": metrics,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_checkpoint(path, model, args):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == SCHEMA and payload["condition"] == condition(args)
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == runner.sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    model.load_state_dict(payload["state"], strict=True)
    return payload


def train(args, protocol):
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    os.chdir(args.output_dir)
    model, _cfg, config, binding = build(args, protocol)
    frozen_before = frozen_signal_digest(model, args.visual_update)
    visual = model.backbone.signal.clip_vision_encoder.base
    visual_before = runner._module_state_sha256(visual)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    visual_names = {"backbone.signal." + VISUAL_PREFIX + name for name, _ in visual.named_parameters()}
    new_parameters = [p for name, p in named.items() if name not in visual_names]
    groups = [{"params": new_parameters, "base_lr": config["OPTIMIZATION"]["NEW_MODULE_LR"],
               "lr": config["OPTIMIZATION"]["NEW_MODULE_LR"], "label": "new"}]
    if args.visual_update == "low_lr":
        assert visual_names <= set(named)
        groups.append({"params": list(visual.parameters()), "base_lr": VISUAL_LR,
                       "lr": VISUAL_LR, "label": "visual"})
    else:
        assert not visual_names.intersection(named)
    optimizer = torch.optim.AdamW(groups, weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    loader = runner.loader_for(protocol, runner.records_for(protocol, "train"), training=True,
                               method="PLAIN_V8", seed=args.seed)
    receipt = {"schema": SCHEMA, "status": "RUNNING", "dataset": args.dataset, "seed": args.seed,
               "epochs": 50, "checkpoint_policy": "best_official_map", "initializer": binding,
               "protocol_sha256": runner.sha256(args.protocol), "condition": condition(args),
               "started_at": datetime.now().astimezone().isoformat(), "history": [],
               "optimizer_groups": [{"label": g["label"], "base_lr": g["base_lr"],
                                     "tensors": len(g["params"]),
                                     "elements": sum(p.numel() for p in g["params"])} for g in groups]}
    receipt_path = args.output_dir / "training.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    best = {"mAP": -1.0, "epoch": None}
    live = set()
    audit_batch = None
    epochs = 1 if args.mode == "m0" else 50
    torch.cuda.reset_peak_memory_stats()
    with (args.output_dir / "training_steps.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            multiplier = runner.learning_rate_multiplier(epoch, max_epochs=50, warmup_epochs=5)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * multiplier
            model.train()
            losses = []
            for index, raw in enumerate(loader):
                if args.mode == "m0" and index == 8:
                    break
                batch, labels = runner._training_batch(raw)
                if audit_batch is None:
                    audit_batch = {"images": {name: value[:2].clone() for name, value in batch["images"].items()},
                                   "camera_ids": batch["camera_ids"][:2].clone()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    identity = F.cross_entropy(output["logits"], labels, label_smoothing=0.1)
                    triplet = runner._batch_hard_triplet(output["fused"], labels, margin=0.3)
                    loss = identity + triplet
                assert bool(torch.isfinite(loss))
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                assert all(torch.isfinite(p.grad).all() for p in named.values() if p.grad is not None)
                live.update(name for name, p in named.items()
                            if p.grad is not None and bool(p.grad.abs().sum() > 0))
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() >= scale
                value = float(loss.detach())
                losses.append(value)
                log.write(json.dumps({"epoch": epoch, "batch": index, "loss": value,
                                      "id": float(identity.detach()), "triplet": float(triplet.detach()),
                                      "lr": {g["label"]: g["lr"] for g in optimizer.param_groups}}) + "\n")
            log.flush()
            row = {"epoch": epoch, "steps": len(losses), "mean_loss": float(np.mean(losses)),
                   "seconds": time.perf_counter() - started}
            if args.mode == "train":
                metrics = runner.official_metrics(model, protocol, args.signal_source)
                row["official_fused"] = metrics
                if metrics["mAP"] >= best["mAP"]:
                    best = {"mAP": metrics["mAP"], "epoch": epoch}
                    save_checkpoint(args.output_dir / "best_map.pth", model, args, epoch, metrics)
            receipt["history"].append(row)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"event": "epoch", "dataset": args.dataset, **row,
                              "best_epoch": best["epoch"]}), flush=True)
    assert frozen_before == frozen_signal_digest(model, args.visual_update)
    visual_after = runner._module_state_sha256(visual)
    assert (visual_before != visual_after) == (args.visual_update == "low_lr")
    receipt.update(frozen_signal_state_unchanged=True, visual_parameters_changed=visual_before != visual_after,
                   visual_state_before_sha256=visual_before, visual_state_after_sha256=visual_after,
                   peak_training_and_epoch_eval_allocated_bytes=torch.cuda.max_memory_allocated())
    if args.mode == "m0":
        assert live == set(named), sorted(set(named) - live)
        probe = args.output_dir / "m0_reload_probe.pth"
        save_checkpoint(probe, model, args, 0, {})
        fresh, _cfg, _config, new_binding = build(args, protocol)
        assert new_binding == binding
        load_checkpoint(probe, fresh, args)
        model.eval()
        fresh.eval()
        with torch.inference_mode():
            original, reloaded = model(audit_batch), fresh(audit_batch)
        assert torch.allclose(original, reloaded, atol=1e-5, rtol=1e-5)
        receipt["m0"] = {"nonzero_gradient_parameters": len(live), "trainable_parameters": len(named),
                         "frozen_signal_state_unchanged": True,
                         "visual_parameters_changed": visual_before != visual_after,
                         "reload_max_abs_difference": float((original - reloaded).abs().max()),
                         "reload_probe_sha256": runner.sha256(probe)}
    receipt.update(peak_stage_allocated_bytes=torch.cuda.max_memory_allocated(),
                   status="M0_PASS" if args.mode == "m0" else "BEST_OFFICIAL_MAP_TRAINING_COMPLETE",
                   best_epoch=best["epoch"], completed_at=datetime.now().astimezone().isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def evaluate(args, protocol):
    os.chdir(args.output_dir)
    training = json.loads((args.output_dir / "training.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    model, _cfg, _config, binding = build(args, protocol)
    assert training["initializer"] == binding
    checkpoint = args.output_dir / "best_map.pth"
    payload = load_checkpoint(checkpoint, model, args)
    selected = max(training["history"], key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    assert selected["epoch"] == payload["epoch"] == training["best_epoch"]
    assert selected["official_fused"] == payload["metrics"]
    distances = args.output_dir / "official_distances.pt"
    metrics = runner.official_metrics(model, protocol, args.signal_source, save_distances=distances)
    assert all(abs(metrics[name] - payload["metrics"][name]) < 1e-5 for name in metrics)
    result = {"schema": SCHEMA, "status": "COMPLETE", "dataset": args.dataset, "seed": args.seed,
              "condition": condition(args), "selected_epoch": payload["epoch"], "training_epochs": 50,
              "protocol_sha256": runner.sha256(args.protocol), "baseline_sha256": args.baseline_sha256,
              "checkpoint_sha256": runner.sha256(checkpoint), "distance_sha256": runner.sha256(distances),
              "metrics": metrics, "reranking": False, "independent_upstream_metrics_equal": True,
              "completed_at": datetime.now().astimezone().isoformat()}
    (args.output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--mode", choices=("m0", "train", "evaluate"), required=True)
    parser.add_argument("--visual-update", choices=("frozen", "low_lr"), required=True)
    parser.add_argument("--readout", choices=("roles", "global_only"), required=True)
    for name in ("protocol", "signal-source", "clip-weight", "baseline-checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(50,), default=50)
    args = parser.parse_args()
    for name in ("protocol", "signal_source", "clip_weight", "baseline_checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    protocol = runner.read_protocol(args.protocol, args.dataset)
    (evaluate if args.mode == "evaluate" else train)(args, protocol)


if __name__ == "__main__":
    main()
