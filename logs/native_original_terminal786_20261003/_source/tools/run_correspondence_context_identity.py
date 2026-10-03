#!/usr/bin/env python3
"""Context/local identity study; reuse fixed data, official scoring and 50-epoch schedule."""

import argparse
from datetime import datetime
from functools import partial
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
from tools import run_correspondence_roles as runner
from tools.official_three_dataset_model import sha256
from trifusion.correspondence_context_identity import ContextIdentityTriFusion

BASE_BUILD = runner.build
CONDITION = {}
loader_for, records_for = runner.loader_for, runner.records_for
_training_batch, _module_state_sha256 = runner._training_batch, runner._module_state_sha256
learning_rate_multiplier, _batch_hard_triplet = runner.learning_rate_multiplier, runner._batch_hard_triplet
official_metrics = runner.official_metrics


def build(args, protocol):
    assert args.m1 and args.m2 and not args.m3 and args.width == 128
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    binding.update(architecture="correspondence_context_identity_v1", condition=CONDITION.copy(),
                   auxiliary_id_weight=1.0, query_context_width=512,
                   entry_sha256=sha256(Path(__file__)),
                   reused_entry_sha256=sha256(ROOT / "tools/run_correspondence_roles.py"),
                   context_source_sha256=sha256(ROOT / "modeling/trifusion/correspondence_context_identity.py"),
                   evidence_source_sha256=sha256(ROOT / "modeling/trifusion/correspondence_evidence_readout.py"))
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-correspondence-context-identity-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": sha256(args.protocol), "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "condition": CONDITION.copy(), "metrics": metrics,
                "state": runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-correspondence-context-identity-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["condition"] == CONDITION
    assert set(payload["state"]) == set(runner.checkpoint_state(model))
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
    receipt = {"schema": "trifusion-correspondence-context-identity-training-v1",
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
                    auxiliary_id = (F.cross_entropy(output["auxiliary_logits"], labels, label_smoothing=0.1)
                                    if model.auxiliary_target != "none" else output["fused"].sum() * 0)
                    loss = id_loss + triplet + auxiliary_id
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
                                            "triplet": float(triplet.detach()), "auxiliary_id": float(auxiliary_id.detach()),
                                            "auxiliary_target": model.auxiliary_target}) + "\n")
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
    """One selected checkpoint; all three paths use complete official metadata."""
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    os.chdir(args.output_dir)
    training = json.loads((args.output_dir / "training.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert len(training["history"]) == 50
    model, _cfg, _config, binding = build(args, protocol)
    from utils import metrics as author_metrics
    assert binding == training["initializer"]
    checkpoint = args.output_dir / "best_map.pth"
    payload = load_checkpoint(checkpoint, model, args, protocol)
    assert Path(author_metrics.__file__).resolve() == (args.signal_source / "utils/metrics.py").resolve()
    names = ("fused", "shared_global", "joint_local")
    embeddings = {}
    model.eval()
    with torch.inference_mode():
        for split in ("query", "gallery"):
            chunks = {name: [] for name in names}
            records = records_for(protocol, split)
            for raw in loader_for(protocol, records, training=False, method="PLAIN_V8"):
                output = model(runner._eval_batch(raw, args.dataset), return_aux=True)
                for name in names:
                    value = output[name].float()
                    if name == "shared_global":
                        value = F.normalize(value, dim=1)
                    chunks[name].append(value.cpu())
            embeddings[split] = {name: torch.cat(values) for name, values in chunks.items()}
            assert all(value.shape == (protocol["counts"][split], 1536)
                       for value in embeddings[split].values())
    qrows, grows = (protocol["records"][name] for name in ("query", "gallery"))
    qids, gids = (np.asarray([r["identity"] for r in records]) for records in (qrows, grows))
    qcameras, gcameras = (np.asarray([r["camera"] for r in records]) for records in (qrows, grows))
    qscenes, gscenes = (np.asarray([r["scene"] for r in records]) for records in (qrows, grows))
    arrays = {"query_ids": qids, "gallery_ids": gids, "query_cameras": qcameras,
              "gallery_cameras": gcameras, "query_scenes": qscenes, "gallery_scenes": gscenes}
    metrics_by_path = {}
    for name in names:
        distances = runner.distance_matrix(embeddings["query"][name], embeddings["gallery"][name]).numpy()
        if args.dataset == "MSVR310":
            scores = scene_scores(distances, qids, gids, qscenes, gscenes)
            cmc, mean_ap = author_metrics.eval_func_msrv(distances, qids, gids, qcameras, gcameras, qscenes, gscenes)
        else:
            scores = camera_scores(distances, qids, gids, qcameras, gcameras)
            cmc, mean_ap = author_metrics.eval_func(distances, qids, gids, qcameras, gcameras)
        metrics = {"mAP":100 * float(mean_ap), "Rank-1":100 * float(cmc[0]),
                   "Rank-5":100 * float(cmc[4]), "Rank-10":100 * float(cmc[9])}
        assert all(abs(metrics[k] - scores["metrics"][k]) < 1e-5 for k in metrics)
        metrics_by_path[name] = metrics
        arrays[name] = torch.from_numpy(distances)
    metrics = metrics_by_path["fused"]
    assert all(abs(metrics[k] - payload["metrics"][k]) < 1e-5 for k in metrics)
    distances_path = args.output_dir / "official_distances.pt"
    torch.save(arrays, distances_path)
    result = {"schema":"trifusion-correspondence-context-identity-retrieval-v1",
              "status":"COMPLETE", "dataset":args.dataset, "seed":args.seed,
              "selected_epoch":payload["epoch"], "training_epochs":50,
              "condition":CONDITION.copy(), "protocol_sha256":sha256(args.protocol),
              "baseline_sha256":args.baseline_sha256, "checkpoint_sha256":sha256(checkpoint),
              "distance_sha256":sha256(distances_path), "metrics":metrics,
              "diagnostic_metrics":{name:metrics_by_path[name] for name in names[1:]},
              "reranking":False, "independent_upstream_metrics_equal":True,
              "completed_at":datetime.now().astimezone().isoformat()}
    (args.output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-mode", choices=("static", "context"), required=True)
    parser.add_argument("--auxiliary-target", choices=("none", "local", "global"), required=True)
    options, remaining = parser.parse_known_args()
    CONDITION.update(query_mode=options.query_mode, auxiliary_target=options.auxiliary_target)
    runner.CorrespondenceTriFusion = partial(ContextIdentityTriFusion, **CONDITION)
    runner.build, runner.train, runner.evaluate = build, train, evaluate
    runner.save_checkpoint, runner.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
