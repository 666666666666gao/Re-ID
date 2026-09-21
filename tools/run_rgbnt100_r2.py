#!/usr/bin/env python3
"""RGBNT100 fixed full-train R2 role-gradient balance run.

The Signal checkpoint, model construction, sampler, and fixed official
evaluation are reused from the registered RGBNT100 V8 pipeline. R2 changes
only the role encoder gradient combination: weighted triplet gradients are
the ranking task and weighted ID gradients are the auxiliary task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from tools.msvr_supported_gradient_balance import SupportedBalance, role_indices
from tools.run_signal_preserving_v5 import (
    _module_state_sha256,
    _set_seed,
    _training_batch,
    learning_rate_multiplier,
    weighted_training_loss,
)
from tools.train_rgbnt100_trifusion_main import load_inputs
from tools.train_rgbnt100_trifusion_oof import (
    build_model,
    engineering_checks,
    evaluate,
    extract,
)
from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion

ROOT = Path(__file__).resolve().parents[1]
EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loss_parts(parts, config):
    weights = config["LOSS"]
    rank = parts["triplet_fused"] * float(weights["TRIPLET_FUSED"])
    auxiliary = parts["id_fused"] * float(weights["ID_FUSED"])
    for expert in EXPERTS:
        rank = rank + parts[f"triplet_{expert}"] * float(weights["TRIPLET_BRANCH"])
        rank = rank + parts[f"triplet_residual_{expert}"] * float(weights["TRIPLET_RESIDUAL"])
        auxiliary = auxiliary + parts[f"id_{expert}"] * float(weights["ID_BRANCH"])
        auxiliary = auxiliary + parts[f"id_residual_{expert}"] * float(weights["ID_RESIDUAL"])
    return rank, auxiliary


def train_roles_r2(model, records, config, directory, *, epochs=20):
    import numpy as np
    from tools.train_rgbnt100_signal_oof import loader_for, write_json

    _set_seed(42)
    model.train()
    initial = _module_state_sha256(model)
    frozen = _module_state_sha256(model.baseline)
    selected = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    names = [name for name, _ in selected]
    parameters = [parameter for _, parameter in selected]
    groups = role_indices(names)
    controller = SupportedBalance()
    optimizer = torch.optim.AdamW(parameters,
                                  lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    criterion = ExpertFormationV8Criterion(
        triplet_margin=config["LOSS"]["TRIPLET_MARGIN"],
        label_smoothing=config["LOSS"]["LABEL_SMOOTHING"],
    ).cuda()
    loader = loader_for(records, True)
    index_by_name = {Path(row[0]).name: index for index, row in enumerate(records)}
    history, steps, live, overflow = [], [], set(), 0
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "steps.jsonl"
    with log_path.open("x", encoding="utf-8") as stream:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * learning_rate_multiplier(
                epoch, max_epochs=epochs, warmup_epochs=5
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            epoch_rows = []
            for raw in loader:
                batch, labels = _training_batch(raw)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    parts = criterion(output, labels)
                    rank_loss, auxiliary_loss = _loss_parts(parts, config)
                    total_loss = rank_loss + auxiliary_loss
                assert torch.isfinite(total_loss).item()
                scale = scaler.get_scale()
                rank_values = torch.autograd.grad(
                    rank_loss * scale, parameters, retain_graph=True, allow_unused=True
                )
                aux_values = torch.autograd.grad(
                    auxiliary_loss * scale, parameters, retain_graph=True, allow_unused=True
                )
                total_values = torch.autograd.grad(
                    total_loss * scale, parameters, allow_unused=True
                )
                rank_grads = [torch.zeros_like(p, dtype=torch.float32) if value is None
                              else value.detach().float() for p, value in zip(parameters, rank_values, strict=True)]
                aux_grads = [torch.zeros_like(p, dtype=torch.float32) if value is None
                             else value.detach().float() for p, value in zip(parameters, aux_values, strict=True)]
                total_grads = [torch.zeros_like(p, dtype=torch.float32) if value is None
                               else value.detach().float() for p, value in zip(parameters, total_values, strict=True)]
                applied = {}
                for role, indexes in groups.items():
                    rank_norm = float(torch.linalg.vector_norm(torch.cat([rank_grads[i].reshape(-1) for i in indexes])) / scale)
                    aux_norm = float(torch.linalg.vector_norm(torch.cat([aux_grads[i].reshape(-1) for i in indexes])) / scale)
                    observation = controller.observe(role, rank_norm, aux_norm, True)
                    wr = observation["proposed_rank_weight"]
                    wa = observation["proposed_auxiliary_weight"]
                    for i in indexes:
                        parameters[i].grad = rank_grads[i] * wr + aux_grads[i] * wa
                    applied[role] = {**observation, "rank_norm": rank_norm, "auxiliary_norm": aux_norm}
                role_indexes = {i for group in groups.values() for i in group}
                for i, parameter in enumerate(parameters):
                    if i not in role_indexes:
                        parameter.grad = total_grads[i]
                scaler.unscale_(optimizer)
                finite = True
                for name, parameter in selected:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                        finite = False
                    elif parameter.grad.abs().sum().item() > 0:
                        live.add(name)
                assert finite
                scaler.step(optimizer)
                scaler.update()
                overflow += int(scaler.get_scale() < scale)
                row = {"step": len(steps) + 1, "epoch": epoch, "loss": float(total_loss.detach()),
                       "rank_loss": float(rank_loss.detach()), "auxiliary_loss": float(auxiliary_loss.detach()),
                       "sampled_record_indices": [index_by_name[Path(path).name] for path in raw[-1]],
                       "amp_scale_before": scale, "amp_scale_after": scaler.get_scale(),
                       "optimizer_update_applied": scaler.get_scale() >= scale,
                       "gradient_balance": applied}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                assert row["optimizer_update_applied"]
                steps.append(row)
                epoch_rows.append(row)
            history.append({"epoch": epoch, "optimizer_steps": len(epoch_rows), "learning_rate": lr,
                            "mean_loss": float(np.mean([row["loss"] for row in epoch_rows])),
                            "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps({"event": "rgbnt100_r2_epoch", **history[-1]}), flush=True)
    training = {"epochs": epochs, "optimizer_steps": len(steps), "initial_state_sha256": initial,
                "final_state_sha256": _module_state_sha256(model), "frozen_state_before_sha256": frozen,
                "frozen_state_after_sha256": _module_state_sha256(model.baseline),
                "trainable_tensors": len(names), "nonzero_gradient_tensors": len(live),
                "missing_nonzero_gradients": sorted(set(names) - live), "overflow_events": overflow,
                "history": history, "steps": steps, "gradient_balance_rule": {
                    "ema_decay": 0.9, "exponent": 0.5, "ratio_min": 0.25, "ratio_max": 4.0,
                    "weight_sum": 2.0, "tasks": ["weighted_triplet_rank", "weighted_id_auxiliary"]},
                "runner_sha256": _sha256(Path(__file__)),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    write_json(directory / "training.json", training)
    return training


def run(args):
    import torch

    config, baseline, protocol, signal_config, binding, fold, records = load_inputs(args.config.resolve())
    assert torch.cuda.is_available()
    assert args.mode in ("main",)
    output_dir = args.output_dir.resolve()
    assert not output_dir.exists()
    output_dir.mkdir(parents=True)
    model, initialization = build_model(config, signal_config, fold, baseline)
    started = time.perf_counter()
    training = train_roles_r2(model, records, config, output_dir, epochs=20)
    checks = {
        "all_trainable_gradients_live": not training["missing_nonzero_gradients"],
        "overflow_zero": training["overflow_events"] == 0,
        "frozen_state_unchanged": training["frozen_state_before_sha256"] == training["frozen_state_after_sha256"],
        "role_state_updated": training["initial_state_sha256"] != training["final_state_sha256"],
        "capacity_below_24gib": training["peak_reserved_mib"] < 24 * 1024,
        "fixed_epoch": training["epochs"] == 20,
    }
    assert all(checks.values()), checks
    checkpoint = output_dir / "roles_epoch20_r2.pth"
    torch.save({"model_state_dict": model.state_dict(), "fold": fold["fold"],
                "source_ids": fold["source_ids"], "heldout_ids": fold["heldout_ids"],
                "r2_rule": training["gradient_balance_rule"]}, checkpoint)
    before = extract(model, records, signal_config)
    del model
    torch.cuda.empty_cache()
    model, _ = build_model(config, signal_config, fold, baseline)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    after = extract(model, records, signal_config)
    assert all(torch.equal(before[name], after[name]) for name in OUTPUTS)
    retrieval = evaluate(after, protocol, fold, output_dir, baseline)
    summary = {"schema": "rgbnt100-r2-fixed-full-v1", "status": "COMPLETE_R2_FIXED_EPOCH20",
               "dataset": "RGBNT100", "seed": 42, "method": "R2_supported_role_gradient_balance",
               "config": str(args.config.resolve()), "config_sha256": _sha256(args.config.resolve()),
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "training_endpoint": 20, "initialization": initialization, "training": training,
               "engineering_checks": checks, "checkpoint": str(checkpoint),
               "checkpoint_sha256": _sha256(checkpoint), "strict_reload_outputs_bitwise_equal": True,
               "retrieval": retrieval, "elapsed_seconds": time.perf_counter() - started,
               "output_mapping": {"Signal": "baseline_only", "CNN": "cnn", "Transformer": "transformer",
                                  "Mamba": "mamba", "fused": "fused"}}
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], "metrics": {k: retrieval["outputs"][k]["metrics"] for k in OUTPUTS}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("main",), default="main")
    run(parser.parse_args())
