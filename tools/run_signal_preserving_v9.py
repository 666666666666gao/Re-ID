#!/usr/bin/env python3
"""Run train-only readiness or final training for TriFusion V9."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any


V9_SCHEMA = "trifusion-v9-orthogonal-triadic-synthesis-v1"


def _build_v9_runtime(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_runtime,
        _module_state_sha256,
        _sha256,
    )

    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V9 is frozen to seed 42")
    if bool(config["PROTOCOL"]["TRAINING_DEV_ACCESS"]):
        raise ValueError("V9 training cannot access dev")
    checkpoint_path = Path(config["INITIALIZATION"]["COMBINED_CHECKPOINT"])
    expected_sha = str(config["INITIALIZATION"]["COMBINED_CHECKPOINT_SHA256"])
    if _sha256(checkpoint_path) != expected_sha:
        raise ValueError("V8 combined checkpoint SHA-256 differs from the contract")

    runtime_config = deepcopy(config)
    runtime_config["MODEL"]["ARCHITECTURE"] = ARCHITECTURE_V8
    runtime = _build_runtime(runtime_config)
    from trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
        OOFMarginRoutedFusion,
    )
    from trifusion.signal_preserving_v9_builder import (
        build_signal_preserving_trifusion_v9,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != "trifusion-v8-phase-a-plus-router-v1":
        raise ValueError("unexpected V8 combined checkpoint schema")
    phase_a = runtime["model"]
    phase_a.load_state_dict(payload["phase_a_model_state_dict"], strict=True)
    router_config = payload["router_config"]
    router = HierarchicalOOFMarginRouter(
        direct_width=int(config["MODEL"]["FEATURE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        hidden_width=int(router_config["HIDDEN_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
        alpha_init=float(router_config["ALPHA_INIT"]),
    )
    router.load_state_dict(payload["router_state_dict"], strict=True)
    phase_b_fusion = OOFMarginRoutedFusion(
        baseline_width=int(config["SIGNAL"]["BASELINE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
    )
    v9 = config["V9"]
    build = build_signal_preserving_trifusion_v9(
        phase_a,
        router,
        phase_b_fusion,
        combined_checkpoint_sha256=expected_sha,
        num_classes=phase_a.num_classes,
        baseline_width=int(config["SIGNAL"]["BASELINE_WIDTH"]),
        phase_b_width=phase_b_fusion.fused_embedding_width,
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        hidden_width=int(v9["HIDDEN_WIDTH"]),
        synergy_modal_width=int(v9["SYNERGY_MODAL_WIDTH"]),
        relay_depth=int(v9["RELAY_DEPTH"]),
        beta_max=float(v9["BETA_MAX"]),
        beta_init=float(v9["BETA_INIT"]),
    )
    build.model.cuda()
    runtime.update(
        {
            "model": build.model,
            "build_provenance": dict(build.provenance),
            "combined_checkpoint": checkpoint_path,
            "combined_checkpoint_sha256": expected_sha,
            "phase_a_state_sha256": _module_state_sha256(build.model.phase_a),
            "router_state_sha256": _module_state_sha256(build.model.router),
        }
    )
    return runtime


def _criterion(config: dict[str, Any]):
    from trifusion.signal_preserving_v9 import SignalPreservingV9Criterion

    return SignalPreservingV9Criterion(
        triplet_margin=float(config["LOSS"]["TRIPLET_MARGIN"]),
        label_smoothing=float(config["LOSS"]["LABEL_SMOOTHING"]),
    ).cuda()


def _weighted_loss(losses: dict[str, Any], config: dict[str, Any]):
    weights = {
        "id_fused": float(config["LOSS"]["ID_FUSED"]),
        "triplet_fused": float(config["LOSS"]["TRIPLET_FUSED"]),
        "id_synergy": float(config["LOSS"]["ID_SYNERGY"]),
        "triplet_synergy": float(config["LOSS"]["TRIPLET_SYNERGY"]),
    }
    for expert in ("cnn", "transformer", "mamba"):
        weights[f"id_{expert}"] = float(config["LOSS"]["ID_BRANCH"])
        weights[f"triplet_{expert}"] = float(config["LOSS"]["TRIPLET_BRANCH"])
    return sum(losses[name] * weights[name] for name in weights)


def _trainable_parameters(model: Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _base_state(model: Any) -> tuple[str, str]:
    from run_signal_preserving_v5 import _module_state_sha256

    return _module_state_sha256(model.phase_a), _module_state_sha256(model.router)


def _relay_diagnostics(output: Any) -> dict[str, float]:
    import torch.nn.functional as F

    cosines = []
    norms = []
    for receivers, messages in zip(
        output.relay.receiver_inputs,
        output.relay.orthogonal_messages,
        strict=True,
    ):
        cosines.append(F.cosine_similarity(receivers, messages, dim=-1).abs())
        norms.append(messages.norm(dim=-1))
    return {
        "max_abs_relay_cosine": float(max(value.max() for value in cosines)),
        "mean_relay_norm": float(sum(value.mean() for value in norms) / len(norms)),
    }


def _run_preflight(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import _training_batch

    started = time.time()
    runtime = _build_v9_runtime(config)
    model = runtime["model"].eval()
    batch, _labels = _training_batch(next(iter(runtime["train_loader"])))
    phase_before, router_before = _base_state(model)
    with torch.no_grad():
        output = model(batch, return_aux=True)
    diagnostics = _relay_diagnostics(output)
    phase_after, router_after = _base_state(model)
    passed = (
        output.diagnostics["all_finite"]
        and output.diagnostics["baseline_exact_prefix"]
        and output.diagnostics["phase_b_exact_prefix"]
        and diagnostics["max_abs_relay_cosine"]
        <= float(config["GATES"]["MAX_ORTHOGONAL_COSINE"])
        and phase_before == phase_after
        and router_before == router_after
    )
    result = {
        "schema_version": V9_SCHEMA,
        "mode": "preflight",
        "status": "PASS" if passed else "FAIL",
        "diagnostics": {**dict(output.diagnostics), **diagnostics},
        "baseline_width": int(output.baseline_embedding.shape[1]),
        "phase_b_width": int(output.phase_b_embedding.shape[1]),
        "fused_width": int(output.fused_embedding.shape[1]),
        "branch_widths": {
            name: int(value.shape[1]) for name, value in output.branch_embeddings.items()
        },
        "phase_a_state_sha256_before": phase_before,
        "phase_a_state_sha256_after": phase_after,
        "router_state_sha256_before": router_before,
        "router_state_sha256_after": router_after,
        "phase_a_state_unchanged": phase_before == phase_after,
        "router_state_unchanged": router_before == router_after,
        "build_provenance": runtime["build_provenance"],
        "optimizer_steps": 0,
        "training_executed": False,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V9 preflight failed")
    return result


def _optimizer_and_scaler(model: Any, config: dict[str, Any]):
    import torch

    optimizer = torch.optim.AdamW(
        _trainable_parameters(model).values(),
        lr=float(config["OPTIMIZATION"]["NEW_MODULE_LR"]),
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"]),
    )
    return optimizer, scaler


def _training_step(
    model: Any,
    criterion: Any,
    optimizer: Any,
    scaler: Any,
    batch: dict[str, Any],
    labels: Any,
    config: dict[str, Any],
) -> tuple[float, bool, set[str]]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(config["OPTIMIZATION"]["AMP"]),
    ):
        output = model(batch, return_aux=True)
        if not output.diagnostics["all_finite"]:
            raise FloatingPointError("V9 emitted a nonfinite tensor")
        loss = _weighted_loss(criterion(output, labels), config)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("V9 loss is nonfinite")
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = set()
    for name, parameter in _trainable_parameters(model).items():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"V9 gradient is nonfinite: {name}")
            gradients.add(name)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), scaler.get_scale() < scale_before, gradients


def _run_capacity(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import _training_batch

    started = time.time()
    runtime = _build_v9_runtime(config)
    model = runtime["model"].train()
    criterion = _criterion(config)
    optimizer, scaler = _optimizer_and_scaler(model, config)
    phase_before, router_before = _base_state(model)
    trainable = _trainable_parameters(model)
    gradient_names: set[str] = set()
    losses = []
    overflow_events = 0
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(runtime["train_loader"])
    steps = int(config["GATES"]["CAPACITY_STEPS"])
    for _step in range(steps):
        batch, labels = _training_batch(next(iterator))
        loss, overflow, gradients = _training_step(
            model, criterion, optimizer, scaler, batch, labels, config
        )
        losses.append(loss)
        overflow_events += int(overflow)
        gradient_names.update(gradients)
    phase_after, router_after = _base_state(model)
    missing = sorted(set(trainable) - gradient_names)
    passed = (
        not missing
        and overflow_events == 0
        and phase_before == phase_after
        and router_before == router_after
    )
    result = {
        "schema_version": V9_SCHEMA,
        "mode": "capacity",
        "status": "PASS" if passed else "FAIL",
        "steps": steps,
        "losses": losses,
        "batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "trainable_parameters": sum(value.numel() for value in trainable.values()),
        "trainable_tensors": len(trainable),
        "trainable_gradient_tensors": len(gradient_names),
        "missing_gradient_tensors": missing,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "phase_a_state_sha256_before": phase_before,
        "phase_a_state_sha256_after": phase_after,
        "router_state_sha256_before": router_before,
        "router_state_sha256_after": router_after,
        "phase_a_state_unchanged": phase_before == phase_after,
        "router_state_unchanged": router_before == router_after,
        "build_provenance": runtime["build_provenance"],
        "optimizer_steps": steps,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V9 capacity gate failed")
    return result


def _label_smoothing_floor(config: dict[str, Any], num_classes: int) -> float:
    smoothing = float(config["LOSS"]["LABEL_SMOOTHING"])
    correct = 1.0 - smoothing + smoothing / num_classes
    other = smoothing / num_classes
    entropy = -correct * math.log(correct)
    entropy -= (num_classes - 1) * other * math.log(other)
    identity_weight = (
        float(config["LOSS"]["ID_FUSED"])
        + float(config["LOSS"]["ID_SYNERGY"])
        + 3.0 * float(config["LOSS"]["ID_BRANCH"])
    )
    return identity_weight * entropy


def _run_overfit(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import _training_batch, evaluate_overfit_gate

    started = time.time()
    runtime = _build_v9_runtime(config)
    model = runtime["model"].train()
    criterion = _criterion(config)
    optimizer, scaler = _optimizer_and_scaler(model, config)
    fixed_batch, fixed_labels = _training_batch(next(iter(runtime["train_loader"])))
    phase_before, router_before = _base_state(model)
    trainable = _trainable_parameters(model)
    gradient_names: set[str] = set()
    losses = []
    overflow_events = 0
    torch.cuda.reset_peak_memory_stats()
    steps = int(config["GATES"]["OVERFIT_STEPS"])
    for _step in range(steps):
        loss, overflow, gradients = _training_step(
            model,
            criterion,
            optimizer,
            scaler,
            fixed_batch,
            fixed_labels,
            config,
        )
        losses.append(loss)
        overflow_events += int(overflow)
        gradient_names.update(gradients)
    phase_after, router_after = _base_state(model)
    gate = evaluate_overfit_gate(
        losses,
        max_ratio=float(config["GATES"]["OVERFIT_MAX_LOSS_RATIO"]),
        minimum_loss=_label_smoothing_floor(config, model.phase_a.num_classes),
    )
    missing = sorted(set(trainable) - gradient_names)
    passed = (
        gate["passed"]
        and not missing
        and overflow_events == 0
        and phase_before == phase_after
        and router_before == router_after
    )
    result = {
        "schema_version": V9_SCHEMA,
        "mode": "overfit",
        "status": "PASS" if passed else "FAIL",
        "steps": steps,
        "losses": losses,
        "overfit_gate": gate,
        "trainable_tensors": len(trainable),
        "trainable_gradient_tensors": len(gradient_names),
        "missing_gradient_tensors": missing,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "phase_a_state_sha256_before": phase_before,
        "phase_a_state_sha256_after": phase_after,
        "router_state_sha256_before": router_before,
        "router_state_sha256_after": router_after,
        "phase_a_state_unchanged": phase_before == phase_after,
        "router_state_unchanged": router_before == router_after,
        "build_provenance": runtime["build_provenance"],
        "optimizer_steps": steps,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "overfit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V9 overfit gate failed")
    return result


def _collaboration_state(model: Any) -> dict[str, Any]:
    return {
        "synthesis": model.synthesis.state_dict(),
        "fused_neck": model.fused_neck.state_dict(),
        "synergy_neck": model.synergy_neck.state_dict(),
        "branch_necks": model.branch_necks.state_dict(),
        "fused_classifier": model.fused_classifier.state_dict(),
        "synergy_classifier": model.synergy_classifier.state_dict(),
        "branch_classifiers": model.branch_classifiers.state_dict(),
    }


def _run_train(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import (
        _sha256,
        _training_batch,
        learning_rate_multiplier,
    )

    started = time.time()
    runtime = _build_v9_runtime(config)
    model = runtime["model"].train()
    criterion = _criterion(config)
    optimizer, scaler = _optimizer_and_scaler(model, config)
    phase_before, router_before = _base_state(model)
    base_lr = float(config["OPTIMIZATION"]["NEW_MODULE_LR"])
    max_epochs = int(config["OPTIMIZATION"]["MAX_EPOCHS"])
    warmup_epochs = int(config["OPTIMIZATION"]["WARMUP_EPOCHS"])
    history = []
    optimizer_steps = 0
    overflow_events = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        multiplier = learning_rate_multiplier(
            epoch,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = base_lr * multiplier
        epoch_losses = []
        for raw_batch in runtime["train_loader"]:
            batch, labels = _training_batch(raw_batch)
            loss, overflow, _gradients = _training_step(
                model, criterion, optimizer, scaler, batch, labels, config
            )
            epoch_losses.append(loss)
            optimizer_steps += 1
            overflow_events += int(overflow)
        history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    phase_after, router_after = _base_state(model)
    if phase_before != phase_after or router_before != router_after:
        raise RuntimeError("V9 training changed the frozen V8 state")
    if overflow_events:
        raise RuntimeError("V9 training recorded AMP overflow")
    checkpoint_path = output_dir / "final_model.pth"
    torch.save(
        {
            "schema_version": V9_SCHEMA,
            "combined_v8_checkpoint_sha256": runtime[
                "combined_checkpoint_sha256"
            ],
            "collaboration_state_dict": _collaboration_state(model),
            "build_provenance": runtime["build_provenance"],
            "epoch": max_epochs,
            "seed": int(config["EXPERIMENT"]["SEED"]),
        },
        checkpoint_path,
    )
    result = {
        "schema_version": V9_SCHEMA,
        "mode": "train",
        "status": "PASS",
        "epochs_completed": max_epochs,
        "model_selection": "none_final_epoch_only",
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "combined_v8_checkpoint_sha256": runtime["combined_checkpoint_sha256"],
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "phase_a_state_sha256_before": phase_before,
        "phase_a_state_sha256_after": phase_after,
        "router_state_sha256_before": router_before,
        "router_state_sha256_after": router_after,
        "phase_a_state_unchanged": phase_before == phase_after,
        "router_state_unchanged": router_before == router_after,
        "build_provenance": runtime["build_provenance"],
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from run_signal_preserving_v5 import load_raw_config

    config = load_raw_config(args.config.resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"V9 output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    runners = {
        "preflight": _run_preflight,
        "capacity": _run_capacity,
        "overfit": _run_overfit,
        "train": _run_train,
    }
    return runners[args.mode](config, args.output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "capacity", "overfit", "train"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
