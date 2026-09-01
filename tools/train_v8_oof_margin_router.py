#!/usr/bin/env python3
"""Train the frozen-expert V8 Router from identity-OOF margin targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


def evaluate_router_phase_gate(
    *,
    learned_oof_margin: float,
    fixed_oof_margin: float,
    learned_top1_accuracy: float,
    majority_top1_accuracy: float,
    corrupted_mass_decreases: dict[str, bool],
    missing_modality_max_mass: float,
) -> dict[str, Any]:
    margin_passed = float(learned_oof_margin) > float(fixed_oof_margin)
    alignment_passed = float(learned_top1_accuracy) > float(majority_top1_accuracy)
    quality_passed = all(
        bool(corrupted_mass_decreases[name]) for name in ("RGB", "NI", "TI")
    )
    missing_passed = float(missing_modality_max_mass) == 0.0
    return {
        "passed": margin_passed and alignment_passed and quality_passed and missing_passed,
        "oof_margin_gain_passed": margin_passed,
        "top1_alignment_passed": alignment_passed,
        "quality_response_passed": quality_passed,
        "missing_modality_zero_mass_passed": missing_passed,
        "learned_oof_margin": float(learned_oof_margin),
        "fixed_oof_margin": float(fixed_oof_margin),
        "learned_top1_accuracy": float(learned_top1_accuracy),
        "majority_top1_accuracy": float(majority_top1_accuracy),
        "corrupted_mass_decreases": {
            name: bool(corrupted_mass_decreases[name])
            for name in ("RGB", "NI", "TI")
        },
        "missing_modality_max_mass": float(missing_modality_max_mass),
    }


def _stack_modal_residual(output: Any) -> Any:
    import torch

    return torch.stack(
        [output.modal_residual_embeddings[name] for name in ("cnn", "transformer", "mamba")],
        dim=1,
    )


def _quality_loader(records: list[Any], runtime: dict[str, Any], config: dict[str, Any]):
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    from torch.utils.data import DataLoader

    return DataLoader(
        ImageDataset(records, runtime["eval_loader"].dataset.transform),
        batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        collate_fn=val_collate_fn,
    )


def _collect_quality_cache(
    model: Any,
    records: list[Any],
    runtime: dict[str, Any],
    config: dict[str, Any],
    identity_to_fold: dict[int, int],
) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import apply_controlled_modality_degradation

    modalities = ("RGB", "NI", "TI")
    conditions = ("clean", *modalities)
    direct = {name: [] for name in conditions}
    residual = {name: [] for name in conditions}
    quality = {name: [] for name in conditions}
    identities: list[int] = []
    folds: list[int] = []
    loader = _quality_loader(records, runtime, config)
    model.eval()
    for images, batch_ids, _cameras, camera_labels, _views, _paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        batch_size = camera_labels.shape[0]
        base_batch = {
            "images": images,
            "modality_mask": torch.ones(batch_size, 3, dtype=torch.bool, device="cuda"),
            "camera_ids": camera_labels,
        }
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config["OPTIMIZATION"]["AMP"]),
        ):
            clean = model(base_batch, return_aux=True)
            direct["clean"].append(clean.direct_modal.float().cpu())
            residual["clean"].append(_stack_modal_residual(clean).float().cpu())
            quality["clean"].append(torch.ones(batch_size, 3))
            selected_samples = torch.ones(batch_size, dtype=torch.bool, device="cuda")
            for modality_index, modality in enumerate(modalities):
                selected_modalities = torch.full(
                    (batch_size,),
                    modality_index,
                    dtype=torch.long,
                    device="cuda",
                )
                degraded_images, targets = apply_controlled_modality_degradation(
                    images,
                    selected_modalities=selected_modalities,
                    selected_samples=selected_samples,
                    degraded_quality=float(config["QUALITY"]["DEGRADED_QUALITY"]),
                )
                degraded = model(
                    {
                        "images": degraded_images,
                        "modality_mask": base_batch["modality_mask"],
                        "camera_ids": camera_labels,
                    },
                    return_aux=True,
                )
                direct[modality].append(degraded.direct_modal.float().cpu())
                residual[modality].append(
                    _stack_modal_residual(degraded).float().cpu()
                )
                quality[modality].append(targets.float().cpu())
        batch_id_list = [int(value) for value in batch_ids]
        identities.extend(batch_id_list)
        folds.extend(identity_to_fold[value] for value in batch_id_list)
    return {
        "schema_version": "trifusion-v8-router-quality-cache-v1",
        "direct_modal": {name: torch.cat(direct[name]) for name in conditions},
        "modal_residual": {name: torch.cat(residual[name]) for name in conditions},
        "modality_quality": {name: torch.cat(quality[name]) for name in conditions},
        "identities": torch.tensor(identities, dtype=torch.long),
        "fold_indices": torch.tensor(folds, dtype=torch.long),
        "conditions": conditions,
    }


def _new_router(config: dict[str, Any], *, direct_width: int, residual_width: int):
    import torch

    from trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
    )

    seed = int(config["EXPERIMENT"]["SEED"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    router_config = config["ROUTER"]
    return HierarchicalOOFMarginRouter(
        direct_width=direct_width,
        residual_width=residual_width,
        hidden_width=int(router_config["HIDDEN_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
        alpha_init=float(router_config["ALPHA_INIT"]),
    ).cuda()


def _fit_router(
    router: Any,
    margin_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    margin_rows: Any,
    quality_rows: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from trifusion.signal_preserving_v8_router import (
        modality_quality_loss,
        oof_margin_router_loss,
    )

    router_config = config["ROUTER"]
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(router_config["LEARNING_RATE"]),
        weight_decay=float(router_config["WEIGHT_DECAY"]),
    )
    direct = margin_cache["direct_modal"][margin_rows].float().cuda()
    residual = margin_cache["modal_residual"][margin_rows].float().cuda()
    target_margin = margin_cache["target_identity_margin"][margin_rows].float().cuda()
    modality_mask = torch.ones(direct.shape[0], 3, dtype=torch.bool, device="cuda")
    conditions = tuple(quality_cache["conditions"])
    quality_direct = torch.cat(
        [quality_cache["direct_modal"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_residual = torch.cat(
        [quality_cache["modal_residual"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_target = torch.cat(
        [quality_cache["modality_quality"][name][quality_rows] for name in conditions]
    ).float().cuda()
    quality_mask = torch.ones(
        quality_direct.shape[0], 3, dtype=torch.bool, device="cuda"
    )
    history = []
    for epoch in range(1, int(router_config["EPOCHS"]) + 1):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        routing = router(direct, residual, modality_mask)
        margin_loss = oof_margin_router_loss(
            routing,
            target_margin,
            modality_mask,
            alpha_max=float(router_config["ALPHA_MAX"]),
            utility_temperature=float(router_config["UTILITY_TEMPERATURE"]),
            alpha_gain_scale=float(router_config["ALPHA_GAIN_SCALE"]),
        )
        quality_routing = router(quality_direct, quality_residual, quality_mask)
        quality_loss = modality_quality_loss(
            quality_routing.modal_probabilities,
            quality_target,
            quality_mask,
        )
        total = margin_loss.total + quality_loss * float(
            router_config["QUALITY_LOSS_WEIGHT"]
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("V8 Router training loss is nonfinite")
        total.backward()
        for name, parameter in router.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"V8 Router gradient is invalid: {name}")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "total": float(total.detach()),
                "utility": float(margin_loss.utility.detach()),
                "alpha": float(margin_loss.alpha.detach()),
                "quality": float(quality_loss.detach()),
            }
        )
    return {
        "optimizer_steps": len(history),
        "first_epoch": history[0],
        "final_epoch": history[-1],
    }


def _evaluate_router_oof(
    margin_cache: dict[str, Any],
    quality_cache: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    import torch

    folds = int(config["PROTOCOL"]["OOF_TARGET_FOLDS"])
    margin_fold = margin_cache["fold_indices"]
    quality_fold = quality_cache["fold_indices"]
    learned_margins = []
    fixed_margins = []
    learned_correct = []
    majority_correct = []
    clean_mass = {name: [] for name in ("RGB", "NI", "TI")}
    corrupted_mass = {name: [] for name in ("RGB", "NI", "TI")}
    missing_mass = []
    fold_receipts = []
    final_router = None
    for fold in range(folds):
        train_margin_rows = margin_fold != fold
        heldout_margin_rows = margin_fold == fold
        train_quality_rows = quality_fold != fold
        heldout_quality_rows = quality_fold == fold
        router = _new_router(
            config,
            direct_width=int(margin_cache["direct_modal"].shape[-1]),
            residual_width=int(margin_cache["modal_residual"].shape[-1]),
        )
        training = _fit_router(
            router,
            margin_cache,
            quality_cache,
            train_margin_rows,
            train_quality_rows,
            config,
        )
        router.eval()
        with torch.no_grad():
            direct = margin_cache["direct_modal"][heldout_margin_rows].float().cuda()
            residual = margin_cache["modal_residual"][heldout_margin_rows].float().cuda()
            target = margin_cache["target_identity_margin"][heldout_margin_rows].float().cuda()
            mask = torch.ones(direct.shape[0], 3, dtype=torch.bool, device="cuda")
            output = router(direct, residual, mask)
            learned_margins.append((output.weights * target).sum(dim=(1, 2)).cpu())
            target_winner = target.flatten(1).argmax(dim=1)
            learned_correct.append((output.weights.flatten(1).argmax(dim=1) == target_winner).cpu())
            fixed_slot = int(
                margin_cache["target_identity_margin"][train_margin_rows]
                .float()
                .mean(dim=0)
                .flatten()
                .argmax()
            )
            fixed_margins.append(target.flatten(1)[:, fixed_slot].cpu())
            majority_correct.append((target_winner == fixed_slot).cpu())

            clean_direct = quality_cache["direct_modal"]["clean"][heldout_quality_rows].float().cuda()
            clean_residual = quality_cache["modal_residual"]["clean"][heldout_quality_rows].float().cuda()
            clean_output = router(
                clean_direct,
                clean_residual,
                torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
            )
            for modality_index, modality in enumerate(("RGB", "NI", "TI")):
                corrupted_output = router(
                    quality_cache["direct_modal"][modality][heldout_quality_rows].float().cuda(),
                    quality_cache["modal_residual"][modality][heldout_quality_rows].float().cuda(),
                    torch.ones(clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"),
                )
                clean_mass[modality].append(
                    clean_output.modal_probabilities[:, modality_index].cpu()
                )
                corrupted_mass[modality].append(
                    corrupted_output.modal_probabilities[:, modality_index].cpu()
                )
                missing_mask = torch.ones(
                    clean_direct.shape[0], 3, dtype=torch.bool, device="cuda"
                )
                missing_mask[:, modality_index] = False
                missing_output = router(clean_direct, clean_residual, missing_mask)
                missing_mass.append(
                    missing_output.modal_probabilities[:, modality_index].cpu()
                )
        fold_receipts.append({"fold": fold, **training})
        final_router = router

    learned_margin = float(torch.cat(learned_margins).mean())
    fixed_margin = float(torch.cat(fixed_margins).mean())
    learned_accuracy = float(torch.cat(learned_correct).float().mean())
    majority_accuracy = float(torch.cat(majority_correct).float().mean())
    quality_response = {
        modality: {
            "clean_mean_mass": float(torch.cat(clean_mass[modality]).mean()),
            "corrupted_mean_mass": float(torch.cat(corrupted_mass[modality]).mean()),
        }
        for modality in ("RGB", "NI", "TI")
    }
    decreases = {
        modality: values["corrupted_mean_mass"] < values["clean_mean_mass"]
        for modality, values in quality_response.items()
    }
    missing_max = float(torch.cat(missing_mass).max())
    gate = evaluate_router_phase_gate(
        learned_oof_margin=learned_margin,
        fixed_oof_margin=fixed_margin,
        learned_top1_accuracy=learned_accuracy,
        majority_top1_accuracy=majority_accuracy,
        corrupted_mass_decreases=decreases,
        missing_modality_max_mass=missing_max,
    )
    return (
        {
            "fold_receipts": fold_receipts,
            "quality_response": quality_response,
            "gate": gate,
        },
        final_router,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from probe_v8_frozen_router import select_cross_camera_records
    from run_signal_preserving_v5 import (
        _build_runtime,
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )

    started = time.time()
    config_path = args.config.resolve()
    config = load_raw_config(config_path)
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V8 Router training is frozen to seed 42")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_ROUTER_TRAINING"]):
        raise ValueError("V8 Router training cannot access dev")
    initialization = config["INITIALIZATION"]
    phase_a_path = Path(initialization["PHASE_A_CHECKPOINT"]).resolve()
    margin_path = Path(initialization["OOF_MARGIN_CACHE"]).resolve()
    if _sha256(phase_a_path) != str(initialization["PHASE_A_CHECKPOINT_SHA256"]):
        raise ValueError("Phase-A checkpoint SHA-256 differs from the contract")
    if _sha256(margin_path) != str(initialization["OOF_MARGIN_CACHE_SHA256"]):
        raise ValueError("OOF margin cache SHA-256 differs from the contract")
    if args.output_dir.exists():
        raise FileExistsError(f"V8 Router output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    runtime = _build_runtime(config)
    model = runtime["model"]
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=True)
    model.load_state_dict(phase_a["model_state_dict"], strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expert_state_before = _module_state_sha256(model)
    margin_cache = torch.load(margin_path, map_location="cpu", weights_only=True)
    if margin_cache["schema_version"] != "trifusion-v8-oof-router-margin-cache-v1":
        raise ValueError("unexpected OOF margin cache schema")
    identity_to_fold = {
        int(identity): int(fold)
        for identity, fold in zip(
            margin_cache["identities"].tolist(),
            margin_cache["fold_indices"].tolist(),
            strict=True,
        )
    }
    eligible_records = select_cross_camera_records(runtime["train_records"])
    torch.cuda.reset_peak_memory_stats()
    quality_cache = _collect_quality_cache(
        model,
        eligible_records,
        runtime,
        config,
        identity_to_fold,
    )
    quality_cache_path = args.output_dir / "router_quality_features.pth"
    torch.save(quality_cache, quality_cache_path)
    oof, _last_fold_router = _evaluate_router_oof(margin_cache, quality_cache, config)
    optimizer_steps = sum(
        int(receipt["optimizer_steps"]) for receipt in oof["fold_receipts"]
    )
    combined_checkpoint = None
    combined_checkpoint_sha256 = None
    final_training = None
    if oof["gate"]["passed"]:
        final_router = _new_router(
            config,
            direct_width=int(margin_cache["direct_modal"].shape[-1]),
            residual_width=int(margin_cache["modal_residual"].shape[-1]),
        )
        final_training = _fit_router(
            final_router,
            margin_cache,
            quality_cache,
            torch.ones(margin_cache["fold_indices"].shape[0], dtype=torch.bool),
            torch.ones(quality_cache["fold_indices"].shape[0], dtype=torch.bool),
            config,
        )
        optimizer_steps += int(final_training["optimizer_steps"])
        combined_checkpoint = args.output_dir / "v8_phase_a_plus_router.pth"
        torch.save(
            {
                "schema_version": "trifusion-v8-phase-a-plus-router-v1",
                "phase_a_model_state_dict": phase_a["model_state_dict"],
                "router_state_dict": {
                    name: value.detach().cpu()
                    for name, value in final_router.state_dict().items()
                },
                "router_config": dict(config["ROUTER"]),
                "phase_a_checkpoint_sha256": _sha256(phase_a_path),
                "oof_margin_cache_sha256": _sha256(margin_path),
            },
            combined_checkpoint,
        )
        combined_checkpoint_sha256 = _sha256(combined_checkpoint)
    expert_state_after = _module_state_sha256(model)
    if expert_state_after != expert_state_before:
        raise RuntimeError("V8 Phase-A experts changed during Router training")
    result = {
        "schema_version": "trifusion-v8-oof-margin-router-result-v1",
        "status": "PASS",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "phase_a_checkpoint": str(phase_a_path),
        "phase_a_checkpoint_sha256": _sha256(phase_a_path),
        "oof_margin_cache": str(margin_path),
        "oof_margin_cache_sha256": _sha256(margin_path),
        "quality_cache": str(quality_cache_path),
        "quality_cache_sha256": _sha256(quality_cache_path),
        "router_oof": oof,
        "final_training": final_training,
        "combined_checkpoint": str(combined_checkpoint) if combined_checkpoint else None,
        "combined_checkpoint_sha256": combined_checkpoint_sha256,
        "next_phase_authorized": bool(oof["gate"]["passed"]),
        "expert_state_sha256_before": expert_state_before,
        "expert_state_sha256_after": expert_state_after,
        "expert_state_unchanged": expert_state_before == expert_state_after,
        "router_training_executed": True,
        "expert_training_executed": False,
        "optimizer_steps": optimizer_steps,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["evaluate_router_phase_gate", "run"]
