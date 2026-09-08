#!/usr/bin/env python3
"""Build identity-OOF expert-by-modality utility targets for the V8 Router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


def build_identity_folds(
    records: list[tuple[Any, int, int, int]] | tuple[tuple[Any, int, int, int], ...],
    *,
    num_folds: int,
) -> tuple[set[int], ...]:
    """Create deterministic folds while balancing cross-camera identities."""

    cameras_by_identity: dict[int, set[int]] = {}
    for _paths, identity, camera, _view in records:
        cameras_by_identity.setdefault(int(identity), set()).add(int(camera))
    eligible = sorted(
        identity
        for identity, cameras in cameras_by_identity.items()
        if len(cameras) >= 2
    )
    ineligible = sorted(set(cameras_by_identity) - set(eligible))
    folds = [set() for _ in range(int(num_folds))]
    for identities in (eligible, ineligible):
        for index, identity in enumerate(identities):
            folds[index % len(folds)].add(identity)
    return tuple(folds)


def evaluate_oof_target_gate(
    *,
    expert_winner_counts: dict[str, int],
    modality_winner_counts: dict[str, int],
    oracle_gain_map: float,
    min_oracle_gain_map: float,
) -> dict[str, Any]:
    """Require non-degenerate OOF utility labels before Router training."""

    expert_diversity = all(int(expert_winner_counts[name]) > 0 for name in EXPERTS)
    modality_diversity = all(
        int(modality_winner_counts[name]) > 0 for name in MODALITIES
    )
    oracle_passed = float(oracle_gain_map) >= float(min_oracle_gain_map)
    return {
        "passed": expert_diversity and modality_diversity and oracle_passed,
        "expert_diversity_passed": expert_diversity,
        "modality_diversity_passed": modality_diversity,
        "oracle_gain_passed": oracle_passed,
        "oracle_gain_mAP": float(oracle_gain_map),
        "minimum_oracle_gain_mAP": float(min_oracle_gain_map),
        "expert_unique_winner_counts": {
            name: int(expert_winner_counts[name]) for name in EXPERTS
        },
        "modality_unique_winner_counts": {
            name: int(modality_winner_counts[name]) for name in MODALITIES
        },
    }


def _eval_loader(records: list[Any], runtime: dict[str, Any], config: dict[str, Any]):
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    from torch.utils.data import DataLoader

    transform = runtime["eval_loader"].dataset.transform
    return DataLoader(
        ImageDataset(records + records, transform),
        batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        collate_fn=val_collate_fn,
    )


def _train_fold(
    model: Any,
    records: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from trifusion.aligned_data import build_aligned_train_loader
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_criterion,
        _criterion_losses,
        _module_state_sha256,
        _training_views,
        learning_rate_multiplier,
        weighted_training_loss,
    )

    loader = build_aligned_train_loader(
        records,
        batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        num_instances=int(config["DATA"]["NUM_INSTANCES"]),
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        seed=int(config["EXPERIMENT"]["SEED"]),
    )
    criterion = _build_criterion(config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_lr = float(config["OPTIMIZATION"]["NEW_MODULE_LR"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=base_lr,
        weight_decay=float(config["OPTIMIZATION"]["WEIGHT_DECAY"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=float(config["OPTIMIZATION"]["AMP_INIT_SCALE"])
    )
    epochs = int(config["OPTIMIZATION"]["OOF_TARGET_EPOCHS"])
    warmup_epochs = int(config["OPTIMIZATION"]["WARMUP_EPOCHS"])
    baseline_before = _module_state_sha256(model.baseline.signal)
    overflow_events = 0
    optimizer_steps = 0
    history = []
    for epoch in range(1, epochs + 1):
        learning_rate = base_lr * learning_rate_multiplier(
            epoch,
            max_epochs=epochs,
            warmup_epochs=warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        total = 0.0
        steps = 0
        for raw_batch in loader:
            batch, _quality_batch, labels = _training_views(raw_batch, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(config["OPTIMIZATION"]["AMP"]),
            ):
                output = model(batch, return_aux=True)
                if not output.diagnostics["all_finite"]:
                    raise FloatingPointError("V8 OOF expert fold emitted nonfinite output")
                parts = _criterion_losses(
                    criterion,
                    output,
                    labels,
                    architecture=ARCHITECTURE_V8,
                    quality_output=None,
                )
                loss = weighted_training_loss(
                    parts,
                    config,
                    phase="expert_formation",
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V8 OOF expert fold loss is nonfinite")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            overflow_events += int(overflow)
            optimizer_steps += int(not overflow)
            total += float(loss.detach())
            steps += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "mean_training_loss": total / steps,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    baseline_after = _module_state_sha256(model.baseline.signal)
    if baseline_before != baseline_after or overflow_events:
        raise RuntimeError("V8 OOF fold integrity gate failed")
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "history": history,
        "signal_state_sha256_before": baseline_before,
        "signal_state_sha256_after": baseline_after,
    }


def _collect_fold(model: Any, loader: Any) -> dict[str, Any]:
    import torch

    collected = {
        "direct_modal": [],
        "modal_residual": [],
        **{f"residual_{expert}": [] for expert in EXPERTS},
        **{
            f"slot_{expert}_{modality}": []
            for expert in EXPERTS
            for modality in MODALITIES
        },
    }
    identities = []
    cameras = []
    model.eval()
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(
                {
                    "images": images,
                    "modality_mask": torch.ones(
                        camera_labels.shape[0], 3, dtype=torch.bool, device="cuda"
                    ),
                    "camera_ids": camera_labels,
                },
                return_aux=True,
            )
        modal = torch.stack(
            [output.modal_residual_embeddings[expert] for expert in EXPERTS],
            dim=1,
        )
        collected["direct_modal"].append(output.direct_modal.float().cpu())
        collected["modal_residual"].append(modal.float().cpu())
        for expert_index, expert in enumerate(EXPERTS):
            collected[f"residual_{expert}"].append(
                output.residual_embeddings[expert].float().cpu()
            )
            for modality_index, modality in enumerate(MODALITIES):
                collected[f"slot_{expert}_{modality}"].append(
                    modal[:, expert_index, modality_index].float().cpu()
                )
        identities.extend(np.asarray(batch_ids).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    return {
        **{name: torch.cat(values) for name, values in collected.items()},
        "identities": np.asarray(identities),
        "cameras": np.asarray(cameras),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from diagnose_v6_oracle_complementarity import (
        QueryRetrievalScores,
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from probe_v8_frozen_router import select_cross_camera_records
    from run_signal_preserving_v5 import (
        ARCHITECTURE_V8,
        _build_runtime,
        _load_config,
        _set_seed,
        _sha256,
    )

    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"OOF output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if str(config["MODEL"]["ARCHITECTURE"]) != ARCHITECTURE_V8:
        raise ValueError("OOF Router targets require V8 expert formation")
    runtime = _build_runtime(config)
    model = runtime["model"]
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    folds = build_identity_folds(
        runtime["train_records"],
        num_folds=int(config["PROTOCOL"]["OOF_TARGET_FOLDS"]),
    )
    eligible_records = select_cross_camera_records(runtime["train_records"])
    eligible_ids = {int(record[1]) for record in eligible_records}
    fold_receipts = []
    query_direct = []
    query_modal_residual = []
    query_target_ap = []
    query_identities = []
    query_cameras = []
    query_folds = []
    expert_scores = {expert: [] for expert in EXPERTS}

    torch.cuda.reset_peak_memory_stats()
    for fold_index, heldout_ids in enumerate(folds):
        _set_seed(int(config["EXPERIMENT"]["SEED"]))
        model.load_state_dict(initial_state, strict=True)
        model.cuda()
        train_records = [
            record for record in runtime["train_records"] if int(record[1]) not in heldout_ids
        ]
        heldout_records = [
            record
            for record in eligible_records
            if int(record[1]) in heldout_ids
        ]
        if not heldout_records:
            raise RuntimeError("each OOF fold requires cross-camera held-out records")
        print(
            json.dumps(
                {
                    "fold": fold_index,
                    "fit_identities": len({int(record[1]) for record in train_records}),
                    "heldout_identities": len({int(record[1]) for record in heldout_records}),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        training = _train_fold(model, train_records, config)
        loader = _eval_loader(heldout_records, runtime, config)
        split = _collect_fold(model, loader)
        num_query = len(heldout_records)
        identities = split["identities"]
        cameras = split["cameras"]
        slot_ap = np.empty((num_query, len(EXPERTS), len(MODALITIES)))
        for expert_index, expert in enumerate(EXPERTS):
            score = _scores_from_features(
                split[f"residual_{expert}"],
                identities,
                cameras,
                num_query=num_query,
            )
            expert_scores[expert].append(score)
            for modality_index, modality in enumerate(MODALITIES):
                slot_score = _scores_from_features(
                    split[f"slot_{expert}_{modality}"],
                    identities,
                    cameras,
                    num_query=num_query,
                )
                slot_ap[:, expert_index, modality_index] = slot_score.average_precision
        query_direct.append(split["direct_modal"][:num_query].half())
        query_modal_residual.append(split["modal_residual"][:num_query].half())
        query_target_ap.append(torch.from_numpy(slot_ap).float())
        query_identities.extend(identities[:num_query].tolist())
        query_cameras.extend(cameras[:num_query].tolist())
        query_folds.extend([fold_index] * num_query)
        expert_state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if not name.startswith("baseline.")
        }
        checkpoint_path = args.output_dir / f"fold_{fold_index}_experts.pth"
        torch.save(
            {
                "fold": fold_index,
                "heldout_identities": sorted(int(value) for value in heldout_ids),
                "expert_state_dict": expert_state,
            },
            checkpoint_path,
        )
        fold_receipts.append(
            {
                "fold": fold_index,
                "fit_identity_count": len({int(record[1]) for record in train_records}),
                "heldout_identity_count": len(heldout_ids),
                "eligible_heldout_identity_count": len(heldout_ids & eligible_ids),
                "eligible_heldout_queries": num_query,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                **training,
            }
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": "trifusion-v8-oof-router-target-progress-v1",
                    "completed_folds": fold_receipts,
                    "official_test_access_count": 0,
                    "dev_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    combined_scores = {
        expert: QueryRetrievalScores(
            average_precision=np.concatenate(
                [score.average_precision for score in expert_scores[expert]]
            ),
            rank1_correct=np.concatenate(
                [score.rank1_correct for score in expert_scores[expert]]
            ),
        )
        for expert in EXPERTS
    }
    expert_oracle = summarize_oracle_complementarity(combined_scores)
    target_ap = torch.cat(query_target_ap)
    maximum = target_ap.amax(dim=(1, 2), keepdim=True)
    ties = torch.isclose(target_ap, maximum, rtol=0.0, atol=1e-12)
    unique = ties.flatten(1).sum(dim=1) == 1
    winners = target_ap.flatten(1).argmax(dim=1)
    expert_counts = {
        expert: int((unique & (winners // len(MODALITIES) == index)).sum())
        for index, expert in enumerate(EXPERTS)
    }
    modality_counts = {
        modality: int((unique & (winners % len(MODALITIES) == index)).sum())
        for index, modality in enumerate(MODALITIES)
    }
    fixed_slot_map = target_ap.mean(dim=0) * 100.0
    oracle_map = float(target_ap.amax(dim=(1, 2)).mean() * 100.0)
    best_fixed_slot_map = float(fixed_slot_map.max())
    oracle_gain_map = oracle_map - best_fixed_slot_map
    gate = evaluate_oof_target_gate(
        expert_winner_counts=expert_counts,
        modality_winner_counts=modality_counts,
        oracle_gain_map=oracle_gain_map,
        min_oracle_gain_map=float(config["GATES"]["FORMATION_MIN_ORACLE_GAIN_MAP"]),
    )
    target_cache_path = args.output_dir / "oof_router_targets.pth"
    torch.save(
        {
            "schema_version": "trifusion-v8-oof-router-target-cache-v1",
            "direct_modal": torch.cat(query_direct),
            "modal_residual": torch.cat(query_modal_residual),
            "target_average_precision": target_ap,
            "identities": torch.tensor(query_identities, dtype=torch.long),
            "cameras": torch.tensor(query_cameras, dtype=torch.long),
            "fold_indices": torch.tensor(query_folds, dtype=torch.long),
            "experts": EXPERTS,
            "modalities": MODALITIES,
        },
        target_cache_path,
    )
    result = {
        "schema_version": "trifusion-v8-oof-router-target-result-v1",
        "status": "PASS",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "folds": len(folds),
        "epochs_per_fold": int(config["OPTIMIZATION"]["OOF_TARGET_EPOCHS"]),
        "fit_identity_count": len({int(record[1]) for record in runtime["train_records"]}),
        "eligible_identity_count": len(eligible_ids),
        "oof_queries": int(target_ap.shape[0]),
        "fold_receipts": fold_receipts,
        "expert_oracle": expert_oracle,
        "slot_fixed_mAP_percent": {
            expert: {
                modality: float(fixed_slot_map[expert_index, modality_index])
                for modality_index, modality in enumerate(MODALITIES)
            }
            for expert_index, expert in enumerate(EXPERTS)
        },
        "slot_oracle_mAP_percent": oracle_map,
        "slot_oracle_minus_best_fixed_mAP_percent": oracle_gain_map,
        "oof_target_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "target_cache": str(target_cache_path),
        "target_cache_sha256": _sha256(target_cache_path),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "training_executed": True,
        "dev_access_count": 0,
        "official_test_access_count": 0,
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


__all__ = ["build_identity_folds", "evaluate_oof_target_gate", "run"]
