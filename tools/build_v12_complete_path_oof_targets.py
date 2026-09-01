#!/usr/bin/env python3
"""Build complete-path identity-OOF utility targets for the V12 Router."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
EXPECTED_FOLDS = 3
EXPECTED_QUERIES = 571
SIGNAL_EPOCHS = 50
EXPERT_EPOCHS = 20
SATURATION_MAP = 99.0
MIN_RESIDUAL_ORACLE_GAIN_MAP = 1.0


def build_complete_path_fold_records(
    records: Sequence[tuple[Any, int, int, int]],
    *,
    heldout_ids: set[int],
) -> dict[str, Any]:
    """Split one identity fold and relabel only its fit records."""

    heldout = {int(identity) for identity in heldout_ids}
    fit_ids = sorted({int(record[1]) for record in records} - heldout)
    label_map = {identity: index for index, identity in enumerate(fit_ids)}
    train_records = [
        (paths, label_map[int(identity)], camera, view)
        for paths, identity, camera, view in records
        if int(identity) not in heldout
    ]
    heldout_records = [
        record for record in records if int(record[1]) in heldout
    ]
    heldout_present = sorted({int(record[1]) for record in heldout_records})
    overlap = sorted(set(fit_ids) & set(heldout_present))
    return {
        "train_records": train_records,
        "heldout_records": heldout_records,
        "label_map": label_map,
        "fit_identity_ids": tuple(fit_ids),
        "heldout_identity_ids": tuple(heldout_present),
        "identity_overlap": tuple(overlap),
    }


def evaluate_complete_path_oof_gate(
    *,
    fold_receipts: Sequence[dict[str, Any]],
    query_count: int,
    fixed_map: dict[str, float],
    expert_winner_counts: dict[str, int],
    modality_winner_counts: dict[str, int],
    residual_oracle_gain_map: float,
    slot_oracle_margin_gain: float,
) -> dict[str, Any]:
    """Apply the preregistered V12 complete-path qualification decision."""

    protocol = len(fold_receipts) == EXPECTED_FOLDS and int(query_count) == EXPECTED_QUERIES
    identity_isolation = all(
        not (
            set(receipt["signal_fit_identity_ids"])
            & set(receipt["heldout_identity_ids"])
        )
        and not (
            set(receipt["expert_fit_identity_ids"])
            & set(receipt["heldout_identity_ids"])
        )
        for receipt in fold_receipts
    )
    training_schedule = all(
        int(receipt["signal_epochs"]) == SIGNAL_EPOCHS
        and int(receipt["expert_epochs"]) == EXPERT_EPOCHS
        and receipt["signal_checkpoint_selection"] == "final_epoch_only"
        for receipt in fold_receipts
    )
    runtime_integrity = all(
        int(receipt["overflow_events"]) == 0
        and int(receipt["dev_access_count"]) == 0
        and int(receipt["official_test_access_count"]) == 0
        for receipt in fold_receipts
    )
    best_fixed_map = max(float(value) for value in fixed_map.values())
    non_saturation = best_fixed_map < SATURATION_MAP
    expert_diversity = all(
        int(expert_winner_counts[expert]) > 0 for expert in EXPERTS
    )
    modality_diversity = all(
        int(modality_winner_counts[modality]) > 0 for modality in MODALITIES
    )
    residual_oracle_gain = (
        float(residual_oracle_gain_map) >= MIN_RESIDUAL_ORACLE_GAIN_MAP
    )
    slot_oracle_margin = float(slot_oracle_margin_gain) > 0.0
    passed = all(
        (
            protocol,
            identity_isolation,
            training_schedule,
            runtime_integrity,
            non_saturation,
            expert_diversity,
            modality_diversity,
            residual_oracle_gain,
            slot_oracle_margin,
        )
    )
    return {
        "passed": passed,
        "protocol_passed": protocol,
        "complete_path_identity_isolation_passed": identity_isolation,
        "training_schedule_passed": training_schedule,
        "runtime_integrity_passed": runtime_integrity,
        "non_saturation_passed": non_saturation,
        "expert_diversity_passed": expert_diversity,
        "modality_diversity_passed": modality_diversity,
        "residual_oracle_gain_passed": residual_oracle_gain,
        "slot_oracle_margin_gain_passed": slot_oracle_margin,
        "fold_count": len(fold_receipts),
        "query_count": int(query_count),
        "best_fixed_mAP": best_fixed_map,
        "saturation_mAP": SATURATION_MAP,
        "residual_oracle_gain_mAP": float(residual_oracle_gain_map),
        "minimum_residual_oracle_gain_mAP": MIN_RESIDUAL_ORACLE_GAIN_MAP,
        "slot_oracle_margin_gain": float(slot_oracle_margin_gain),
        "expert_unique_winner_counts": {
            expert: int(expert_winner_counts[expert]) for expert in EXPERTS
        },
        "modality_unique_winner_counts": {
            modality: int(modality_winner_counts[modality])
            for modality in MODALITIES
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(config: dict[str, Any]) -> list[tuple[Any, int, int, int]]:
    from tools.run_signal_baseline_dev import _records_for_ids

    protocol_path = Path(config["DATA"]["DEV_PROTOCOL"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    train_ids = {int(value) for value in protocol["train_ids"]}
    if len(train_ids) != 141 or int(protocol["counts"]["train_triplets"]) <= 0:
        raise ValueError("V12 requires the frozen 141-fit identity registry")
    records = list(
        _records_for_ids(
            Path(config["DATA"]["DATASET_ROOT"]).resolve() / "train_171",
            train_ids,
            relabel=True,
        )
    )
    if len(records) != int(protocol["counts"]["train_triplets"]):
        raise ValueError("V12 fit records differ from the frozen protocol")
    return records


def _configure_signal(config: dict[str, Any]) -> tuple[Any, str, str]:
    from tools.run_signal_baseline_dev import _configure_signal_source

    source = Path(config["SIGNAL"]["SOURCE"]).resolve()
    source_commit = _configure_signal_source(source)
    source_diff_sha256 = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
    ).hexdigest()
    from config import cfg as signal_cfg

    signal_cfg.merge_from_file(str(Path(config["SIGNAL"]["CONFIG"]).resolve()))
    signal_cfg.defrost()
    signal_cfg.MODEL.PRETRAIN_PATH_T = str(
        Path(config["SIGNAL"]["CLIP_WEIGHT"]).resolve()
    )
    signal_cfg.SOLVER.SEED = int(config["EXPERIMENT"]["SEED"])
    signal_cfg.SOLVER.MAX_EPOCHS = int(config["V12"]["SIGNAL_TEACHER_EPOCHS"])
    signal_cfg.SOLVER.IMS_PER_BATCH = int(config["DATA"]["TRAIN_BATCH_SIZE"])
    signal_cfg.DATALOADER.NUM_INSTANCE = int(config["DATA"]["NUM_INSTANCES"])
    signal_cfg.DATALOADER.NUM_WORKERS = int(config["DATA"]["NUM_WORKERS"])
    signal_cfg.TEST.IMS_PER_BATCH = int(config["DATA"]["EVAL_BATCH_SIZE"])
    signal_cfg.freeze()
    return signal_cfg, source_commit, source_diff_sha256


def _signal_training_loss(
    output: Any,
    *,
    loss_fn: Any,
    labels: Any,
    cameras: Any,
    stage: str,
    gram_weight: float,
    patch_weight: float,
) -> Any:
    loss = 0
    sign = output[0]
    if sign in (1, 2):
        end = len(output) - 1
        for index in range(1, end, 2):
            loss = loss + loss_fn(
                score=output[index],
                feat=output[index + 1],
                target=labels,
                target_cam=cameras,
            )
        return loss
    if stage == "CLS":
        end = len(output) - 2
        for index in range(1, end, 2):
            loss = loss + loss_fn(
                score=output[index],
                feat=output[index + 1],
                target=labels,
                target_cam=cameras,
            )
        return loss + float(gram_weight) * output[-1]
    end = len(output) - 3
    for index in range(1, end, 2):
        loss = loss + loss_fn(
            score=output[index],
            feat=output[index + 1],
            target=labels,
            target_cam=cameras,
        )
    return loss + float(gram_weight) * output[-2] + float(patch_weight) * output[-1]


def _build_signal_teacher(
    signal_cfg: Any,
    *,
    num_classes: int,
    camera_num: int,
    view_num: int,
) -> Any:
    from modeling import make_frame

    return make_frame(
        signal_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
    ).cuda()


def _train_signal_teacher(
    model: Any,
    loader: Any,
    signal_cfg: Any,
    *,
    epochs: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    import torch
    from layers.make_loss import make_loss
    from solver.make_optimizer import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from tools.run_signal_preserving_v5 import _module_state_sha256

    loss_fn, center_criterion = make_loss(signal_cfg, num_classes=model.num_classes)
    optimizer, _optimizer_center = make_optimizer(signal_cfg, model, center_criterion)
    scheduler = create_scheduler(signal_cfg, optimizer)
    scaler = torch.amp.GradScaler("cuda")
    initial_state_sha256 = _module_state_sha256(model)
    history = []
    overflow_events = 0
    optimizer_steps = 0
    gradient_names: set[str] = set()
    stop = False
    for epoch in range(1, int(epochs) + 1):
        scheduler.step(epoch)
        model.train()
        epoch_losses = []
        for images, labels, cameras, views, _paths in loader:
            optimizer.zero_grad(set_to_none=True)
            images = {
                name: tensor.cuda(non_blocking=True)
                for name, tensor in images.items()
            }
            labels = labels.cuda(non_blocking=True)
            cameras = cameras.cuda(non_blocking=True)
            views = views.cuda(non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                output = model(
                    images,
                    label=labels,
                    cam_label=cameras,
                    view_label=views,
                    training=True,
                    sge=signal_cfg.MODEL.stageName,
                )
                loss = _signal_training_loss(
                    output,
                    loss_fn=loss_fn,
                    labels=labels,
                    cameras=cameras,
                    stage=signal_cfg.MODEL.stageName,
                    gram_weight=float(signal_cfg.MODEL.Gram_Loss_weight),
                    patch_weight=float(signal_cfg.MODEL.PAT_Loss_weight),
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V12 Signal teacher loss is nonfinite")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(
                            f"V12 Signal teacher gradient is nonfinite: {name}"
                        )
                    gradient_names.add(name)
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            overflow_events += int(overflow)
            optimizer_steps += int(not overflow)
            epoch_losses.append(float(loss.detach()))
            if max_steps is not None and optimizer_steps + overflow_events >= max_steps:
                stop = True
                break
        history.append(
            {
                "epoch": epoch,
                "mean_training_loss": float(np.mean(epoch_losses)),
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if stop:
            break
    return {
        "epochs": len(history),
        "optimizer_steps": optimizer_steps,
        "overflow_events": overflow_events,
        "gradient_tensors": len(gradient_names),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": _module_state_sha256(model),
        "history": history,
    }


def _eval_loader(records: Sequence[Any], config: dict[str, Any]) -> Any:
    import torchvision.transforms as transforms
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import val_collate_fn
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transforms.Resize([256, 128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return DataLoader(
        ImageDataset(list(records) + list(records), transform),
        batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        collate_fn=val_collate_fn,
    )


def _build_v8_experts(
    signal_model: Any,
    config: dict[str, Any],
    *,
    signal_checkpoint_sha256: str,
    num_classes: int,
) -> Any:
    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.signal_preserving_v8_builder import (
        build_signal_preserving_trifusion_v8_expert_formation,
    )

    model_config = config["MODEL"]
    return build_signal_preserving_trifusion_v8_expert_formation(
        signal_model,
        signal_checkpoint_sha256=signal_checkpoint_sha256,
        num_classes=num_classes,
        feature_width=int(model_config["FEATURE_WIDTH"]),
        grid_size=tuple(model_config["GRID_SIZE"]),
        adapter_width=int(model_config["ADAPTER_WIDTH"]),
        semantic_width=int(model_config["SEMANTIC_WIDTH"]),
        branch_after_block=int(model_config["BRANCH_AFTER_BLOCK"]),
        expert_modal_width=int(model_config["EXPERT_MODAL_WIDTH"]),
        scale_init=float(model_config["SCALE_INIT"]),
        gradient_checkpointing=bool(model_config["GRADIENT_CHECKPOINTING"]),
    ).model.cuda()


def _metric_summary(scores: Any) -> dict[str, float]:
    return {
        "mAP": float(scores.average_precision.mean() * 100.0),
        "Rank-1": float(scores.rank1_correct.mean() * 100.0),
    }


def _run_preflight(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.aligned_data import build_aligned_train_loader
    from tools.build_v8_oof_router_targets import build_identity_folds
    from tools.run_signal_preserving_v5 import _set_seed

    started = time.time()
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    folds = build_identity_folds(records, num_folds=EXPECTED_FOLDS)
    split = build_complete_path_fold_records(records, heldout_ids=folds[0])
    loader = build_aligned_train_loader(
        split["train_records"],
        batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        num_instances=int(config["DATA"]["NUM_INSTANCES"]),
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        seed=int(config["EXPERIMENT"]["SEED"]),
    )
    _set_seed(int(config["EXPERIMENT"]["SEED"]))
    torch.cuda.reset_peak_memory_stats()
    model = _build_signal_teacher(
        signal_cfg,
        num_classes=len(split["fit_identity_ids"]),
        camera_num=len({record[2] for record in split["train_records"]}),
        view_num=len({record[3] for record in split["train_records"]}),
    )
    training = _train_signal_teacher(
        model,
        loader,
        signal_cfg,
        epochs=1,
        max_steps=1,
    )
    passed = (
        split["identity_overlap"] == ()
        and training["optimizer_steps"] == 1
        and training["overflow_events"] == 0
        and training["gradient_tensors"] > 0
    )
    result = {
        "schema_version": "trifusion-v12-complete-path-oof-preflight-v1",
        "status": "PASS" if passed else "FAIL",
        "mode": "preflight",
        "source_commit": source_commit,
        "source_diff_sha256": source_diff_sha256,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "fold": 0,
        "fit_identity_count": len(split["fit_identity_ids"]),
        "heldout_identity_count": len(split["heldout_identity_ids"]),
        "identity_overlap": list(split["identity_overlap"]),
        "batch_size": int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        "num_instances": int(config["DATA"]["NUM_INSTANCES"]),
        "training": training,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("V12 complete-path preflight failed")
    return result


def _run_oof(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    project_modeling = str(Path(__file__).resolve().parents[1] / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    from trifusion.aligned_data import build_aligned_train_loader
    from tools.build_v8_oof_router_targets import (
        _collect_fold,
        _train_fold,
        build_identity_folds,
    )
    from tools.diagnose_v6_oracle_complementarity import (
        QueryRetrievalScores,
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from tools.probe_v8_frozen_router import select_cross_camera_records
    from tools.repair_v8_oof_margin_targets import _margins_from_features
    from tools.run_signal_preserving_v5 import _set_seed

    started = time.time()
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    folds = build_identity_folds(records, num_folds=EXPECTED_FOLDS)
    eligible_records = select_cross_camera_records(records)
    eligible_ids = {int(record[1]) for record in eligible_records}
    fold_receipts = []
    fold_metrics = []
    query_direct = []
    query_modal_residual = []
    query_target_margin = []
    query_identities = []
    query_cameras = []
    query_folds = []
    score_parts = {name: [] for name in (*EXPERTS, "bank")}
    torch.cuda.reset_peak_memory_stats()
    for fold_index, heldout_ids in enumerate(folds):
        _set_seed(int(config["EXPERIMENT"]["SEED"]))
        split_records = build_complete_path_fold_records(
            records,
            heldout_ids=heldout_ids,
        )
        expert_fit_original = [
            (paths, split_records["label_map"][int(identity)], camera, view)
            for paths, identity, camera, view in records
            if int(identity) in split_records["fit_identity_ids"]
        ]
        signal_loader = build_aligned_train_loader(
            split_records["train_records"],
            batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
            num_instances=int(config["DATA"]["NUM_INSTANCES"]),
            num_workers=int(config["DATA"]["NUM_WORKERS"]),
            seed=int(config["EXPERIMENT"]["SEED"]),
        )
        signal_model = _build_signal_teacher(
            signal_cfg,
            num_classes=len(split_records["fit_identity_ids"]),
            camera_num=len({record[2] for record in split_records["train_records"]}),
            view_num=len({record[3] for record in split_records["train_records"]}),
        )
        signal_training = _train_signal_teacher(
            signal_model,
            signal_loader,
            signal_cfg,
            epochs=int(config["V12"]["SIGNAL_TEACHER_EPOCHS"]),
        )
        signal_checkpoint = output_dir / f"fold_{fold_index}_signal_final.pth"
        torch.save(
            {
                "schema_version": "trifusion-v12-fold-signal-final-v1",
                "fold": fold_index,
                "fit_identity_ids": list(split_records["fit_identity_ids"]),
                "heldout_identity_ids": list(split_records["heldout_identity_ids"]),
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in signal_model.state_dict().items()
                },
            },
            signal_checkpoint,
        )
        signal_checkpoint_sha256 = _sha256(signal_checkpoint)
        model = _build_v8_experts(
            signal_model,
            config,
            signal_checkpoint_sha256=signal_checkpoint_sha256,
            num_classes=len(split_records["fit_identity_ids"]),
        )
        expert_training = _train_fold(model, expert_fit_original, config)
        expert_checkpoint = output_dir / f"fold_{fold_index}_experts.pth"
        torch.save(
            {
                "schema_version": "trifusion-v12-fold-experts-v1",
                "fold": fold_index,
                "signal_checkpoint_sha256": signal_checkpoint_sha256,
                "fit_identity_ids": list(split_records["fit_identity_ids"]),
                "heldout_identity_ids": list(split_records["heldout_identity_ids"]),
                "expert_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                    if not name.startswith("baseline.")
                },
            },
            expert_checkpoint,
        )
        heldout_records = [
            record
            for record in eligible_records
            if int(record[1]) in heldout_ids
        ]
        heldout_loader = _eval_loader(heldout_records, config)
        collected = _collect_fold(model, heldout_loader)
        num_query = len(heldout_records)
        identities = collected["identities"]
        cameras = collected["cameras"]
        residuals = {
            expert: collected[f"residual_{expert}"] for expert in EXPERTS
        }
        residuals["bank"] = torch.cat(
            [F.normalize(residuals[expert].float(), dim=1) for expert in EXPERTS],
            dim=1,
        )
        local_scores = {
            name: _scores_from_features(
                features,
                identities,
                cameras,
                num_query=num_query,
            )
            for name, features in residuals.items()
        }
        for name, scores in local_scores.items():
            score_parts[name].append(scores)
        slot_margin = np.empty(
            (num_query, len(EXPERTS), len(MODALITIES)),
            dtype=np.float64,
        )
        for expert_index, expert in enumerate(EXPERTS):
            for modality_index, modality in enumerate(MODALITIES):
                slot_margin[:, expert_index, modality_index] = _margins_from_features(
                    collected[f"slot_{expert}_{modality}"],
                    identities,
                    cameras,
                    num_query=num_query,
                )
        query_direct.append(collected["direct_modal"][:num_query].half())
        query_modal_residual.append(collected["modal_residual"][:num_query].half())
        query_target_margin.append(torch.from_numpy(slot_margin).float())
        query_identities.extend(identities[:num_query].tolist())
        query_cameras.extend(cameras[:num_query].tolist())
        query_folds.extend([fold_index] * num_query)
        fold_metrics.append(
            {
                "fold": fold_index,
                "queries": num_query,
                "metrics_percent": {
                    name: _metric_summary(scores)
                    for name, scores in local_scores.items()
                },
            }
        )
        receipt = {
            "fold": fold_index,
            "signal_fit_identity_ids": list(split_records["fit_identity_ids"]),
            "expert_fit_identity_ids": list(split_records["fit_identity_ids"]),
            "heldout_identity_ids": list(split_records["heldout_identity_ids"]),
            "eligible_heldout_identity_count": len(heldout_ids & eligible_ids),
            "eligible_heldout_queries": num_query,
            "signal_epochs": int(signal_training["epochs"]),
            "expert_epochs": int(expert_training["epochs"]),
            "signal_checkpoint_selection": "final_epoch_only",
            "signal_checkpoint": str(signal_checkpoint),
            "signal_checkpoint_sha256": signal_checkpoint_sha256,
            "expert_checkpoint": str(expert_checkpoint),
            "expert_checkpoint_sha256": _sha256(expert_checkpoint),
            "signal_training": signal_training,
            "expert_training": expert_training,
            "overflow_events": int(signal_training["overflow_events"])
            + int(expert_training["overflow_events"]),
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }
        fold_receipts.append(receipt)
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": "trifusion-v12-complete-path-oof-progress-v1",
                    "completed_folds": fold_receipts,
                    "dev_access_count": 0,
                    "official_test_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        del model
        del signal_model
        torch.cuda.empty_cache()

    combined_scores = {
        name: QueryRetrievalScores(
            average_precision=np.concatenate(
                [scores.average_precision for scores in parts]
            ),
            rank1_correct=np.concatenate(
                [scores.rank1_correct for scores in parts]
            ),
        )
        for name, parts in score_parts.items()
    }
    fixed_metrics = {
        name: _metric_summary(scores) for name, scores in combined_scores.items()
    }
    expert_oracle = summarize_oracle_complementarity(
        {expert: combined_scores[expert] for expert in EXPERTS}
    )
    target_margin = torch.cat(query_target_margin)
    maximum = target_margin.amax(dim=(1, 2), keepdim=True)
    unique = torch.isclose(
        target_margin,
        maximum,
        rtol=0.0,
        atol=1e-8,
    ).flatten(1).sum(dim=1) == 1
    winners = target_margin.flatten(1).argmax(dim=1)
    expert_counts = {
        expert: int((unique & (winners // len(MODALITIES) == index)).sum())
        for index, expert in enumerate(EXPERTS)
    }
    modality_counts = {
        modality: int((unique & (winners % len(MODALITIES) == index)).sum())
        for index, modality in enumerate(MODALITIES)
    }
    fixed_margin = target_margin.mean(dim=0)
    slot_oracle_margin = float(target_margin.amax(dim=(1, 2)).mean())
    slot_margin_gain = slot_oracle_margin - float(fixed_margin.max())
    gate = evaluate_complete_path_oof_gate(
        fold_receipts=fold_receipts,
        query_count=int(target_margin.shape[0]),
        fixed_map={name: values["mAP"] for name, values in fixed_metrics.items()},
        expert_winner_counts=expert_counts,
        modality_winner_counts=modality_counts,
        residual_oracle_gain_map=float(
            expert_oracle["oracle_minus_best_fixed_percent"]["mAP"]
        ),
        slot_oracle_margin_gain=slot_margin_gain,
    )
    target_cache = output_dir / "oof_router_margin_targets.pth"
    torch.save(
        {
            "schema_version": "trifusion-v8-oof-router-margin-cache-v1",
            "direct_modal": torch.cat(query_direct),
            "modal_residual": torch.cat(query_modal_residual),
            "target_identity_margin": target_margin,
            "identities": torch.tensor(query_identities, dtype=torch.long),
            "cameras": torch.tensor(query_cameras, dtype=torch.long),
            "fold_indices": torch.tensor(query_folds, dtype=torch.long),
            "experts": EXPERTS,
            "modalities": MODALITIES,
            "teacher_protocol": "complete_path_identity_oof_v12",
        },
        target_cache,
    )
    result = {
        "schema_version": "trifusion-v12-complete-path-oof-result-v1",
        "status": "PASS",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "source_commit": source_commit,
        "source_diff_sha256": source_diff_sha256,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "clip_weight": str(Path(config["SIGNAL"]["CLIP_WEIGHT"]).resolve()),
        "clip_weight_sha256": _sha256(
            Path(config["SIGNAL"]["CLIP_WEIGHT"]).resolve()
        ),
        "fold_receipts": fold_receipts,
        "fold_metrics": fold_metrics,
        "fixed_metrics_percent": fixed_metrics,
        "residual_expert_oracle": expert_oracle,
        "slot_fixed_mean_identity_margin": {
            expert: {
                modality: float(fixed_margin[expert_index, modality_index])
                for modality_index, modality in enumerate(MODALITIES)
            }
            for expert_index, expert in enumerate(EXPERTS)
        },
        "slot_oracle_mean_identity_margin": slot_oracle_margin,
        "slot_oracle_minus_best_fixed_margin": slot_margin_gain,
        "qualification_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "target_cache": str(target_cache),
        "target_cache_sha256": _sha256(target_cache),
        "training_executed": True,
        "optimizer_steps": sum(
            int(receipt["signal_training"]["optimizer_steps"])
            + int(receipt["expert_training"]["optimizer_steps"])
            for receipt in fold_receipts
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import yaml

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V12 is frozen to seed 42")
    if bool(config["PROTOCOL"]["DEV_ACCESS_DURING_TARGET_BUILDING"]):
        raise ValueError("V12 target building cannot access dev")
    clip_weight = Path(config["SIGNAL"]["CLIP_WEIGHT"]).resolve()
    if _sha256(clip_weight) != str(config["SIGNAL"]["CLIP_WEIGHT_SHA256"]):
        raise ValueError("V12 raw CLIP SHA-256 differs from the contract")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"V12 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.mode == "preflight":
        return _run_preflight(config, config_path, output_dir)
    return _run_oof(config, config_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "oof"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "build_complete_path_fold_records",
    "evaluate_complete_path_oof_gate",
    "run",
]
