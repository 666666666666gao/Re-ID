#!/usr/bin/env python3
"""Build deployment-aligned actual-path OOF targets for TriFusion V13."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sha256(values: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sample_key(paths: Any) -> str:
    return hashlib.sha256(
        json.dumps(list(paths), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_paired_cache_payload(
    *,
    sample_keys: tuple[str, ...],
    identities: Any,
    cameras: Any,
    folds: Any,
    teacher_baseline: Any,
    teacher_modal_residual: Any,
    teacher_utility: Any,
    student_direct_modal: Any,
    student_modal_residual: Any,
    phase_a_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the immutable row-aligned teacher/student cache payload."""

    return {
        "schema_version": "trifusion-v13-paired-target-cache-v1",
        "sample_keys": sample_keys,
        "sample_order_sha256": _ordered_sha256(sample_keys),
        "identities": identities,
        "cameras": cameras,
        "fold_indices": folds,
        "teacher_oof_baseline": teacher_baseline,
        "teacher_oof_modal_residual": teacher_modal_residual,
        "teacher_identity_utility": teacher_utility,
        "student_direct_modal": student_direct_modal,
        "student_modal_residual": student_modal_residual,
        "phase_a_checkpoint_sha256": phase_a_checkpoint_sha256,
        "experts": EXPERTS,
        "modalities": MODALITIES,
        "fixed_alpha": 0.2,
    }


def evaluate_v13_q0_gate(
    *,
    expert_unique_positive_wins: dict[str, int],
    modality_unique_positive_wins: dict[str, int],
    oracle_mean_utility: float,
    best_fixed_mean_utility: float,
    transfer_per_fold_noninferior: tuple[bool, bool, bool],
    transfer_aggregate_gain: float,
    reference_bank_immutable: bool,
    dev_access_count: int,
    official_test_access_count: int,
) -> dict[str, Any]:
    expert_diversity = all(expert_unique_positive_wins[name] > 0 for name in EXPERTS)
    modality_diversity = all(
        modality_unique_positive_wins[name] > 0 for name in MODALITIES
    )
    oracle_gain = float(oracle_mean_utility) > float(best_fixed_mean_utility)
    action_transfer = (
        all(transfer_per_fold_noninferior) and float(transfer_aggregate_gain) > 0.0
    )
    access_clean = int(dev_access_count) == 0 and int(official_test_access_count) == 0
    passed = (
        expert_diversity
        and modality_diversity
        and oracle_gain
        and action_transfer
        and bool(reference_bank_immutable)
        and access_clean
    )
    return {
        "passed": passed,
        "expert_diversity_passed": expert_diversity,
        "modality_diversity_passed": modality_diversity,
        "oracle_gain_passed": oracle_gain,
        "action_transfer_passed": action_transfer,
        "reference_bank_immutable_passed": bool(reference_bank_immutable),
        "access_boundary_passed": access_clean,
        "expert_unique_positive_wins": dict(expert_unique_positive_wins),
        "modality_unique_positive_wins": dict(modality_unique_positive_wins),
        "oracle_mean_utility": float(oracle_mean_utility),
        "best_fixed_mean_utility": float(best_fixed_mean_utility),
        "transfer_aggregate_gain": float(transfer_aggregate_gain),
    }


def _collect_features(model: Any, records: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from build_v12_complete_path_oof_targets import _eval_loader

    values = {"baseline": [], "direct": [], "residual": []}
    identities = []
    cameras = []
    model.eval()
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in _eval_loader(
        records,
        config,
    ):
        images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(
                {
                    "images": images,
                    "modality_mask": torch.ones(
                        camera_labels.shape[0],
                        3,
                        dtype=torch.bool,
                        device="cuda",
                    ),
                    "camera_ids": camera_labels,
                },
                return_aux=True,
            )
        residual = torch.stack(
            [output.modal_residual_embeddings[name] for name in EXPERTS],
            dim=1,
        )
        values["baseline"].append(output.baseline_embedding.float().cpu())
        values["direct"].append(output.direct_modal.float().cpu())
        values["residual"].append(residual.float().cpu())
        identities.extend(int(value) for value in batch_ids)
        cameras.extend(int(value) for value in batch_cameras)
    count = len(records)
    result = {
        name: torch.cat(parts)[:count] for name, parts in values.items()
    }
    result["identities"] = torch.tensor(identities[:count], dtype=torch.long)
    result["cameras"] = torch.tensor(cameras[:count], dtype=torch.long)
    expected_identities = torch.tensor([int(record[1]) for record in records])
    expected_cameras = torch.tensor([int(record[2]) for record in records])
    if not torch.equal(result["identities"], expected_identities):
        raise RuntimeError("feature identities do not match ordered records")
    if not torch.equal(result["cameras"], expected_cameras):
        raise RuntimeError("feature cameras do not match ordered records")
    return result


def _load_fold_teacher(
    *,
    fold: int,
    heldout_ids: set[int],
    records: list[Any],
    config: dict[str, Any],
    v12_dir: Path,
) -> tuple[Any, dict[str, str]]:
    import torch
    from config import cfg as signal_cfg

    from build_v12_complete_path_oof_targets import (
        _build_signal_teacher,
        _build_v8_experts,
        build_complete_path_fold_records,
    )

    split = build_complete_path_fold_records(records, heldout_ids=heldout_ids)
    signal_path = v12_dir / f"fold_{fold}_signal_final.pth"
    expert_path = v12_dir / f"fold_{fold}_experts.pth"
    expected = config["V13"]["FOLD_CHECKPOINT_SHA256"][fold]
    signal_sha = _sha256(signal_path)
    expert_sha = _sha256(expert_path)
    if signal_sha != expected["SIGNAL"] or expert_sha != expected["EXPERT"]:
        raise ValueError(f"V13 fold {fold} checkpoint SHA-256 differs from contract")
    signal_payload = torch.load(signal_path, map_location="cpu", weights_only=True)
    signal_model = _build_signal_teacher(
        signal_cfg,
        num_classes=len(split["fit_identity_ids"]),
        camera_num=len({int(record[2]) for record in split["train_records"]}),
        view_num=len({int(record[3]) for record in split["train_records"]}),
    )
    signal_model.load_state_dict(signal_payload["model_state_dict"], strict=True)
    model = _build_v8_experts(
        signal_model,
        config,
        signal_checkpoint_sha256=signal_sha,
        num_classes=len(split["fit_identity_ids"]),
    )
    expert_payload = torch.load(expert_path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    state.update(expert_payload["expert_state_dict"])
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"signal": signal_sha, "expert": expert_sha}


def _utility(features: dict[str, Any]) -> Any:
    import torch
    from trifusion.signal_preserving_v13 import query_side_counterfactual_utilities

    result = query_side_counterfactual_utilities(
        features["baseline"].cuda(),
        features["residual"].cuda(),
        features["identities"].cuda(),
        features["cameras"].cuda(),
    )
    return {
        "utilities": result.utilities.float().cpu(),
        "full_margins": result.full_margins.float().cpu(),
        "reference_sha_before": result.reference_embedding_sha256_before,
        "reference_sha_after": result.reference_embedding_sha256_after,
    }


def _unique_positive_counts(utilities: Any) -> tuple[dict[str, int], dict[str, int]]:
    import torch

    maximum = utilities.amax(dim=(1, 2), keepdim=True)
    unique = torch.isclose(utilities, maximum, rtol=0.0, atol=1e-8).flatten(1).sum(1) == 1
    positive = maximum.flatten() > 0.0
    winners = utilities.flatten(1).argmax(1)
    valid = unique & positive
    expert = {
        name: int((valid & (winners // len(MODALITIES) == index)).sum())
        for index, name in enumerate(EXPERTS)
    }
    modality = {
        name: int((valid & (winners % len(MODALITIES) == index)).sum())
        for index, name in enumerate(MODALITIES)
    }
    return expert, modality


def _fold_health(utilities: Any, *, temperature: float) -> dict[str, Any]:
    import torch

    probability = torch.softmax(utilities.flatten(1) / float(temperature), dim=1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
    entropy = entropy / np.log(EXPERTS.__len__() * MODALITIES.__len__())
    return {
        "mean": float(utilities.mean()),
        "std": float(utilities.std()),
        "min": float(utilities.min()),
        "max": float(utilities.max()),
        "positive_ratio": float((utilities > 0.0).float().mean()),
        "normalized_target_entropy": float(entropy.mean()),
        "slot_mean": utilities.mean(0).tolist(),
    }


def _action_transfer(
    teacher_utility: Any,
    deployment_utility: Any,
    folds: Any,
) -> dict[str, Any]:
    import torch

    fold_receipts = []
    differences = []
    for fold in range(3):
        train_rows = folds != fold
        heldout_rows = folds == fold
        fixed_slot = int(teacher_utility[train_rows].mean(0).flatten().argmax())
        oracle_slot = teacher_utility[heldout_rows].flatten(1).argmax(1)
        deployment = deployment_utility[heldout_rows].flatten(1)
        oracle_value = deployment.gather(1, oracle_slot[:, None]).squeeze(1)
        fixed_value = deployment[:, fixed_slot]
        difference = oracle_value - fixed_value
        differences.append(difference)
        fold_receipts.append(
            {
                "fold": fold,
                "fixed_slot": fixed_slot,
                "oracle_mean": float(oracle_value.mean()),
                "fixed_mean": float(fixed_value.mean()),
                "gain": float(difference.mean()),
                "noninferior": float(difference.mean()) >= 0.0,
            }
        )
    flat_teacher = teacher_utility.flatten().numpy()
    flat_deployment = deployment_utility.flatten().numpy()
    teacher_rank = np.argsort(np.argsort(flat_teacher))
    deployment_rank = np.argsort(np.argsort(flat_deployment))
    combined = torch.cat(differences)
    return {
        "folds": fold_receipts,
        "aggregate_gain": float(combined.mean()),
        "per_fold_noninferior": tuple(row["noninferior"] for row in fold_receipts),
        "sign_agreement": float(
            ((teacher_utility > 0.0) == (deployment_utility > 0.0)).float().mean()
        ),
        "top_slot_agreement": float(
            (
                teacher_utility.flatten(1).argmax(1)
                == deployment_utility.flatten(1).argmax(1)
            ).float().mean()
        ),
        "spearman_rank_correlation": float(
            np.corrcoef(teacher_rank, deployment_rank)[0, 1]
        ),
    }


def _run_preflight(
    config: dict[str, Any],
    output_dir: Path,
    allfit_model: Any,
    records: list[Any],
    folds: tuple[set[int], ...],
) -> dict[str, Any]:
    import torch

    from probe_v8_frozen_router import select_cross_camera_records

    eligible = select_cross_camera_records(records)
    heldout = [record for record in eligible if int(record[1]) in folds[0]][:8]
    teacher, checkpoint_sha = _load_fold_teacher(
        fold=0,
        heldout_ids=folds[0],
        records=records,
        config=config,
        v12_dir=Path(config["V13"]["V12_OOF_DIR"]).resolve(),
    )
    teacher_features = _collect_features(teacher, heldout, config)
    student_features = _collect_features(allfit_model, heldout, config)
    shape_passed = (
        teacher_features["baseline"].shape == (8, 3072)
        and teacher_features["residual"].shape == (8, 3, 3, 512)
        and student_features["direct"].shape == (8, 3, 512)
        and student_features["residual"].shape == (8, 3, 3, 512)
    )
    result = {
        "schema_version": "trifusion-v13-deployment-aligned-preflight-v1",
        "status": "PASS" if shape_passed else "FAIL",
        "shape_passed": shape_passed,
        "fold": 0,
        "samples": len(heldout),
        "fold_checkpoint_sha256": checkpoint_sha,
        "phase_a_checkpoint_sha256": config["INITIALIZATION"]["PHASE_A_CHECKPOINT_SHA256"],
        "training_executed": False,
        "optimizer_steps": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _run_q0(
    config: dict[str, Any],
    output_dir: Path,
    allfit_model: Any,
    records: list[Any],
    folds: tuple[set[int], ...],
) -> dict[str, Any]:
    import torch

    from probe_v8_frozen_router import select_cross_camera_records

    eligible = select_cross_camera_records(records)
    teacher_baseline = []
    teacher_residual = []
    teacher_utility = []
    student_direct = []
    student_residual = []
    deployment_utility = []
    identities = []
    cameras = []
    fold_indices = []
    sample_keys = []
    fold_receipts = []
    immutable = True
    v12_dir = Path(config["V13"]["V12_OOF_DIR"]).resolve()
    for fold, heldout_ids in enumerate(folds):
        heldout = [record for record in eligible if int(record[1]) in heldout_ids]
        teacher, checkpoint_sha = _load_fold_teacher(
            fold=fold,
            heldout_ids=heldout_ids,
            records=records,
            config=config,
            v12_dir=v12_dir,
        )
        teacher_features = _collect_features(teacher, heldout, config)
        student_features = _collect_features(allfit_model, heldout, config)
        teacher_result = _utility(teacher_features)
        deployment_result = _utility(student_features)
        immutable = immutable and (
            teacher_result["reference_sha_before"]
            == teacher_result["reference_sha_after"]
            and deployment_result["reference_sha_before"]
            == deployment_result["reference_sha_after"]
        )
        keys = tuple(_sample_key(record[0]) for record in heldout)
        teacher_baseline.append(teacher_features["baseline"].half())
        teacher_residual.append(teacher_features["residual"].half())
        teacher_utility.append(teacher_result["utilities"])
        student_direct.append(student_features["direct"].half())
        student_residual.append(student_features["residual"].half())
        deployment_utility.append(deployment_result["utilities"])
        identities.append(teacher_features["identities"])
        cameras.append(teacher_features["cameras"])
        fold_indices.append(torch.full((len(heldout),), fold, dtype=torch.long))
        sample_keys.extend(keys)
        fold_receipts.append(
            {
                "fold": fold,
                "queries": len(heldout),
                "checkpoint_sha256": checkpoint_sha,
                "target_health": _fold_health(
                    teacher_result["utilities"],
                    temperature=float(config["ROUTER"]["UTILITY_TEMPERATURE"]),
                ),
                "reference_embedding_sha256": teacher_result[
                    "reference_sha_before"
                ],
            }
        )
        del teacher
        torch.cuda.empty_cache()
    teacher_utility_tensor = torch.cat(teacher_utility)
    deployment_utility_tensor = torch.cat(deployment_utility)
    identity_tensor = torch.cat(identities)
    camera_tensor = torch.cat(cameras)
    fold_tensor = torch.cat(fold_indices)
    if teacher_utility_tensor.shape[0] != int(config["V13"]["EXPECTED_QUERIES"]):
        raise RuntimeError("V13 paired target query count differs from contract")
    expert_wins, modality_wins = _unique_positive_counts(teacher_utility_tensor)
    transfer = _action_transfer(
        teacher_utility_tensor,
        deployment_utility_tensor,
        fold_tensor,
    )
    oracle_mean = float(teacher_utility_tensor.amax(dim=(1, 2)).mean())
    best_fixed = float(teacher_utility_tensor.mean(0).max())
    gate = evaluate_v13_q0_gate(
        expert_unique_positive_wins=expert_wins,
        modality_unique_positive_wins=modality_wins,
        oracle_mean_utility=oracle_mean,
        best_fixed_mean_utility=best_fixed,
        transfer_per_fold_noninferior=transfer["per_fold_noninferior"],
        transfer_aggregate_gain=float(transfer["aggregate_gain"]),
        reference_bank_immutable=immutable,
        dev_access_count=0,
        official_test_access_count=0,
    )
    cache = build_paired_cache_payload(
        sample_keys=tuple(sample_keys),
        identities=identity_tensor,
        cameras=camera_tensor,
        folds=fold_tensor,
        teacher_baseline=torch.cat(teacher_baseline),
        teacher_modal_residual=torch.cat(teacher_residual),
        teacher_utility=teacher_utility_tensor,
        student_direct_modal=torch.cat(student_direct),
        student_modal_residual=torch.cat(student_residual),
        phase_a_checkpoint_sha256=config["INITIALIZATION"][
            "PHASE_A_CHECKPOINT_SHA256"
        ],
    )
    cache["teacher_fold_checkpoint_sha256"] = tuple(
        receipt["checkpoint_sha256"] for receipt in fold_receipts
    )
    cache_path = output_dir / "paired_target_cache.pth"
    torch.save(cache, cache_path)
    result = {
        "schema_version": "trifusion-v13-deployment-aligned-q0-result-v1",
        "status": "PASS",
        "seed": int(config["EXPERIMENT"]["SEED"]),
        "fold_receipts": fold_receipts,
        "query_count": int(teacher_utility_tensor.shape[0]),
        "target_health": {
            "expert_unique_positive_wins": expert_wins,
            "modality_unique_positive_wins": modality_wins,
            "oracle_mean_utility": oracle_mean,
            "best_fixed_mean_utility": best_fixed,
            "oracle_minus_fixed": oracle_mean - best_fixed,
        },
        "action_transfer_audit": transfer,
        "gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "paired_target_cache": str(cache_path),
        "paired_target_cache_sha256": _sha256(cache_path),
        "training_executed": False,
        "optimizer_steps": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from build_v8_oof_router_targets import build_identity_folds
    from run_signal_preserving_v5 import (
        _build_runtime,
        _module_state_sha256,
        load_raw_config,
    )

    started = time.time()
    config_path = args.config.resolve()
    config = load_raw_config(config_path)
    if int(config["EXPERIMENT"]["SEED"]) != 42:
        raise ValueError("V13 target building is frozen to seed 42")
    if args.output_dir.exists():
        raise FileExistsError(f"V13 output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    phase_a_path = Path(config["INITIALIZATION"]["PHASE_A_CHECKPOINT"]).resolve()
    if _sha256(phase_a_path) != config["INITIALIZATION"]["PHASE_A_CHECKPOINT_SHA256"]:
        raise ValueError("V13 Phase-A checkpoint SHA-256 differs from contract")
    runtime = _build_runtime(config)
    allfit_model = runtime["model"]
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=True)
    allfit_model.load_state_dict(phase_a["model_state_dict"], strict=True)
    allfit_model.cuda().eval()
    for parameter in allfit_model.parameters():
        parameter.requires_grad_(False)
    allfit_state_before = _module_state_sha256(allfit_model)
    records = list(runtime["train_records"])
    folds = build_identity_folds(records, num_folds=3)
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "preflight":
        result = _run_preflight(config, args.output_dir, allfit_model, records, folds)
    else:
        result = _run_q0(config, args.output_dir, allfit_model, records, folds)
    allfit_state_after = _module_state_sha256(allfit_model)
    if allfit_state_before != allfit_state_after:
        raise RuntimeError("V13 target building changed the Phase-A model")
    result.update(
        {
            "mode": args.mode,
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary"])
            ).hexdigest(),
            "phase_a_state_sha256_before": allfit_state_before,
            "phase_a_state_sha256_after": allfit_state_after,
            "elapsed_seconds": time.time() - started,
        }
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "q0"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "build_paired_cache_payload",
    "evaluate_v13_q0_gate",
    "run",
]
