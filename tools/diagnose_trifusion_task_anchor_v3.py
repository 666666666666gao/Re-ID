#!/usr/bin/env python3
"""Read-only representation diagnostics for a frozen task-anchored V3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


EXPERTS = ("cnn", "transformer", "mamba")
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


@dataclass(frozen=True, eq=False)
class DiagnosticViews:
    """Retrieval views and scale statistics reconstructed from one model output."""

    views: Mapping[str, torch.Tensor]
    routing_weights: torch.Tensor
    expert_to_anchor_norm_ratio: torch.Tensor
    routed_to_anchor_norm_ratio: torch.Tensor
    normalized_routing_entropy: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))


def derive_diagnostic_views(
    *,
    anchor_embedding: torch.Tensor,
    contribution_embeddings: torch.Tensor,
    reliability: torch.Tensor,
    uncertainty: torch.Tensor,
    modality_mask: torch.Tensor,
) -> DiagnosticViews:
    """Reconstruct anchor, expert, routed-residual, and fused retrieval views."""

    if contribution_embeddings.ndim != 4:
        raise ValueError("contribution_embeddings must have shape B,E,M,D")
    batch, experts, modalities, width = contribution_embeddings.shape
    if experts != len(EXPERTS) or modalities != 3:
        raise ValueError("diagnostic expects exactly three experts and modalities")
    if anchor_embedding.shape != (batch, modalities * width):
        raise ValueError("anchor_embedding has the wrong flattened shape")
    if reliability.shape != (batch, experts, modalities):
        raise ValueError("reliability has the wrong shape")
    if uncertainty.shape != reliability.shape:
        raise ValueError("uncertainty and reliability shapes differ")
    if modality_mask.shape != (batch, modalities) or modality_mask.dtype != torch.bool:
        raise ValueError("modality_mask must be a Bx3 boolean tensor")

    valid = modality_mask[..., None].to(anchor_embedding.dtype)
    anchor_modal = anchor_embedding.reshape(batch, modalities, width) * valid
    contributions = contribution_embeddings * valid[:, None]
    confidence = reliability * (1.0 - uncertainty).clamp_min(0.05)
    confidence = confidence * modality_mask[:, None].to(confidence.dtype)
    routing_weights = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
    routed = (contributions * routing_weights[..., None]).sum(dim=1)

    anchor_flat = anchor_modal.flatten(1)
    routed_flat = routed.flatten(1)
    views: dict[str, torch.Tensor] = {
        "anchor": anchor_flat,
        "fused": torch.cat((anchor_flat, routed_flat), dim=1),
    }
    for index, expert in enumerate(EXPERTS):
        views[expert] = torch.cat((anchor_flat, contributions[:, index].flatten(1)), dim=1)
    views["routed_residual"] = routed_flat

    anchor_norm = anchor_modal.norm(dim=-1).clamp_min(1e-12)
    expert_ratio = contributions.norm(dim=-1) / anchor_norm[:, None]
    routed_ratio = routed.norm(dim=-1) / anchor_norm
    expert_ratio = expert_ratio * modality_mask[:, None].to(expert_ratio.dtype)
    routed_ratio = routed_ratio * modality_mask.to(routed_ratio.dtype)
    entropy = -(
        routing_weights
        * routing_weights.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(experts)
    entropy = entropy * modality_mask.to(entropy.dtype)

    return DiagnosticViews(
        views=views,
        routing_weights=routing_weights,
        expert_to_anchor_norm_ratio=expert_ratio,
        routed_to_anchor_norm_ratio=routed_ratio,
        normalized_routing_entropy=entropy,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_frozen_dev_artifacts(
    *,
    output_dir: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Fail closed unless the requested checkpoint is a completed dev selection."""

    best_path = output_dir / "best_dev_receipt.json"
    worker_path = output_dir / "dev_worker_result.json"
    if not best_path.is_file() or not worker_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("completed dev receipt, worker result, or config is missing")
    if (output_dir / "official_test_metrics.json").exists() or (
        output_dir / "official_test_access_guard.json"
    ).exists():
        raise RuntimeError("diagnostic input unexpectedly contains official-test artifacts")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(best.get("checkpoint", ""))).expanduser().resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError as error:
        raise RuntimeError("best checkpoint is outside the completed run directory") from error
    required_best = {
        "phase": "complete",
        "contract_testing": False,
        "scientific_evidence_eligible": True,
        "official_test_access_count": 0,
        "dev_evaluation_count": 60,
        "selection_output": "fused",
    }
    required_worker = {
        "status": "COMPLETE",
        "phase": "complete",
        "epoch": 60,
        "dev_evaluation_count": 60,
        "official_test_access_count": 0,
        "fatal_or_nonfinite_detected": False,
        "scientific_evidence_eligible": True,
    }
    best_mismatches = {
        key: {"expected": expected, "actual": best.get(key)}
        for key, expected in required_best.items()
        if best.get(key) != expected
    }
    worker_mismatches = {
        key: {"expected": expected, "actual": worker.get(key)}
        for key, expected in required_worker.items()
        if worker.get(key) != expected
    }
    if best_mismatches or worker_mismatches:
        raise RuntimeError(
            f"dev completion evidence mismatch: best={best_mismatches}, worker={worker_mismatches}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    if (
        best.get("checkpoint_sha256") != checkpoint_hash
        or worker.get("best_checkpoint_sha256") != checkpoint_hash
        or best.get("config_sha256") != _sha256(config_path)
        or int(best.get("epoch", -1)) != int(worker.get("best_epoch", -2))
        or best.get("metrics_percent") != worker.get("metrics_percent")
    ):
        raise RuntimeError("best checkpoint, config, epoch, or metrics binding is invalid")
    return best, worker, checkpoint


def _reid_metrics(
    features: torch.Tensor,
    *,
    num_query: int,
    identities: np.ndarray,
    cameras: np.ndarray,
) -> dict[str, float]:
    from utils.reid_evaluation import evaluate_reid

    normalized = F.normalize(features.float(), dim=1)
    distances = torch.cdist(
        normalized[:num_query], normalized[num_query:], p=2
    ).numpy()
    cmc, mean_ap = evaluate_reid(
        distances,
        identities[:num_query],
        identities[num_query:],
        cameras[:num_query],
        cameras[num_query:],
        max_rank=50,
    )
    return {
        "mAP": float(mean_ap * 100.0),
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[4] * 100.0),
        "Rank-10": float(cmc[9] * 100.0),
    }


def _nested_means(values: torch.Tensor, names: tuple[str, ...]) -> dict[str, Any]:
    if values.ndim == 2:
        return {
            names[index]: float(values[:, index].mean())
            for index in range(values.shape[1])
        }
    if values.ndim == 3:
        return {
            EXPERTS[expert]: {
                names[modality]: float(values[:, expert, modality].mean())
                for modality in range(values.shape[2])
            }
            for expert in range(values.shape[1])
        }
    raise ValueError("mean table expects a two- or three-dimensional tensor")


def run_diagnostic(
    *,
    config_path: Path,
    output_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Evaluate only the frozen held-out dev split and emit a bound JSON receipt."""

    from modeling.trifusion.data import build_rgbnt201_dev_loaders
    from modeling.trifusion.task_anchor_v3_builder import (
        build_task_anchored_trifusion_v3_from_clip,
    )

    best, worker, checkpoint = validate_frozen_dev_artifacts(
        output_dir=output_dir,
        config_path=config_path,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain a mapping")
    if config.get("MODEL", {}).get("ARCHITECTURE") != "task_anchored_collaborative_v3":
        raise ValueError("diagnostic accepts only task-anchored V3")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen diagnostic requires the remote CUDA environment")

    seed = int(config["EXPERIMENT"]["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = build_rgbnt201_dev_loaders(
        dataset_root=Path(config["DATA"]["DATASET_ROOT"]),
        protocol_path=PROJECT / config["DATA"]["DEV_PROTOCOL"],
        train_batch_size=int(config["DATA"]["TRAIN_BATCH_SIZE"]),
        num_instances=int(config["DATA"]["NUM_INSTANCES"]),
        eval_batch_size=int(config["DATA"]["EVAL_BATCH_SIZE"]),
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
    )
    built = build_task_anchored_trifusion_v3_from_clip(
        config["MODEL"]["CLIP_CHECKPOINT"],
        num_classes=data.num_classes,
        image_size=tuple(config["MODEL"]["IMAGE_SIZE"]),
        patch_size=int(config["MODEL"]["PATCH_SIZE"]),
        cnn_width=int(config["MODEL"]["CNN_WIDTH"]),
        mamba_width=int(config["MODEL"]["MAMBA_WIDTH"]),
        relay_rank=int(config["MODEL"]["RELAY_RANK"]),
        embedding_width=int(config["MODEL"]["EMBEDDING_WIDTH"]),
        private_width=int(config["MODEL"]["PRIVATE_WIDTH"]),
        reliability_mode="joint_beta",
        architecture=str(config["MODEL"]["ARCHITECTURE"]),
        adapter_width=int(config["MODEL"]["ADAPTER_WIDTH"]),
        gradient_checkpointing=bool(config["MODEL"]["GRADIENT_CHECKPOINTING"]),
        residual_scale_init=float(config["MODEL"]["RESIDUAL_SCALE_INIT"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    built.model.load_state_dict(state, strict=True)
    model = built.model.cuda().eval()
    amp_enabled = bool(config["OPTIMIZATION"]["AMP"])

    feature_chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("anchor", "fused", *EXPERTS, "routed_residual")
    }
    routing_chunks: list[torch.Tensor] = []
    expert_ratio_chunks: list[torch.Tensor] = []
    routed_ratio_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    reliability_chunks: list[torch.Tensor] = []
    uncertainty_chunks: list[torch.Tensor] = []
    residual_cosines: dict[str, list[torch.Tensor]] = {
        "cnn__transformer": [],
        "cnn__mamba": [],
        "transformer__mamba": [],
    }
    fused_branch_cosines: dict[str, list[torch.Tensor]] = {
        expert: [] for expert in EXPERTS
    }
    identities: list[int] = []
    cameras: list[int] = []
    with torch.no_grad():
        for images, pids, camids, _camids_batch, _viewids, _paths in data.eval_loader:
            images = {name: value.cuda(non_blocking=False) for name, value in images.items()}
            mask = torch.ones(len(pids), 3, dtype=torch.bool, device="cuda")
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(
                    {"images": images, "modality_mask": mask},
                    return_aux=True,
                )
            derived = derive_diagnostic_views(
                anchor_embedding=output.anchor_embedding,
                contribution_embeddings=output.contribution_embeddings,
                reliability=output.reliability.r,
                uncertainty=output.reliability.u,
                modality_mask=output.modality_mask,
            )
            for name, value in derived.views.items():
                feature_chunks[name].append(value.detach().float().cpu())
            routing_chunks.append(derived.routing_weights.detach().float().cpu())
            expert_ratio_chunks.append(
                derived.expert_to_anchor_norm_ratio.detach().float().cpu()
            )
            routed_ratio_chunks.append(
                derived.routed_to_anchor_norm_ratio.detach().float().cpu()
            )
            entropy_chunks.append(
                derived.normalized_routing_entropy.detach().float().cpu()
            )
            reliability_chunks.append(output.reliability.r.detach().float().cpu())
            uncertainty_chunks.append(output.reliability.u.detach().float().cpu())
            residuals = output.contribution_embeddings.detach().float().flatten(2)
            for left, right in ((0, 1), (0, 2), (1, 2)):
                key = f"{EXPERTS[left]}__{EXPERTS[right]}"
                residual_cosines[key].append(
                    F.cosine_similarity(residuals[:, left], residuals[:, right], dim=1).cpu()
                )
            fused = derived.views["fused"].detach().float()
            for expert in EXPERTS:
                fused_branch_cosines[expert].append(
                    F.cosine_similarity(fused, derived.views[expert].detach().float(), dim=1).cpu()
                )
            identities.extend(int(pid) for pid in pids)
            cameras.extend(int(camid) for camid in camids.tolist())

    all_features = {name: torch.cat(chunks) for name, chunks in feature_chunks.items()}
    identity_array = np.asarray(identities)
    camera_array = np.asarray(cameras)
    metrics = {
        name: _reid_metrics(
            values,
            num_query=data.num_query,
            identities=identity_array,
            cameras=camera_array,
        )
        for name, values in all_features.items()
    }
    expected_metrics = dict(best["metrics_percent"])
    parity_deltas = {
        name: {
            metric: metrics[name][metric] - float(expected_metrics[name][metric])
            for metric in expected_metrics[name]
        }
        for name in ("fused", *EXPERTS)
    }
    max_parity_delta = max(
        abs(value)
        for output_deltas in parity_deltas.values()
        for value in output_deltas.values()
    )
    if max_parity_delta > 1e-9:
        raise RuntimeError(f"frozen metric parity failed: max delta {max_parity_delta}")

    routing = torch.cat(routing_chunks)
    expert_ratios = torch.cat(expert_ratio_chunks)
    routed_ratios = torch.cat(routed_ratio_chunks)
    entropy = torch.cat(entropy_chunks)
    reliability = torch.cat(reliability_chunks)
    uncertainty = torch.cat(uncertainty_chunks)
    modality_names = ("RGB", "NI", "TI")
    payload: dict[str, Any] = {
        "schema_version": "trifusion-task-anchor-v3-diagnostic-v1",
        "status": "PASS",
        "mode": "frozen_best_heldout_dev_read_only",
        "training_executed": False,
        "optimizer_steps": 0,
        "official_test_access_count": 0,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_commit": os.popen(
            f"git -C {PROJECT} rev-parse HEAD"
        ).read().strip(),
        "query_records": data.num_query,
        "gallery_records": len(data.eval_loader.dataset) - data.num_query,
        "metrics_percent": metrics,
        "registered_metric_parity": {
            "passed": True,
            "max_absolute_delta": max_parity_delta,
            "deltas": parity_deltas,
        },
        "feature_dimensions": {
            name: int(values.shape[1]) for name, values in all_features.items()
        },
        "learned_residual_scales": {
            expert: float(value)
            for expert, value in zip(
                EXPERTS,
                torch.sigmoid(model.fusion.residual_scale_logits.detach().float()).cpu(),
                strict=True,
            )
        },
        "mean_routing_weight": _nested_means(routing, modality_names),
        "mean_reliability": _nested_means(reliability, modality_names),
        "mean_uncertainty": _nested_means(uncertainty, modality_names),
        "mean_expert_to_anchor_norm_ratio": _nested_means(
            expert_ratios, modality_names
        ),
        "mean_routed_to_anchor_norm_ratio": _nested_means(
            routed_ratios, modality_names
        ),
        "mean_normalized_routing_entropy": {
            modality_names[index]: float(entropy[:, index].mean())
            for index in range(entropy.shape[1])
        },
        "mean_residual_pair_cosine": {
            key: float(torch.cat(values).mean())
            for key, values in residual_cosines.items()
        },
        "mean_fused_branch_cosine": {
            key: float(torch.cat(values).mean())
            for key, values in fused_branch_cosines.items()
        },
        "input_receipts": {
            "best_dev_receipt_sha256": _sha256(output_dir / "best_dev_receipt.json"),
            "dev_worker_result_sha256": _sha256(output_dir / "dev_worker_result.json"),
            "worker_best_checkpoint_sha256": worker["best_checkpoint_sha256"],
        },
    }
    _atomic_json(result_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = (
        args.result_json.expanduser().resolve()
        if args.result_json is not None
        else output_dir / "representation_diagnostic_v3.json"
    )
    payload = run_diagnostic(
        config_path=config_path,
        output_dir=output_dir,
        result_path=result_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagnosticViews",
    "derive_diagnostic_views",
    "run_diagnostic",
    "validate_frozen_dev_artifacts",
]
