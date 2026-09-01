#!/usr/bin/env python3
"""Measure whether cascade V2 actually preserves and diversifies CLIP anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from modeling.trifusion.cascade_v2_builder import (  # noqa: E402
    build_trifusion_cascade_v2_from_clip,
)
from modeling.trifusion.data import build_rgbnt201_dev_loaders  # noqa: E402
from modeling.trifusion.state import EXPERT_ORDER, MODALITY_ORDER  # noqa: E402
from utils.reid_evaluation import evaluate_reid  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project / path
    return path.resolve()


def _metric(
    chunks: list[torch.Tensor],
    identities: list[int],
    cameras: list[int],
    *,
    num_query: int,
) -> dict[str, float]:
    feature = F.normalize(torch.cat(chunks).float(), dim=1)
    distances = torch.cdist(
        feature[:num_query], feature[num_query:], p=2
    ).numpy()
    pid_array = np.asarray(identities)
    camera_array = np.asarray(cameras)
    cmc, mean_ap = evaluate_reid(
        distances,
        pid_array[:num_query],
        pid_array[num_query:],
        camera_array[:num_query],
        camera_array[num_query:],
        max_rank=50,
    )
    return {
        "mAP": float(mean_ap * 100.0),
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[4] * 100.0),
        "Rank-10": float(cmc[9] * 100.0),
    }


def _build_kwargs(config: dict[str, Any], num_classes: int) -> dict[str, Any]:
    model = config["MODEL"]
    return {
        "num_classes": num_classes,
        "image_size": tuple(model["IMAGE_SIZE"]),
        "patch_size": int(model["PATCH_SIZE"]),
        "cnn_width": int(model["CNN_WIDTH"]),
        "mamba_width": int(model["MAMBA_WIDTH"]),
        "relay_rank": int(model["RELAY_RANK"]),
        "embedding_width": int(model["EMBEDDING_WIDTH"]),
        "private_width": int(model["PRIVATE_WIDTH"]),
        "reliability_mode": "uniform",
        "architecture": "shared_semantic_cascade_v2",
        "adapter_width": int(model["ADAPTER_WIDTH"]),
        "gradient_checkpointing": False,
    }


def _exact_anchor(
    model: torch.nn.Module,
    images: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stacked = torch.stack([images[name] for name in MODALITY_ORDER], dim=1)
    packed = stacked[mask]
    modality_ids = torch.arange(3, device=mask.device).view(1, -1).expand(
        mask.shape[0], -1
    )
    tokens = model.encoder.tokenizer(packed, modality_ids[mask])["cnn"]
    # APSD centers its local residuals, so this mean is exactly the tokenizer CLS.
    cls = tokens.mean(dim=1).reshape(mask.shape[0], 3, -1)
    projected = model.fusion.semantic_projection(cls)
    combined = torch.cat(
        (projected.flatten(1), projected.mean(dim=1)), dim=1
    )
    return cls, projected, combined


def _loader(config: dict[str, Any]):
    data = config["DATA"]
    return build_rgbnt201_dev_loaders(
        dataset_root=_resolve(PROJECT, str(data["DATASET_ROOT"])),
        protocol_path=_resolve(PROJECT, str(data["DEV_PROTOCOL"])),
        train_batch_size=int(data["TRAIN_BATCH_SIZE"]),
        num_instances=int(data["NUM_INSTANCES"]),
        eval_batch_size=int(data["EVAL_BATCH_SIZE"]),
        num_workers=int(data["NUM_WORKERS"]),
    )


def _collect_initial_clip(
    model: torch.nn.Module,
    data: Any,
    *,
    amp: bool,
) -> tuple[dict[str, dict[str, float]], list[int], list[int]]:
    features = {f"clip_{name}": [] for name in MODALITY_ORDER}
    features.update({"clip_concat": [], "clip_mean": []})
    identities: list[int] = []
    cameras: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, pids, camids, *_ in data.eval_loader:
            images = {
                name: tensor.cuda(non_blocking=False)
                for name, tensor in images.items()
            }
            mask = torch.ones(len(pids), 3, dtype=torch.bool, device="cuda")
            with torch.cuda.amp.autocast(enabled=amp):
                _cls, projected, combined = _exact_anchor(model, images, mask)
            for index, name in enumerate(MODALITY_ORDER):
                features[f"clip_{name}"].append(
                    projected[:, index].detach().float().cpu()
                )
            features["clip_concat"].append(combined.detach().float().cpu())
            features["clip_mean"].append(
                projected.mean(dim=1).detach().float().cpu()
            )
            identities.extend(int(value) for value in pids)
            cameras.extend(int(value) for value in camids.tolist())
    metrics = {
        name: _metric(
            chunks,
            identities,
            cameras,
            num_query=data.num_query,
        )
        for name, chunks in features.items()
    }
    return metrics, identities, cameras


def _summarize(chunks: list[torch.Tensor]) -> dict[str, float]:
    values = torch.cat(chunks).float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _collect_trained(
    model: torch.nn.Module,
    data: Any,
    identities: list[int],
    cameras: list[int],
    *,
    amp: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    features = {f"trained_cls_{name}": [] for name in MODALITY_ORDER}
    features.update({"trained_cls_concat": [], "trained_cls_mean": [], "fused": []})
    features.update({expert: [] for expert in EXPERT_ORDER})
    statistic_names = [
        *(f"cls_to_{expert}" for expert in EXPERT_ORDER),
        "anchor_norm",
        "residual_norm",
        "residual_ratio",
        "branch_cos_cnn_transformer",
        "branch_cos_cnn_mamba",
        "branch_cos_transformer_mamba",
        "contrib_cos_cnn_transformer",
        "contrib_cos_cnn_mamba",
        "contrib_cos_transformer_mamba",
    ]
    statistics: dict[str, list[torch.Tensor]] = {
        name: [] for name in statistic_names
    }
    observed_identities: list[int] = []
    observed_cameras: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, pids, camids, *_ in data.eval_loader:
            images = {
                name: tensor.cuda(non_blocking=False)
                for name, tensor in images.items()
            }
            mask = torch.ones(len(pids), 3, dtype=torch.bool, device="cuda")
            with torch.cuda.amp.autocast(enabled=amp):
                cls, projected, combined = _exact_anchor(model, images, mask)
                states = model.encoder(images, mask)
                fusion = model.fusion(states, states.reliability, mask)
            for index, name in enumerate(MODALITY_ORDER):
                features[f"trained_cls_{name}"].append(
                    projected[:, index].detach().float().cpu()
                )
            features["trained_cls_concat"].append(
                combined.detach().float().cpu()
            )
            features["trained_cls_mean"].append(
                projected.mean(dim=1).detach().float().cpu()
            )
            features["fused"].append(fusion.fused_embedding.detach().float().cpu())
            for expert in EXPERT_ORDER:
                features[expert].append(
                    fusion.branch_embeddings[expert].detach().float().cpu()
                )
                statistics[f"cls_to_{expert}"].append(
                    F.cosine_similarity(
                        cls.float(), states[expert].global_embedding.float(), dim=-1
                    ).flatten().cpu()
                )
            anchors = []
            residuals = []
            for expert_index, expert in enumerate(EXPERT_ORDER):
                global_embedding = states[expert].global_embedding
                anchor = model.fusion.semantic_projection(global_embedding)
                residual = model.fusion.residual_projections[expert](
                    model.fusion.residual_norms[expert](global_embedding)
                )
                residual = residual * model.fusion.residual_scales[expert_index]
                anchors.append(anchor.float())
                residuals.append(residual.float())
            anchor_tensor = torch.stack(anchors, dim=1)
            residual_tensor = torch.stack(residuals, dim=1)
            anchor_norm = anchor_tensor.norm(dim=-1)
            residual_norm = residual_tensor.norm(dim=-1)
            statistics["anchor_norm"].append(anchor_norm.flatten().cpu())
            statistics["residual_norm"].append(residual_norm.flatten().cpu())
            statistics["residual_ratio"].append(
                (residual_norm / anchor_norm.clamp_min(1e-12)).flatten().cpu()
            )
            pairs = (
                (0, 1, "cnn_transformer"),
                (0, 2, "cnn_mamba"),
                (1, 2, "transformer_mamba"),
            )
            for left, right, name in pairs:
                statistics[f"branch_cos_{name}"].append(
                    F.cosine_similarity(
                        fusion.branch_embeddings[EXPERT_ORDER[left]].float(),
                        fusion.branch_embeddings[EXPERT_ORDER[right]].float(),
                        dim=-1,
                    ).cpu()
                )
                statistics[f"contrib_cos_{name}"].append(
                    F.cosine_similarity(
                        fusion.contribution_embeddings[:, left].float(),
                        fusion.contribution_embeddings[:, right].float(),
                        dim=-1,
                    ).flatten().cpu()
                )
            observed_identities.extend(int(value) for value in pids)
            observed_cameras.extend(int(value) for value in camids.tolist())
    if observed_identities != identities or observed_cameras != cameras:
        raise RuntimeError("development loader ordering changed between passes")
    metrics = {
        name: _metric(
            chunks,
            identities,
            cameras,
            num_query=data.num_query,
        )
        for name, chunks in features.items()
    }
    return metrics, {name: _summarize(chunks) for name, chunks in statistics.items()}


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = _loader(config)
    kwargs = _build_kwargs(config, data.num_classes)
    clip_path = _resolve(PROJECT, str(config["MODEL"]["CLIP_CHECKPOINT"]))

    initial = build_trifusion_cascade_v2_from_clip(
        clip_path, **kwargs
    ).model.cuda()
    initial_metrics, identities, cameras = _collect_initial_clip(
        initial, data, amp=args.amp
    )
    del initial
    torch.cuda.empty_cache()

    trained = build_trifusion_cascade_v2_from_clip(
        clip_path, **kwargs
    ).model.cuda()
    trained.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    trained_metrics, representation_stats = _collect_trained(
        trained,
        data,
        identities,
        cameras,
        amp=args.amp,
    )
    result = {
        "schema_version": "trifusion-cascade-v2-representation-diagnostic-v1",
        "claim_boundary": "held-out train_171 development identities only",
        "official_test_access_count": 0,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "initial_exact_clip_metrics": initial_metrics,
        "trained_exact_cls_metrics": trained_metrics,
        "representation_stats": representation_stats,
        "fusion_residual_scales": [
            float(value) for value in trained.fusion.residual_scales.detach().cpu()
        ],
        "num_query": data.num_query,
        "num_gallery": len(identities) - data.num_query,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = diagnose(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
