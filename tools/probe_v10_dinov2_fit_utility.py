#!/usr/bin/env python3
"""Qualify frozen DINOv2 complementarity on fit identities only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Set
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


MODALITIES = ("RGB", "NI", "TI")
DINO_IMAGE_SIZE = (252, 126)
DINO_TOKEN_SHAPE = (163, 768)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_dinov2_input(images: torch.Tensor) -> torch.Tensor:
    """Convert the existing [-1,1] image tensor to fixed DINOv2 input."""

    pixels = images * 0.5 + 0.5
    resized = F.interpolate(
        pixels,
        size=DINO_IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


def prepare_dinov2_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    model_keys: Set[str],
) -> dict[str, torch.Tensor]:
    """Remove the one pretraining-only key, then require an exact load."""

    if "mask_token" not in state_dict:
        raise ValueError("DINOv2 checkpoint must contain exactly one mask_token")
    prepared = dict(state_dict)
    del prepared["mask_token"]
    if set(prepared) != set(model_keys):
        raise ValueError("DINOv2 state keys do not match the feature model")
    return prepared


def dinov2_global_embedding(tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate final CLS and patch mean for each of three modalities."""

    cls = tokens[:, :, 0]
    patch_mean = tokens[:, :, 1:].mean(dim=2)
    return torch.cat((cls, patch_mean), dim=-1).flatten(1)


def compose_equal_block_embedding(
    phase_b: torch.Tensor,
    dinov2: torch.Tensor,
) -> torch.Tensor:
    """Use the one preregistered equal-energy two-foundation composition."""

    return torch.cat(
        (F.normalize(phase_b, dim=1), F.normalize(dinov2, dim=1)),
        dim=1,
    )


def evaluate_qualification_gate(
    *,
    phase_b_map: float,
    concat_map: float,
    oracle_gain_map: float,
    unique_ap_wins: Mapping[str, int],
    min_concat_gain_map: float,
    min_oracle_gain_map: float,
) -> dict[str, Any]:
    concat_gain = float(concat_map) - float(phase_b_map)
    two_source_wins = all(
        int(unique_ap_wins[name]) > 0 for name in ("phase_b", "dinov2")
    )
    passed = (
        concat_gain >= float(min_concat_gain_map)
        and float(oracle_gain_map) >= float(min_oracle_gain_map)
        and two_source_wins
    )
    return {
        "passed": passed,
        "concat_minus_phase_b_mAP": concat_gain,
        "minimum_concat_gain_mAP": float(min_concat_gain_map),
        "oracle_minus_best_fixed_mAP": float(oracle_gain_map),
        "minimum_oracle_gain_mAP": float(min_oracle_gain_map),
        "unique_ap_wins": {
            name: int(unique_ap_wins[name]) for name in ("phase_b", "dinov2")
        },
        "both_sources_have_unique_ap_wins": two_source_wins,
    }


def _build_frozen_phase_b(
    runtime: dict[str, Any],
    config: dict[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    from trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
        OOFMarginRoutedFusion,
    )

    phase_a = runtime["model"]
    phase_a.load_state_dict(checkpoint["phase_a_model_state_dict"], strict=True)
    router_config = checkpoint["router_config"]
    router = HierarchicalOOFMarginRouter(
        direct_width=int(config["MODEL"]["FEATURE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        hidden_width=int(router_config["HIDDEN_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
        alpha_init=float(router_config["ALPHA_INIT"]),
    )
    router.load_state_dict(checkpoint["router_state_dict"], strict=True)
    fusion = OOFMarginRoutedFusion(
        baseline_width=int(config["SIGNAL"]["BASELINE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
    )

    for module in (phase_a, router, fusion):
        module.cuda().eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return phase_a, router, fusion


def _phase_b_forward(
    phase_a: torch.nn.Module,
    router: torch.nn.Module,
    fusion: torch.nn.Module,
    images: Mapping[str, torch.Tensor],
    camera_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size = camera_ids.shape[0]
    batch = {
        "images": images,
        "modality_mask": torch.ones(
            batch_size, 3, dtype=torch.bool, device=camera_ids.device
        ),
        "camera_ids": camera_ids,
    }
    phase = phase_a(batch, return_aux=True)
    modal_residual = torch.stack(
        [
            phase.modal_residual_embeddings[name]
            for name in ("cnn", "transformer", "mamba")
        ],
        dim=1,
    )
    routing = router(phase.direct_modal, modal_residual, batch["modality_mask"])
    return fusion(
        phase.baseline_embedding,
        modal_residual,
        routing,
    ).fused_embedding


def _build_dinov2(weight_path: Path) -> torch.nn.Module:
    import timm

    model = timm.create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        dynamic_img_size=True,
        num_classes=0,
    )
    raw_state = torch.load(weight_path, map_location="cpu", weights_only=False)
    prepared = prepare_dinov2_state_dict(
        raw_state,
        model_keys=set(model.state_dict()),
    )
    load_result = model.load_state_dict(prepared, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("DINOv2 strict load did not match every feature key")
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _collect_fit_features(
    *,
    phase_a: torch.nn.Module,
    router: torch.nn.Module,
    fusion: torch.nn.Module,
    dinov2: torch.nn.Module,
    loader: Any,
) -> dict[str, Any]:
    phase_features = []
    dino_features = []
    identities: list[int] = []
    cameras: list[int] = []
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
        images = {
            name: value.cuda(non_blocking=True) for name, value in images.items()
        }
        camera_labels = camera_labels.cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            phase_b = _phase_b_forward(
                phase_a,
                router,
                fusion,
                images,
                camera_labels,
            )
            modality_tokens = []
            for modality in MODALITIES:
                tokens = dinov2.forward_features(
                    prepare_dinov2_input(images[modality])
                )
                if tuple(tokens.shape[1:]) != DINO_TOKEN_SHAPE:
                    raise RuntimeError(
                        f"DINOv2 tokens must have shape B,{DINO_TOKEN_SHAPE[0]},{DINO_TOKEN_SHAPE[1]}"
                    )
                modality_tokens.append(tokens)
            dino = dinov2_global_embedding(torch.stack(modality_tokens, dim=1))
        phase_features.append(phase_b.float().cpu())
        dino_features.append(dino.float().cpu())
        identities.extend(torch.as_tensor(batch_ids).tolist())
        cameras.extend(torch.as_tensor(batch_cameras).tolist())
    phase_b = torch.cat(phase_features)
    dinov2_embedding = torch.cat(dino_features)
    return {
        "phase_b": phase_b,
        "dinov2": dinov2_embedding,
        "concat": compose_equal_block_embedding(phase_b, dinov2_embedding),
        "identities": identities,
        "cameras": cameras,
    }


def _metric_summary(scores: Any) -> dict[str, float]:
    return {
        "mAP": float(scores.average_precision.mean() * 100.0),
        "Rank-1": float(scores.rank1_correct.mean() * 100.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    from diagnose_v6_oracle_complementarity import (
        _scores_from_features,
        summarize_oracle_complementarity,
    )
    from probe_v8_frozen_router import _fit_eval_loader
    from run_signal_preserving_v5 import (
        _build_runtime,
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )

    started = time.time()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"V10 qualification output already exists: {output_path}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dino_weight_path = args.dino_weight.resolve()
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError("combined V8 checkpoint SHA-256 differs from the contract")
    if _sha256(dino_weight_path) != args.dino_weight_sha256:
        raise ValueError("DINOv2 weight SHA-256 differs from the contract")

    config = load_raw_config(config_path)
    runtime = _build_runtime(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != "trifusion-v8-phase-a-plus-router-v1":
        raise ValueError("unexpected combined V8 checkpoint schema")
    phase_a, router, fusion = _build_frozen_phase_b(runtime, config, checkpoint)
    dinov2 = _build_dinov2(dino_weight_path)

    phase_state_before = _module_state_sha256(phase_a)
    router_state_before = _module_state_sha256(router)
    dino_state_before = _module_state_sha256(dinov2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    loader = _fit_eval_loader(runtime, config)
    collected = _collect_fit_features(
        phase_a=phase_a,
        router=router,
        fusion=fusion,
        dinov2=dinov2,
        loader=loader,
    )
    phase_state_after = _module_state_sha256(phase_a)
    router_state_after = _module_state_sha256(router)
    dino_state_after = _module_state_sha256(dinov2)
    state_unchanged = (
        phase_state_before == phase_state_after
        and router_state_before == router_state_after
        and dino_state_before == dino_state_after
    )
    if not state_unchanged:
        raise RuntimeError("V10 qualification changed a frozen checkpoint state")

    num_query = len(loader.dataset) // 2
    identities = np.asarray(collected["identities"])
    cameras = np.asarray(collected["cameras"])
    scores = {
        name: _scores_from_features(
            collected[name],
            identities,
            cameras,
            num_query=num_query,
        )
        for name in ("phase_b", "dinov2", "concat")
    }
    metrics = {name: _metric_summary(value) for name, value in scores.items()}
    oracle = summarize_oracle_complementarity(
        {name: scores[name] for name in ("phase_b", "dinov2")}
    )
    gate = evaluate_qualification_gate(
        phase_b_map=metrics["phase_b"]["mAP"],
        concat_map=metrics["concat"]["mAP"],
        oracle_gain_map=float(oracle["oracle_minus_best_fixed_percent"]["mAP"]),
        unique_ap_wins=oracle["unique_ap_wins"],
        min_concat_gain_map=float(args.min_concat_gain_map),
        min_oracle_gain_map=float(args.min_oracle_gain_map),
    )
    result = {
        "schema_version": "trifusion-v10-dinov2-fit-qualification-v1",
        "status": "PASS",
        "scope": "141-fit cross-camera identities only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "phase_b_checkpoint": str(checkpoint_path),
        "phase_b_checkpoint_sha256": _sha256(checkpoint_path),
        "dinov2_weight": str(dino_weight_path),
        "dinov2_weight_sha256": _sha256(dino_weight_path),
        "dinov2_model": "vit_base_patch14_dinov2",
        "dinov2_source": "https://github.com/facebookresearch/dinov2",
        "dinov2_input_size": list(DINO_IMAGE_SIZE),
        "dinov2_token_shape": list(DINO_TOKEN_SHAPE),
        "strict_load": True,
        "removed_pretraining_only_key": "mask_token",
        "fit_queries": num_query,
        "fit_identities": int(np.unique(identities[:num_query]).size),
        "metrics_percent": metrics,
        "phase_b_dinov2_oracle": oracle,
        "qualification_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "phase_a_state_sha256_before": phase_state_before,
        "phase_a_state_sha256_after": phase_state_after,
        "router_state_sha256_before": router_state_before,
        "router_state_sha256_after": router_state_after,
        "dinov2_state_sha256_before": dino_state_before,
        "dinov2_state_sha256_after": dino_state_after,
        "checkpoint_states_unchanged": state_unchanged,
        "feature_widths": {
            name: int(collected[name].shape[1])
            for name in ("phase_b", "dinov2", "concat")
        },
        "optimizer_steps": 0,
        "training_executed": False,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--dino-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-concat-gain-map", type=float, default=1.0)
    parser.add_argument("--min-oracle-gain-map", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "compose_equal_block_embedding",
    "dinov2_global_embedding",
    "evaluate_qualification_gate",
    "prepare_dinov2_input",
    "prepare_dinov2_state_dict",
    "run",
]
