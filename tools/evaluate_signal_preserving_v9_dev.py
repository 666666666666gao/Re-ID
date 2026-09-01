#!/usr/bin/env python3
"""Evaluate one frozen final-only TriFusion V9 checkpoint on held-out dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


COMPARISON_OUTPUTS = ("baseline_only", "phase_b", "cnn", "transformer", "mamba")


def evaluate_v9_dev_gate(
    metrics: dict[str, dict[str, float]],
    *,
    min_map: float,
) -> dict[str, Any]:
    fused_map = float(metrics["fused"]["mAP"])
    strictly_beaten = {
        name: fused_map > float(metrics[name]["mAP"])
        for name in COMPARISON_OUTPUTS
    }
    return {
        "passed": fused_map >= float(min_map) and all(strictly_beaten.values()),
        "fused_mAP": fused_map,
        "minimum_mAP": float(min_map),
        "strictly_beaten": strictly_beaten,
    }


def _load_collaboration(model: Any, state: dict[str, Any]) -> None:
    model.synthesis.load_state_dict(state["synthesis"], strict=True)
    model.fused_neck.load_state_dict(state["fused_neck"], strict=True)
    model.synergy_neck.load_state_dict(state["synergy_neck"], strict=True)
    model.branch_necks.load_state_dict(state["branch_necks"], strict=True)
    model.fused_classifier.load_state_dict(state["fused_classifier"], strict=True)
    model.synergy_classifier.load_state_dict(state["synergy_classifier"], strict=True)
    model.branch_classifiers.load_state_dict(
        state["branch_classifiers"],
        strict=True,
    )


def _evaluate(
    model: Any,
    loader: Any,
    *,
    num_query: int,
    retrieval_outputs: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, float]]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from utils.metrics import R1_mAP_eval

    evaluators = {
        name: R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")
        for name in retrieval_outputs
    }
    widths = {name: 0 for name in retrieval_outputs}
    beta_values = []
    relay_cosines = []
    relay_norms = []
    model.eval()
    for images, identities, camera_ids, camera_ids_batch, _view_ids, paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
        camera_ids_cuda = camera_ids_batch.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_ids_cuda.shape[0],
                3,
                dtype=torch.bool,
                device="cuda",
            ),
            "camera_ids": camera_ids_cuda,
        }
        with torch.no_grad():
            output = model(batch, return_aux=True)
        if not output.diagnostics["all_finite"]:
            raise FloatingPointError("V9 dev evaluation emitted a nonfinite tensor")
        if not output.diagnostics["baseline_exact_prefix"]:
            raise RuntimeError("V9 dev output lost the exact Signal prefix")
        if not output.diagnostics["phase_b_exact_prefix"]:
            raise RuntimeError("V9 dev output lost the exact Phase-B prefix")
        features = {
            "baseline_only": output.baseline_embedding,
            "phase_b": output.phase_b_embedding,
            "fused": output.fused_embedding,
            **dict(output.branch_embeddings),
        }
        for name in retrieval_outputs:
            widths[name] = int(features[name].shape[1])
            evaluators[name].update((features[name], identities, camera_ids, paths))
        beta_values.append(output.beta.detach().cpu())
        for receivers, messages in zip(
            output.relay.receiver_inputs,
            output.relay.orthogonal_messages,
            strict=True,
        ):
            relay_cosines.append(
                F.cosine_similarity(receivers, messages, dim=-1).abs().cpu()
            )
            relay_norms.append(messages.norm(dim=-1).cpu())

    metrics = {}
    for name in retrieval_outputs:
        cmc, mean_ap, *_ = evaluators[name].compute()
        metrics[name] = {
            "mAP": float(mean_ap * 100.0),
            "Rank-1": float(cmc[0] * 100.0),
            "Rank-5": float(cmc[4] * 100.0),
            "Rank-10": float(cmc[9] * 100.0),
        }
        if not all(np.isfinite(value) for value in metrics[name].values()):
            raise FloatingPointError(f"V9 metric is nonfinite: {name}")
    beta = torch.cat(beta_values)
    diagnostics = {
        "beta_mean": float(beta.mean()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "max_abs_relay_cosine": float(max(value.max() for value in relay_cosines)),
        "mean_relay_norm": float(sum(value.mean() for value in relay_norms) / len(relay_norms)),
    }
    return metrics, widths, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from run_signal_preserving_v5 import (
        _module_state_sha256,
        _sha256,
        load_raw_config,
    )
    from run_signal_preserving_v9 import _build_v9_runtime

    started = time.time()
    config = load_raw_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError("V9 checkpoint SHA-256 differs from the frozen contract")
    if args.output_dir.exists():
        raise FileExistsError(f"V9 dev output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    runtime = _build_v9_runtime(config)
    model = runtime["model"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != "trifusion-v9-orthogonal-triadic-synthesis-v1":
        raise ValueError("unexpected V9 checkpoint schema")
    if payload["combined_v8_checkpoint_sha256"] != runtime[
        "combined_checkpoint_sha256"
    ]:
        raise ValueError("V9 checkpoint references another V8 base")
    _load_collaboration(model, payload["collaboration_state_dict"])
    model.cuda().eval()
    state_before = _module_state_sha256(model)
    phase_before = _module_state_sha256(model.phase_a)
    router_before = _module_state_sha256(model.router)
    torch.cuda.reset_peak_memory_stats()
    retrieval_outputs = tuple(config["MODEL"]["RETRIEVAL_OUTPUTS"])
    metrics, widths, diagnostics = _evaluate(
        model,
        runtime["eval_loader"],
        num_query=len(runtime["dev_records"]),
        retrieval_outputs=retrieval_outputs,
    )
    state_after = _module_state_sha256(model)
    phase_after = _module_state_sha256(model.phase_a)
    router_after = _module_state_sha256(model.router)
    if state_before != state_after:
        raise RuntimeError("V9 frozen dev evaluation changed checkpoint state")
    gate = evaluate_v9_dev_gate(
        metrics,
        min_map=float(config["GATES"]["DEV_MIN_MAP"]),
    )
    result = {
        "schema_version": "trifusion-v9-frozen-dev-result-v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "combined_v8_checkpoint_sha256": runtime["combined_checkpoint_sha256"],
        "metrics_percent": metrics,
        "feature_widths": widths,
        "diagnostics": diagnostics,
        "promotion_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "checkpoint_state_sha256_before": state_before,
        "checkpoint_state_sha256_after": state_after,
        "checkpoint_state_unchanged": state_before == state_after,
        "phase_a_state_sha256_before": phase_before,
        "phase_a_state_sha256_after": phase_after,
        "router_state_sha256_before": router_before,
        "router_state_sha256_after": router_after,
        "optimizer_steps": 0,
        "training_executed": False,
        "dev_access_count": 1,
        "official_test_access_count": 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
