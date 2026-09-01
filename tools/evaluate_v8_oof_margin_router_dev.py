#!/usr/bin/env python3
"""Evaluate one frozen V8 Phase-A-plus-Router checkpoint on held-out dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    from run_signal_preserving_v5 import (
        _build_runtime,
        _evaluate_outputs,
        _module_state_sha256,
        _sha256,
        evaluate_dev_gate,
        load_raw_config,
    )
    started = time.time()
    config = load_raw_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError("combined V8 checkpoint SHA-256 differs from the contract")
    if args.output_dir.exists():
        raise FileExistsError(f"V8 Router dev output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    runtime = _build_runtime(config)
    from trifusion.signal_preserving_v8_router import (
        HierarchicalOOFMarginRouter,
        OOFMarginRoutedFusion,
    )

    phase_a_model = runtime["model"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != "trifusion-v8-phase-a-plus-router-v1":
        raise ValueError("unexpected combined V8 checkpoint schema")
    phase_a_model.load_state_dict(payload["phase_a_model_state_dict"], strict=True)
    phase_a_model.cuda().eval()
    for parameter in phase_a_model.parameters():
        parameter.requires_grad_(False)
    router_config = payload["router_config"]
    router = HierarchicalOOFMarginRouter(
        direct_width=int(config["MODEL"]["FEATURE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        hidden_width=int(router_config["HIDDEN_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
        alpha_init=float(router_config["ALPHA_INIT"]),
    )
    router.load_state_dict(payload["router_state_dict"], strict=True)
    router.cuda().eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    fusion = OOFMarginRoutedFusion(
        baseline_width=int(config["SIGNAL"]["BASELINE_WIDTH"]),
        residual_width=int(config["MODEL"]["EXPERT_MODAL_WIDTH"]),
        alpha_max=float(router_config["ALPHA_MAX"]),
    ).cuda().eval()

    class RoutedV8(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phase_a = phase_a_model
            self.router = router
            self.fusion = fusion
            self.alpha_values: list[torch.Tensor] = []
            self.modal_values: list[torch.Tensor] = []
            self.expert_values: list[torch.Tensor] = []

        def forward(self, batch: dict[str, Any], return_aux: bool = False):
            if not return_aux:
                raise ValueError("V8 Router dev evaluation requires auxiliary output")
            phase = self.phase_a(batch, return_aux=True)
            modal_residual = torch.stack(
                [
                    phase.modal_residual_embeddings[name]
                    for name in ("cnn", "transformer", "mamba")
                ],
                dim=1,
            )
            routing = self.router(
                phase.direct_modal,
                modal_residual,
                batch["modality_mask"],
            )
            routed = self.fusion(
                phase.baseline_embedding,
                modal_residual,
                routing,
            )
            self.alpha_values.append(routing.alpha.detach().cpu())
            self.modal_values.append(routing.modal_probabilities.detach().cpu())
            self.expert_values.append(routing.expert_probabilities.detach().cpu())
            finite = (
                phase.diagnostics["all_finite"]
                and bool(torch.isfinite(routed.fused_embedding).all())
                and bool(torch.isfinite(routing.weights).all())
                and bool(torch.isfinite(routing.alpha).all())
            )
            return SimpleNamespace(
                baseline_embedding=phase.baseline_embedding,
                fused_embedding=routed.fused_embedding,
                branch_embeddings=phase.branch_embeddings,
                diagnostics={
                    "all_finite": finite,
                    "baseline_exact_prefix": torch.equal(
                        routed.fused_embedding[
                            :, : phase.baseline_embedding.shape[1]
                        ],
                        phase.baseline_embedding,
                    ),
                },
            )

    model = RoutedV8().cuda().eval()
    phase_state_before = _module_state_sha256(phase_a_model)
    router_state_before = _module_state_sha256(router)
    torch.cuda.reset_peak_memory_stats()
    retrieval_outputs = tuple(config["MODEL"]["RETRIEVAL_OUTPUTS"])
    metrics, widths = _evaluate_outputs(
        model,
        runtime["eval_loader"],
        num_query=len(runtime["dev_records"]),
        retrieval_outputs=retrieval_outputs,
    )
    phase_state_after = _module_state_sha256(phase_a_model)
    router_state_after = _module_state_sha256(router)
    if phase_state_after != phase_state_before or router_state_after != router_state_before:
        raise RuntimeError("V8 Router dev evaluation changed checkpoint state")
    gate = evaluate_dev_gate(
        metrics,
        min_map=float(config["GATES"]["DEV_MIN_MAP"]),
    )
    alpha = torch.cat(model.alpha_values)
    modal = torch.cat(model.modal_values)
    expert = torch.cat(model.expert_values)
    result = {
        "schema_version": "trifusion-v8-oof-margin-router-dev-result-v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metrics_percent": metrics,
        "feature_widths": widths,
        "router_diagnostics": {
            "alpha_mean": float(alpha.mean()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "modal_probability_mean": {
                name: float(modal[:, index].mean())
                for index, name in enumerate(("RGB", "NI", "TI"))
            },
            "expert_probability_mean": {
                name: float(expert[:, index].mean())
                for index, name in enumerate(("cnn", "transformer", "mamba"))
            },
        },
        "promotion_gate": gate,
        "next_phase_authorized": bool(gate["passed"]),
        "phase_a_state_sha256_before": phase_state_before,
        "phase_a_state_sha256_after": phase_state_after,
        "router_state_sha256_before": router_state_before,
        "router_state_sha256_after": router_state_after,
        "checkpoint_state_unchanged": (
            phase_state_before == phase_state_after
            and router_state_before == router_state_after
        ),
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = ["run"]
