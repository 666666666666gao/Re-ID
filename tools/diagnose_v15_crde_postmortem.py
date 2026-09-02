#!/usr/bin/env python3
"""Read-only postmortem of V15 exchange geometry on registered OOF folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


EXPERT_ORDER = ("cnn", "transformer", "mamba")
OUTPUT_ORDER = ("fused", *EXPERT_ORDER)


def exchange_effect_statistics_v15(
    own_delta: torch.Tensor,
    incoming: torch.Tensor,
) -> dict[str, float]:
    """Summarize the actual injected tensor relative to the receiver delta."""

    own = own_delta.flatten(1).float()
    received = incoming.flatten(1).float()
    own_norm = own.norm(dim=1)
    incoming_norm = received.norm(dim=1)
    return {
        "own_norm_mean": float(own_norm.mean()),
        "incoming_norm_mean": float(incoming_norm.mean()),
        "incoming_to_own_norm_mean": float((incoming_norm / own_norm).mean()),
        "incoming_own_cosine_mean": float(
            F.cosine_similarity(received, own, dim=1).mean()
        ),
    }


def matched_embedding_statistics_v15(
    exchange_on: torch.Tensor,
    exchange_off: torch.Tensor,
    ap_gain: torch.Tensor,
) -> dict[str, float | int]:
    """Relate matched representation displacement to per-query AP change."""

    on = F.normalize(exchange_on.float(), dim=1)
    off = F.normalize(exchange_off.float(), dim=1)
    displacement = (on - off).norm(dim=1)
    gain = ap_gain.float()
    return {
        "cosine_mean": float(F.cosine_similarity(on, off, dim=1).mean()),
        "l2_displacement_mean": float(displacement.mean()),
        "ap_gain_mean": float(gain.mean()),
        "ap_gain_median": float(torch.quantile(gain, 0.5)),
        "positive_queries": int((gain > 0).sum()),
        "negative_queries": int((gain < 0).sum()),
        "zero_queries": int((gain == 0).sum()),
        "displacement_ap_gain_pearson": float(
            torch.corrcoef(torch.stack((displacement, gain)))[0, 1]
        ),
    }


def edge_scale_stability_v15(scales: torch.Tensor) -> list[dict[str, Any]]:
    """Report fold-wise stability of every directed, non-self exchange edge."""

    records = []
    for stage in range(scales.shape[1]):
        for source_index, source in enumerate(EXPERT_ORDER):
            for target_index, target in enumerate(EXPERT_ORDER):
                if source == target:
                    continue
                values = scales[:, stage, source_index, target_index].float()
                records.append(
                    {
                        "stage": stage,
                        "edge": f"{source}__{target}",
                        "values": [round(float(value), 7) for value in values],
                        "mean_abs": float(values.abs().mean()),
                        "sign_agreement": float(values.sign().sum().abs() / len(values)),
                        "range": float(values.max() - values.min()),
                    }
                )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fold_checkpoint(model: Any, checkpoint_path: Path, split: Mapping[str, Any]) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(payload["fit_identity_ids"]) != tuple(split["fit_identity_ids"]):
        raise ValueError("V15 diagnostic checkpoint fit identities differ")
    if tuple(payload["heldout_identity_ids"]) != tuple(split["heldout_identity_ids"]):
        raise ValueError("V15 diagnostic checkpoint heldout identities differ")
    trainable_state = payload["trainable_state_dict"]
    expected = {
        name
        for name in model.state_dict()
        if name.startswith("encoder.exchange_stages.")
        or name.startswith("fused_neck.")
        or name.startswith("branch_necks.")
        or name.startswith("residual_necks.")
        or name.startswith("fused_classifier.")
        or name.startswith("branch_classifiers.")
        or name.startswith("residual_classifiers.")
    }
    if set(trainable_state) != expected:
        raise RuntimeError("V15 diagnostic checkpoint trainable keys differ")
    incompatible = model.load_state_dict(trainable_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("V15 diagnostic checkpoint has unexpected keys")


def _register_exchange_observers(model: Any) -> tuple[dict[str, Any], list[Any]]:
    storage = {
        "own": {
            stage: {expert: [] for expert in EXPERT_ORDER} for stage in range(2)
        },
        "incoming": {
            stage: {expert: [] for expert in EXPERT_ORDER} for stage in range(2)
        },
        "edges": {
            stage: {
                f"{source}__{target}": []
                for source in EXPERT_ORDER
                for target in EXPERT_ORDER
                if source != target
            }
            for stage in range(2)
        },
    }
    handles = []

    def make_hook(stage_index: int):
        def observe(module: Any, args: tuple[Any, ...], output: Any) -> None:
            before, after = args
            for target in EXPERT_ORDER:
                storage["own"][stage_index][target].append(
                    (after[target] - before[target]).detach().float().cpu()
                )
                storage["incoming"][stage_index][target].append(
                    (output.states[target] - after[target]).detach().float().cpu()
                )
            for source_index, source in enumerate(EXPERT_ORDER):
                for target_index, target in enumerate(EXPERT_ORDER):
                    if source == target:
                        continue
                    projected = module.edge_projections[f"{source}__{target}"](
                        output.messages[source]
                    )
                    contribution = (
                        output.edge_scales[source_index, target_index] * projected
                    )
                    storage["edges"][stage_index][f"{source}__{target}"].append(
                        contribution.detach().float().cpu()
                    )

        return observe

    for stage_index, stage in enumerate(model.encoder.exchange_stages):
        handles.append(stage.register_forward_hook(make_hook(stage_index)))
    return storage, handles


def _collect_fold(
    model: Any,
    loader: Any,
    *,
    num_query: int,
) -> dict[str, Any]:
    import numpy as np

    from tools.diagnose_v6_oracle_complementarity import _scores_from_features

    storage, handles = _register_exchange_observers(model)
    features = {
        side: {name: [] for name in OUTPUT_ORDER} for side in ("on", "off")
    }
    identities = []
    cameras = []
    stage_scales = None
    model.eval()
    for images, batch_ids, batch_cameras, camera_labels, _views, _paths in loader:
        images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
        camera_labels = camera_labels.cuda(non_blocking=True)
        batch = {
            "images": images,
            "modality_mask": torch.ones(
                camera_labels.shape[0], 3, dtype=torch.bool, device="cuda"
            ),
            "camera_ids": camera_labels,
        }
        with torch.no_grad():
            paired = model.forward_paired(batch, with_on_heads=False)
        if stage_scales is None:
            stage_scales = torch.stack(
                [value.detach().float().cpu() for value in paired.exchange_on.exchange_edge_scales]
            )
        for side, output in (
            ("on", paired.exchange_on),
            ("off", paired.exchange_off),
        ):
            features[side]["fused"].append(output.fused_embedding.float().cpu())
            for expert in EXPERT_ORDER:
                features[side][expert].append(
                    output.branch_embeddings[expert].float().cpu()
                )
        identities.extend(np.asarray(batch_ids).tolist())
        cameras.extend(np.asarray(batch_cameras).tolist())
    for handle in handles:
        handle.remove()

    feature_tensors = {
        side: {name: torch.cat(parts) for name, parts in outputs.items()}
        for side, outputs in features.items()
    }
    identities_array = np.asarray(identities)
    cameras_array = np.asarray(cameras)
    scores = {
        side: {
            name: _scores_from_features(
                value,
                identities_array,
                cameras_array,
                num_query=num_query,
            )
            for name, value in outputs.items()
        }
        for side, outputs in feature_tensors.items()
    }
    embedding_statistics = {}
    for name in OUTPUT_ORDER:
        ap_gain = torch.from_numpy(
            scores["on"][name].average_precision
            - scores["off"][name].average_precision
        )
        embedding_statistics[name] = matched_embedding_statistics_v15(
            feature_tensors["on"][name][:num_query],
            feature_tensors["off"][name][:num_query],
            ap_gain,
        )

    exchange_statistics = {}
    edge_statistics = {}
    for stage in range(2):
        exchange_statistics[str(stage)] = {}
        edge_statistics[str(stage)] = {}
        for target in EXPERT_ORDER:
            own = torch.cat(storage["own"][stage][target])
            incoming = torch.cat(storage["incoming"][stage][target])
            exchange_statistics[str(stage)][target] = exchange_effect_statistics_v15(
                own,
                incoming,
            )
        for edge, parts in storage["edges"][stage].items():
            target = edge.split("__", 1)[1]
            own = torch.cat(storage["own"][stage][target])
            contribution = torch.cat(parts)
            edge_statistics[str(stage)][edge] = exchange_effect_statistics_v15(
                own,
                contribution,
            )
    return {
        "stage_scales": stage_scales,
        "embedding_statistics": embedding_statistics,
        "exchange_statistics": exchange_statistics,
        "edge_statistics": edge_statistics,
        "query_count": num_query,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from tools.build_v12_complete_path_oof_targets import (
        _configure_signal,
        _eval_loader,
        _load_records,
        build_complete_path_fold_records,
    )
    from tools.probe_v8_frozen_router import select_cross_camera_records
    from tools.run_signal_preserving_v5 import _set_seed
    from tools.train_signal_preserving_v15 import (
        _build_fold_model,
        _frozen_state_sha256,
        _load_contract,
    )

    started = time.time()
    if args.output.exists():
        raise FileExistsError(f"V15 postmortem output exists: {args.output}")
    contract = _load_contract(args.config.resolve())
    q1_path = args.q1_summary.resolve()
    q1 = json.loads(q1_path.read_text(encoding="utf-8"))
    if q1["gate"]["passed"] or q1["next_phase_authorized"]:
        raise ValueError("V15 postmortem requires the sealed negative Q1 receipt")
    config = contract["config"]
    signal_cfg, source_commit, source_diff_sha256 = _configure_signal(config)
    records = _load_records(config)
    eligible_records = select_cross_camera_records(records)
    torch.cuda.reset_peak_memory_stats()
    folds = []
    fold_scales = []
    for fold_index, registry in enumerate(contract["v12_summary"]["fold_receipts"]):
        heldout = {int(value) for value in registry["heldout_identity_ids"]}
        split = build_complete_path_fold_records(records, heldout_ids=heldout)
        _set_seed(42)
        model, _binding = _build_fold_model(
            config,
            signal_cfg,
            fold_index=fold_index,
            split=split,
        )
        checkpoint_path = Path(q1["fold_receipts"][fold_index]["checkpoint"])
        if _sha256(checkpoint_path) != q1["fold_receipts"][fold_index]["checkpoint_sha256"]:
            raise ValueError("V15 postmortem checkpoint SHA-256 differs from Q1")
        _load_fold_checkpoint(model, checkpoint_path, split)
        frozen_before = _frozen_state_sha256(model)
        heldout_records = [
            record for record in eligible_records if int(record[1]) in heldout
        ]
        collected = _collect_fold(
            model,
            _eval_loader(heldout_records, config),
            num_query=len(heldout_records),
        )
        frozen_after = _frozen_state_sha256(model)
        fold_scales.append(collected.pop("stage_scales"))
        folds.append(
            {
                "fold": fold_index,
                "query_count": collected.pop("query_count"),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "frozen_state_unchanged": frozen_before == frozen_after,
                **collected,
            }
        )
        del model
        torch.cuda.empty_cache()
    stacked_scales = torch.stack(fold_scales)
    result = {
        "schema_version": "trifusion-v15-crde-postmortem-v1",
        "status": "PASS",
        "evaluation_type": "real_gt_train_only_identity_oof_read_only_postmortem",
        "seed": 42,
        "folds": folds,
        "edge_scale_stability": edge_scale_stability_v15(stacked_scales),
        "q1_summary": str(q1_path),
        "q1_summary_sha256": _sha256(q1_path),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "repository_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"])
        ).hexdigest(),
        "source_commit": source_commit,
        "signal_source_diff_sha256": source_diff_sha256,
        "optimizer_steps": 0,
        "training_reexecuted": False,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "d1_executed": False,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


__all__ = [
    "edge_scale_stability_v15",
    "exchange_effect_statistics_v15",
    "matched_embedding_statistics_v15",
    "run",
]
