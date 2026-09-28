#!/usr/bin/env python3
"""Read-only decomposition of a completed correspondence-role checkpoint."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.run_correspondence_roles import (
    _eval_batch, build, distance_matrix, load_checkpoint, loader_for, read_protocol,
    records_for, sha256,
)
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def collect(model, initial_adapters, protocol, split):
    features = {name: [] for name in ("baseline", "adapted_global", "fused")}
    measurements = {name: [] for name in ("correction_ratio", "anchor_shift", "cnn_transformer_cosine",
                                         "cnn_mamba_cosine", "transformer_mamba_cosine")}
    captured = []
    handle = model.roles.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    with torch.inference_mode():
        for raw in loader_for(protocol, records_for(protocol, split), training=False, method="PLAIN_V8"):
            batch = _eval_batch(raw, protocol["dataset"])
            output = model(batch, return_aux=True)
            evidence = captured.pop()
            assert not captured
            _, baseline = model.backbone(batch["images"], batch["camera_ids"], adapter_bank=initial_adapters)
            pooled = [role.mean(dim=(1, 2)) for role in (evidence.cnn, evidence.transformer, evidence.mamba)]
            correction = model.readout_gain * model.readout(torch.cat(pooled, dim=1)).float()
            assert torch.allclose(output["fused"], F.normalize(output["shared_global"].float() + correction, dim=1))
            for name, value in (("baseline", baseline), ("adapted_global", output["shared_global"]), ("fused", output["fused"])):
                features[name].append(value.float().cpu())
            measurements["correction_ratio"].append((correction.norm(dim=1) / output["shared_global"].float().norm(dim=1)).cpu())
            measurements["anchor_shift"].append((evidence.positions - model.roles.reference_points[None, None]).norm(dim=-1).mean(dim=(1, 2)).cpu())
            for name, first, second in (("cnn_transformer_cosine", 0, 1), ("cnn_mamba_cosine", 0, 2), ("transformer_mamba_cosine", 1, 2)):
                measurements[name].append(F.cosine_similarity(pooled[first].float(), pooled[second].float()).cpu())
    handle.remove()
    return ({name: torch.cat(values) for name, values in features.items()},
            {name: torch.cat(values) for name, values in measurements.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BASELINES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    run = args.run_dir.resolve()
    training = json.loads((run / "training.json").read_text())
    evaluation = json.loads((run / "official_metrics.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE" and evaluation["status"] == "COMPLETE"
    assert len(training["history"]) == 50
    binding = training["initializer"]
    weight, digest = BASELINES[args.dataset]
    model_args = argparse.Namespace(dataset=args.dataset, signal_source=SOURCE,
                                    clip_weight=WEIGHTS / "ViT-B-16.pt", baseline_checkpoint=WEIGHTS / weight,
                                    baseline_sha256=digest, seed=training["seed"], width=binding["width"],
                                    m1=binding["m1"], m2=binding["m2"], m3=binding["m3"],
                                    pred_weight=binding["prediction_weight"], protocol=PROTOCOLS / f"{args.dataset}.json")
    protocol = read_protocol(model_args.protocol, args.dataset)
    model, _cfg, _config, actual = build(model_args, protocol)
    assert actual == binding
    initial_adapters = deepcopy(model.backbone.adapters)
    checkpoint = run / "best_map.pth"
    assert sha256(checkpoint) == evaluation["checkpoint_sha256"]
    load_checkpoint(checkpoint, model, model_args, protocol)
    model.eval()
    query, measurements = collect(model, initial_adapters, protocol, "query")
    gallery, _ = collect(model, initial_adapters, protocol, "gallery")
    qrows, grows = protocol["records"]["query"], protocol["records"]["gallery"]
    qids, gids = (np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows))
    field = "scene" if args.dataset == "MSVR310" else "camera"
    qenvironments, genvironments = (np.asarray([row[field] for row in rows]) for rows in (qrows, grows))
    scorer = scene_scores if field == "scene" else camera_scores
    scores = {name: scorer(distance_matrix(query[name], gallery[name]).numpy(), qids, gids, qenvironments, genvironments)
              for name in query}
    assert all(abs(scores["fused"]["metrics"][name] - evaluation["metrics"][name]) < 1e-5 for name in evaluation["metrics"])
    baseline_rank = np.asarray(scores["baseline"]["first_match_rank"])
    fused_rank = np.asarray(scores["fused"]["first_match_rank"])
    baseline_ap = np.asarray(scores["baseline"]["average_precision"])
    fused_ap = np.asarray(scores["fused"]["average_precision"])
    identity_changes = [float((fused_ap - baseline_ap)[qids == identity].mean()) for identity in np.unique(qids)]
    report = {"dataset": args.dataset, "status": "READ_ONLY_COMPLETE", "selected_epoch": evaluation["selected_epoch"],
              "checkpoint_sha256": sha256(checkpoint), "evaluation_sha256": sha256(run / "official_metrics.json"),
              "components": {name: value["metrics"] for name, value in scores.items()},
              "readout_gain": float(model.readout_gain.detach()),
              "measurements": {name: {"mean": float(values.mean()), "median": float(values.median()), "maximum": float(values.max())}
                               for name, values in measurements.items()},
              "repairs": int(((baseline_rank != 1) & (fused_rank == 1)).sum()),
              "new_errors": int(((baseline_rank == 1) & (fused_rank != 1)).sum()),
              "query_ap_improved": int((fused_ap > baseline_ap + 1e-8).sum()),
              "query_ap_worsened": int((fused_ap < baseline_ap - 1e-8).sum()),
              "identity_ap_improved": sum(value > 1e-8 for value in identity_changes),
              "identity_ap_worsened": sum(value < -1e-8 for value in identity_changes),
              "evidence_boundary": "Post-selection official diagnosis; component removal is not independently trained ablation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
