#!/usr/bin/env python3
"""Read attention/slot statistics from each accepted full-gallery endpoint."""

import argparse
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_slot_competition_roles as patch
from tools.collect_correspondence_roles import sha
from trifusion.slot_competition_roles import allocation_weights
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores

STAT_NAMES = ("entropy_fraction", "effective_patch_count", "maximum_attention",
              "attention_slot_cosine", "selected_content_slot_cosine",
              "distinct_top1_patches")


def measure(roles, patches, context, role, selected):
    width = patches.shape[-1]
    queries = roles.anchor_queries[None] + context[:, role, None]
    queries = roles.query_projections[role](F.layer_norm(queries, (width,)))
    keys = roles.key_projections[role](F.layer_norm(patches, (width,)))
    scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
    weights = allocation_weights(scores, roles.attention_normalization).to(patches.dtype).float()
    entropy = -torch.special.xlogy(weights, weights).sum(dim=-1)
    off_diagonal = ~torch.eye(roles.anchor_count, dtype=torch.bool, device=weights.device)
    normalized = F.normalize(weights, dim=-1)
    attention_cosine = (normalized @ normalized.transpose(-1, -2))[..., off_diagonal].mean(-1)
    contents = F.normalize(selected.float(), dim=-1)
    content_cosine = (contents @ contents.transpose(-1, -2))[..., off_diagonal].mean(-1)
    top1 = weights.argmax(-1).sort(-1).values
    distinct = 1 + (top1[..., 1:] != top1[..., :-1]).sum(-1)
    support_size = 128
    return torch.stack((entropy.mean(-1) / math.log(support_size),
                        entropy.exp().mean(-1), weights.max(-1).values.mean(-1),
                        attention_cosine, content_cosine, distinct.float()), dim=-1).cpu()


def analyze(row, protocol, output_dir):
    entry = patch.entry
    patch.NORMALIZATION = row["attention_normalization"]
    entry.CONDITION.update(query_mode="context", auxiliary_target="none")
    entry.runner.CorrespondenceTriFusion = partial(
        patch.SlotCompetitionTriFusion, attention_normalization=patch.NORMALIZATION, **entry.CONDITION)
    weight, digest = BASELINES[row["dataset"]]
    args = SimpleNamespace(dataset=row["dataset"], seed=42, epochs=50, width=128,
                           m1=True, m2=True, m3=False, pred_weight=0.1,
                           protocol=PROTOCOLS / (row["dataset"] + ".json"),
                           signal_source=SOURCE, clip_weight=WEIGHTS / "ViT-B-16.pt",
                           baseline_checkpoint=WEIGHTS / weight, baseline_sha256=digest,
                           output_dir=Path(row["run_dir"]), mode="evaluate")
    training = json.loads((args.output_dir / "training.json").read_text())
    assert sha(args.output_dir / "best_map.pth") == row["checkpoint_sha256"]
    assert sha(args.output_dir / "official_distances.pt") == row["distance_sha256"]
    model, _, _, binding = patch.build(args, protocol)
    assert binding == training["initializer"]
    payload = patch.load_checkpoint(args.output_dir / "best_map.pth", model, args, protocol)
    assert payload["epoch"] == row["best_epoch"]
    model.eval()
    model_hash = entry._module_state_sha256(model)
    original_sample = model.roles.sample_context
    sums = torch.zeros(3, 3, len(STAT_NAMES), dtype=torch.float64)
    calls = torch.zeros(3, dtype=torch.int64)

    def instrumented(roles, patches, positions, context, role):
        selected = original_sample(patches, positions, context, role)
        values = measure(roles, patches, context, role, selected)
        assert torch.isfinite(values).all()
        sums[role] += values.double().sum(0)
        calls[role] += patches.shape[0]
        return selected

    embeddings, statistics, parity = {}, {}, {}
    with torch.inference_mode():
        for split in ("query", "gallery"):
            sums.zero_()
            calls.zero_()
            chunks = []
            for index, raw in enumerate(entry.loader_for(
                    protocol, entry.records_for(protocol, split), training=False, method="PLAIN_V8")):
                batch = entry.runner._eval_batch(raw, args.dataset)
                if index == 0:
                    model.roles.sample_context = original_sample
                    original = model(batch)
                model.roles.sample_context = MethodType(instrumented, model.roles)
                output = model(batch)
                if index == 0:
                    parity[split] = float((original - output).abs().max())
                    assert torch.allclose(original, output, atol=1e-5, rtol=1e-5)
                chunks.append(output.float().cpu())
            count = protocol["counts"][split]
            assert calls.tolist() == [count] * 3
            embeddings[split] = torch.cat(chunks)
            assert embeddings[split].shape == (count, 1536)
            statistics[split] = {"rows": count, "role_modality_mean": (sums / count).tolist()}
    assert entry._module_state_sha256(model) == model_hash
    distances = entry.runner.distance_matrix(embeddings["query"], embeddings["gallery"]).numpy()
    environment = "scene" if args.dataset == "MSVR310" else "camera"
    scorer = scene_scores if args.dataset == "MSVR310" else camera_scores
    qrows, grows = (protocol["records"][split] for split in ("query", "gallery"))
    scores = scorer(distances, np.asarray([r["identity"] for r in qrows]),
                    np.asarray([r["identity"] for r in grows]),
                    np.asarray([r[environment] for r in qrows]),
                    np.asarray([r[environment] for r in grows]))
    difference = {key: abs(value - row["metrics"][key]) for key, value in scores["metrics"].items()}
    assert max(difference.values()) < 1e-5
    report = {"status": "COMPLETE_FULL_GALLERY_SLOT_DIAGNOSIS", "dataset": args.dataset,
              "variant": row["variant"], "completed_at": datetime.now().astimezone().isoformat(),
              "checkpoint_sha256": row["checkpoint_sha256"], "best_epoch": payload["epoch"],
              "protocol_sha256": sha(args.protocol), "source_sha256": sha(Path(__file__)),
              "roles": ["CNN", "Transformer", "Mamba"], "modalities": ["RGB", "NIR", "TIR"],
              "statistic_names": list(STAT_NAMES), "statistics": statistics,
              "first_batch_instrumentation_max_difference": parity,
              "full_gallery_metric_difference_pp": difference, "model_state_unchanged": True,
              "boundary": "All official query and gallery inputs from one selected fixed checkpoint. "
                          "Hook returns the original sample output; statistics recompute its attention. "
                          "No optimizer, augmentation, model/weight update, reranking or inference selection. "
                          "Slot similarity is not proof of identity-task redundancy or a unique cause."}
    target = output_dir / (row["variant"] + ".json")
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variant": row["variant"], "query_rows": statistics["query"]["rows"],
                      "gallery_rows": statistics["gallery"]["rows"], "max_replay_error_pp": max(difference.values())}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows = [r for r in matrix["rows"] if r["dataset"] == args.dataset]
    assert {r["variant"] for r in rows} == {"independent", "competitive"}
    assert all(r["status"] == "VERIFIED_COMPLETE" for r in rows)
    assert len({r["initial_model_state_sha256"] for r in rows}) == 1
    assert len({r["trainable_parameters"] for r in rows}) == 1
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    protocol = patch.entry.runner.read_protocol(PROTOCOLS / (args.dataset + ".json"), args.dataset)
    for row in rows:
        analyze(row, protocol, args.output_dir)
    summary = {"status": "COMPLETE_PAIRED_SLOT_DIAGNOSIS", "dataset": args.dataset,
               "completed_at": datetime.now().astimezone().isoformat(),
               "matrix_sha256": sha(args.matrix), "source_sha256": sha(Path(__file__)),
               "endpoint_reports": {r["variant"]: sha(args.output_dir / (r["variant"] + ".json")) for r in rows}}
    (args.output_dir / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
