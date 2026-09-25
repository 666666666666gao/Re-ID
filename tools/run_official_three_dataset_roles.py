#!/usr/bin/env python3
"""Train fixed R2/V27 endpoints and evaluate the full official query/gallery."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.official_three_dataset_data import records_for, loader_for
from tools.official_three_dataset_model import build_model, sha256
from tools.train_msvr310_trifusion_oof import OUTPUT_WIDTHS, output_mapping
from tools.train_official_three_dataset_roles import PLAIN_WIDTHS


def configure_style(model, dataset, method):
    if method == "V27":
        if dataset == "RGBNT201":
            from trifusion.source_style_v27 import SourceStyleFrozenSignalBackbone
            wrapper = SourceStyleFrozenSignalBackbone
        else:
            from trifusion.source_style_msvr import SourceStyleMSVRBackbone
            wrapper = SourceStyleMSVRBackbone
        model.baseline = wrapper(model.baseline.signal, fold=0)
    return model


def read_protocol(path, dataset):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["schema"] == "trifusion-official-three-dataset-protocol-v1"
    assert protocol["dataset"] == dataset and protocol["seed"] == 42
    assert protocol["counts"] == {
        "RGBNT201": {"train": 3951, "query": 836, "gallery": 836},
        "RGBNT100": {"train": 8675, "query": 1715, "gallery": 8575},
        "MSVR310": {"train": 1032, "query": 591, "gallery": 1055},
    }[dataset]
    assert protocol["environment_key"] == ("scene" if dataset == "MSVR310" else "camera")
    return protocol


def initialize(args, protocol):
    from tools.run_signal_preserving_v5 import _module_state_sha256

    model, cfg, config, binding = build_model(protocol, args.signal_source, args.clip_weight,
                                               args.signal_checkpoint, args.signal_sha256, seed=args.seed,
                                               plain_baseline=args.method == "PLAIN_V8")
    before = _module_state_sha256(model)
    model = configure_style(model, args.dataset, args.method)
    assert _module_state_sha256(model) == before
    return model, cfg, config, binding


def train(args, protocol):
    import torch
    from tools.train_official_three_dataset_roles import train_r2, train_v27

    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    model, _cfg, config, binding = initialize(args, protocol)
    receipt = dict(schema=("trifusion-official-plain-v8-training-v1" if args.method == "PLAIN_V8"
                           else "trifusion-official-r2-v27-training-v1"), dataset=args.dataset,
                   method=args.method, mode=args.mode, status="RUNNING",
                   started_at=datetime.now().astimezone().isoformat(),
                   commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   protocol=str(args.protocol), protocol_sha256=sha256(args.protocol),
                   source_count=protocol["counts"]["train"], seed=args.seed,
                   initializer=binding, official_model_forwards=0)
    (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    records = records_for(protocol, "train")
    if args.method in ("V27", "PLAIN_V8"):
        result = train_v27(model, protocol, records, config, m0=args.mode == "m0",
                           directory=args.output_dir, seed=args.seed,
                           style=args.method == "V27")
    else:
        result = train_r2(model, protocol, records, config, m0=args.mode == "m0",
                          directory=args.output_dir, seed=args.seed,
                          top1=args.method == "R2_TOP1",
                          balanced=args.method != "R2_UNIFORM")
    receipt["training"] = result
    receipt["status"] = "M0_PASS" if args.mode == "m0" else "FIXED_EPOCH20_TRAINING_COMPLETE"
    if args.mode == "train":
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                 if not name.startswith("baseline.")}
        path = args.output_dir / "roles_epoch20.pth"
        torch.save(dict(schema="trifusion-official-role-checkpoint-v1", dataset=args.dataset,
                        method=args.method, protocol_sha256=receipt["protocol_sha256"],
                        author_checkpoint_sha256=args.signal_sha256,
                        final_state_sha256=result["final_state_sha256"], role_state_dict=state), path)
        receipt["checkpoint"] = str(path)
        receipt["checkpoint_sha256"] = sha256(path)
    receipt["completed_at"] = datetime.now().astimezone().isoformat()
    (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=receipt["status"], dataset=args.dataset, method=args.method,
                          training=result, checkpoint=receipt.get("checkpoint"))), flush=True)


def extract(model, protocol, split, method, *, baseline_only=False):
    import torch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from tools.run_signal_preserving_v5 import _training_batch
    from tools.train_signal_preserving_v18 import image_batch

    records = records_for(protocol, split)
    model.eval()
    if method == "V27":
        model.baseline.style_plan = None
    widths = PLAIN_WIDTHS if method == "PLAIN_V8" else OUTPUT_WIDTHS
    parts = {name: [] for name in (("baseline_only",) if baseline_only else widths)}
    context = torch.inference_mode() if protocol["dataset"] == "RGBNT201" else torch.no_grad()
    with context:
        for raw in loader_for(protocol, records, training=False, method=method):
            if protocol["dataset"] == "RGBNT201":
                images, _, _, camera_ids, _, _ = raw
                batch = image_batch(images, camera_ids)
            else:
                batch, _ = _training_batch(raw)
            if baseline_only:
                values = {"baseline_only": model(batch, retrieval_output="baseline_only")}
            else:
                output = (model(batch, return_aux=True)
                          if protocol["dataset"] == "RGBNT201" or method == "PLAIN_V8"
                          else exact_signal_forward(model, batch))
                values = output_mapping(output, widths=widths)
            for name, value in values.items():
                parts[name].append(value.float().cpu())
    result = {name: torch.cat(values) for name, values in parts.items()}
    assert all(value.shape == (protocol["counts"][split], widths[name])
               for name, value in result.items())
    return result


def distance_matrix(query, gallery):
    import torch

    query = torch.nn.functional.normalize(query.float(), dim=1)
    gallery = torch.nn.functional.normalize(gallery.float(), dim=1)
    result = query.square().sum(1, keepdim=True) + gallery.square().sum(1)[None]
    result.addmm_(query, gallery.T, beta=1, alpha=-2)
    assert torch.isfinite(result).all()
    return result


def preflight_plain(args, protocol):
    import numpy as np
    import torch
    from tools.run_signal_preserving_v5 import _training_batch
    from tools.train_signal_preserving_v18 import image_batch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    assert args.method == "PLAIN_V8" and args.baseline_receipt is not None
    model, cfg, _config, binding = initialize(args, protocol)
    model.eval()
    assert model.baseline.baseline_width == 1536
    reference = json.loads(args.baseline_receipt.read_text(encoding="utf-8"))
    assert reference["dataset"] == args.dataset
    assert reference["checkpoint_sha256"] == args.signal_sha256
    raw = next(iter(loader_for(protocol, records_for(protocol, "query"),
                               training=False, method=args.method)))
    batch = (image_batch(raw[0], raw[3]) if args.dataset == "RGBNT201"
             else _training_batch(raw)[0])
    with torch.no_grad():
        author = model.baseline.signal(batch["images"], cam_label=batch["camera_ids"],
                                       training=False, sge=cfg.MODEL.stageName)
        wrapped = model(batch, retrieval_output="baseline_only")
    assert torch.equal(author, wrapped)
    query = extract(model, protocol, "query", args.method, baseline_only=True)["baseline_only"]
    gallery = extract(model, protocol, "gallery", args.method, baseline_only=True)["baseline_only"]
    qrows, grows = protocol["records"]["query"], protocol["records"]["gallery"]
    qids, gids = [np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows)]
    if args.dataset == "MSVR310":
        qenv, genv = [np.asarray([row["scene"] for row in rows]) for rows in (qrows, grows)]
        result = scene_scores(distance_matrix(query, gallery).numpy(), qids, gids, qenv, genv)
    else:
        qenv, genv = [np.asarray([row["camera"] for row in rows]) for rows in (qrows, grows)]
        result = camera_scores(distance_matrix(query, gallery).numpy(), qids, gids, qenv, genv)
    errors = {name: result["metrics"][name] - value for name, value in reference["metrics"].items()}
    assert all(abs(value) < 1e-4 for value in errors.values()), errors
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "baseline_parity.json").write_text(json.dumps(dict(
        status="PASS", dataset=args.dataset, seed=args.seed,
        checkpoint_sha256=args.signal_sha256, baseline_receipt=str(args.baseline_receipt),
        baseline_receipt_sha256=sha256(args.baseline_receipt),
        initializer=binding, single_batch_author_forward_bitwise_equal=True,
        metrics=result["metrics"], errors=errors), indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(dict(status="PASS", dataset=args.dataset, metrics=result["metrics"])), flush=True)


def evaluate(args, protocol):
    import numpy as np
    import torch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores
    from tools.run_signal_preserving_v5 import _module_state_sha256

    summary_path = args.output_dir / "training.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
    assert summary["dataset"] == args.dataset and summary["method"] == args.method
    assert summary["seed"] == args.seed
    assert summary["protocol_sha256"] == sha256(args.protocol)
    assert summary["initializer"]["author_checkpoint_sha256"] == args.signal_sha256
    assert sha256(summary["checkpoint"]) == summary["checkpoint_sha256"]
    assert not (args.output_dir / "official_metrics.json").exists()
    model, _cfg, _config, binding = initialize(args, protocol)
    from utils import metrics as upstream_metrics

    assert Path(upstream_metrics.__file__).resolve() == (
        args.signal_source / "utils" / "metrics.py").resolve()
    assert binding == summary["initializer"]
    payload = torch.load(summary["checkpoint"], map_location="cpu", weights_only=True)
    assert payload["dataset"] == args.dataset and payload["method"] == args.method
    assert payload["protocol_sha256"] == summary["protocol_sha256"]
    assert payload["author_checkpoint_sha256"] == args.signal_sha256
    state = model.state_dict()
    assert set(payload["role_state_dict"]) == {name for name in state if not name.startswith("baseline.")}
    state.update(payload["role_state_dict"])
    model.load_state_dict(state, strict=True)
    assert _module_state_sha256(model) == summary["training"]["final_state_sha256"]
    query = extract(model, protocol, "query", args.method)
    gallery = extract(model, protocol, "gallery", args.method)
    assert _module_state_sha256(model) == summary["training"]["final_state_sha256"]
    qrows, grows = protocol["records"]["query"], protocol["records"]["gallery"]
    qids, gids = [np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows)]
    qcameras, gcameras = [np.asarray([row["camera"] for row in rows]) for rows in (qrows, grows)]
    qscenes, gscenes = [np.asarray([row["scene"] for row in rows]) for rows in (qrows, grows)]
    scores, arrays = {}, {}
    os.chdir(args.output_dir)
    for name in (PLAIN_WIDTHS if args.method == "PLAIN_V8" else OUTPUT_WIDTHS):
        distances = distance_matrix(query[name], gallery[name])
        assert distances.shape == (len(qrows), len(grows))
        if args.dataset == "MSVR310":
            row = scene_scores(distances.numpy(), qids, gids, qscenes, gscenes)
            cmc, mean_ap = upstream_metrics.eval_func_msrv(
                distances.numpy(), qids, gids, qcameras, gcameras, qscenes, gscenes)
        else:
            row = camera_scores(distances.numpy(), qids, gids, qcameras, gcameras)
            cmc, mean_ap = upstream_metrics.eval_func(
                distances.numpy(), qids, gids, qcameras, gcameras)
        assert abs(row["metrics"]["mAP"] - mean_ap * 100) < 1e-10
        assert all(abs(row["metrics"][f"Rank-{rank}"] - cmc[rank - 1] * 100) < 1e-5
                   for rank in (1, 5, 10))
        scores[name] = row
        arrays[name] = distances
    path = args.output_dir / "official_distances.pt"
    torch.save(dict(distances=arrays, query_ids=qids, gallery_ids=gids,
                    query_cameras=qcameras, gallery_cameras=gcameras,
                    query_scenes=qscenes, gallery_scenes=gscenes,
                    protocol_sha256=summary["protocol_sha256"]), path)
    result = dict(schema=("trifusion-official-plain-v8-retrieval-v1" if args.method == "PLAIN_V8"
                          else "trifusion-official-r2-v27-retrieval-v1"), status="COMPLETE",
                  dataset=args.dataset, method=args.method,
                  query_count=len(qrows), gallery_count=len(grows),
                  author_checkpoint_sha256=args.signal_sha256,
                  role_checkpoint_sha256=summary["checkpoint_sha256"],
                  protocol_sha256=summary["protocol_sha256"],
                  model_state_sha256=summary["training"]["final_state_sha256"],
                  fixed_epoch=20, seed=args.seed, reranking=False,
                  filter=protocol["filter"], outputs=scores,
                  distance_arrays=str(path), distance_arrays_sha256=sha256(path),
                  independent_upstream_metrics_equal=True,
                  completed_at=datetime.now().astimezone().isoformat())
    (args.output_dir / "official_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE", dataset=args.dataset, method=args.method,
                          metrics={name: row["metrics"] for name, row in scores.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("RGBNT201", "RGBNT100", "MSVR310"), required=True)
    parser.add_argument("--method", choices=("R2", "V27", "R2_TOP1", "R2_UNIFORM", "PLAIN_V8"), required=True)
    parser.add_argument("--mode", choices=("preflight", "m0", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--signal-source", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--signal-checkpoint", type=Path, required=True)
    parser.add_argument("--signal-sha256", required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("protocol", "signal_source", "clip_weight", "signal_checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    protocol = read_protocol(args.protocol, args.dataset)
    if args.mode == "preflight":
        preflight_plain(args, protocol)
    elif args.mode == "evaluate":
        evaluate(args, protocol)
    else:
        train(args, protocol)


if __name__ == "__main__":
    main()
