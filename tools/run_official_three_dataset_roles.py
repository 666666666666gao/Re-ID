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
from tools.train_official_three_dataset_roles import PLAIN_WIDTHS, SIM_JOINT_LOWLR

SIM_JOINT_METHODS = ("SIGNAL_SIM_JOINT", "SIGNAL_SIM_JOINT_LOWLR")
FEEDBACK_METHODS = ("SIGNAL_SIM_FEEDBACK", "SIGNAL_SIM_FEEDBACK_MATCHED")


def configure_style(model, dataset, method):
    if method in ("V27", "PLAIN_V27"):
        if dataset == "RGBNT201":
            from trifusion.source_style_v27 import SourceStyleFrozenSignalBackbone
            wrapper = SourceStyleFrozenSignalBackbone
        else:
            from trifusion.source_style_msvr import SourceStyleMSVRBackbone
            wrapper = SourceStyleMSVRBackbone
        model.baseline = wrapper(model.baseline.signal, fold=0)
    return model


def checkpoint_names(model, method):
    return {name for name in model.state_dict()
            if not name.startswith("baseline.") or
            (method in SIM_JOINT_METHODS and
             name.startswith("baseline.signal.SIM.modal_interactive."))}


def save_role_checkpoint(path, model, args, receipt, state_sha256):
    import torch

    names = checkpoint_names(model, args.method)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
             if name in names}
    torch.save(dict(schema=("trifusion-official-sim-joint-checkpoint-v1"
                            if args.method in SIM_JOINT_METHODS
                            else "trifusion-official-role-checkpoint-v1"), dataset=args.dataset,
                    method=args.method, protocol_sha256=receipt["protocol_sha256"],
                    author_checkpoint_sha256=args.signal_sha256,
                    final_state_sha256=state_sha256, role_state_dict=state), path)


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
                                               plain_baseline=args.method in ("PLAIN_V8", "PLAIN_V27"),
                                               sim_feedback=args.method in FEEDBACK_METHODS,
                                               matched_feedback_reference=args.method == "SIGNAL_SIM_FEEDBACK_MATCHED")
    if args.method in SIM_JOINT_METHODS:
        names = []
        for name, parameter in model.baseline.signal.SIM.modal_interactive.named_parameters():
            parameter.requires_grad_(True)
            names.append(f"baseline.signal.SIM.modal_interactive.{name}")
        binding["joint_signal_parameters"] = names
        if args.method == "SIGNAL_SIM_JOINT_LOWLR":
            binding["joint_signal_base_lr"] = SIM_JOINT_LOWLR
    before = _module_state_sha256(model)
    model = configure_style(model, args.dataset, args.method)
    assert _module_state_sha256(model) == before
    return model, cfg, config, binding


def train(args, protocol):
    from tools.train_official_three_dataset_roles import train_r2, train_v27
    from tools.run_signal_preserving_v5 import _module_state_sha256

    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    model, _cfg, config, binding = initialize(args, protocol)
    receipt = dict(schema=("trifusion-official-plain-v8-training-v1" if args.method in ("PLAIN_V8", "PLAIN_V27")
                           else "trifusion-official-signal-sim-joint-training-v1" if args.method in SIM_JOINT_METHODS
                           else "trifusion-official-signal-sim-feedback-training-v1" if args.method in FEEDBACK_METHODS
                           else "trifusion-official-signal-v8-training-v1" if args.method == "SIGNAL_V8"
                           else "trifusion-official-r2-v27-training-v1"), dataset=args.dataset,
                   method=args.method, mode=args.mode, status="RUNNING",
                   started_at=datetime.now().astimezone().isoformat(),
                   commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   protocol=str(args.protocol), protocol_sha256=sha256(args.protocol),
                   source_count=protocol["counts"]["train"], seed=args.seed,
                   initializer=binding, official_epoch_evaluations=0,
                   checkpoint_policy=args.checkpoint_policy)
    (args.output_dir / "training.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    records = records_for(protocol, "train")
    best = dict(mAP=-1.0, epoch=None, checkpoint_state_sha256=None)

    def select_epoch(model, epoch):
        metrics = official_fused_metrics(model, protocol, args.method, args.signal_source)
        receipt["official_epoch_evaluations"] += 1
        row = dict(epoch=epoch, metrics=metrics)
        with (args.output_dir / "epoch_official_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if metrics["mAP"] >= best["mAP"]:
            state_sha256 = _module_state_sha256(model)
            path = args.output_dir / ("joint_best_map.pth" if args.method in SIM_JOINT_METHODS
                                      else "roles_best_map.pth")
            save_role_checkpoint(path, model, args, receipt, state_sha256)
            best.update(mAP=metrics["mAP"], epoch=epoch,
                        checkpoint_state_sha256=state_sha256, checkpoint=str(path))
        print(json.dumps(dict(event="official_epoch_selection", dataset=args.dataset,
                              method=args.method, **row, best_epoch=best["epoch"],
                              best_mAP=best["mAP"])), flush=True)

    on_epoch_end = (select_epoch if args.mode == "train" and
                    args.checkpoint_policy == "best_official_map" else None)
    if args.method in ("V27", "PLAIN_V27", "PLAIN_V8", "SIGNAL_V8", *SIM_JOINT_METHODS, *FEEDBACK_METHODS):
        result = train_v27(model, protocol, records, config, m0=args.mode == "m0",
                           directory=args.output_dir, seed=args.seed,
                           style=args.method in ("V27", "PLAIN_V27"),
                           plain_baseline=args.method in ("PLAIN_V8", "PLAIN_V27"),
                           joint_sim=args.method in SIM_JOINT_METHODS,
                           joint_sim_low_lr=args.method == "SIGNAL_SIM_JOINT_LOWLR",
                           sim_feedback=args.method in FEEDBACK_METHODS,
                           on_epoch_end=on_epoch_end)
    else:
        result = train_r2(model, protocol, records, config, m0=args.mode == "m0",
                          directory=args.output_dir, seed=args.seed,
                          top1=args.method == "R2_TOP1",
                          balanced=args.method != "R2_UNIFORM",
                          on_epoch_end=on_epoch_end)
    receipt["training"] = result
    receipt["status"] = ("M0_PASS" if args.mode == "m0" else
                         "BEST_OFFICIAL_MAP_TRAINING_COMPLETE" if on_epoch_end else
                         "FIXED_EPOCH20_TRAINING_COMPLETE")
    if args.mode == "train":
        if on_epoch_end:
            assert best["epoch"] is not None
            path = Path(best["checkpoint"])
            receipt["selected_epoch"] = best["epoch"]
            receipt["selected_mAP"] = best["mAP"]
            receipt["checkpoint_state_sha256"] = best["checkpoint_state_sha256"]
        else:
            path = args.output_dir / ("joint_epoch20.pth" if args.method in SIM_JOINT_METHODS
                                      else "roles_epoch20.pth")
            save_role_checkpoint(path, model, args, receipt, result["final_state_sha256"])
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
    if method in ("V27", "PLAIN_V27"):
        model.baseline.style_plan = None
    widths = PLAIN_WIDTHS if method in ("PLAIN_V8", "PLAIN_V27") else OUTPUT_WIDTHS
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
                          if protocol["dataset"] == "RGBNT201" or method in ("PLAIN_V8", "PLAIN_V27", *SIM_JOINT_METHODS, *FEEDBACK_METHODS)
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


def official_fused_metrics(model, protocol, method, signal_source):
    import numpy as np
    from utils import metrics as upstream_metrics

    assert Path(upstream_metrics.__file__).resolve() == (
        signal_source / "utils" / "metrics.py").resolve()
    query = extract(model, protocol, "query", method)["fused"]
    gallery = extract(model, protocol, "gallery", method)["fused"]
    qrows, grows = protocol["records"]["query"], protocol["records"]["gallery"]
    qids, gids = [np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows)]
    qcameras, gcameras = [np.asarray([row["camera"] for row in rows]) for rows in (qrows, grows)]
    distances = distance_matrix(query, gallery).numpy()
    if protocol["dataset"] == "MSVR310":
        qscenes, gscenes = [np.asarray([row["scene"] for row in rows]) for rows in (qrows, grows)]
        cmc, mean_ap = upstream_metrics.eval_func_msrv(
            distances, qids, gids, qcameras, gcameras, qscenes, gscenes)
    else:
        cmc, mean_ap = upstream_metrics.eval_func(
            distances, qids, gids, qcameras, gcameras)
    return {"mAP": 100 * float(mean_ap), "Rank-1": 100 * float(cmc[0]),
            "Rank-5": 100 * float(cmc[4]), "Rank-10": 100 * float(cmc[9])}


def preflight_plain(args, protocol):
    import numpy as np
    import torch
    from tools.run_signal_preserving_v5 import _training_batch
    from tools.train_signal_preserving_v18 import image_batch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    assert args.method in ("PLAIN_V8", "PLAIN_V27") and args.baseline_receipt is not None
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
    assert summary["status"] == ("BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
                                  if args.checkpoint_policy == "best_official_map"
                                  else "FIXED_EPOCH20_TRAINING_COMPLETE")
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
    assert payload["schema"] == ("trifusion-official-sim-joint-checkpoint-v1"
                                 if args.method in SIM_JOINT_METHODS
                                 else "trifusion-official-role-checkpoint-v1")
    assert payload["dataset"] == args.dataset and payload["method"] == args.method
    assert payload["protocol_sha256"] == summary["protocol_sha256"]
    assert payload["author_checkpoint_sha256"] == args.signal_sha256
    state = model.state_dict()
    assert set(payload["role_state_dict"]) == checkpoint_names(model, args.method)
    state.update(payload["role_state_dict"])
    model.load_state_dict(state, strict=True)
    expected_state = (summary["checkpoint_state_sha256"]
                      if args.checkpoint_policy == "best_official_map"
                      else summary["training"]["final_state_sha256"])
    assert payload["final_state_sha256"] == expected_state
    assert _module_state_sha256(model) == expected_state
    query = extract(model, protocol, "query", args.method)
    gallery = extract(model, protocol, "gallery", args.method)
    assert _module_state_sha256(model) == expected_state
    qrows, grows = protocol["records"]["query"], protocol["records"]["gallery"]
    qids, gids = [np.asarray([row["identity"] for row in rows]) for rows in (qrows, grows)]
    qcameras, gcameras = [np.asarray([row["camera"] for row in rows]) for rows in (qrows, grows)]
    qscenes, gscenes = [np.asarray([row["scene"] for row in rows]) for rows in (qrows, grows)]
    scores, arrays = {}, {}
    os.chdir(args.output_dir)
    for name in (PLAIN_WIDTHS if args.method in ("PLAIN_V8", "PLAIN_V27") else OUTPUT_WIDTHS):
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
    result = dict(schema=("trifusion-official-plain-v8-retrieval-v1" if args.method in ("PLAIN_V8", "PLAIN_V27")
                          else "trifusion-official-signal-sim-joint-retrieval-v1" if args.method in SIM_JOINT_METHODS
                          else "trifusion-official-signal-sim-feedback-retrieval-v1" if args.method in FEEDBACK_METHODS
                          else "trifusion-official-signal-v8-retrieval-v1" if args.method == "SIGNAL_V8"
                          else "trifusion-official-r2-v27-retrieval-v1"), status="COMPLETE",
                  dataset=args.dataset, method=args.method,
                  query_count=len(qrows), gallery_count=len(grows),
                  author_checkpoint_sha256=args.signal_sha256,
                  role_checkpoint_sha256=summary["checkpoint_sha256"],
                  protocol_sha256=summary["protocol_sha256"],
                  model_state_sha256=expected_state,
                  fixed_epoch=20 if args.checkpoint_policy == "fixed_final_epoch" else None,
                  checkpoint_policy=args.checkpoint_policy,
                  selected_epoch=summary["selected_epoch"] if args.checkpoint_policy == "best_official_map" else None,
                  seed=args.seed, reranking=False,
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
    parser.add_argument("--method", choices=("R2", "V27", "R2_TOP1", "R2_UNIFORM", "PLAIN_V8", "PLAIN_V27", "SIGNAL_V8", *SIM_JOINT_METHODS, *FEEDBACK_METHODS), required=True)
    parser.add_argument("--mode", choices=("preflight", "m0", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--signal-source", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--signal-checkpoint", type=Path, required=True)
    parser.add_argument("--signal-sha256", required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-policy", choices=("fixed_final_epoch", "best_official_map"),
                        default="fixed_final_epoch")
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
