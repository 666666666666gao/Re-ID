#!/usr/bin/env python3
"""Registered real-input operator regression and full-batch check of local FP32 Gram."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run(args):
    import torch
    import torch.nn.functional as F
    from tools import train_rgbnt100_signal_oof as runner
    from tools.run_signal_preserving_v5 import _set_seed

    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    for name, digest in plan["project_files"].items():
        assert runner.sha256(ROOT / name) == digest, name
    assert runner.sha256(args.fixture) == plan["fixture_sha256"]
    assert runner.sha256(args.volume_inputs) == plan["volume_inputs_sha256"]
    config = json.loads((ROOT / plan["config"]).read_text())
    cfg, binding = runner.configure(config)
    for name, digest in plan["extra_signal_files"].items():
        assert runner.sha256(Path(config["signal_source"]) / name) == digest, name
    from utils.volume import volume_computation3
    from modeling.AddModule import useB
    from layers.make_loss import make_loss
    from tools.build_v12_complete_path_oof_targets import _signal_training_loss
    from tools.signal_gram_fp32 import signal_gram_volume_fp32

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    calls = torch.load(args.volume_inputs, map_location="cpu", weights_only=True)
    assert len(calls) == 2
    for key in calls[0]:
        assert torch.equal(calls[0][key], calls[1][key]), key
    assert fixture["fold"] == 0 and fixture["step"] == 34 and fixture["amp_scale"] == 256.0
    started = time.perf_counter()
    context = {
        "created_at": datetime.datetime.now().astimezone().isoformat(), **binding,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "plan_sha256": runner.sha256(args.plan), "fixture_sha256": plan["fixture_sha256"],
        "volume_inputs_sha256": plan["volume_inputs_sha256"],
        "optimizer_steps": 0, "image_decodes": 0, "heldout_image_forwards": 0,
        "torch_version": str(torch.__version__),
    }
    operator_rows = []
    operator_arrays = {}
    recorded_calls = []

    def trace(frame, event, arg):
        if frame.f_code is not volume_computation3.__code__:
            return None
        if event == "return":
            state = frame.f_locals
            recorded_calls.append({key: state[key].detach().cpu().clone()
                                   for key in ("G", "gram_det", "res")})
        return trace

    for mode, function in [("original_amp", volume_computation3),
                           ("local_gram_fp32", signal_gram_volume_fp32)]:
        inputs = [calls[0][name].cuda().clone().requires_grad_(True)
                  for name in ("language", "video", "audio")]
        temperature = fixture["model_state_dict"]["AlignM.contra_temp"].cuda().clone().requires_grad_(True)
        targets = torch.arange(64, device="cuda")
        recorded_calls.clear()
        sys.settrace(trace)
        with torch.autocast("cuda", dtype=torch.float16):
            volume = function(*inputs) / temperature
            volume_t = function(*inputs).T / temperature
            loss = (F.cross_entropy(-volume, targets, label_smoothing=.1) +
                    F.cross_entropy(-volume_t, targets, label_smoothing=.1)) / 2
        sys.settrace(None)
        assert len(recorded_calls) == 2 and torch.isfinite(loss).item()
        (loss * .1 * 256).backward()
        gradient_rows = [
            {"input": name, "elements": value.numel(),
             "nan_count": int(torch.isnan(value.grad).sum()),
             "inf_count": int(torch.isinf(value.grad).sum())}
            for name, value in zip(("language", "video", "audio", "temperature"), [*inputs, temperature])]
        stats = [{
            "G_dtype": str(call["G"].dtype), "zero_determinants": int((call["gram_det"] == 0).sum()),
            "negative_determinants": int((call["gram_det"] < 0).sum()),
            "min_absolute_determinant": float(call["gram_det"].abs().min()),
            "zero_coordinates": torch.nonzero(call["gram_det"] == 0).tolist(),
        } for call in recorded_calls]
        operator_rows.append({"mode": mode, "gram_loss": float(loss.detach()),
                              "gradient_counts": gradient_rows, "volume_calls": stats})
        operator_arrays[mode] = [{**call} for call in recorded_calls]
        runner.write_json(args.output_dir / "operator_results.json", {**context, "operators": operator_rows})
        del inputs, temperature, loss, volume, volume_t

    arrays_path = args.output_dir / "operator_arrays.pt"
    torch.save(operator_arrays, arrays_path)
    assert operator_rows[0]["gram_loss"] == plan["original_gram_loss"]
    assert any(row["nan_count"] + row["inf_count"] for row in operator_rows[0]["gradient_counts"])
    assert all(row["nan_count"] + row["inf_count"] == 0 for row in operator_rows[1]["gradient_counts"])
    assert all(row["zero_determinants"] == 0 for row in operator_rows[1]["volume_calls"])

    protocol = json.loads((ROOT / config["protocol"]).read_text())
    fold = protocol["folds"][0]
    _set_seed(42)
    model = runner.new_model(cfg, fold)
    model.load_state_dict(fixture["model_state_dict"], strict=True)
    images = {name: value.cuda() for name, value in fixture["images"].items()}
    labels, cameras, scenes = (fixture[name].cuda() for name in ("labels", "cameras", "scenes"))
    assert labels.numel() == 64
    model.eval()
    with torch.no_grad():
        before = model(images, cam_label=cameras, view_label=scenes, training=False,
                       sge=cfg.MODEL.stageName).float().cpu()
    assert useB.volume_computation3 is volume_computation3
    useB.volume_computation3 = signal_gram_volume_fp32
    with torch.no_grad():
        after = model(images, cam_label=cameras, view_label=scenes, training=False,
                      sge=cfg.MODEL.stageName).float().cpu()
    assert before.shape == after.shape == (64, 3072) and torch.equal(before, after)
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn, _ = make_loss(cfg, num_classes=model.num_classes)
    components = []

    def recorded_loss(**kwargs):
        value = loss_fn(**kwargs)
        components.append(float(value.detach()))
        return value

    torch.cuda.reset_peak_memory_stats()
    torch.set_rng_state(fixture["cpu_rng_state"])
    torch.cuda.set_rng_state(fixture["cuda_rng_state"])
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(images, label=labels, cam_label=cameras, view_label=scenes,
                       training=True, sge=cfg.MODEL.stageName)
        loss = _signal_training_loss(output, loss_fn=recorded_loss, labels=labels, cameras=cameras,
                                    stage=cfg.MODEL.stageName, gram_weight=cfg.MODEL.Gram_Loss_weight,
                                    patch_weight=cfg.MODEL.PAT_Loss_weight)
    forward = {"loss": float(loss.detach()), "id_triplet_head_losses": components,
               "gram_loss": float(output[-2].detach()), "patch_loss": float(output[-1].detach()),
               "scale": 256.0, "retrieval_feature_max_difference": float((before-after).abs().max()),
               "retrieval_feature_bitwise_equal": torch.equal(before, after),
               "source_record_forwards": 192, "operator_model_forwards": 0}
    runner.write_json(args.output_dir / "full_batch_forward.json", {**context, **forward})
    assert torch.isfinite(loss).item()
    assert components == plan["original_id_triplet_head_losses"]
    assert forward["patch_loss"] == plan["original_patch_loss"]
    (loss * 256.0).backward()
    gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.div_(256.0)
            gradients.append({"name": name, "elements": parameter.numel(),
                              "nan_count": int(torch.isnan(parameter.grad).sum()),
                              "inf_count": int(torch.isinf(parameter.grad).sum())})
    runner.write_json(args.output_dir / "full_batch_gradients.json", gradients)
    assert len(gradients) == 195
    assert all(row["nan_count"] + row["inf_count"] == 0 for row in gradients)
    useB.volume_computation3 = volume_computation3
    runner.write_json(args.output_dir / "summary.json", {
        **context, "status": "PASS_LOCAL_FP32_GRAM_REAL_OPERATOR_AND_BATCH_REGRESSION",
        "operators": operator_rows, "operator_arrays_sha256": runner.sha256(arrays_path),
        "full_batch": forward, "gradient_tensors": 195, "all_gradients_finite": True,
        "operator_backward_passes": 2, "full_model_backward_passes": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "limits": "one saved real failed batch, numerical repair only, no full epoch or retrieval-performance claim",
    })
    print(json.dumps({"event": "local_gram_fp32_regression_pass", "source_record_forwards": 192,
                      "optimizer_steps": 0, "all195_gradients_finite": True}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--volume-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
