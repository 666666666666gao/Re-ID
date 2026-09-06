#!/usr/bin/env python3
"""Zero-update precision/anomaly checks of the saved real RGBNT100 failed batch."""
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
    from tools import train_rgbnt100_signal_oof as runner
    from tools.run_signal_preserving_v5 import _set_seed

    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    assert runner.sha256(__file__) == plan["probe_sha256"]
    assert runner.sha256(args.fixture) == plan["fixture_sha256"]
    config_path = ROOT / plan["config"]
    assert runner.sha256(config_path) == plan["config_sha256"]
    config = json.loads(config_path.read_text())
    cfg, binding = runner.configure(config)
    from tools.build_v12_complete_path_oof_targets import _signal_training_loss
    for name, digest in plan["extra_signal_files"].items():
        assert runner.sha256(Path(config["signal_source"]) / name) == digest, name
    from layers.make_loss import make_loss
    from utils.volume import volume_computation3

    protocol = json.loads((ROOT / config["protocol"]).read_text())
    fold = protocol["folds"][0]
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    assert fixture["fold"] == 0 and fixture["step"] == 34 and fixture["amp_scale"] == 256.0
    assert fixture["source_ids"] == fold["source_ids"]
    assert set(fixture["source_record_indices"]) <= set(fold["source_record_indices"])
    _set_seed(42)
    model = runner.new_model(cfg, fold)
    model.load_state_dict(fixture["model_state_dict"], strict=True)
    model.train()
    loss_fn, _ = make_loss(cfg, num_classes=model.num_classes)
    images = {name: value.cuda() for name, value in fixture["images"].items()}
    labels, cameras, scenes = (fixture[name].cuda() for name in ("labels", "cameras", "scenes"))
    assert labels.numel() == 64
    model.zero_grad(set_to_none=True)
    volume_calls = []
    components = []

    def trace(frame, event, arg):
        if frame.f_code is not volume_computation3.__code__:
            return None
        if event == "return":
            values = frame.f_locals
            volume_calls.append({name: values[name].detach().cpu().clone()
                                 for name in ("language", "video", "audio", "G", "gram_det", "res")})
        return trace

    def recorded_loss(**kwargs):
        value = loss_fn(**kwargs)
        components.append(float(value.detach()))
        return value

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    torch.set_rng_state(fixture["cpu_rng_state"])
    torch.cuda.set_rng_state(fixture["cuda_rng_state"])
    with torch.autograd.set_detect_anomaly(args.mode == "fp16_anomaly"):
        sys.settrace(trace)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.mode != "fp32"):
            output = model(images, label=labels, cam_label=cameras, view_label=scenes,
                           training=True, sge=cfg.MODEL.stageName)
            loss = _signal_training_loss(output, loss_fn=recorded_loss, labels=labels, cameras=cameras,
                                        stage=cfg.MODEL.stageName, gram_weight=cfg.MODEL.Gram_Loss_weight,
                                        patch_weight=cfg.MODEL.PAT_Loss_weight)
        sys.settrace(None)
        assert torch.isfinite(loss).item() and len(volume_calls) == 2
        arrays_path = args.output_dir / "volume_inputs.pt"
        torch.save(volume_calls, arrays_path)
        volume_stats = []
        for call in volume_calls:
            determinant = call["gram_det"]
            assert torch.isfinite(determinant).all().item()
            volume_stats.append({
                "G_dtype": str(call["G"].dtype), "input_dtype": str(call["language"].dtype),
                "determinant_dtype": str(determinant.dtype), "shape": list(determinant.shape),
                "exact_zero_count": int((determinant == 0).sum()),
                "negative_count": int((determinant < 0).sum()),
                "min_absolute_determinant": float(determinant.abs().min()),
                "max_absolute_determinant": float(determinant.abs().max()),
                "zero_coordinates": torch.nonzero(determinant == 0).tolist(),
            })
        forward = {
            "mode": args.mode, "created_at": datetime.datetime.now().astimezone().isoformat(),
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "probe_sha256": runner.sha256(__file__), "fixture_sha256": plan["fixture_sha256"],
            "torch_version": str(torch.__version__),
            "plan_sha256": runner.sha256(args.plan), "loss": float(loss.detach()),
            "id_triplet_head_losses": components, "gram_loss": float(output[-2].detach()),
            "patch_loss": float(output[-1].detach()), "scale": 256.0,
            "capture_loss_exact": float(loss.detach()) == plan["capture_loss"],
            "capture_loss_difference": float(loss.detach()) - plan["capture_loss"],
            "volume_calls": volume_stats, "volume_inputs_sha256": runner.sha256(arrays_path),
            "source_record_forwards": 64, "optimizer_steps": 0,
            "heldout_image_forwards": 0, "image_decodes": 0, **binding,
        }
        runner.write_json(args.output_dir / "forward.json", forward)
        (loss * 256.0).backward()
    gradient_rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.div_(256.0)
            gradient = parameter.grad.detach()
            gradient_rows.append({
                "name": name, "elements": gradient.numel(),
                "nan_count": int(torch.isnan(gradient).sum()),
                "positive_inf_count": int(torch.isposinf(gradient).sum()),
                "negative_inf_count": int(torch.isneginf(gradient).sum()),
            })
    bad = [row["name"] for row in gradient_rows
           if row["nan_count"] + row["positive_inf_count"] + row["negative_inf_count"]]
    runner.write_json(args.output_dir / "result.json", {
        **forward, "status": "NONFINITE_GRADIENT_REPRODUCED" if bad else "ALL_GRADIENTS_FINITE",
        "all_gradient_counts": gradient_rows, "nonfinite_gradient_names": bad,
        "gradient_tensors": len(gradient_rows), "backward_passes": 1,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(json.dumps({"mode": args.mode, "loss": float(loss.detach()),
                      "nonfinite_gradient_tensors": len(bad), "optimizer_steps": 0}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fp16", "fp16_anomaly", "fp32"], required=True)
    run(parser.parse_args())
