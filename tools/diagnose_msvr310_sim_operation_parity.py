#!/usr/bin/env python3
"""Fixed-input, zero-update localization of the observed MSVR310 SIM drift."""

import argparse
from collections import Counter
import datetime
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    import torch
    from tools.train_msvr310_signal_oof import configure, loader_for, new_model, records_for, sha256, write_json
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch

    started = time.perf_counter()
    run = args.run_root.resolve()
    output = run / "sim_operation_parity_diagnosis"
    assert not output.exists()
    assert (run / "comparison_exit.txt").read_text().strip() == "1"
    prior = json.loads((run / "baseline_parity_diagnosis/summary.json").read_text())
    assert prior["baseline_comparisons"]["standalone_before_wrapping"]["bitwise_equal"]
    assert not prior["baseline_comparisons"]["standalone_after_wrapping"]["bitwise_equal"]
    config_path = ROOT / "configs/MSVR310/TriFusion-source-oof-v1.json"
    assert sha256(config_path) == "49d82696fbf5d0d71431ed8d7f1e5f80d7c13215a8bec6cafe77969ee8e276a5"
    config = json.loads(config_path.read_text())
    base = json.loads((ROOT / config["BASELINE"]["CONFIG"]).read_text())
    cfg, binding = configure(base)
    baseline_path = Path(config["BASELINE"]["SUMMARY"])
    assert sha256(baseline_path) == config["BASELINE"]["SUMMARY_SHA256"]
    baseline = json.loads(baseline_path.read_text())
    protocol = json.loads((ROOT / base["protocol"]).read_text())
    fold, b0 = protocol["folds"][0], baseline["folds"][0]
    checkpoint = run / "comparison/fold_0/roles_epoch20.pth"
    assert sha256(checkpoint) == "b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c"
    assert sha256(b0["checkpoint"]) == b0["checkpoint_sha256"]
    original_path = Path(b0["checkpoint"]).parent / "retrieval_arrays.pt"
    assert sha256(original_path) == b0["retrieval"]["retrieval_arrays_sha256"]
    original = torch.load(original_path, map_location="cpu", weights_only=True)["features"][:64]
    _set_seed(42)
    signal = new_model(cfg, fold)
    payload = torch.load(b0["checkpoint"], map_location="cpu", weights_only=True)
    signal.load_state_dict(payload["model_state_dict"], strict=True)
    signal.eval()
    state_sha = _module_state_sha256(signal)
    assert state_sha == b0["training"]["final_state_sha256"]
    initial_flags = {n: p.requires_grad for n, p in signal.named_parameters()}
    records = records_for(base, protocol, fold, False)
    raw = next(iter(loader_for(records[:64], False)))
    assert len(raw[1]) == 64
    batch, _ = _training_batch(raw)
    output.mkdir()

    def flat(value):
        if isinstance(value, torch.Tensor):
            return [value]
        assert isinstance(value, (tuple, list)), type(value)
        return [tensor for item in value for tensor in flat(item)]

    def metadata(tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "device": str(tensor.device),
                "storage_offset": tensor.storage_offset(), "data_ptr": tensor.data_ptr(),
                "requires_grad": tensor.requires_grad, "contiguous": tensor.is_contiguous()}

    def difference(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype
        delta = (a.double() - b.double()).abs()
        return {"bitwise_equal": torch.equal(a, b), "different_elements": int((a != b).sum()),
                "maximum_absolute_difference": float(delta.max()),
                "mean_absolute_difference": float(delta.mean())}

    def backend():
        return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
                "num_threads": torch.get_num_threads(), "grad_enabled": torch.is_grad_enabled(),
                "cuda_autocast_enabled": torch.is_autocast_enabled("cuda")}

    stages, reference, saved_arrays = [], {}, {}
    fixed_inputs = []

    def capture_inputs(_module, inputs):
        fixed_inputs.extend(inputs)

    def observe(label, call):
        captured, handles = {}, []

        def hook(name):
            def save(_module, inputs, result):
                assert name not in captured
                captured[name] = {"input": flat(inputs), "output": flat(result)}
            return save

        for name, module in signal.SIM.named_modules():
            handles.append(module.register_forward_hook(hook(name or "SIM")))
        with torch.no_grad(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            value = call()
            torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        trace_path = output / (label + ".trace.json")
        profiler.export_chrome_trace(str(trace_path))
        rows, arrays = {}, {}
        for name, sides in captured.items():
            rows[name], arrays[name] = {}, {}
            for side, tensors in sides.items():
                copies = [t.detach().cpu().clone() for t in tensors]
                arrays[name][side] = copies
                rows[name][side] = [{"metadata": metadata(t),
                    "vs_initial": difference(copy, reference[name][side][i]) if reference else None}
                    for i, (t, copy) in enumerate(zip(tensors, copies, strict=True))]
        if not reference:
            reference.update(arrays)
            saved_arrays["initial_module_tensors"] = arrays
        if label == "08_original_signal_after_load":
            saved_arrays["final_module_tensors"] = arrays
        counts = Counter((str(e.device_type), e.name) for e in profiler.events())
        stages.append({"name": label, "modules": rows, "backend": backend(),
                       "signal_state_unchanged": _module_state_sha256(signal) == state_sha,
                       "sim_parameters": {n: metadata(p) for n, p in signal.SIM.named_parameters()},
                       "signal_requires_grad_names": [n for n, p in signal.named_parameters() if p.requires_grad],
                       "trace_sha256": sha256(trace_path),
                       "operator_counts": [{"device": d, "name": n, "count": c}
                                           for (d, n), c in sorted(counts.items())]})
        return value.detach().float().cpu()

    def original_signal():
        return signal(batch["images"], cam_label=batch["camera_ids"], view_label=raw[3].cuda(),
                      training=False, sge=cfg.MODEL.stageName)

    handle = signal.SIM.register_forward_pre_hook(capture_inputs)
    first = observe("00_original_signal", original_signal)
    handle.remove()
    assert len(fixed_inputs) == 6
    saved_arrays["fixed_sim_inputs"] = [x.detach().cpu() for x in fixed_inputs]
    values = {"00_original_signal": first}
    values["01_repeat_same_sim"] = observe("01_repeat_same_sim", lambda: signal.SIM(*fixed_inputs))
    signal.SIM.requires_grad_(False)
    values["02_freeze_sim_only"] = observe("02_freeze_sim_only", lambda: signal.SIM(*fixed_inputs))
    for name, parameter in signal.named_parameters():
        parameter.requires_grad_(initial_flags[name])
    values["03_restore_requires_grad"] = observe("03_restore_requires_grad", lambda: signal.SIM(*fixed_inputs))
    project_modeling = str(ROOT / "modeling")
    if project_modeling not in sys.path:
        sys.path.append(project_modeling)
    importlib.import_module("trifusion.signal_preserving_v8_builder")
    values["04_import_builder"] = observe("04_import_builder", lambda: signal.SIM(*fixed_inputs))
    importlib.import_module("mamba_ssm")
    values["05_import_mamba"] = observe("05_import_mamba", lambda: signal.SIM(*fixed_inputs))
    from tools.build_v12_complete_path_oof_targets import _build_v8_experts
    model = _build_v8_experts(signal, config, signal_checkpoint_sha256=b0["checkpoint_sha256"],
                             num_classes=len(fold["source_ids"]))
    model.eval()
    values["06_build_before_load"] = observe("06_build_before_load", lambda: signal.SIM(*fixed_inputs))
    final = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert final["source_ids"] == fold["source_ids"] and final["heldout_ids"] == fold["heldout_ids"]
    model.load_state_dict(final["model_state_dict"], strict=True)
    model.eval()
    values["07_load_final_roles"] = observe("07_load_final_roles", lambda: signal.SIM(*fixed_inputs))
    values["08_original_signal_after_load"] = observe("08_original_signal_after_load", original_signal)
    training = json.loads((run / "comparison/fold_0/training.json").read_text())
    assert _module_state_sha256(model) == training["final_state_sha256"]
    assert all(s["signal_state_unchanged"] for s in stages)
    assert all(not p.requires_grad for p in signal.parameters())
    assert all(not m.training for m in signal.modules())
    saved_arrays["stage_outputs"] = values
    array_path = output / "probe_arrays.pt"
    torch.save(saved_arrays, array_path)
    implementation = output / "installed_pytorch_multihead_attention.py.txt"
    implementation.write_text(inspect.getsource(torch.nn.MultiheadAttention.forward) + "\n"
                              + inspect.getsource(torch.nn.functional.multi_head_attention_forward))
    report = {"observed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": sha256(__file__), "role_checkpoint_sha256": sha256(checkpoint),
              "b0_checkpoint_sha256": b0["checkpoint_sha256"], **binding,
              "scope": "Fixed first64 previously visited fold0 gallery records; exact same six cached CLIP inputs; reversible flags/import/build/load stages",
              "gallery_record_indices": fold["gallery_record_indices"][:64],
              "initial_full_signal_vs_stored_b0": difference(first, original),
              "final_full_signal_vs_stored_b0": difference(values["08_original_signal_after_load"], original),
              "stage_sim_vs_initial": {k: difference(v[:, 1536:] if v.shape[1] == 3072 else v, first[:, 1536:])
                                       for k, v in values.items()},
              "stages": stages, "source_arrays_sha256": sha256(array_path),
              "installed_torch_version": torch.__version__, "installed_pytorch_mha_source_sha256": sha256(implementation),
              "record_forward_calls": 576, "full_signal_record_forwards": 128,
              "additional_cached_sim_record_forwards": 448, "distinct_image_records": 64,
              "optimizer_updates": 0, "backward_calls": 0, "checkpoint_writes": 0,
              "retrieval_metric_evaluations": 0, "backend_settings_changed": False,
              "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    main(parser.parse_args())
