#!/usr/bin/env python3
"""One fixed read-only full-gallery probe of the failed MSVR310 parity gate."""

import argparse
import datetime
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
    from tools.build_v12_complete_path_oof_targets import _build_v8_experts
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch

    started = time.perf_counter()
    runroot = args.run_root.resolve()
    output = runroot / "baseline_parity_diagnosis"
    assert not output.exists()
    assert (runroot / "comparison_exit.txt").read_text().strip() == "1"
    config_path = ROOT / "configs/MSVR310/TriFusion-source-oof-v1.json"
    assert sha256(config_path) == "49d82696fbf5d0d71431ed8d7f1e5f80d7c13215a8bec6cafe77969ee8e276a5"
    config = json.loads(config_path.read_text())
    base = json.loads((ROOT / config["BASELINE"]["CONFIG"]).read_text())
    cfg, binding = configure(base)
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text())
    assert sha256(config["BASELINE"]["SUMMARY"]) == config["BASELINE"]["SUMMARY_SHA256"]
    protocol = json.loads((ROOT / base["protocol"]).read_text())
    fold, baseline_fold = protocol["folds"][0], baseline["folds"][0]
    role_checkpoint = runroot / "comparison/fold_0/roles_epoch20.pth"
    assert sha256(role_checkpoint) == "b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c"
    training = json.loads((runroot / "comparison/fold_0/training.json").read_text())
    original_path = Path(baseline_fold["checkpoint"]).parent / "retrieval_arrays.pt"
    assert sha256(original_path) == baseline_fold["retrieval"]["retrieval_arrays_sha256"]
    original = torch.load(original_path, map_location="cpu", weights_only=True)["features"]
    _set_seed(42)
    signal = new_model(cfg, fold)
    payload = torch.load(baseline_fold["checkpoint"], map_location="cpu", weights_only=True)
    signal.load_state_dict(payload["model_state_dict"], strict=True)
    signal.eval()
    signal_sha = _module_state_sha256(signal)
    assert signal_sha == baseline_fold["training"]["final_state_sha256"]
    records = records_for(base, protocol, fold, False)
    raw_batches = list(loader_for(records, False))
    assert len(records) == 360 and [len(r[1]) for r in raw_batches] == [64] * 5 + [40]

    def backend():
        return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
                "num_threads": torch.get_num_threads()}

    backend_before = backend()
    values = {}
    before = []
    with torch.no_grad():
        for raw in raw_batches:
            batch, _ = _training_batch(raw)
            before.append(signal(
                batch["images"], cam_label=batch["camera_ids"], view_label=raw[3].cuda(),
                training=False, sge=cfg.MODEL.stageName,
            ).float().cpu())
    values["standalone_before_wrapping"] = torch.cat(before)
    signal_before_wrap_sha = _module_state_sha256(signal)
    model = _build_v8_experts(signal, config,
                             signal_checkpoint_sha256=baseline_fold["checkpoint_sha256"],
                             num_classes=len(fold["source_ids"]))
    final = torch.load(role_checkpoint, map_location="cpu", weights_only=True)
    assert final["source_ids"] == fold["source_ids"] and final["heldout_ids"] == fold["heldout_ids"]
    model.load_state_dict(final["model_state_dict"], strict=True)
    model.eval()
    assert _module_state_sha256(model) == training["final_state_sha256"]
    assert _module_state_sha256(model.baseline.signal) == signal_sha
    backend_after = backend()
    after, hierarchical, complete, exact_prefix = [], [], [], []
    with torch.no_grad():
        for raw in raw_batches:
            batch, _ = _training_batch(raw)
            after.append(model.baseline.signal(
                batch["images"], cam_label=batch["camera_ids"], view_label=raw[3].cuda(),
                training=False, sge=cfg.MODEL.stageName,
            ).float().cpu())
            hierarchical.append(model.baseline(batch).baseline_embedding.float().cpu())
            full = model(batch, return_aux=True)
            complete.append(full.baseline_embedding.float().cpu())
            exact_prefix.append(full.diagnostics["baseline_exact_prefix"])
    values.update({"standalone_after_wrapping": torch.cat(after),
                   "hierarchical_baseline": torch.cat(hierarchical),
                   "full_model_baseline": torch.cat(complete)})
    model_state_unchanged = _module_state_sha256(model) == training["final_state_sha256"]
    signal_state_unchanged = _module_state_sha256(model.baseline.signal) == signal_before_wrap_sha == signal_sha

    def difference(a, b):
        delta = (a.double() - b.double()).abs()
        return {"bitwise_equal": torch.equal(a, b), "shape": list(a.shape),
                "exact_rows": int((a == b).all(dim=1).sum()),
                "different_elements": int((a != b).sum()),
                "maximum_absolute_difference": float(delta.max()),
                "mean_absolute_difference": float(delta.mean()),
                "direct_1536_max_absolute_difference": float(delta[:, :1536].max()),
                "sim_1536_max_absolute_difference": float(delta[:, 1536:].max()),
                "maximum_relative_l2_difference": float((delta.norm(dim=1) / b.double().norm(dim=1)).max()),
                "minimum_cosine": float(torch.nn.functional.cosine_similarity(a.double(), b.double()).min())}

    output.mkdir()
    arrays = output / "probe_arrays.pt"
    torch.save({"original_b0": original, **values,
                "gallery_record_indices": fold["gallery_record_indices"]}, arrays)
    report = {"observed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "scope": "All360 previously extracted fold0 gallery records, original five64 and one40 batches; four inference paths, zero ranking/AP and zero optimization",
              "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "diagnostic_source_sha256": sha256(__file__), "role_checkpoint_sha256": sha256(role_checkpoint),
              "b0_checkpoint_sha256": baseline_fold["checkpoint_sha256"], **binding,
              "baseline_comparisons": {k: difference(v, original) for k, v in values.items()},
              "within_process_comparisons": {k: difference(v, values["standalone_before_wrapping"]) for k, v in values.items()},
              "per_batch_vs_original_b0": {k: [difference(v[start:start + 64], original[start:start + 64])
                                                 for start in range(0, 360, 64)] for k, v in values.items()},
              "backend_before_wrapping": backend_before, "backend_after_wrapping": backend_after,
              "backend_flags_changed_by_probe": False,
              "frozen_signal_state_unchanged": signal_state_unchanged,
              "role_checkpoint_model_state_unchanged": model_state_unchanged,
              "all_signal_submodules_eval": all(not m.training for m in model.baseline.signal.modules()),
              "full_fusion_exact_signal_prefix": all(exact_prefix),
              "record_forward_calls": 1440, "distinct_gallery_records": 360,
              "standalone_signal_record_forwards": 720, "hierarchical_record_forwards": 360,
              "full_role_record_forwards": 360, "optimizer_updates": 0, "backward_calls": 0,
              "checkpoint_writes": 0, "retrieval_metric_evaluations": 0,
              "source_probe_arrays_sha256": sha256(arrays), "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    main(parser.parse_args())
