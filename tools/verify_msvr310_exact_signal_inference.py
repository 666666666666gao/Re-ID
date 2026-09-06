#!/usr/bin/env python3
"""Full360 verification of the measured SIM dispatch repair, without ranking."""

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
    from tools.train_msvr310_signal_oof import configure, loader_for, records_for, sha256, write_json
    from tools.train_msvr310_trifusion_oof import build_model, output_mapping, frozen_state_sha
    from tools.run_signal_preserving_v5 import _module_state_sha256, _training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward

    started = time.perf_counter()
    run = args.run_root.resolve()
    out = run / "exact_signal_inference_verification"
    assert not out.exists()
    registration = ROOT / "evidence/trifusion_msvr310_trifusion_v1_exact_inference_verification_plan_20260906.json"
    plan = json.loads(registration.read_text())
    for name, expected in plan["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    config_path = ROOT / "configs/MSVR310/TriFusion-source-oof-v1.json"
    assert sha256(config_path) == "49d82696fbf5d0d71431ed8d7f1e5f80d7c13215a8bec6cafe77969ee8e276a5"
    config = json.loads(config_path.read_text())
    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    base = json.loads((ROOT / config["BASELINE"]["CONFIG"]).read_text())
    cfg, binding = configure(base)
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text())
    assert sha256(config["BASELINE"]["SUMMARY"]) == config["BASELINE"]["SUMMARY_SHA256"]
    protocol = json.loads((ROOT / base["protocol"]).read_text())
    fold, b0 = protocol["folds"][0], baseline["folds"][0]
    checkpoint = run / "comparison/fold_0/roles_epoch20.pth"
    assert sha256(checkpoint) == "b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c"
    original_path = Path(b0["checkpoint"]).parent / "retrieval_arrays.pt"
    assert sha256(original_path) == b0["retrieval"]["retrieval_arrays_sha256"]
    original = torch.load(original_path, map_location="cpu", weights_only=True)
    model, _ = build_model(config, cfg, fold, b0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    parameter_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    training = json.loads((run / "comparison/fold_0/training.json").read_text())
    state = _module_state_sha256(model)
    assert state == training["final_state_sha256"]
    frozen = frozen_state_sha(model)
    signal_sha = _module_state_sha256(model.baseline.signal)
    assert signal_sha == b0["training"]["final_state_sha256"]
    records = records_for(base, protocol, fold, False)
    assert len(records) == 360
    repaired_parts, unchanged_parts, batch_checks = {}, [], []
    with torch.no_grad():
        for raw in loader_for(records, False):
            batch, _ = _training_batch(raw)
            before = model(batch, return_aux=True)
            after = exact_signal_forward(model, batch)
            checks = {
                "direct_modal_unchanged": torch.equal(before.direct_modal, after.direct_modal),
                "all_role_residuals_unchanged": all(torch.equal(before.residual_embeddings[e], after.residual_embeddings[e])
                                                  for e in ("cnn", "transformer", "mamba")),
                "all_modal_residuals_unchanged": all(torch.equal(before.modal_residual_embeddings[e], after.modal_residual_embeddings[e])
                                                   for e in ("cnn", "transformer", "mamba")),
                "registered_signal_parameters_still_frozen": all(not p.requires_grad for p in model.baseline.signal.parameters()),
                "fused_exact_signal_prefix": after.diagnostics["baseline_exact_prefix"],
                "batch_size": len(raw[1]),
            }
            assert all(v for k, v in checks.items() if k != "batch_size"), checks
            batch_checks.append(checks)
            unchanged_parts.append(before.baseline_embedding.float().cpu())
            for name, value in output_mapping(after).items():
                repaired_parts.setdefault(name, []).append(value.float().cpu())
    repaired = {k: torch.cat(v) for k, v in repaired_parts.items()}
    unchanged = torch.cat(unchanged_parts)
    positions = [q["gallery_position"] for q in fold["query_rows"]]
    assert original["query_gallery_positions"] == positions
    normalized = torch.nn.functional.normalize(repaired["baseline_only"].float(), dim=1)
    qf, gf = normalized[positions], normalized
    distances = qf.square().sum(1, keepdim=True) + gf.square().sum(1)[None]
    distances.addmm_(qf, gf.T, beta=1, alpha=-2)
    checks = {"original_unrepaired_difference_reproduced": not torch.equal(unchanged, original["features"]),
              "repaired_all360_baseline_features_bitwise_equal": torch.equal(repaired["baseline_only"], original["features"]),
              "repaired_all210x360_baseline_distances_bitwise_equal": torch.equal(distances, original["distances"]),
              "model_state_unchanged": _module_state_sha256(model) == state,
              "all_frozen_state_unchanged": frozen_state_sha(model) == frozen,
              "signal_state_unchanged": _module_state_sha256(model.baseline.signal) == signal_sha,
              "all_parameter_flags_unchanged": parameter_flags == {n: p.requires_grad for n, p in model.named_parameters()},
              "original_batches_preserved": [x["batch_size"] for x in batch_checks] == [64] * 5 + [40]}
    out.mkdir()
    arrays = out / "verification_arrays.pt"
    torch.save({"repaired_features": repaired, "unrepaired_baseline": unchanged,
                "repaired_baseline_distances": distances, "gallery_record_indices": fold["gallery_record_indices"]}, arrays)
    report = {"observed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "status": "PASS_EXACT_INFERENCE_ENGINEERING" if all(checks.values()) else "FAIL_EXACT_INFERENCE_ENGINEERING",
              "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "verification_script_sha256": sha256(__file__), "registration_sha256": sha256(registration),
              "helper_sha256": sha256(ROOT / "tools/msvr310_exact_signal_inference.py"), **binding,
              "role_checkpoint_sha256": sha256(checkpoint), "checks": checks, "batch_checks": batch_checks,
              "torch_version": str(torch.__version__), "torch_git_version": torch.version.git_version,
              "arrays_sha256": sha256(arrays), "record_forward_calls": 720,
              "distinct_already_visited_gallery_records": 360, "optimizer_updates": 0, "backward_calls": 0,
              "checkpoint_writes": 0, "ranking_AP_Rank_evaluations": 0,
              "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "summary.json", report)
    print(json.dumps(report), flush=True)
    assert all(checks.values()), checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    main(parser.parse_args())
