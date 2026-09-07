#!/usr/bin/env python3
"""Same-forward precision pairing after the archived cross-process replay failure."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.train_signal_preserving_v28 import (
    EXPERTS, build_model, check_batch, load_contract, new_optimizer, step, training_loader,
)
from tools.train_signal_preserving_v19 import criterion_for, seed_everything
from tools.train_signal_preserving_v17 import _raw_batch_receipt, _record_index_by_path, _tensor_mapping_sha256
from tools.run_signal_preserving_v5 import _training_batch
from tools.build_v12_complete_path_oof_targets import _configure_signal, _load_records, build_complete_path_fold_records


def grad_stats(value):
    assert value is not None and torch.isfinite(value).all()
    return {"nonzero": int(torch.count_nonzero(value)), "elements": value.numel(),
            "abs_max": float(value.abs().max()), "abs_sum": float(value.abs().sum()),
            "l2": float(value.float().norm()), "dtype": str(value.dtype)}


def run(args):
    args.output_dir.mkdir(exist_ok=False, parents=True)
    (args.output_dir / "started.json").write_text(json.dumps({"status": "STARTED", "maximum_reconstruction_updates": 1}))
    original = json.loads(args.original_summary.read_bytes())
    assert original["status"] == "M0_FAIL" and not original["folds"]
    assert original["m0"]["capacities"][1]["missing_nonzero_gradients"] == ["joint.mixer.dt_proj.weight"]
    assert original["m0"]["overfit"]["missing_nonzero_gradients"] == ["joint.mixer.dt_proj.weight"]
    config, sources = load_contract(args.config)
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == original["config_sha256"]
    signal_cfg, _, _ = _configure_signal(config)
    records = _load_records(config)
    split = build_complete_path_fold_records(records, heldout_ids=set(sources["fold_receipts"][0]["heldout_identity_ids"]))
    model, binding = build_model(config, signal_cfg, 0, split, enabled=True)
    seed_everything()
    model.train()
    iterator = iter(training_loader(split["train_records"], config, "joint_tokens"))
    raw0, raw1 = next(iterator), next(iterator)
    index = _record_index_by_path(split["train_records"])
    receipts = [_raw_batch_receipt(raw, record_index_by_path=index) for raw in (raw0, raw1)]
    assert receipts == original["m0"]["capacities"][1]["batch_receipts"][:2]
    optimizer, scaler = new_optimizer(model, config)
    criterion = criterion_for(config)
    first, overflow, _, _ = step(model, criterion, raw0, optimizer, scaler, config, enabled=True, step_index=0)
    (args.output_dir / "reconstruction_step.json").write_text(json.dumps({"completed_optimizer_updates": 1, "losses": first, "overflow": overflow}))
    assert not overflow and first == original["m0"]["capacities"][1]["components"][0]
    # Exactly one reconstruction update opens the zero-output projection.
    model.zero_grad(set_to_none=True)
    parameters_before = _tensor_mapping_sha256(dict(model.named_parameters()))
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    captured = {}

    def capture(_module, inputs, output):
        captured["input"] = inputs[0].detach().clone()
        captured["output"] = output
        for value in output.values():
            value.retain_grad()

    batch, labels = _training_batch(raw1)
    from trifusion.source_style_v27 import make_style_plan
    model.baseline.style_enabled = True
    model.baseline.style_plan = make_style_plan(raw1[2].numpy(), fold=0, step=1)
    with model.joint.register_forward_hook(capture), torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
        parts = criterion(output, labels)
        w = config["LOSS"]
        common = w["ID_FUSED"] * parts["id_fused"] + w["TRIPLET_FUSED"] * parts["triplet_fused"]
        for expert in EXPERTS:
            common = common + w["ID_BRANCH"] * parts["id_" + expert] + w["TRIPLET_BRANCH"] * parts["triplet_" + expert]
            common = common + w["ID_RESIDUAL"] * parts["id_residual_" + expert]
        ordinary = w["TRIPLET_RESIDUAL"] * sum(parts["triplet_residual_" + expert] for expert in EXPERTS)
    total = common.float() + ordinary.float()
    expected = original["m0"]["capacities"][1]["components"][1]
    actual = {"total": float(total), "common_identity_and_branch_triplet": float(common),
              "ordinary_residual_triplet": float(ordinary),
              **{key: float(value) for key, value in parts.items()}, **model.baseline.last_style_stats}
    comparison = {"actual": actual, "expected": {key: expected[key] for key in actual},
                  "differences": {key: actual[key] - expected[key] for key in actual},
                  "style_plan_exact": model.baseline.style_plan == original["m0"]["capacities"][1]["style_plans"][1],
                  "actual_total_exact": actual["total"] == expected["total"]}
    (args.output_dir / "second_batch_loss_comparison.json").write_bytes((json.dumps(comparison, indent=2) + chr(10)).encode())
    # The old cross-process second-loss check failed and remains archived.
    # This separately registered probe pairs precision within this one forward.
    # Its actual old-loss difference is recorded, not hidden by a tolerance.
    scale = scaler.get_scale()
    (total * scale).backward()
    full_grad = grad_stats(model.joint.mixer.dt_proj.weight.grad)
    assert full_grad["nonzero"] == 0, "the original zero-gradient symptom was not reproduced"
    upstream = {name: value.grad.detach().clone() for name, value in captured["output"].items()}
    expected_output = {name: value.detach().clone() for name, value in captured["output"].items()}
    assert all(value.abs().sum() > 0 for value in upstream.values())
    assert parameters_before == _tensor_mapping_sha256(dict(model.named_parameters()))
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    fixture = args.output_dir / "joint_input_state_upstream.pt"
    torch.save({"input": captured["input"].cpu(), "state": {k: v.detach().cpu() for k, v in model.joint.state_dict().items()},
                "upstream_scaled": {k: v.cpu() for k, v in upstream.items()},
                "expected_amp_output": {k: v.cpu() for k, v in expected_output.items()},
                "amp_scale": scale}, fixture)
    module = model.joint
    x = captured["input"]
    del captured, output, parts, total, common, ordinary, model, optimizer, criterion, buffers
    joint_state_before = _tensor_mapping_sha256(module.state_dict())
    cases = []
    for name, autocast, fast in [("amp_fast", True, True), ("fp32_fast", False, True), ("fp32_unfused", False, False)]:
        module.zero_grad(set_to_none=True)
        module.mixer.use_fast_path = fast
        leaf = (x.detach().clone() if autocast else x.detach().float().clone()).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out = module(leaf)
        difference = max(float((out[e].float() - expected_output[e].float()).abs().max()) for e in EXPERTS)
        torch.autograd.backward([out[e] for e in EXPERTS], [upstream[e].to(out[e].dtype) for e in EXPERTS])
        row = {"mode": name, "output_abs_max_difference_from_original_amp": difference,
               "parameters": {k: grad_stats(v.grad) for k, v in module.named_parameters()},
               "input_gradient": grad_stats(leaf.grad)}
        cases.append(row)
        if name == "amp_fast":
            assert difference == 0 and row["parameters"]["mixer.dt_proj.weight"] == full_grad
        del out, leaf
    assert joint_state_before == _tensor_mapping_sha256(module.state_dict())
    report = {
        "status": "COMPLETE_SAME_FORWARD_SOURCE_PRECISION_DERIVATIVE_COMPARISON",
        "original_m0_summary_sha256": hashlib.sha256(args.original_summary.read_bytes()).hexdigest(),
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_binding": binding, "raw_batch_receipts": receipts,
        "reconstruction_optimizer_updates": 1, "new_q1_or_m0_updates": 0,
        "source_model_forwards": 2, "joint_only_derivative_replays": 3,
        "first_update_losses_exact": True, "second_batch_loss_exact": comparison["actual_total_exact"],
        "second_batch_loss_difference": comparison["differences"]["total"],
        "same_forward_precision_pairing": True,
        "old_cross_process_loss_replay_failure_not_reclassified": True,
        "full_model_scaled_dt_gradient": full_grad, "amp_scale": scale,
        "cases": cases, "parameters_unchanged_during_derivative_comparison": True,
        "fixture": {"path": str(fixture), "bytes": fixture.stat().st_size,
                    "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()},
        "dev_access": 0, "official_access": 0, "saved_retrieval_checkpoint": False,
        "independent_audit": False,
    }
    (args.output_dir / "diagnosis.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({"status": report["status"], "modes": [{"mode": row["mode"],
          "dt_weight": row["parameters"]["mixer.dt_proj.weight"],
          "output_error": row["output_abs_max_difference_from_original_amp"]} for row in cases]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
