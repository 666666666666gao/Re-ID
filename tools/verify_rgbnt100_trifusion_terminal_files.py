#!/usr/bin/env python3
"""Check every completed RGBNT100 role checkpoint, feature, distance and full order."""

import argparse
import datetime
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    import numpy as np
    import torch

    started = time.perf_counter()
    run, project = args.run_root.resolve(), args.project_root.resolve()
    directory = run / "comparison"
    output = run / "comparison_terminal_files.json"
    assert not output.exists()
    assert (run / "comparison_exit.txt").read_text().strip() == "0"
    assert json.loads((run / "comparison_terminal.json").read_text())["exit_code"] == 0
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] in ("COMPLETE_COMPARISON_SUPPORT_PASS", "COMPLETE_COMPARISON_SUPPORT_FAIL")
    assert summary["mode"] == "comparison" and len(summary["folds"]) == 3
    assert summary["heldout_record_forwards"] == 8675
    config = json.loads(args.config.read_text())
    assert sha(args.config) == summary["config_sha256"]
    assert sha(project / "tools/train_rgbnt100_trifusion_oof.py") == summary["runner_sha256"]
    assert config["project_source_file_sha256"] == summary["project_source_file_sha256"]
    for name, expected in config["project_source_file_sha256"].items():
        assert sha(project / name) == expected, name
    base_path = project / config["BASELINE"]["CONFIG"]
    assert sha(base_path) == config["BASELINE"]["CONFIG_SHA256"]
    base = json.loads(base_path.read_text())
    for name, expected in base["project_source_file_sha256"].items():
        assert sha(project / name) == expected, name
    for name, expected in base["signal_source_file_sha256"].items():
        assert sha(Path(base["signal_source"]) / name) == expected, name
    assert subprocess.check_output(["git", "-C", base["signal_source"], "rev-parse", "HEAD"], text=True).strip() == base["signal_commit"]
    assert hashlib.sha256(subprocess.check_output(["git", "-C", base["signal_source"], "diff", "--binary"])).hexdigest() == base["signal_diff_sha256"]
    assert sha(base["clip_weight"]) == base["clip_weight_sha256"]
    assert sha(project / base["protocol"]) == base["protocol_sha256"] == summary["protocol_sha256"]
    protocol = json.loads((project / base["protocol"]).read_text())
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text())
    assert sha(config["BASELINE"]["SUMMARY"]) == summary["baseline_summary_sha256"] == config["BASELINE"]["SUMMARY_SHA256"]
    assert baseline["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION" and baseline["engineering_revision"] == 2
    m0_path = run / "m0/summary.json"
    assert sha(m0_path) == summary["m0_receipt_sha256"]
    m0 = json.loads(m0_path.read_text())
    assert m0["status"] == "PASS_ENGINEERING_ONLY" and m0["config_sha256"] == summary["config_sha256"]
    files = {}
    selected = sorted(directory.rglob("*"))
    selected += [run / name for name in ("comparison.log", "comparison_exit.txt",
                 "comparison_launch.json", "comparison_wrapper.py", "comparison_wrapper.log",
                 "comparison_terminal.json", "m0/summary.json", "m0_files_verification.json")]
    for path in selected:
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    checks = []
    for actual, fold, b0 in zip(summary["folds"], protocol["folds"], baseline["folds"], strict=True):
        index = fold["fold"]
        local = directory / f"fold_{index}"
        assert actual == json.loads((local / "receipt.json").read_text())
        assert actual["training"] == json.loads((local / "training.json").read_text())
        assert actual["training"]["steps"] == [json.loads(line) for line in (local / "steps.jsonl").read_text().splitlines()]
        assert actual["training"]["epochs"] == 20
        assert all(row["optimizer_update_applied"] and all(row["gradient_finite"].values()) for row in actual["training"]["steps"])
        assert actual["initialization"] == m0["folds"][index]["initialization"]
        assert sha(actual["checkpoint"]) == actual["checkpoint_sha256"]
        payload = torch.load(actual["checkpoint"], map_location="cpu", weights_only=True)
        assert payload["fold"] == index and payload["config_sha256"] == summary["config_sha256"]
        assert payload["source_ids"] == fold["source_ids"] and payload["heldout_ids"] == fold["heldout_ids"]
        digest = hashlib.sha256()
        for name, tensor in sorted(payload["model_state_dict"].items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        assert digest.hexdigest() == actual["training"]["final_state_sha256"] == actual["strict_reload_state_sha256"]
        base_payload = torch.load(b0["checkpoint"], map_location="cpu", weights_only=True)
        assert sha(b0["checkpoint"]) == b0["checkpoint_sha256"] == actual["initialization"]["signal_checkpoint_sha256"]
        signal_state = {name.removeprefix("baseline.signal."): value
                        for name, value in payload["model_state_dict"].items() if name.startswith("baseline.signal.")}
        assert set(signal_state) == set(base_payload["model_state_dict"])
        assert all(torch.equal(value, base_payload["model_state_dict"][name]) for name, value in signal_state.items())
        del payload, base_payload, signal_state
        array_path = local / "retrieval_arrays.pt"
        assert sha(array_path) == actual["retrieval"]["retrieval_arrays_sha256"]
        rank_path = local / "rankings.json.gz"
        assert sha(rank_path) == actual["retrieval"]["rankings_sha256"]
        with gzip.open(rank_path, "rt", encoding="utf-8") as handle:
            ranked = json.load(handle)
        arrays = torch.load(array_path, map_location="cpu", weights_only=True)
        b0_array_path = Path(b0["checkpoint"]).parent / "retrieval_arrays.pt"
        assert sha(b0_array_path) == b0["retrieval"]["retrieval_arrays_sha256"]
        b0_arrays = torch.load(b0_array_path, map_location="cpu", weights_only=True)
        positions = [q["gallery_position"] for q in fold["query_rows"]]
        assert arrays["gallery_record_indices"] == fold["gallery_record_indices"]
        assert arrays["query_gallery_positions"] == positions
        assert torch.equal(arrays["features"]["baseline_only"], b0_arrays["features"])
        assert torch.equal(arrays["distances"]["baseline_only"], b0_arrays["distances"])
        for name, width in {"baseline_only": 3072, "fused": 7680, "cnn": 4608, "transformer": 4608, "mamba": 4608}.items():
            features, distances = arrays["features"][name], arrays["distances"][name]
            assert list(features.shape) == [len(fold["gallery_record_indices"]), width]
            assert list(distances.shape) == [len(positions), len(fold["gallery_record_indices"])]
            assert torch.isfinite(features).all() and torch.isfinite(distances).all()
            normalized = torch.nn.functional.normalize(features.float(), dim=1)
            qf = normalized[positions]
            computed = qf.square().sum(1, keepdim=True) + normalized.square().sum(1)[None]
            computed.addmm_(qf, normalized.T, beta=1, alpha=-2)
            equal = torch.equal(computed, distances)
            assert equal
            assert np.argsort(distances.numpy(), axis=1).tolist() == ranked[name]
            checks.append({"fold": index, "output": name, "feature_shape": list(features.shape),
                           "distance_shape": list(distances.shape), "distance_recomputation_bitwise_equal": equal,
                           "saved_rankings_equal_full_saved_distance_sort": True,
                           "maximum_absolute_distance_difference": float((computed - distances).abs().max())})
        assert torch.equal(arrays["features"]["fused"][:, :3072], arrays["features"]["baseline_only"])
        del arrays, b0_arrays, ranked
    assert summary["optimizer_steps"] == sum(len(row["training"]["steps"]) for row in summary["folds"])
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat(), "status": "PASS_WHOLE_FILES_AND_ALL15_ARRAYS",
              "summary_sha256": sha(directory / "summary.json"), "execution_commit": summary["project_commit"],
              "verification_project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
              "verifier_sha256": sha(__file__), "files": files, "array_checks": checks,
              "all_three_baselines_bitwise_equal_b0_features_and_distances": True,
              "all_three_saved_signal_states_bitwise_equal_b0": True,
              "three_fold_receipts_equal_summary": True, "all_update_training_files_equal_summary_and_jsonl": True,
              "training_steps_verified": summary["optimizer_steps"],
              "checkpoint_files_verified": 3, "retrieval_array_files_verified": 3,
              "project_source_bindings_verified": len(config["project_source_file_sha256"]),
              "signal_source_bindings_verified": len(base["signal_source_file_sha256"]),
              "model_image_forwards": 0, "optimizer_updates": 0, "backward_calls": 0,
              "scope": "Remote saved-file/tensor arithmetic only; all15 feature-distance-rank paths, no model/image execution",
              "elapsed_seconds": time.perf_counter() - started}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    main(parser.parse_args())
