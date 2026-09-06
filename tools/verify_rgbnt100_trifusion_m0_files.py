#!/usr/bin/env python3
"""Verify completed RGBNT100 role M0 files and saved states on the remote host."""

import argparse
import datetime
import hashlib
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


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(args):
    import torch

    started = time.perf_counter()
    run, project = args.run_root.resolve(), args.project_root.resolve()
    directory, output = run / "m0", run / "m0_files_verification.json"
    assert not output.exists()
    assert (run / "m0_exit.txt").read_text().strip() == "0"
    terminal = json.loads((run / "m0_terminal.json").read_text())
    assert terminal["exit_code"] == 0
    config = json.loads(args.config.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["mode"] == "m0" and summary["status"] in ("PASS_ENGINEERING_ONLY", "FAIL_FIXED_M0")
    assert summary["optimizer_steps"] == 124 and summary["heldout_record_forwards"] == 0
    assert summary["config_sha256"] == sha(args.config)
    assert summary["runner_sha256"] == sha(project / "tools/train_rgbnt100_trifusion_oof.py")
    assert summary["project_source_file_sha256"] == config["project_source_file_sha256"]
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
    protocol_path = project / base["protocol"]
    assert sha(protocol_path) == base["protocol_sha256"] == summary["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    baseline = json.loads(Path(config["BASELINE"]["SUMMARY"]).read_text())
    assert sha(config["BASELINE"]["SUMMARY"]) == summary["baseline_summary_sha256"] == config["BASELINE"]["SUMMARY_SHA256"]
    assert baseline["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION" and baseline["engineering_revision"] == 2
    assert len(summary["folds"]) == len(protocol["folds"]) == len(baseline["folds"]) == 3
    selected = sorted(directory.rglob("*")) + [run / name for name in (
        "m0.log", "m0_exit.txt", "m0_launch.json", "m0_terminal.json", "m0_wrapper.py", "m0_wrapper.log")]
    files = {p.relative_to(run).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
             for p in selected if p.is_file()}
    checks = []
    for actual, fold, b0 in zip(summary["folds"], protocol["folds"], baseline["folds"], strict=True):
        index = fold["fold"]
        local = directory / f"fold_{index}"
        assert actual == json.loads((local / "receipt.json").read_text())
        training = actual["training"]
        assert training == json.loads((local / "training.json").read_text())
        assert training["steps"] == [json.loads(line) for line in (local / "steps.jsonl").read_text().splitlines()]
        assert training["optimizer_steps"] == len(training["steps"]) == 8
        checkpoint = local / "roles_m0.pth"
        assert Path(actual["checkpoint"]) == checkpoint and sha(checkpoint) == actual["checkpoint_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["fold"] == index and payload["config_sha256"] == summary["config_sha256"]
        assert payload["source_ids"] == fold["source_ids"] and payload["heldout_ids"] == fold["heldout_ids"]
        content_sha = state_sha(payload["model_state_dict"])
        assert content_sha == training["final_state_sha256"] == actual["strict_reload_state_sha256"]
        assert sha(b0["checkpoint"]) == b0["checkpoint_sha256"] == actual["initialization"]["signal_checkpoint_sha256"]
        base_payload = torch.load(b0["checkpoint"], map_location="cpu", weights_only=True)
        base_state = base_payload["model_state_dict"]
        signal_state = {name.removeprefix("baseline.signal."): value
                        for name, value in payload["model_state_dict"].items() if name.startswith("baseline.signal.")}
        assert set(signal_state) == set(base_state)
        assert all(torch.equal(value, base_state[name]) for name, value in signal_state.items())
        assert state_sha(base_state) == b0["training"]["final_state_sha256"] == actual["initialization"]["signal_state_sha256"]
        checks.append({"fold": index, "checkpoint_content_state_sha256": content_sha,
                       "signal_saved_state_bitwise_equal_b0": True, "all_step_logs_equal_training": True})
        del payload, base_payload, signal_state, base_state
    overfit = summary["overfit"]["training"]
    local = directory / "overfit_fold0"
    assert overfit == json.loads((local / "training.json").read_text())
    assert overfit["steps"] == [json.loads(line) for line in (local / "steps.jsonl").read_text().splitlines()]
    assert overfit["optimizer_steps"] == len(overfit["steps"]) == 100
    result = {"verified_at": datetime.datetime.now().astimezone().isoformat(),
              "status": "PASS_COMPLETE_M0_FILES_AND_CHECKPOINTS", "engineering_status": summary["status"],
              "summary_sha256": sha(directory / "summary.json"), "execution_commit": summary["project_commit"],
              "verification_project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
              "verifier_sha256": sha(__file__), "files": files, "folds": checks,
              "checkpoint_files_verified": 3, "saved_training_steps_verified": 124,
              "project_source_bindings_verified": len(config["project_source_file_sha256"]),
              "signal_source_bindings_verified": len(base["signal_source_file_sha256"]),
              "model_image_forwards": 0, "optimizer_updates": 0, "backward_calls": 0,
              "scope": "Saved files and CPU state tensors only; source feature parity remains the original M0 runtime receipt",
              "elapsed_seconds": time.perf_counter() - started}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run-root", "project-root", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
