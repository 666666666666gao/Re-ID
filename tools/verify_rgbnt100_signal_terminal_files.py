#!/usr/bin/env python3
"""Verify complete RGBNT100 baseline checkpoints, arrays, ranks, and source bindings."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(args):
    import numpy as np
    import torch

    started = time.perf_counter()
    run, project = args.run_root.resolve(), args.project_root.resolve()
    output = run / "baseline_terminal_file_verification.json"
    assert not output.exists()
    assert (run / "baseline_exit.txt").read_text().strip() == "0"
    summary = json.loads((run / "baseline/summary.json").read_text())
    m0 = json.loads((run / "m0/summary.json").read_text())
    config_path = project / "configs/RGBNT100/Signal-source-oof-v1.json"
    config = json.loads(config_path.read_text())
    protocol = json.loads((project / config["protocol"]).read_text())
    assert summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert summary["mode"] == "train" and summary["checkpoint_selection"] == "fixed_epoch_30"
    assert summary["config_sha256"] == sha(config_path)
    assert summary["protocol_sha256"] == sha(project / config["protocol"])
    assert summary["preflight_receipt_sha256"] == sha(run / "m0/summary.json")
    assert summary["protocol_receipt_sha256"] == sha(run / "t0.json")
    assert summary["heldout_image_forwards"] == protocol["counts"]["gallery_records"] == 8675
    for name, expected in config["project_source_file_sha256"].items():
        assert sha(project / name) == expected, name
    signal = Path(config["signal_source"])
    for name, expected in config["signal_source_file_sha256"].items():
        assert sha(signal / name) == expected, name
    assert subprocess.check_output(["git", "-C", str(signal), "rev-parse", "HEAD"], text=True).strip() == config["signal_commit"]
    assert hashlib.sha256(subprocess.check_output(["git", "-C", str(signal), "diff", "--binary"])).hexdigest() == config["signal_diff_sha256"]
    assert sha(config["clip_weight"]) == config["clip_weight_sha256"]
    files = {}
    selected = sorted((run / "baseline").rglob("*"))
    selected += [run / name for name in ("baseline.log", "baseline_exit.txt", "baseline_launch.json",
                 "baseline_wrapper.log", "baseline_wrapper.py", "original_baseline_launch.json", "baseline_terminal.json")]
    for path in selected:
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    checks = []
    for actual, fold, capacity in zip(summary["folds"], protocol["folds"], m0["folds"], strict=True):
        index = fold["fold"]
        directory = run / "baseline" / f"fold_{index}"
        assert actual == json.loads((directory / "receipt.json").read_text())
        training = actual["training"]
        assert training == json.loads((directory / "training.json").read_text())
        assert training["epochs"] == len(training["history"]) == 30
        assert training["initial_state_sha256"] == capacity["training"]["initial_state_sha256"]
        assert training["initial_state_sha256"] != capacity["training"]["final_state_sha256"]
        checkpoint = directory / "signal_epoch30.pth"
        assert str(checkpoint) == actual["checkpoint"] and sha(checkpoint) == actual["checkpoint_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["fold"] == index and payload["source_ids"] == fold["source_ids"]
        assert payload["heldout_ids"] == fold["heldout_ids"]
        assert payload["config_sha256"] == summary["config_sha256"]
        assert payload["protocol_sha256"] == summary["protocol_sha256"]
        digest = state_sha(payload["model_state_dict"])
        assert digest == training["final_state_sha256"]
        del payload
        path = directory / "retrieval_arrays.pt"
        assert sha(path) == actual["retrieval"]["retrieval_arrays_sha256"]
        arrays = torch.load(path, map_location="cpu", weights_only=True)
        features, distances = arrays["features"], arrays["distances"]
        indices = fold["gallery_record_indices"]
        positions = [q["gallery_position"] for q in fold["query_rows"]]
        assert arrays["gallery_record_indices"] == indices and arrays["query_gallery_positions"] == positions
        assert list(features.shape) == [len(indices), 3072]
        assert list(distances.shape) == [len(positions), len(indices)]
        assert torch.isfinite(features).all() and torch.isfinite(distances).all()
        normalized = torch.nn.functional.normalize(features.float(), dim=1)
        qf = normalized[positions]
        expected = qf.square().sum(1, keepdim=True) + normalized.square().sum(1)[None]
        expected.addmm_(qf, normalized.T, beta=1, alpha=-2)
        assert torch.equal(expected, distances)
        ranks_path = directory / "full_rankings.json.gz"
        assert sha(ranks_path) == actual["retrieval"]["full_rankings_sha256"]
        with gzip.open(ranks_path, "rt", encoding="utf-8") as handle:
            ranks = json.load(handle)
        assert ranks["gallery_record_indices"] == indices and ranks["query_gallery_positions"] == positions
        order = np.argsort(distances.numpy(), axis=1)
        assert np.array_equal(order, np.asarray(ranks["ordered_gallery_positions"]))
        checks.append({"fold": index, "feature_shape": list(features.shape), "distance_shape": list(distances.shape),
                       "distances_bitwise_equal_recomputation": True, "complete_rankings_equal_full_sort": True,
                       "checkpoint_content_state_sha256": digest, "same_fresh_initial_state_as_m0": True,
                       "initial_state_not_m0_trained_state": True})
    # Replay the actual author scheduler without model, image or optimizer updates.
    import sys
    sys.path.insert(0, str(signal))
    from config import cfg
    cfg.merge_from_file(str(signal / "configs/RGBNT100/Signal.yml"))
    cfg.defrost()
    cfg.SOLVER.MAX_EPOCHS = 30
    cfg.freeze()
    from solver.scheduler_factory import create_scheduler
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([{"params": [parameter], "lr": 5e-6},
                                  {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": .0007},
                                  {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": .0014}])
    scheduler = create_scheduler(cfg, optimizer)
    schedule = []
    for epoch in range(1, 31):
        scheduler.step(epoch)
        lrs = sorted({g["lr"] for g in optimizer.param_groups})
        for fold in summary["folds"]:
            assert fold["training"]["history"][epoch - 1]["learning_rates"] == lrs
        schedule.append({"epoch": epoch, "learning_rates": lrs})
    result = {"status": "PASS_COMPLETE_BASELINE_FILES_ARRAYS_RANKS_AND_AUTHOR_LR",
              "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "execution_commit": summary["project_commit"],
              "verification_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
              "verifier_sha256": sha(__file__), "summary_sha256": sha(run / "baseline/summary.json"),
              "files": files, "folds": checks, "author_lr_schedule": schedule,
              "complete_query_rankings": 8675, "checkpoints_verified": 3, "arrays_verified": 3,
              "source_bindings": {"project": len(config["project_source_file_sha256"]), "signal": len(config["signal_source_file_sha256"])},
              "model_forwards": 0, "image_decodes": 0, "optimizer_updates": 0, "backward_calls": 0,
              "elapsed_seconds": time.perf_counter() - started}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    main(parser.parse_args())

