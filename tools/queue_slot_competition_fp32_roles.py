#!/usr/bin/env python3
"""Run the registered independent/competitive allocation panel on four GPUs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha

CONDITIONS = {"competitive": "competitive", "independent": "independent"}
DATASETS = ("RGBNT100", "RGBNT201", "MSVR310")
SOURCE_PATHS = (
    "modeling/trifusion/slot_competition_fp32_roles.py",
    "refine-logs/slot_competition_roles_v1/FP32_PLAN_20261001.md",
    "modeling/trifusion/slot_competition_roles.py",
    "modeling/trifusion/correspondence_roles.py",
    "modeling/trifusion/correspondence_evidence_readout.py",
    "modeling/trifusion/correspondence_context_identity.py",
    "modeling/trifusion/patch_memory_roles.py",
    "tools/run_correspondence_roles.py",
    "tools/run_correspondence_context_identity.py",
    "tools/run_slot_competition_fp32_roles.py",
    "tools/queue_slot_competition_fp32_roles.py",
    "tools/collect_slot_competition_fp32_roles.py",
    "tools/check_slot_competition_fp32_roles_cpu.py",
    "refine-logs/slot_competition_roles_v1/EXPERIMENT_PLAN.md",
    "tools/queue_correspondence_refinement.py",
    "tools/queue_correspondence_roles.py",
    "tools/collect_correspondence_roles.py",
    "tools/audit_correspondence_context_identity_losses.py",
    "tools/train_msvr310_signal_oof.py",
    "tools/train_rgbnt100_signal_oof.py",
    "tools/train_msvr310_trifusion_oof.py",
    "tools/train_official_three_dataset_roles.py",
    "tools/audit_v17_full_gallery.py",
    "tools/official_three_dataset_data.py",
    "tools/official_three_dataset_model.py",
    "tools/run_official_three_dataset_roles.py",
    "tools/run_signal_preserving_v5.py",
    "tools/build_v12_complete_path_oof_targets.py",
    "tools/run_signal_baseline_dev.py",
    "tools/train_signal_preserving_v18.py",
    "tools/train_signal_preserving_v17.py",
    "modeling/trifusion/aligned_data.py",
    "modeling/trifusion/criterion.py",
    "modeling/trifusion/experts/mamba.py",
    "configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json",
    "configs/RGBNT100/TriFusion-main-v1.json",
    "configs/MSVR310/TriFusion-source-style-paired-v1-r2.json",
)


def require_sources(campaign):
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert all(sha(ROOT / name) == digest for name, digest in manifest["source_sha256"].items())
    return manifest


def command(dataset, variant, mode, output):
    weight, digest = BASELINES[dataset]
    return [sys.executable, "-B", str(ROOT / "tools/run_slot_competition_fp32_roles.py"),
            "--dataset", dataset, "--mode", mode,
            "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE), "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight), "--baseline-sha256", digest,
            "--output-dir", str(output), "--seed", "42", "--epochs", "50", "--width", "128",
            "--pred-weight", "0.1", "--m1", "--m2", "--no-m3",
            "--query-mode", "context", "--auxiliary-target", "none",
            "--attention-normalization", CONDITIONS[variant]]


def worker(args):
    manifest = require_sources(args.campaign)
    campaign = queue.child_campaign(args.campaign, "slot_competition_fp32", args.dataset, args.variant)
    assert not campaign.exists()
    campaign.mkdir()
    state = {"status": "RUNNING", "dataset": args.dataset, "variant": args.variant,
             "gpu": args.gpu, "controller_pid": os.getpid(), "started_at": queue.stamp(), "jobs": []}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for mode in ("m0", "train", "evaluate"):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        suffix = "m0" if mode == "m0" else "full"
        output = ROOT / f"trained-model/{campaign.name}_seed42_{suffix}"
        row = {"mode": mode, "status": "RUNNING", "started_at": queue.stamp(),
               "output_dir": str(output), "command": command(args.dataset, args.variant, mode, output)}
        with (campaign / f"{mode}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(row["command"], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            state["jobs"].append(row)
            queue.write(campaign / "campaign.json", state)
            code = process.wait()
        if code == 0:
            receipt = json.loads((output / (
                "official_metrics.json" if mode == "evaluate" else "training.json")).read_text())
            expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE",
                        "evaluate": "COMPLETE"}[mode]
            assert receipt["status"] == expected
            if mode == "m0":
                assert len(receipt["history"]) == 1 and receipt["history"][0]["steps"] == 8
                assert receipt["m0"]["frozen_signal_unchanged"]
                assert receipt["m0"]["nonzero_gradient_parameters"] == receipt["m0"]["trainable_parameters"]
                assert receipt["m0"]["reload_max_abs_difference"] <= 1e-5
                assert sha(output / "m0_reload_probe.pth") == receipt["m0"]["reload_probe_sha256"]
                steps = [json.loads(line) for line in (output / "training_steps.jsonl").read_text().splitlines()]
                assert [step["batch"] for step in steps] == list(range(8))
        row.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code,
                   completed_at=queue.stamp())
        state["status"] = "FAILED" if code else "RUNNING"
        queue.write(campaign / "campaign.json", state)
        if code:
            return code
    from tools.collect_slot_competition_fp32_roles import verify
    verification = verify(output, Path(state["jobs"][0]["output_dir"]),
                          args.dataset, args.variant, manifest)
    state.update(status="COMPLETE", completed_at=queue.stamp(), verification=verification)
    queue.write(campaign / "campaign.json", state)
    return 0


def start_command(campaign, job, gpu):
    return [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
            "--campaign", str(campaign), "--dataset", job["dataset"],
            "--variant", job["variant"], "--gpu", str(gpu)]


def coordinate(args):
    predecessor = json.loads((args.after_campaign / "campaign.json").read_text())
    assert predecessor["status"] == "COMPLETE" and len(predecessor["jobs"]) == 6
    assert all(job["status"] == "COMPLETE" and job["exit_code"] == 0
               for job in predecessor["jobs"])
    matrix_path = args.after_campaign / "accepted_matrix.json"
    assert sha(matrix_path) == args.after_matrix_sha256
    prior = json.loads(matrix_path.read_text())
    assert prior["verified_complete"] == prior["expected_endpoints"] == len(prior["rows"]) == 6
    previous_sources = json.loads((args.after_campaign / "manifest.json").read_text())["source_sha256"]
    assert len(previous_sources) == 210
    assert all(sha(ROOT / name) == digest for name, digest in previous_sources.items())
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{"phase": "slot_competition_fp32", "dataset": dataset, "variant": variant, "status": "PENDING"}
            for variant in CONDITIONS for dataset in DATASETS]
    upstream = [path.relative_to(ROOT).as_posix() for path in sorted(SOURCE.rglob("*.py"))]
    local_model = [path.relative_to(ROOT).as_posix()
                   for path in sorted((ROOT / "modeling/trifusion").rglob("*.py"))]
    inputs = [path.relative_to(ROOT).as_posix() for dataset in DATASETS
              for path in (PROTOCOLS / f"{dataset}.json", SOURCE / "configs" / dataset / "Signal.yml")]
    assert upstream
    queue.write(args.campaign / "manifest.json", {
        "schema": "trifusion-slot-competition-fp32-roles-panel-v2", "seed": 42, "epochs": 50,
        "poll_seconds": queue.POLL_SECONDS, "after_campaign": str(args.after_campaign),
        "after_matrix_sha256": args.after_matrix_sha256, "jobs": jobs,
        "source_sha256": {name: sha(ROOT / name)
                          for name in (*SOURCE_PATHS, *local_model, *upstream, *inputs)},
        "fixed_contract": {"m1": True, "m2": True, "m3": False, "width": 128,
                           "fused_width": 1536, "selection": "query", "readout": "regions",
                           "auxiliary_target": "none", "query_mode": "context",
                           "slot_count": 16, "patch_count": 128, "attention_normalizations": list(CONDITIONS), "normalization_math": "FP32", "attention_subgraph": "FP32",
                           "full_candidates": 128, "fixed_slot_positions": True,
                           "checkpoint_policy": "best_official_map"},
        "boundary": "Independent versus competitive allocation uses the same trainable tensors, seed, full128 support, data, budget, "
                    "role operators, fixed positions, attention projections and readout. Six real eight-batch M0 checks precede each full50 run; "
                    "single mAP-best strict reload and complete-gallery CPU verification. No auto-retry.",
    })
    queue.write(args.campaign / "parent_matrix.json", prior)
    state = {"status": "RUNNING", "phase": "slot_competition_fp32", "controller_pid": os.getpid(),
             "started_at": queue.stamp(), "jobs": jobs}
    queue.start_command = start_command
    code = queue.run_phase(args.campaign, state, "slot_competition_fp32")
    if code == 0:
        require_sources(args.campaign)
        from tools.collect_slot_competition_fp32_roles import collect
        accepted = collect(args.campaign)
        assert accepted["expected_endpoints"] == accepted["verified_complete"] == len(accepted["rows"]) == 6
        queue.write(args.campaign / "accepted_matrix.json", accepted)
    state.update(status="FAILED" if code else "COMPLETE", completed_at=queue.stamp())
    queue.write(args.campaign / "campaign.json", state)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-matrix-sha256")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=queue.DATASETS)
    parser.add_argument("--variant", choices=tuple(CONDITIONS))
    parser.add_argument("--gpu", type=int, choices=range(4))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return worker(args)
    assert args.after_campaign is not None and args.after_matrix_sha256
    args.after_campaign = args.after_campaign.resolve()
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
