#!/usr/bin/env python3
"""Run five controls on free GPUs once all original M3 jobs have started."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools.collect_correspondence_role_prediction import collect as collect_m3
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha

CONDITIONS = {"static_none": ("static", "none"), "context_none": ("context", "none"),
              "static_local": ("static", "local"), "context_local": ("context", "local"),
              "context_global": ("context", "global")}
SOURCE_PATHS = ("modeling/trifusion/correspondence_roles.py",
                "modeling/trifusion/correspondence_evidence_readout.py",
                "tools/run_correspondence_roles.py", "tools/queue_correspondence_refinement.py",
                "modeling/trifusion/correspondence_context_identity.py",
                "tools/run_correspondence_context_identity.py",
                "tools/queue_correspondence_context_identity.py",
                "tools/collect_correspondence_context_identity.py",
                "tools/audit_correspondence_context_identity_losses.py")


def require_sources(campaign):
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert all(sha(ROOT / name) == digest for name, digest in manifest["source_sha256"].items())
    return manifest


def command(dataset, variant, mode, output):
    query, auxiliary = CONDITIONS[variant]
    weight, digest = BASELINES[dataset]
    return [sys.executable, "-B", str(ROOT / "tools/run_correspondence_context_identity.py"),
            "--dataset", dataset, "--mode", mode, "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE), "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight), "--baseline-sha256", digest,
            "--output-dir", str(output), "--seed", "42", "--epochs", "50", "--width", "128",
            "--pred-weight", "0.1", "--m1", "--m2", "--no-m3", "--query-mode", query,
            "--auxiliary-target", auxiliary]


def worker(args):
    manifest = require_sources(args.campaign)
    campaign = queue.child_campaign(args.campaign, "context", args.dataset, args.variant)
    assert not campaign.exists()
    campaign.mkdir()
    state = {"status": "RUNNING", "dataset": args.dataset, "variant": args.variant,
             "gpu": args.gpu, "controller_pid": os.getpid(), "started_at": queue.stamp(), "jobs": []}
    # The old worker releases memory between train and evaluate; its GPU stays reserved.
    while True:
        predecessor = json.loads((Path(manifest["after_campaign"]) / "campaign.json").read_text())
        assert predecessor["status"] in ("RUNNING", "COMPLETE")
        assert not any(job["status"] == "FAILED" for job in predecessor["jobs"])
        reserved = [job for job in predecessor["jobs"]
                    if job["status"] == "RUNNING" and job["gpu"] == args.gpu]
        if not reserved:
            break
        state.update(status="WAIT_PREDECESSOR_GPU", updated_at=queue.stamp())
        queue.write(campaign / "campaign.json", state)
        time.sleep(queue.POLL_SECONDS)
    state["status"] = "RUNNING"
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
            process = subprocess.Popen(row["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            state["jobs"].append(row)
            queue.write(campaign / "campaign.json", state)
            code = process.wait()
        if code == 0:
            receipt = json.loads((output / ("official_metrics.json" if mode == "evaluate" else "training.json")).read_text())
            expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE", "evaluate": "COMPLETE"}[mode]
            assert receipt["status"] == expected
        row.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code, completed_at=queue.stamp())
        state["status"] = "FAILED" if code else "RUNNING"
        queue.write(campaign / "campaign.json", state)
        if code:
            return code
    from tools.collect_correspondence_context_identity import verify
    verification = verify(output, Path(state["jobs"][0]["output_dir"]), args.dataset, args.variant, manifest)
    state.update(status="COMPLETE", completed_at=queue.stamp(), verification=verification)
    queue.write(campaign / "campaign.json", state)
    return 0


def start_command(campaign, job, gpu):
    return [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--campaign", str(campaign),
            "--dataset", job["dataset"], "--variant", job["variant"], "--gpu", str(gpu)]


def coordinate(args):
    assert sha(args.after_campaign / "manifest.json") == args.after_manifest_sha256
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{"phase": "context", "dataset": dataset, "variant": variant, "status": "PENDING"}
            for variant in CONDITIONS for dataset in queue.DATASETS]
    queue.write(args.campaign / "manifest.json", {
        "schema": "trifusion-correspondence-context-identity-panel-v1", "seed": 42, "epochs": 50,
        "poll_seconds": queue.POLL_SECONDS, "after_campaign": str(args.after_campaign),
        "after_manifest_sha256": args.after_manifest_sha256, "after_pid": args.after_pid,
        "jobs": jobs, "source_sha256": {name: sha(ROOT / name) for name in SOURCE_PATHS},
        "fixed_contract": {"m1": True, "m2": True, "m3": False, "width": 128, "fused_width": 1536,
                           "selection": "query", "readout": "regions", "auxiliary_id_weight": 1.0,
                           "context_width": 512, "seed": 42, "checkpoint_policy": "best_official_map"},
        "boundary": "All twelve original M3 jobs have started before new jobs use free GPUs; no preemption. "
                    "Five controls on all three datasets; real M0 precedes each full50/best/reload. "
                    "Original M3 full12 verification is required again at final completion. "
                    "No retry, loss scan, seed search or automatic configuration selection.",
    })
    state = {"status": "WAITING_M3", "phase": "predecessor", "controller_pid": os.getpid(),
             "started_at": queue.stamp(), "jobs": jobs}
    while True:
        require_sources(args.campaign)
        assert sha(args.after_campaign / "manifest.json") == args.after_manifest_sha256
        previous = json.loads((args.after_campaign / "campaign.json").read_text())
        assert previous["status"] in ("RUNNING", "COMPLETE")
        assert not any(job["status"] == "FAILED" for job in previous["jobs"])
        if previous["status"] == "RUNNING":
            previous_command = Path(f"/proc/{args.after_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            assert "tools/queue_correspondence_role_prediction.py" in previous_command
            assert str(args.after_campaign) in previous_command
            state["predecessor_command"] = previous_command
        if not any(job["status"] == "PENDING" for job in previous["jobs"]):
            break
        state.update(updated_at=queue.stamp())
        queue.write(args.campaign / "campaign.json", state)
        time.sleep(queue.POLL_SECONDS)
    prior = collect_m3(args.after_campaign)
    assert prior["expected_endpoints"] == len(prior["rows"]) == 12
    assert all(row["status"] == "VERIFIED_COMPLETE" for row in prior["rows"] if row["dataset"] in ("RGBNT201", "MSVR310"))
    queue.write(args.campaign / "parent_matrix_at_launch.json", prior)
    queue.start_command = start_command
    code = queue.run_phase(args.campaign, state, "context")
    if code == 0:
        prior = collect_m3(args.after_campaign)
        assert prior["expected_endpoints"] == prior["verified_complete"] == len(prior["rows"]) == 12
        queue.write(args.campaign / "parent_matrix.json", prior)
        from tools.collect_correspondence_context_identity import collect
        accepted = collect(args.campaign)
        assert accepted["expected_endpoints"] == accepted["verified_complete"] == len(accepted["rows"]) == 15
        queue.write(args.campaign / "accepted_matrix.json", accepted)
    state.update(status="FAILED" if code else "COMPLETE", completed_at=queue.stamp())
    queue.write(args.campaign / "campaign.json", state)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-manifest-sha256")
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=queue.DATASETS)
    parser.add_argument("--variant", choices=tuple(CONDITIONS))
    parser.add_argument("--gpu", type=int, choices=range(4))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return worker(args)
    assert args.after_campaign is not None and args.after_pid is not None and args.after_manifest_sha256
    args.after_campaign = args.after_campaign.resolve()
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
