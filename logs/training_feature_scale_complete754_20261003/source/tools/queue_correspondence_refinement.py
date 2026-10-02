#!/usr/bin/env python3
"""Finish the registered matrix, then global-only controls, then five M2 cells."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_correspondence_roles import collect
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS

DATASETS = ("RGBNT201", "RGBNT100", "MSVR310")
VARIANTS = {"single_pooled": ("single", False), "query_pooled": ("query", False),
            "single_regions": ("single", True), "query_regions": ("query", True),
            "uniform_pooled": ("uniform", False)}
POLL_SECONDS = 240


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def child_campaign(campaign, phase, dataset, variant=None):
    suffix = dataset if variant is None else f"{variant}_{dataset}"
    return campaign / f"{campaign.name}_{phase}_{suffix}"


def require_matrix():
    matrix = collect()
    assert matrix["expected_endpoints"] == matrix["verified_complete"] == len(matrix["rows"]) == 24
    return matrix


def require_complete(campaign, dataset):
    state = json.loads((campaign / "campaign.json").read_text())
    assert state["status"] == "COMPLETE" and state["dataset"] == dataset
    assert [job["mode"] for job in state["jobs"]] == ["m0", "train", "evaluate"]
    assert all(job["status"] == "COMPLETE" and job["exit_code"] == 0 for job in state["jobs"])
    folder = Path(state["jobs"][-1]["output_dir"])
    training = json.loads((folder / "training.json").read_text())
    result = json.loads((folder / "official_metrics.json").read_text())
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    assert result["status"] == "COMPLETE" and result["dataset"] == dataset
    assert result["seed"] == 42 and result["training_epochs"] == 50
    assert result["selected_epoch"] == training["best_epoch"]
    assert result["independent_upstream_metrics_equal"] and not result["reranking"]
    return result


def require_global_controls(campaign):
    return {dataset: require_complete(child_campaign(campaign, "global", dataset), dataset)
            for dataset in DATASETS}


def m2_command(dataset, variant, mode, output_dir):
    weight, digest = BASELINES[dataset]
    selection, structured = VARIANTS[variant]
    return [sys.executable, "-B", str(ROOT / "tools/run_correspondence_evidence_readout.py"),
            "--dataset", dataset, "--mode", mode,
            "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE), "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight), "--baseline-sha256", digest,
            "--output-dir", str(output_dir), "--seed", "42", "--epochs", "50",
            "--width", "128", "--pred-weight", "0.1", "--m1", "--m2", "--m3",
            "--selection", selection,
            "--structured-readout" if structured else "--no-structured-readout"]


def m2_worker(args):
    require_global_controls(args.campaign)
    campaign = child_campaign(args.campaign, "m2", args.dataset, args.variant)
    assert not campaign.exists()
    campaign.mkdir()
    state = {"status": "RUNNING", "dataset": args.dataset, "variant": args.variant,
             "gpu": args.gpu, "controller_pid": os.getpid(), "started_at": stamp(), "jobs": []}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for mode in ("m0", "train", "evaluate"):
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        suffix = "m0" if mode == "m0" else "full"
        output = ROOT / f"trained-model/{campaign.name}_seed42_{suffix}"
        row = {"mode": mode, "status": "RUNNING", "started_at": stamp(),
               "output_dir": str(output), "command": m2_command(args.dataset, args.variant, mode, output)}
        with (campaign / f"{mode}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(row["command"], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            state["jobs"].append(row)
            write(campaign / "campaign.json", state)
            code = process.wait()
        if code == 0:
            receipt = json.loads((output / ("official_metrics.json" if mode == "evaluate" else "training.json")).read_text())
            expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE",
                        "evaluate": "COMPLETE"}[mode]
            code = 0 if receipt["status"] == expected else 1
        row.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code, completed_at=stamp())
        state["status"] = "FAILED" if code else "RUNNING"
        write(campaign / "campaign.json", state)
        if code:
            return code
    state.update(status="COMPLETE", completed_at=stamp())
    write(campaign / "campaign.json", state)
    require_complete(campaign, args.dataset)
    return 0


def planned_jobs(campaign):
    return ([{"phase": "global", "dataset": dataset, "variant": None, "status": "PENDING"}
             for dataset in DATASETS]
            + [{"phase": "m2", "dataset": dataset, "variant": variant, "status": "PENDING"}
               for variant in VARIANTS for dataset in DATASETS])


def start_command(campaign, job, gpu):
    if job["phase"] == "global":
        return [sys.executable, "-B", str(ROOT / "tools/queue_correspondence_global_only.py"),
                "--dataset", job["dataset"], "--gpu", str(gpu), "--campaign",
                str(child_campaign(campaign, "global", job["dataset"]))]
    return [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--campaign", str(campaign),
            "--dataset", job["dataset"], "--variant", job["variant"], "--gpu", str(gpu)]


def run_phase(campaign, state, phase):
    pending = [job for job in state["jobs"] if job["phase"] == phase]
    active = []
    failed = False
    state.update(status="RUNNING", phase=phase, updated_at=stamp())
    while pending or active:
        for job, process in list(active):
            code = process.poll()
            if code is None:
                continue
            if code == 0:
                folder = child_campaign(campaign, phase, job["dataset"], job["variant"])
                job["result"] = require_complete(folder, job["dataset"])
            job.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code, completed_at=stamp())
            failed |= code != 0
            active.remove((job, process))
            print(json.dumps({"event": "completed", **job}), flush=True)
        if not failed:
            memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                              "--format=csv,noheader,nounits"], text=True)
            occupied = {job["gpu"] for job, _ in active}
            available = [int(row.split(",")[0]) for row in memory.splitlines()
                         if int(row.split(",")[0]) in range(4)
                         and int(row.split(",")[0]) not in occupied and int(row.split(",")[1]) < 500]
            for gpu in available:
                if not pending:
                    break
                assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
                job = pending.pop(0)
                job.update(gpu=gpu, status="RUNNING", started_at=stamp(),
                           command=start_command(campaign, job, gpu))
                name = f"{phase}_{job['variant'] or 'control'}_{job['dataset']}"
                with (campaign / f"{name}.log").open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(job["command"], cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = process.pid
                active.append((job, process))
                print(json.dumps({"event": "started", **job}), flush=True)
        state["updated_at"] = stamp()
        if failed:
            state["status"] = "FAILED"
        write(campaign / "campaign.json", state)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(POLL_SECONDS)
    return 0


def coordinate(args):
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = planned_jobs(args.campaign)
    write(args.campaign / "manifest.json", {"seed": 42, "epochs": 50, "poll_seconds": POLL_SECONDS,
                                           "after_campaign": str(args.after_campaign),
                                           "after_pid": args.after_pid, "jobs": jobs})
    state = {"status": "WAITING_MATRIX", "phase": "matrix", "controller_pid": os.getpid(),
             "started_at": stamp(), "jobs": jobs}
    while True:
        previous = json.loads((args.after_campaign / "campaign.json").read_text())
        if previous["status"] == "COMPLETE":
            break
        assert previous["status"] == "RUNNING"
        command = Path(f"/proc/{args.after_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert "tools/transfer_correspondence_variant.py transfer" in command
        assert str(args.after_campaign) in command
        state.update(updated_at=stamp(), predecessor_pid=args.after_pid, predecessor_command=command)
        write(args.campaign / "campaign.json", state)
        time.sleep(POLL_SECONDS)
    write(args.campaign / "parent_matrix.json", require_matrix())
    if run_phase(args.campaign, state, "global"):
        return 1
    write(args.campaign / "global_controls.json", require_global_controls(args.campaign))
    if run_phase(args.campaign, state, "m2"):
        return 1
    state.update(status="COMPLETE", completed_at=stamp())
    write(args.campaign / "campaign.json", state)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--gpu", type=int, choices=range(4))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return m2_worker(args)
    assert args.after_campaign is not None and args.after_pid is not None
    args.after_campaign = args.after_campaign.resolve()
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
