"""Recover two finished trainings and run the four missing registered endpoints."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools import queue_patch_memory_roles as patch
from tools.collect_patch_memory_roles import verify
from tools.collect_correspondence_roles import read, sha

REUSE = (("MSVR310", "local_memory"), ("RGBNT201", "full_memory"))
NEW = (("RGBNT201", "local_memory"), ("RGBNT100", "local_memory"),
       ("RGBNT100", "full_memory"), ("MSVR310", "full_memory"))


def old_runs(original, dataset, variant):
    folder = queue.child_campaign(original, "patch_memory", dataset, variant)
    return tuple(ROOT / f"trained-model/{folder.name}_seed42_{suffix}"
                 for suffix in ("m0", "full"))


def endpoint(folder, dataset, manifest):
    state = read(folder / "campaign.json")
    assert state["status"] == "COMPLETE" and state["dataset"] == dataset
    assert all(job["status"] == "COMPLETE" and job["exit_code"] == 0
               for job in state["jobs"])
    if state.get("recovery_action") == "evaluate_existing":
        assert [job["mode"] for job in state["jobs"]] == ["evaluate"]
        m0, run = Path(state["m0_run"]), Path(state["jobs"][0]["output_dir"])
        assert sha(run / "training.json") == state["preserved_training_sha256"]
        assert sha(run / "best_map.pth") == state["preserved_checkpoint_sha256"]
    else:
        assert [job["mode"] for job in state["jobs"]] == ["m0", "train", "evaluate"]
        m0, run = (Path(state["jobs"][i]["output_dir"]) for i in (0, 2))
        assert state["jobs"][1]["output_dir"] == str(run)
    result = verify(run, m0, dataset, state["variant"], manifest)
    return {"dataset": dataset, "variant": state["variant"],
            "run_dir": str(run), "m0_run_dir": str(m0),
            "campaign_dir": str(folder),
            "recovery_action": state.get("recovery_action", "fresh_full50"), **result}


def evaluate_existing(args):
    assert (args.dataset, args.variant) in REUSE
    manifest = patch.require_sources(args.campaign)
    m0, run = old_runs(args.original, args.dataset, args.variant)
    training = read(run / "training.json")
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    assert not (run / "official_metrics.json").exists()
    assert not (run / "official_distances.pt").exists()
    folder = queue.child_campaign(args.campaign, "patch_memory", args.dataset, args.variant)
    assert not folder.exists()
    folder.mkdir()
    row = {"mode": "evaluate", "status": "RUNNING", "started_at": queue.stamp(),
           "output_dir": str(run),
           "command": patch.command(args.dataset, args.variant, "evaluate", run)}
    state = {"status": "RUNNING", "dataset": args.dataset, "variant": args.variant,
             "gpu": args.gpu, "controller_pid": os.getpid(), "started_at": queue.stamp(),
             "recovery_action": "evaluate_existing", "m0_run": str(m0),
             "preserved_training_sha256": sha(run / "training.json"),
             "preserved_checkpoint_sha256": sha(run / "best_map.pth"), "jobs": [row]}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with (folder / "evaluate.log").open("x") as log:
        process = subprocess.Popen(row["command"], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        row["pid"] = process.pid
        queue.write(folder / "campaign.json", state)
        code = process.wait()
    row.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code,
               completed_at=queue.stamp())
    state.update(status=row["status"], completed_at=queue.stamp())
    queue.write(folder / "campaign.json", state)
    if code == 0:
        state["verification"] = endpoint(folder, args.dataset, manifest)
        queue.write(folder / "campaign.json", state)
    return code


def collect(campaign):
    manifest = patch.require_sources(campaign)
    rows = []
    for variant in patch.CONDITIONS:
        for dataset in queue.DATASETS:
            folder = queue.child_campaign(campaign, "patch_memory", dataset, variant)
            if not (folder / "campaign.json").exists():
                rows.append({"dataset": dataset, "variant": variant, "status": "PENDING"})
            elif read(folder / "campaign.json")["status"] != "COMPLETE":
                rows.append({"dataset": dataset, "variant": variant, "status": "UNACCEPTED"})
            else:
                rows.append(endpoint(folder, dataset, manifest))
    for dataset in queue.DATASETS:
        complete = [row for row in rows if row["dataset"] == dataset
                    and row["status"] == "VERIFIED_COMPLETE"]
        assert len({row["initial_model_state_sha256"] for row in complete}) <= 1
        assert len({row["trainable_parameters"] for row in complete}) <= 1
    return {"schema": "trifusion-patch-memory-roles-recovery-verification-v1",
            "collected_at": queue.stamp(), "expected_endpoints": 6,
            "verified_complete": sum(row["status"] == "VERIFIED_COMPLETE" for row in rows),
            "seed": 42, "scope": "Original verify() unchanged; two old full50 runs and four fresh attempts",
            "rows": rows}


def coordinate(args):
    assert read(args.original / "campaign.json")["status"] == "FAILED"
    manifest = patch.require_sources(args.original)
    witness = read(ROOT / "logs/patch_memory_resumed_device_check_20261001.json")
    assert len(witness["cuda_witnesses"]) == 4
    assert all(row["returncode"] == 0 for row in witness["cuda_witnesses"])
    assert all(witness["processes"][str(pid)]["state"] == "absent"
               for pid in (3680049, 3680054, 3680055))
    assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    shutil.copyfile(args.original / "manifest.json", args.campaign / "manifest.json")
    queue.write(args.campaign / "recovery_manifest.json", {
        "original_campaign": str(args.original), "original_campaign_sha256": sha(args.original / "campaign.json"),
        "original_manifest_sha256": sha(args.original / "manifest.json"),
        "recovery_script_sha256": sha(Path(__file__)),
        "device_witness_sha256": sha(ROOT / "logs/patch_memory_resumed_device_check_20261001.json"),
        "scope": "No training code, environment, seed, initialization, schedule, gate or failed receipt changes"})
    jobs = [{"phase": "patch_memory", "dataset": dataset, "variant": variant,
             "action": action, "status": "PENDING"}
            for pairs, action in ((REUSE, "evaluate_existing"), (NEW, "fresh_full50"))
            for dataset, variant in pairs]
    state = {"status": "RUNNING", "controller_pid": os.getpid(),
             "started_at": queue.stamp(), "jobs": jobs}

    def command(campaign, job, gpu):
        if job["action"] == "fresh_full50":
            return patch.start_command(campaign, job, gpu)
        return [sys.executable, "-B", str(Path(__file__).resolve()), "--evaluate-existing",
                "--campaign", str(campaign), "--original", str(args.original),
                "--dataset", job["dataset"], "--variant", job["variant"], "--gpu", str(gpu)]

    queue.start_command = command
    queue.require_complete = lambda folder, dataset: endpoint(folder, dataset, manifest)
    code = queue.run_phase(args.campaign, state, "patch_memory")
    if code == 0:
        accepted = collect(args.campaign)
        assert accepted["verified_complete"] == 6
        queue.write(args.campaign / "accepted_matrix.json", accepted)
    state.update(status="FAILED" if code else "COMPLETE", completed_at=queue.stamp())
    queue.write(args.campaign / "campaign.json", state)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--collect-only", type=Path)
    parser.add_argument("--dataset", choices=queue.DATASETS)
    parser.add_argument("--variant", choices=tuple(patch.CONDITIONS))
    parser.add_argument("--gpu", type=int, choices=range(4))
    args = parser.parse_args()
    args.campaign, args.original = args.campaign.resolve(), args.original.resolve()
    if args.collect_only:
        assert not args.collect_only.exists()
        queue.write(args.collect_only, collect(args.campaign))
        return 0
    if args.evaluate_existing:
        assert args.dataset and args.variant and args.gpu is not None
        return evaluate_existing(args)
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
