#!/usr/bin/env python3
"""Run the registered 50-epoch panel after each GPU's existing queue finishes."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "logs/official_50_epoch_panel_20260926"
GPU_UUIDS = {
    0: "GPU-6a0b725d-15aa-9ec3-a292-1367f3c33f8f",
    1: "GPU-bc13d99e-845a-f985-262e-1918d2a39d6b",
    2: "GPU-442835f1-3e38-07d1-c823-57f7b12ad5bd",
    3: "GPU-a594c014-6f57-08e1-3b65-06e99b6e312a",
}
LANES = {
    0: ("logs/rgbnt100_r2_seed43_wait_20260926/status.json",
        (("RGBNT100", "R2"),)),
    1: ("logs/branch_only_launch_20260926/driver_status.json",
        (("MSVR310", "R2"),)),
    2: ("logs/rgbnt201_r2_seed44_wait_20260926/status.json",
        (("RGBNT201", "V27"), ("RGBNT201", "R2"))),
    3: ("logs/rgbnt100_v27_seed45_wait_20260926/status.json",
        (("MSVR310", "V27"), ("RGBNT100", "V27"))),
}


def stamp():
    return datetime.now().astimezone().isoformat()


def gpu_idle(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    return not any(line.split(",", 1)[0].strip() == GPU_UUIDS[gpu]
                   for line in output.splitlines())


def campaign_path(dataset, method):
    name = f"official_extra_seed42_{dataset}_{method}_horizon50_v1_bestmap_e50_20260926"
    return ROOT / "logs" / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=tuple(LANES), required=True)
    args = parser.parse_args()
    gpu = args.gpu
    prerequisite, jobs = LANES[gpu]
    prerequisite = ROOT / prerequisite
    PANEL.mkdir(parents=True, exist_ok=True)
    status_path = PANEL / f"lane_gpu{gpu}.json"
    assert not status_path.exists()
    status = dict(status="WAITING", gpu=gpu, gpu_uuid=GPU_UUIDS[gpu],
                  prerequisite=str(prerequisite), jobs=[dict(dataset=d, method=m, status="PENDING")
                                                     for d, m in jobs], started_at=stamp())

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save()
    while True:
        previous = json.loads(prerequisite.read_text(encoding="utf-8"))
        if previous["status"] in ("COMPLETE", "SKIPPED_TARGET_MET"):
            break
        assert previous["status"] in ("WAITING", "RUNNING", "TRAINING", "EVALUATING")
        time.sleep(240)

    for row in status["jobs"]:
        while not gpu_idle(gpu):
            time.sleep(240)
        assert shutil.disk_usage(ROOT).free > 10 * 1024**3
        dataset, method = row["dataset"], row["method"]
        campaign = campaign_path(dataset, method)
        assert not campaign.exists()
        row.update(status="RUNNING", campaign=str(campaign), started_at=stamp())
        save()
        command = [sys.executable, "-B", str(ROOT / "tools/queue_official_extra_seed.py"),
                   "--seed", "42", "--machine", "new", "--dataset", dataset,
                   "--method", method, "--gpu", str(gpu), "--checkpoint-policy",
                   "best_official_map", "--epochs", "50", "--run-label", "horizon50_v1"]
        with (PANEL / f"{dataset}_{method}_gpu{gpu}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        receipt = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "COMPLETE" and receipt["training_epochs"] == 50
        assert len(receipt["jobs"]) == 1 and receipt["jobs"][0]["status"] == "COMPLETE"
        row.update(status="COMPLETE", completed_at=stamp(),
                   metrics_path=receipt["jobs"][0]["metrics_path"])
        save()
    status.update(status="COMPLETE", completed_at=stamp())
    save()


if __name__ == "__main__":
    main()
