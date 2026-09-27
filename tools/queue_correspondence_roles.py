#!/usr/bin/env python3
"""Run M0 then 50-epoch correspondence-role experiments after the V8 control."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "logs/official_extra_seed42_RGBNT201_SIGNAL_V8_signal_v8_horizon50_control_v1_bestmap_e50_20260926/campaign.json"
AFTER_PID = 820904
CAMPAIGN = ROOT / "logs/correspondence_roles_v1_20260928"
WEIGHTS = ROOT / "pertrained-model"
PROTOCOLS = ROOT / "logs/official_three_dataset_protocols_20260923"
SOURCE = ROOT / "comparators/Signal-cd1b0a6"
BASELINES = {
    "RGBNT201": ("RGBNT201_PlainBaseline_50.pth", "789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407"),
    "MSVR310": ("MSVR310_PlainBaseline_50.pth", "69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37"),
    "RGBNT100": ("RGBNT100_PlainBaseline_30.pth", "299a28bfb3e3180eeae0736cf8638cd162525dce0f2192a940b62b97f6e67dcd"),
}


def stamp():
    return datetime.now().astimezone().isoformat()


def save(status):
    (CAMPAIGN / "campaign.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def command(dataset, mode, output_dir):
    weight, digest = BASELINES[dataset]
    return [sys.executable, "-B", str(ROOT / "tools/run_correspondence_roles.py"),
            "--dataset", dataset, "--mode", mode,
            "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE),
            "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight),
            "--baseline-sha256", digest,
            "--output-dir", str(output_dir), "--seed", "42", "--epochs", "50"]


def run(dataset, mode, output_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    path = CAMPAIGN / f"{dataset}.{mode}.log"
    with path.open("x", encoding="utf-8") as log:
        return subprocess.run(command(dataset, mode, output_dir), cwd=ROOT,
                              env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    assert not CAMPAIGN.exists()
    CAMPAIGN.mkdir(parents=True)
    status = {"schema": "trifusion-correspondence-roles-queue-v1",
              "status": "WAITING", "gpu": 2, "after": str(AFTER),
              "after_pid": AFTER_PID,
              "started_at": stamp(), "jobs": []}
    save(status)
    while True:
        parent = json.loads(AFTER.read_text(encoding="utf-8"))
        if parent["status"] == "COMPLETE":
            break
        if parent["status"] == "FAILED":
            status.update(status="STOPPED_PARENT_FAILED", completed_at=stamp())
            save(status)
            return 1
        if not Path(f"/proc/{AFTER_PID}").exists():
            status.update(status="STOPPED_PARENT_INCOMPLETE", completed_at=stamp())
            save(status)
            return 1
        time.sleep(240)
    status["status"] = "RUNNING"
    save(status)
    for dataset in BASELINES:
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            status.update(status="STOPPED_LOW_DISK", completed_at=stamp())
            save(status)
            return 1
        for mode in ("m0", "train", "evaluate"):
            suffix = "m0" if mode == "m0" else "full"
            output_dir = ROOT / f"trained-model/correspondence_roles_v1_{dataset}_seed42_{suffix}_20260928"
            row = {"dataset": dataset, "mode": mode, "output_dir": str(output_dir),
                   "status": "RUNNING", "started_at": stamp()}
            status["jobs"].append(row)
            save(status)
            code = run(dataset, mode, output_dir)
            row.update(status="COMPLETE" if code == 0 else "FAILED",
                       exit_code=code, completed_at=stamp())
            save(status)
            if code:
                status.update(status="FAILED", completed_at=stamp())
                save(status)
                return code
    status.update(status="COMPLETE", completed_at=stamp())
    save(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
