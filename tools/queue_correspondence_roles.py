#!/usr/bin/env python3
"""Run M0 then 50-epoch correspondence-role experiments after the V8 control."""

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
AFTER = ROOT / "logs/official_extra_seed42_RGBNT201_SIGNAL_V8_signal_v8_horizon50_control_v1_bestmap_e50_20260926/campaign.json"
AFTER_PID = 820904
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


def save(campaign, status):
    (campaign / "campaign.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def command(dataset, variant, mode, output_dir):
    weight, digest = BASELINES[dataset]
    result = [sys.executable, "-B", str(ROOT / "tools/run_correspondence_roles.py"),
            "--dataset", dataset, "--mode", mode,
            "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE),
            "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight),
            "--baseline-sha256", digest,
            "--output-dir", str(output_dir), "--seed", "42", "--epochs", "50"]
    result.extend(f"--m{index}" if enabled == "1" else f"--no-m{index}"
                  for index, enabled in enumerate(variant, start=1))
    return result


def run(args, dataset, variant, mode, output_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    path = args.campaign / f"{dataset}.{variant}.{mode}.log"
    with path.open("x", encoding="utf-8") as log:
        return subprocess.run(command(dataset, variant, mode, output_dir), cwd=ROOT,
                              env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=range(4), default=2)
    parser.add_argument("--variants", nargs="+", choices=("000", "001", "010", "011", "100", "101", "110", "111"), default=["111"])
    parser.add_argument("--datasets", nargs="+", choices=tuple(BASELINES), default=list(BASELINES))
    parser.add_argument("--campaign", type=Path, default=ROOT / "logs/correspondence_roles_v1_20260928")
    parser.add_argument("--after-campaign", type=Path, default=AFTER)
    parser.add_argument("--after-pid", type=int, default=AFTER_PID)
    args = parser.parse_args()
    assert len(args.variants) == len(set(args.variants))
    assert len(args.datasets) == len(set(args.datasets))
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    status = {"schema": "trifusion-correspondence-roles-queue-v1",
              "status": "WAITING", "gpu": args.gpu, "after": str(args.after_campaign),
              "after_pid": args.after_pid, "variants": args.variants, "datasets": args.datasets,
              "started_at": stamp(), "jobs": []}
    save(args.campaign, status)
    while True:
        parent = json.loads(args.after_campaign.read_text(encoding="utf-8"))
        if parent["status"] == "COMPLETE":
            break
        if parent["status"] == "FAILED":
            status.update(status="STOPPED_PARENT_FAILED", completed_at=stamp())
            save(args.campaign, status)
            return 1
        if not Path(f"/proc/{args.after_pid}").exists():
            status.update(status="STOPPED_PARENT_INCOMPLETE", completed_at=stamp())
            save(args.campaign, status)
            return 1
        time.sleep(240)
    status["status"] = "RUNNING"
    save(args.campaign, status)
    for variant, dataset in ((variant, dataset) for variant in args.variants for dataset in args.datasets):
        for mode in ("m0", "train", "evaluate"):
            suffix = "m0" if mode == "m0" else "full"
            memory = int(subprocess.check_output(
                ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            while memory >= 500:
                status["status"] = "WAITING_GPU"
                save(args.campaign, status)
                time.sleep(240)
                memory = int(subprocess.check_output(
                    ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                status.update(status="STOPPED_LOW_DISK", completed_at=stamp())
                save(args.campaign, status)
                return 1
            status["status"] = "RUNNING"
            output_dir = ROOT / f"trained-model/{args.campaign.name}_{dataset}_m{variant}_seed42_{suffix}"
            row = {"dataset": dataset, "variant": variant, "mode": mode, "output_dir": str(output_dir),
                   "status": "RUNNING", "started_at": stamp()}
            status["jobs"].append(row)
            save(args.campaign, status)
            code = run(args, dataset, variant, mode, output_dir)
            if code == 0:
                receipt_path = output_dir / ("official_metrics.json" if mode == "evaluate" else "training.json")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE", "evaluate": "COMPLETE"}[mode]
                code = 0 if receipt["status"] == expected else 1
                row["receipt"] = str(receipt_path)
            row.update(status="COMPLETE" if code == 0 else "FAILED",
                       exit_code=code, completed_at=stamp())
            save(args.campaign, status)
            if code:
                status.update(status="FAILED", completed_at=stamp())
                save(args.campaign, status)
                return code
    status.update(status="COMPLETE", completed_at=stamp())
    save(args.campaign, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
