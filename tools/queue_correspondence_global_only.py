#!/usr/bin/env python3
"""Run one global-only control after all 24 registered endpoints pass."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_correspondence_roles import collect
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS


def stamp():
    return datetime.now().astimezone().isoformat()


def command(dataset, mode, output_dir):
    weight, digest = BASELINES[dataset]
    return [sys.executable, "-B", str(ROOT / "tools/run_correspondence_global_only.py"),
            "--dataset", dataset, "--mode", mode,
            "--protocol", str(PROTOCOLS / f"{dataset}.json"),
            "--signal-source", str(SOURCE),
            "--clip-weight", str(WEIGHTS / "ViT-B-16.pt"),
            "--baseline-checkpoint", str(WEIGHTS / weight),
            "--baseline-sha256", digest,
            "--output-dir", str(output_dir), "--seed", "42", "--epochs", "50"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BASELINES), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    assert not args.campaign.exists()
    matrix = collect()
    assert matrix["expected_endpoints"] == matrix["verified_complete"] == len(matrix["rows"]) == 24
    args.campaign.mkdir(parents=True)
    (args.campaign / "parent_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    status = {"schema": "trifusion-correspondence-global-only-queue-v1",
              "status": "RUNNING", "dataset": args.dataset, "gpu": args.gpu,
              "controller_pid": os.getpid(), "started_at": stamp(), "jobs": []}
    state_path = args.campaign / "campaign.json"
    state_path.write_text(json.dumps(status, indent=2) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for mode in ("m0", "train", "evaluate"):
        memory = int(subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip())
        assert memory < 500
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        suffix = "m0" if mode == "m0" else "full"
        output_dir = ROOT / f"trained-model/{args.campaign.name}_{args.dataset}_seed42_{suffix}"
        row = {"mode": mode, "output_dir": str(output_dir), "status": "RUNNING",
               "started_at": stamp(), "command": command(args.dataset, mode, output_dir)}
        with (args.campaign / f"{mode}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(row["command"], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            status["jobs"].append(row)
            state_path.write_text(json.dumps(status, indent=2) + "\n")
            code = process.wait()
        if code == 0:
            receipt_path = output_dir / ("official_metrics.json" if mode == "evaluate" else "training.json")
            receipt = json.loads(receipt_path.read_text())
            expected = {"m0": "M0_PASS", "train": "BEST_OFFICIAL_MAP_TRAINING_COMPLETE",
                        "evaluate": "COMPLETE"}[mode]
            code = 0 if receipt["status"] == expected else 1
            row["receipt"] = str(receipt_path)
        row.update(status="COMPLETE" if code == 0 else "FAILED",
                   exit_code=code, completed_at=stamp())
        status["status"] = "FAILED" if code else "RUNNING"
        state_path.write_text(json.dumps(status, indent=2) + "\n")
        if code:
            return code
    status.update(status="COMPLETE", completed_at=stamp())
    state_path.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
