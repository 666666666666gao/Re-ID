#!/usr/bin/env python3
"""Run fixed-epoch R2/V27 seeds 43 and 44 after the seed-42 campaign."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.queue_official_three_dataset_campaign import BASE, CAMPAIGN as FIRST_CAMPAIGN, JOBS, command

CAMPAIGN = BASE / "artifacts/official_r2_v27_seeds43_44_20260923"
FIRST_PID = 10763


def stamp():
    return datetime.now().astimezone().isoformat()


def write_status(value):
    (CAMPAIGN / "campaign.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_command(seed, method, dataset, mode, directory, log_path):
    cmd = command(method, dataset, mode, directory) + ["--seed", str(seed)]
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    assert not CAMPAIGN.exists()
    CAMPAIGN.mkdir(parents=True)
    jobs = [dict(seed=seed, method=method, dataset=dataset, status="PENDING")
            for seed in (43, 44) for method, dataset in JOBS]
    status = dict(schema="trifusion-official-additional-seeds-campaign-v1",
                  status="WAITING_FOR_SEED42", started_at=stamp(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  fixed_epoch=20, jobs=jobs)
    write_status(status)
    while True:
        first = json.loads((FIRST_CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))
        if first["status"] == "COMPLETE":
            assert all(row["status"] == "COMPLETE" for row in first["jobs"])
            break
        assert (Path("/proc") / str(FIRST_PID) / "cmdline").exists()
        assert b"queue_official_three_dataset_campaign.py" in (
            Path("/proc") / str(FIRST_PID) / "cmdline").read_bytes()
        time.sleep(240)
    status["status"] = "RUNNING"
    write_status(status)
    for row in status["jobs"]:
        seed, method, dataset = row["seed"], row["method"], row["dataset"]
        assert shutil.disk_usage(BASE).free > 3 * 1024**3
        tag = f"seed{seed}_{dataset}_{method}"
        m0 = CAMPAIGN / "m0" / tag
        row["status"] = "M0"
        row["started_at"] = stamp()
        write_status(status)
        run_command(seed, method, dataset, "m0", m0, CAMPAIGN / f"{tag}.m0.log")
        m0_receipt = json.loads((m0 / "training.json").read_text(encoding="utf-8"))
        assert m0_receipt["status"] == "M0_PASS" and m0_receipt["seed"] == seed
        directory = CAMPAIGN / "train" / tag
        row["status"] = "TRAINING"
        write_status(status)
        run_command(seed, method, dataset, "train", directory, CAMPAIGN / f"{tag}.train.log")
        training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        assert training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE" and training["seed"] == seed
        row["status"] = "EVALUATING"
        row["trained_at"] = stamp()
        write_status(status)
        run_command(seed, method, dataset, "evaluate", directory, CAMPAIGN / f"{tag}.evaluate.log")
        retrieval = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
        assert retrieval["status"] == "COMPLETE" and retrieval["seed"] == seed
        row["status"] = "COMPLETE"
        row["completed_at"] = stamp()
        row["metrics_path"] = str(directory / "official_metrics.json")
        row["free_disk_bytes"] = shutil.disk_usage(BASE).free
        write_status(status)
    status["status"] = "COMPLETE"
    status["completed_at"] = stamp()
    write_status(status)


if __name__ == "__main__":
    main()
