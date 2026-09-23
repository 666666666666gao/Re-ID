#!/usr/bin/env python3
"""Resume the active official campaign on four GPUs without repeating completed work."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import queue_official_three_dataset_two_gpu as base


ADOPTED_TRAIN_PID = 2408556
PREVIOUS_QUEUE_PID = 2404577


def complete_adopted_training(status, row):
    directory = base.TRAIN_DIR / "RGBNT100_R2_seed43"
    receipt = directory / "training.json"
    while True:
        training = json.loads(receipt.read_text(encoding="utf-8"))
        if training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE":
            break
        assert Path(f"/proc/{ADOPTED_TRAIN_PID}").exists()
        time.sleep(180)
    assert training["seed"] == 43
    base.set_status(status, row, "EVALUATING", trained_at=base.stamp())
    base.run_command(row, "evaluate", directory)
    retrieval = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
    assert retrieval["status"] == "COMPLETE" and retrieval["seed"] == 43
    base.set_status(status, row, "COMPLETE", completed_at=base.stamp(),
                    metrics_path=str(directory / "official_metrics.json"),
                    free_disk_bytes=base.shutil.disk_usage(base.BASE).free)


def main():
    assert ROOT == Path("/data/gaob/Re-ID/Trifusion")
    assert Path(f"/proc/{PREVIOUS_QUEUE_PID}").exists()
    command = Path(f"/proc/{ADOPTED_TRAIN_PID}/cmdline").read_bytes()
    assert b"--dataset\x00RGBNT100" in command and b"--method\x00R2" in command
    assert b"--mode\x00train" in command and b"--seed\x0043" in command
    status = json.loads((base.CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))
    jobs = status["jobs"]
    adopted = next(row for row in jobs if row["dataset"] == "RGBNT100"
                   and row["method"] == "R2" and row["seed"] == 43)
    completed = next(row for row in jobs if row["dataset"] == "RGBNT100"
                     and row["method"] == "V27" and row["seed"] == 43)
    assert adopted["status"] == "TRAINING" and completed["status"] == "COMPLETE"
    assert status["status"] == "RUNNING"
    assert base.shutil.disk_usage(base.BASE).free > 3 * 1024**3

    pending = iter(row for row in jobs if row["status"] == "PENDING")
    status.update(schema="trifusion-official-four-gpu-resume-v1",
                  resumed_at=base.stamp(), previous_queue_pid=PREVIOUS_QUEUE_PID,
                  adopted_training_pid=ADOPTED_TRAIN_PID,
                  resume_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip())
    base.save(status)

    def worker(gpu):
        if gpu == 0:
            complete_adopted_training(status, adopted)
        while True:
            with base.LOCK:
                row = next(pending, None)
                if row is None:
                    return
                row["gpu"] = gpu
                base.save(status)
            base.run_job(status, row)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(4)]
        for future in futures:
            future.result()
    status["status"] = "COMPLETE"
    status["completed_at"] = base.stamp()
    base.save(status)


if __name__ == "__main__":
    main()
