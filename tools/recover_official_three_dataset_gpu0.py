#!/usr/bin/env python3
"""Continue the seed43/44 campaign after GPU 0 disappeared during RGBNT100 R2."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_official_three_dataset_four_gpu as original

BASE = Path("/data/gb")
SOURCE = BASE / "artifacts/official_r2_v27_four_gpu_20260923/campaign.json"
CAMPAIGN = BASE / "artifacts/official_r2_v27_three_gpu_recovery_20260923"
WEIGHTS = ROOT / "pretained/official_r2_v27_three_gpu_recovery_20260923"
ORIGINAL_PID = 147101
original.CAMPAIGN = CAMPAIGN


def stamp():
    return datetime.now().astimezone().isoformat()


def source_jobs():
    return json.loads(SOURCE.read_text(encoding="utf-8"))["jobs"]


def run_job(status, row):
    assert shutil.disk_usage(BASE).free > 3 * 1024**3
    tag = f"{row['dataset']}_{row['method']}_seed{row['seed']}"
    m0 = CAMPAIGN / "m0" / tag
    original.set_status(status, row, "M0", started_at=stamp())
    original.run_command(row, "m0", m0)
    receipt = json.loads((m0 / "training.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "M0_PASS" and receipt["seed"] == row["seed"]
    directory = WEIGHTS / tag
    original.set_status(status, row, "TRAINING", m0_at=stamp())
    original.run_command(row, "train", directory)
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    assert training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
    assert training["seed"] == row["seed"]
    original.set_status(status, row, "EVALUATING", trained_at=stamp())
    original.run_command(row, "evaluate", directory)
    metrics = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "COMPLETE" and metrics["seed"] == row["seed"]
    original.set_status(status, row, "COMPLETE", completed_at=stamp(),
                        metrics_path=str(directory / "official_metrics.json"),
                        free_disk_bytes=shutil.disk_usage(BASE).free)


def run_wave(status, dataset):
    wave = [row for row in status["jobs"] if row["dataset"] == dataset]
    lanes = [[row for row in wave if row["gpu"] == gpu] for gpu in (1, 2, 3)]

    def run_lane(rows):
        for row in rows:
            run_job(status, row)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_lane, rows) for rows in lanes]
        for future in futures:
            future.result()


def main():
    assert not CAMPAIGN.exists()
    assert SOURCE.is_file()
    assert Path("/proc/147101").exists()
    assert not Path("/proc/149985").exists()
    rows = [dict(dataset="RGBNT100", method="R2", seed=43, gpu=2, status="WAITING_FOR_GPU2")]
    for dataset in ("MSVR310", "RGBNT201"):
        rows.extend(dict(dataset=dataset, method=method, seed=seed, gpu=gpu, status="PENDING")
                    for method, seed, gpu in (("R2", 43, 1), ("R2", 44, 2),
                                              ("V27", 43, 3), ("V27", 44, 3)))
    CAMPAIGN.mkdir(parents=True)
    status = dict(schema="trifusion-official-gpu0-recovery-v1", status="WAITING_FOR_GPU2",
                  started_at=stamp(), commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  cause="GPU 0 lost after RGBNT100 R2 seed43 CUDA failure",
                  source_campaign=str(SOURCE), fixed_epoch=20, jobs=rows)
    original.save(status)

    while not any(j["dataset"] == "RGBNT100" and j["method"] == "V27" and
                  j["seed"] == 43 and j["status"] == "COMPLETE" for j in source_jobs()):
        time.sleep(240)
    status["status"] = "RUNNING"
    original.save(status)
    run_job(status, rows[0])

    while Path(f"/proc/{ORIGINAL_PID}").exists():
        time.sleep(240)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert source["jobs"][0]["status"] == "TRAINING"
    assert all(row["status"] == "COMPLETE" for row in source["jobs"][1:4])
    source["jobs"][0]["status"] = "HARDWARE_FAILED"
    source["status"] = "INTERRUPTED_GPU0_LOST"
    source["recovery_campaign"] = str(CAMPAIGN / "campaign.json")
    SOURCE.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    for dataset in ("MSVR310", "RGBNT201"):
        run_wave(status, dataset)
    status["status"] = "COMPLETE"
    status["completed_at"] = stamp()
    original.save(status)


if __name__ == "__main__":
    main()
