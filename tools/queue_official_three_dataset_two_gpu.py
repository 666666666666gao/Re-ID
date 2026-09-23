#!/usr/bin/env python3
"""Run the fixed-epoch R2/V27 seeds 43/44 on two GPUs, one dataset at a time."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT.parent
CAMPAIGN = ROOT / "logs/official_r2_v27_two_gpu_20260923"
WEIGHT_DIR = ROOT / "pertrained-model"
TRAIN_DIR = ROOT / "trained-model/official_r2_v27_two_gpu_20260923"
PROTOCOL_DIR = ROOT / "logs/official_three_dataset_protocols_20260923"
SOURCE = ROOT / "comparators/Signal-cd1b0a6"
DATASETS = ("RGBNT100", "MSVR310", "RGBNT201")
WEIGHTS = {
    "RGBNT100": ("RGBNT100_Signal_30.pth", "09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860"),
    "MSVR310": ("MSVR310_Signal_50.pth", "b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a"),
    "RGBNT201": ("RGBNT201_Signal_50.pth", "ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c"),
}
LOCK = Lock()


def stamp():
    return datetime.now().astimezone().isoformat()


def save(status):
    (CAMPAIGN / "campaign.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def set_status(status, row, value, **fields):
    with LOCK:
        row.update(status=value, **fields)
        save(status)


def run_command(row, mode, directory):
    name, digest = WEIGHTS[row["dataset"]]
    tag = f'{row["dataset"]}_{row["method"]}_seed{row["seed"]}'
    command = [sys.executable, "-B", str(ROOT / "tools/run_official_three_dataset_roles.py"),
               "--dataset", row["dataset"], "--method", row["method"], "--mode", mode,
               "--protocol", str(PROTOCOL_DIR / f'{row["dataset"]}.json'),
               "--signal-source", str(SOURCE), "--clip-weight", str(WEIGHT_DIR / "ViT-B-16.pt"),
               "--signal-checkpoint", str(WEIGHT_DIR / name), "--signal-sha256", digest,
               "--output-dir", str(directory), "--seed", str(row["seed"])]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    with (CAMPAIGN / f"{tag}.{mode}.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run_job(status, row):
    assert shutil.disk_usage(BASE).free > 3 * 1024**3
    tag = f'{row["dataset"]}_{row["method"]}_seed{row["seed"]}'
    m0 = CAMPAIGN / "m0" / tag
    set_status(status, row, "M0", started_at=stamp())
    run_command(row, "m0", m0)
    receipt = json.loads((m0 / "training.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "M0_PASS" and receipt["seed"] == row["seed"]
    directory = TRAIN_DIR / tag
    set_status(status, row, "TRAINING", m0_at=stamp())
    run_command(row, "train", directory)
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    assert training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
    assert training["seed"] == row["seed"]
    set_status(status, row, "EVALUATING", trained_at=stamp())
    run_command(row, "evaluate", directory)
    retrieval = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
    assert retrieval["status"] == "COMPLETE" and retrieval["seed"] == row["seed"]
    set_status(status, row, "COMPLETE", completed_at=stamp(),
               metrics_path=str(directory / "official_metrics.json"),
               free_disk_bytes=shutil.disk_usage(BASE).free)


def main():
    from tools.official_three_dataset_model import sha256

    assert ROOT == Path("/data/gaob/Re-ID/Trifusion")
    assert not CAMPAIGN.exists()
    assert (SOURCE / "utils/metrics.py").is_file()
    assert (WEIGHT_DIR / "ViT-B-16.pt").is_file()
    for dataset in DATASETS:
        name, digest = WEIGHTS[dataset]
        assert sha256(WEIGHT_DIR / name) == digest
        assert (PROTOCOL_DIR / f"{dataset}.json").is_file()
    assert shutil.disk_usage(BASE).free > 3 * 1024**3
    CAMPAIGN.mkdir(parents=True)
    jobs = [dict(dataset=dataset, method=method, seed=seed, gpu=gpu, status="PENDING")
            for dataset in DATASETS for seed in (43, 44)
            for gpu, method in enumerate(("R2", "V27"))]
    status = dict(schema="trifusion-official-two-gpu-seeds43-44-v1", status="RUNNING",
                  started_at=stamp(), fixed_epoch=20, dataset_order=DATASETS,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(), jobs=jobs)
    save(status)
    for dataset in DATASETS:
        for seed in (43, 44):
            wave = [row for row in jobs if row["dataset"] == dataset and row["seed"] == seed]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_job, status, row) for row in wave]
                for future in futures:
                    future.result()
    status["status"] = "COMPLETE"
    status["completed_at"] = stamp()
    save(status)


if __name__ == "__main__":
    main()
