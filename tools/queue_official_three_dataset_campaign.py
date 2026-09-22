#!/usr/bin/env python3
"""Run the six reviewed fixed-endpoint jobs sequentially on one GPU."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path("/root/autodl-tmp/trifusion-v2")
CAMPAIGN = BASE / "artifacts/official_r2_v27_campaign_20260923"
SOURCE = BASE / "comparators/Signal-cd1b0a6"
CLIP = BASE / "pretrained/ViT-B-16.pt"
WEIGHTS = {
    "MSVR310": ("MSVR310_Signal_50.pth", "b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a"),
    "RGBNT100": ("RGBNT100_Signal_30.pth", "09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860"),
    "RGBNT201": ("RGBNT201_Signal_50.pth", "ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c"),
}
JOBS = [(method, dataset) for method in ("R2", "V27")
        for dataset in ("MSVR310", "RGBNT100", "RGBNT201")]


def stamp():
    return datetime.now().astimezone().isoformat()


def write_status(value):
    (CAMPAIGN / "campaign.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(method, dataset, mode, directory):
    weight, digest = WEIGHTS[dataset]
    return [sys.executable, "-B", str(ROOT / "tools/run_official_three_dataset_roles.py"),
            "--dataset", dataset, "--method", method, "--mode", mode,
            "--protocol", str(BASE / "artifacts/official_three_dataset_protocols_20260923" / f"{dataset}.json"),
            "--signal-source", str(SOURCE), "--clip-weight", str(CLIP),
            "--signal-checkpoint", str(BASE / "author_signal_pretrained_20260923" / weight),
            "--signal-sha256", digest, "--output-dir", str(directory)]


def main():
    assert not (CAMPAIGN / "campaign.json").exists()
    from tools.official_three_dataset_model import sha256

    for method, dataset in JOBS:
        receipt = json.loads((CAMPAIGN / "m0" / f"{dataset}_{method}" / "training.json").read_text())
        assert receipt["status"] == "M0_PASS"
        weight, digest = WEIGHTS[dataset]
        assert sha256(BASE / "author_signal_pretrained_20260923" / weight) == digest
    status = dict(schema="trifusion-official-six-job-campaign-v1", status="RUNNING",
                  started_at=stamp(), commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  jobs=[dict(method=method, dataset=dataset, status="PENDING") for method, dataset in JOBS])
    write_status(status)
    for row in status["jobs"]:
        method, dataset = row["method"], row["dataset"]
        assert shutil.disk_usage(BASE).free > 3 * 1024**3
        directory = CAMPAIGN / "train" / f"{dataset}_{method}"
        assert not directory.exists()
        row["status"] = "TRAINING"
        row["started_at"] = stamp()
        write_status(status)
        with (CAMPAIGN / f"{dataset}_{method}.train.log").open("x", encoding="utf-8") as log:
            subprocess.run(command(method, dataset, "train", directory), cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        training = json.loads((directory / "training.json").read_text())
        assert training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
        row["status"] = "EVALUATING"
        row["trained_at"] = stamp()
        write_status(status)
        with (CAMPAIGN / f"{dataset}_{method}.evaluate.log").open("x", encoding="utf-8") as log:
            subprocess.run(command(method, dataset, "evaluate", directory), cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        retrieval = json.loads((directory / "official_metrics.json").read_text())
        assert retrieval["status"] == "COMPLETE"
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
