#!/usr/bin/env python3
"""Run one additional fixed-epoch seed under the existing official protocol."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.official_three_dataset_model import sha256


DATASETS = ("RGBNT100", "MSVR310", "RGBNT201")
METHODS = ("R2", "V27")
WEIGHTS = {
    "RGBNT100": ("RGBNT100_Signal_30.pth", "09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860"),
    "MSVR310": ("MSVR310_Signal_50.pth", "b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a"),
    "RGBNT201": ("RGBNT201_Signal_50.pth", "ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c"),
}
PLAIN_WEIGHTS = {
    "RGBNT201": ("RGBNT201_PlainBaseline_50.pth", "789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407"),
    "RGBNT100": ("RGBNT100_PlainBaseline_30.pth", "299a28bfb3e3180eeae0736cf8638cd162525dce0f2192a940b62b97f6e67dcd"),
    "MSVR310": ("MSVR310_PlainBaseline_50.pth", "69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37"),
}
LOCK = Lock()


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--machine", choices=("old", "new"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=(*METHODS, "R2_TOP1", "R2_UNIFORM", "PLAIN_V8"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--top1-pair", action="store_true")
    parser.add_argument("--skip-cell", action="append",
                        choices=[f"{dataset}:{method}" for dataset in DATASETS for method in METHODS])
    parser.add_argument("--overlap-previous", action="store_true")
    args = parser.parse_args()
    assert args.seed >= 42
    if args.top1_pair:
        assert args.machine == "new" and args.dataset is None and args.method is None
        assert args.gpu is None and not args.skip_cell and not args.overlap_previous

    if args.machine == "old":
        assert not args.overlap_previous
        assert ROOT == Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
        base = ROOT.parent
        source = base / "comparators/Signal-cd1b0a6"
        weights = base / "author_signal_pretrained_20260923"
        clip = base / "pretrained/ViT-B-16.pt"
        protocols = base / "artifacts/official_three_dataset_protocols_20260923"
        gpus = (0,)
    else:
        assert ROOT == Path("/data/gaob/Re-ID/Trifusion")
        base = ROOT
        source = ROOT / "comparators/Signal-cd1b0a6"
        weights = ROOT / "pertrained-model"
        clip = weights / "ViT-B-16.pt"
        protocols = ROOT / "logs/official_three_dataset_protocols_20260923"
        gpus = (0, 1, 2, 3)
        previous = ROOT / "logs/official_r2_v27_two_gpu_20260923/campaign.json"
        if args.overlap_previous:
            prior = json.loads(previous.read_text(encoding="utf-8"))
            if args.seed == 46:
                assert args.dataset is None and args.method is None and args.gpu is None
                assert all(row["status"] != "PENDING" for row in prior["jobs"])
            else:
                assert args.seed >= 48
                allowed = {("RGBNT100", "R2", 0), ("RGBNT100", "V27", 1),
                           ("MSVR310", "R2", 2), ("MSVR310", "V27", 3)}
                assert (args.dataset, args.method, args.gpu) in allowed
                seed46 = json.loads((ROOT / "logs/official_extra_seed46_20260924/campaign.json").read_text(encoding="utf-8"))
                gpu_rows = [row for row in seed46["jobs"] if row.get("gpu") == args.gpu]
                assert gpu_rows and all(row["status"] == "COMPLETE" for row in gpu_rows)
                assert all(row["status"] == "COMPLETE" for row in prior["jobs"] if row.get("gpu") == args.gpu)
        else:
            while json.loads(previous.read_text(encoding="utf-8"))["status"] != "COMPLETE":
                time.sleep(240)

    if args.gpu is not None:
        assert args.gpu in gpus
        gpus = (args.gpu,)
    datasets = (args.dataset,) if args.dataset else (("MSVR310", "RGBNT201", "RGBNT100")
                                                  if args.top1_pair else DATASETS)
    methods = (args.method,) if args.method else (("R2", "R2_TOP1") if args.top1_pair else METHODS)

    assert (source / "utils/metrics.py").is_file() and clip.is_file()
    for dataset in datasets:
        name, digest = (PLAIN_WEIGHTS if args.method == "PLAIN_V8" else WEIGHTS)[dataset]
        assert sha256(weights / name) == digest
        assert (protocols / f"{dataset}.json").is_file()
    assert shutil.disk_usage(base).free > 3 * 1024**3

    suffix = ("_top1_pair" if args.top1_pair else
              f"_{args.dataset}_{args.method}" if args.dataset and args.method else "")
    date_suffix = "20260925" if args.method == "PLAIN_V8" else "20260924"
    campaign = ROOT / f"logs/official_extra_seed{args.seed}{suffix}_{date_suffix}"
    train_root = ROOT / f"trained-model/official_extra_seed{args.seed}{suffix}_{date_suffix}"
    assert not campaign.exists() and not train_root.exists()
    campaign.mkdir(parents=True)
    train_root.mkdir(parents=True)
    skipped = set(args.skip_cell or ())
    jobs = [dict(dataset=dataset, method=method, seed=args.seed, status="PENDING")
            for dataset in datasets for method in methods
            if f"{dataset}:{method}" not in skipped]
    assert jobs
    status = dict(schema="trifusion-official-extra-seed-v1", status="RUNNING",
                  seed=args.seed, machine=args.machine, fixed_epoch=20,
                  started_at=stamp(), dataset_order=datasets,
                  skipped_cells=sorted(skipped),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(), jobs=jobs)

    def save():
        (campaign / "campaign.json").write_text(json.dumps(status, indent=2) + "\n",
                                                  encoding="utf-8")

    def set_status(row, value, **fields):
        with LOCK:
            row.update(status=value, **fields)
            save()

    def run_command(row, mode, directory):
        name, digest = (PLAIN_WEIGHTS if row["method"] == "PLAIN_V8" else WEIGHTS)[row["dataset"]]
        tag = f'{row["dataset"]}_{row["method"]}_seed{args.seed}'
        command = [sys.executable, "-B", str(ROOT / "tools/run_official_three_dataset_roles.py"),
                   "--dataset", row["dataset"], "--method", row["method"], "--mode", mode,
                   "--protocol", str(protocols / f'{row["dataset"]}.json'),
                   "--signal-source", str(source), "--clip-weight", str(clip),
                   "--signal-checkpoint", str(weights / name), "--signal-sha256", digest,
                   "--output-dir", str(directory), "--seed", str(args.seed)]
        if row["method"] == "PLAIN_V8":
            command.extend(["--baseline-receipt", str(ROOT / "logs/signal_plain_baseline_20260924_r2" /
                                                        row["dataset"] / "metrics.json")])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
        with (campaign / f"{tag}.{mode}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    def run_job(row):
        assert shutil.disk_usage(base).free > 3 * 1024**3
        tag = f'{row["dataset"]}_{row["method"]}_seed{args.seed}'
        if row["method"] == "PLAIN_V8":
            preflight = campaign / "preflight" / tag
            set_status(row, "PREFLIGHT", started_at=stamp())
            run_command(row, "preflight", preflight)
            assert json.loads((preflight / "baseline_parity.json").read_text(encoding="utf-8"))["status"] == "PASS"
        m0 = campaign / "m0" / tag
        set_status(row, "M0", started_at=stamp())
        run_command(row, "m0", m0)
        receipt = json.loads((m0 / "training.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "M0_PASS" and receipt["seed"] == args.seed
        directory = train_root / tag
        set_status(row, "TRAINING", m0_at=stamp())
        run_command(row, "train", directory)
        training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        assert training["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
        assert training["seed"] == args.seed
        set_status(row, "EVALUATING", trained_at=stamp())
        run_command(row, "evaluate", directory)
        retrieval = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
        assert retrieval["status"] == "COMPLETE" and retrieval["seed"] == args.seed
        set_status(row, "COMPLETE", completed_at=stamp(),
                   metrics_path=str(directory / "official_metrics.json"),
                   free_disk_bytes=shutil.disk_usage(base).free)

    def worker(gpu, pending):
        while True:
            with LOCK:
                row = next(pending, None)
                if row is None:
                    return
                row["gpu"] = gpu
                save()
            if args.overlap_previous:
                while any(previous_row.get("gpu") == gpu and previous_row["status"] != "COMPLETE"
                          for previous_row in json.loads(previous.read_text(encoding="utf-8"))["jobs"]):
                    time.sleep(240)
            run_job(row)

    save()
    pending = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, pending) for gpu in gpus]
        for future in futures:
            future.result()
    status["status"] = "COMPLETE"
    status["completed_at"] = stamp()
    save()


if __name__ == "__main__":
    main()
