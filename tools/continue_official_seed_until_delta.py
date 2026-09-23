#!/usr/bin/env python3
"""Continue official seeds until each assigned cell meets every required metric."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.official_three_dataset_model import sha256


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=("old", "new"), required=True)
    parser.add_argument("--min-gain", type=float, required=True)
    parser.add_argument("--early-msvr-r2", action="store_true")
    parser.add_argument("--new-cell", choices=("rgbnt100_r2", "rgbnt100_v27", "msvr310_v27"))
    args = parser.parse_args()
    assert 0 < args.min_gain <= 1.0
    assert not (args.early_msvr_r2 and args.new_cell)

    if args.machine == "old":
        assert not args.early_msvr_r2 and not args.new_cell
        assert ROOT == Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
        cells = (("RGBNT201", "R2"), ("RGBNT201", "V27"))
        gpus = (0,)
        next_seed = {cell: 47 for cell in cells}
        prior = ROOT / "logs/official_extra_seed45_20260924/campaign.json"
        historic = ROOT.parent / "artifacts/official_r2_v27_campaign_20260923/train"
        roots = (historic, ROOT / "trained-model/official_extra_seed45_20260924")
        baseline_paths = {"RGBNT201": historic / "RGBNT201_R2/official_metrics.json"}
    else:
        assert ROOT == Path("/data/gaob/Re-ID/Trifusion")
        assignments = {"rgbnt100_r2": (("RGBNT100", "R2"), 0),
                       "rgbnt100_v27": (("RGBNT100", "V27"), 1),
                       "msvr310_v27": (("MSVR310", "V27"), 3)}
        if args.early_msvr_r2:
            cells, gpus = (("MSVR310", "R2"),), (2,)
        elif args.new_cell:
            cell, gpu = assignments[args.new_cell]
            cells, gpus = (cell,), (gpu,)
        else:
            cells = (("RGBNT100", "R2"), ("RGBNT100", "V27"), ("MSVR310", "V27"))
            gpus = (0, 1, 3)
        next_seed = {cell: 48 for cell in cells}
        prior = ROOT / "logs/official_extra_seed46_20260924/campaign.json"
        historic = ROOT / "trained-model/official_r2_v27_two_gpu_20260923"
        roots = (historic, ROOT / "trained-model/official_extra_seed46_20260924")
        baseline_paths = {
            "RGBNT100": historic / "RGBNT100_V27_seed44/official_metrics.json",
            "MSVR310": historic / "MSVR310_R2_seed43/official_metrics.json",
        }

    suffix = "_msvr_r2" if args.early_msvr_r2 else (f"_{args.new_cell}" if args.new_cell else "")
    status_path = ROOT / f"logs/official_target_continuation_{args.machine}{suffix}_20260924.json"
    assert not status_path.exists()
    watch_gpu = gpus[0] if args.early_msvr_r2 or args.new_cell else None
    while True:
        if prior.exists():
            previous = json.loads(prior.read_text(encoding="utf-8"))
            if watch_gpu is not None:
                rows = [row for row in previous["jobs"] if row.get("gpu") == watch_gpu]
                assert rows
                if all(row["status"] == "COMPLETE" for row in rows):
                    break
            elif previous["status"] == "COMPLETE":
                break
        time.sleep(240)

    names = {"RGBNT201": ("mAP", "Rank-1", "Rank-5", "Rank-10"),
             "RGBNT100": ("mAP", "Rank-1"), "MSVR310": ("mAP", "Rank-1")}
    baseline = {}
    for dataset, path in baseline_paths.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["status"] == "COMPLETE" and result["dataset"] == dataset
        baseline[dataset] = result["outputs"]["baseline_only"]["metrics"]

    def rank(cell, metrics):
        dataset, _ = cell
        gains = [metrics[name] - baseline[dataset][name] for name in names[dataset]]
        passed = all(gain >= args.min_gain for gain in gains)
        return (passed, metrics["mAP"], metrics["Rank-1"])

    selected = {}
    for root in roots:
        for path in root.glob("*/official_metrics.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            cell = (result["dataset"], result["method"])
            if cell not in cells:
                continue
            assert result["status"] == "COMPLETE"
            actual = result["outputs"]["baseline_only"]["metrics"]
            assert all(actual[name] == baseline[cell[0]][name] for name in names[cell[0]])
            entry = dict(seed=result["seed"], metrics=result["outputs"]["fused"]["metrics"],
                         receipt=str(path))
            if cell not in selected or rank(cell, entry["metrics"]) > rank(cell, selected[cell]["metrics"]):
                selected[cell] = entry
    assert set(selected) == set(cells)

    status = dict(schema="trifusion-official-target-continuation-v1", status="RUNNING",
                  machine=args.machine, min_gain_pp=args.min_gain,
                  required_metrics={dataset: names[dataset] for dataset, _ in cells},
                  baseline=baseline, started_at=stamp(), selected={}, jobs=[])
    adaptive_winner = {}

    def save():
        status["selected"] = {f"{dataset}_{method}": entry for (dataset, method), entry in selected.items()}
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def run_job(cell, seed, gpu):
        dataset, method = cell
        command = [sys.executable, "-u", str(ROOT / "tools/queue_official_extra_seed.py"),
                   "--machine", args.machine, "--seed", str(seed),
                   "--dataset", dataset, "--method", method, "--gpu", str(gpu)]
        if watch_gpu is not None:
            command.append("--overlap-previous")
        log = ROOT / f"logs/official_extra_seed{seed}_{dataset}_{method}_20260924.launch.log"
        with log.open("x", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        directory = ROOT / f"trained-model/official_extra_seed{seed}_{dataset}_{method}_20260924/{dataset}_{method}_seed{seed}"
        result = json.loads((directory / "official_metrics.json").read_text(encoding="utf-8"))
        training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        assert result["status"] == "COMPLETE" and result["seed"] == seed
        assert (result["dataset"], result["method"]) == cell
        assert result["role_checkpoint_sha256"] == training["checkpoint_sha256"]
        return result, training, directory

    save()
    rotation = 0
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        running = {}

        def schedule(gpu):
            nonlocal rotation
            active = [cell for cell in cells if not rank(cell, selected[cell]["metrics"])[0]]
            if not active:
                return
            cell = active[rotation % len(active)]
            rotation += 1
            seed = next_seed[cell]
            next_seed[cell] += 2
            future = pool.submit(run_job, cell, seed, gpu)
            running[future] = (cell, seed, gpu)

        for gpu in gpus:
            schedule(gpu)
        while running:
            future = next(as_completed(tuple(running)))
            cell, seed, gpu = running.pop(future)
            result, training, directory = future.result()
            score = result["outputs"]["fused"]["metrics"]
            path = Path(training["checkpoint"])
            assert path == directory / "roles_epoch20.pth"
            assert sha256(path) == training["checkpoint_sha256"]
            better = rank(cell, score) > rank(cell, selected[cell]["metrics"])
            if better:
                if cell in adaptive_winner:
                    old_path, old_sha, old_row = adaptive_winner[cell]
                    assert sha256(old_path) == old_sha
                    old_path.unlink()
                    old_row["checkpoint_retained"] = False
                selected[cell] = dict(seed=seed, metrics=score,
                                      receipt=str(directory / "official_metrics.json"))
            else:
                path.unlink()
            row = dict(dataset=cell[0], method=cell[1], seed=seed, gpu=gpu,
                       metrics=score, checkpoint_retained=better, completed_at=stamp())
            if better:
                adaptive_winner[cell] = (path, training["checkpoint_sha256"], row)
            status["jobs"].append(row)
            save()
            schedule(gpu)
    status["status"] = "TARGET_MET"
    status["completed_at"] = stamp()
    save()


if __name__ == "__main__":
    main()
