#!/usr/bin/env python3
"""Plot completed 000/011/111 trajectories without new inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("RGBNT201", "MSVR310"), default="RGBNT201")
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "logs/correspondence_roles_matrix_011_window_20260928.json")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows = {row["variant"]: row for row in matrix["rows"]
            if row["dataset"] == args.dataset and row["status"] == "VERIFIED_COMPLETE"}
    colors = {"000": "#566573", "011": "#d97732", "111": "#2475b5"}
    labels = {"000": "000: role core, all modules off", "011": "011: M2 + M3",
              "111": "111: M1 + M2 + M3"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, layout="constrained")
    for bits in ("000", "011", "111"):
        run = Path(rows[bits]["run_dir"])
        history = json.loads((run / "training.json").read_text())["history"]
        assert [row["epoch"] for row in history] == list(range(1, 51))
        metrics = [row["official_fused"]["mAP"] for row in history]
        axes[0].plot(range(1, 51), metrics, color=colors[bits], label=labels[bits], linewidth=1.6)
        epoch = rows[bits]["best_epoch"]
        assert abs(metrics[epoch - 1] - rows[bits]["metrics"]["mAP"]) < 1e-5
        axes[0].scatter([epoch], [metrics[epoch - 1]], color=colors[bits], s=36, zorder=3)
        offset = (-7, -16) if bits == "000" else (5, 7)
        axes[0].annotate(f"E{epoch}: {metrics[epoch - 1]:.2f}", (epoch, metrics[epoch - 1]),
                         xytext=offset, textcoords="offset points", fontsize=8,
                         color=colors[bits], ha="right" if bits == "000" else "left")
        steps = [json.loads(line) for line in (run / "training_steps.jsonl").read_text().splitlines()]
        id_means = [np.mean([row["id"] for row in steps if row["epoch"] == epoch])
                    for epoch in range(1, 51)]
        assert np.isfinite(id_means).all()
        axes[1].plot(range(1, 51), id_means, color=colors[bits], linewidth=1.6)
    baseline = {"RGBNT201": 69.64150314770879, "MSVR310": 50.52198102035894}[args.dataset]
    axes[0].axhline(baseline, linestyle="--", linewidth=1, color="#8c8c8c",
                    label="Frozen pure CLIP ReID baseline")
    axes[0].set_ylabel("Official mAP (%)")
    axes[0].set_title(f"{args.dataset} | seed 42 | all 50 epochs completed")
    legend = {"RGBNT201": {"loc": "upper right"},
              "MSVR310": {"loc": "upper left", "bbox_to_anchor": (1.01, 1)}}
    axes[0].legend(fontsize=8, frameon=False, **legend[args.dataset])
    axes[1].set_ylabel("Mean identity loss")
    axes[1].set_xlabel("Training epoch")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1, 50)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    path = ROOT / f"docs/assets/correspondence_roles_{args.dataset}_trajectories_20260928.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
