"""Analyze the closed FP32 slot panel using bound full-gallery evidence only."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.audit_correspondence_context_identity_losses import audit
from tools.diagnose_patch_memory_readout import analyze

DATASETS = ("RGBNT201", "RGBNT100", "MSVR310")
VARIANTS = ("independent", "competitive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    assert matrix["schema"] == "trifusion-slot-competition-fp32-roles-verification-v2"
    assert matrix["verified_complete"] == matrix["expected_endpoints"] == len(matrix["rows"]) == 6
    assert {(row["dataset"], row["variant"]) for row in matrix["rows"]} == {
        (dataset, variant) for dataset in DATASETS for variant in VARIANTS}
    assert all(row["status"] == "VERIFIED_COMPLETE" for row in matrix["rows"])
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    sources = {str(args.matrix): sha(args.matrix)}
    rows, histories, pairs, slot_rows = [], {}, [], []
    for row in matrix["rows"]:
        readout = analyze(row)
        run = Path(row["run_dir"])
        losses = audit(run, "none")
        histories[row["dataset"], row["variant"]] = losses["history"]
        steps = [json.loads(line) for line in (run / "training_steps.jsonl").read_text().splitlines()]
        for name in ("training.json", "training_steps.jsonl", "official_metrics.json"):
            sources[str(run / name)] = sha(run / name)
        rows.append({**row, "readout": readout,
                     "final_metrics": losses["last"]["official_fused"],
                     "final_minus_best_metrics_pp": {
                         key: losses["last"]["official_fused"][key] - value
                         for key, value in row["metrics"].items()},
                     "best_mean_loss": losses["best"]["loss"],
                     "final_mean_loss": losses["last"]["loss"],
                     "after_best_steps": sum(step["epoch"] > row["best_epoch"] for step in steps),
                     "after_best_positive_triplet_steps": sum(
                         step["epoch"] > row["best_epoch"] and step["triplet"] > 0 for step in steps)})
    for dataset in DATASETS:
        control, candidate = (next(row for row in rows if row["dataset"] == dataset
                                   and row["variant"] == variant) for variant in VARIANTS)
        assert control["initial_model_state_sha256"] == candidate["initial_model_state_sha256"]
        assert control["trainable_parameters"] == candidate["trainable_parameters"]
        paired = compare(matrix, dataset, *VARIANTS)
        paired["boundary"] = ("Matched initial state and trainable parameter count; fixed seed42, full50, "
                              "one official mAP best per endpoint. Official post-selection diagnosis. "
                              "Bootstrap resamples fixed-model identities, not training seeds or "
                              "an untouched test set. No new inference or selection.")
        required_map = 0.5 if dataset in ("RGBNT201", "MSVR310") else 0.0
        delta = paired["delta_metrics"]
        passed = delta["mAP"] > 0 and delta["mAP"] >= required_map and delta["Rank-1"] >= 0
        pairs.append({"dataset": dataset, "required_positive_map_pp": required_map,
                      "registered_gate_passed": passed, "paired_diagnosis": paired})
        complete_path = args.slots / dataset / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        assert complete["status"] == "COMPLETE_PAIRED_SLOT_DIAGNOSIS"
        assert complete["dataset"] == dataset and complete["matrix_sha256"] == sha(args.matrix)
        assert complete["source_sha256"] == sha(ROOT / "tools/diagnose_slot_competition_fp32_slots.py")
        sources[str(complete_path)] = sha(complete_path)
        for variant in VARIANTS:
            path = args.slots / dataset / (variant + ".json")
            measure = json.loads(path.read_text())
            row = next(row for row in rows if row["dataset"] == dataset and row["variant"] == variant)
            assert complete["endpoint_reports"][variant] == sha(path)
            assert measure["status"] == "COMPLETE_FULL_GALLERY_SLOT_DIAGNOSIS"
            assert measure["dataset"] == dataset and measure["variant"] == variant
            assert measure["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert measure["protocol_sha256"] == row["readout"]["bindings"]["protocol_sha256"]
            assert measure["source_sha256"] == complete["source_sha256"]
            assert measure["model_state_unchanged"]
            assert all(value == 0 for value in measure["first_batch_instrumentation_max_difference"].values())
            assert max(measure["full_gallery_metric_difference_pp"].values()) < 1e-5
            for split in ("query", "gallery"):
                assert measure["statistics"][split]["rows"] == row["readout"][split + "_count"]
            means = np.asarray(measure["statistics"]["query"]["role_modality_mean"]).mean(axis=(0, 1))
            slot_rows.append({"dataset": dataset, "variant": variant,
                              "query_rows": measure["statistics"]["query"]["rows"],
                              "gallery_rows": measure["statistics"]["gallery"]["rows"],
                              "query_role_modality_mean": dict(zip(measure["statistic_names"], means.tolist()))})
            sources[str(path)] = sha(path)
    args.output_dir.mkdir()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 2, figsize=(9.5, 8.5), layout="constrained")
    for index, dataset in enumerate(DATASETS):
        for variant, color in zip(VARIANTS, ("#D55E00", "#0072B2")):
            history = histories[dataset, variant]
            selected = next(row for row in rows if row["dataset"] == dataset
                            and row["variant"] == variant)["best_epoch"]
            epochs = [item["epoch"] for item in history]
            maps = [item["official_fused"]["mAP"] for item in history]
            axes[index, 0].plot(epochs, maps, color=color, label=variant, linewidth=1.5)
            axes[index, 0].scatter(selected, maps[selected - 1], color=color, s=22, zorder=3)
            axes[index, 1].plot(epochs, [item["loss"] for item in history], color=color, linewidth=1.5)
        axes[index, 0].set_title(dataset + ": official mAP (dot = selected best)")
        axes[index, 1].set_title(dataset + ": mean training loss")
        axes[index, 0].set_ylabel("mAP (%)")
        axes[index, 1].set_ylabel("ID + Triplet")
        for axis in axes[index]:
            axis.set_xlabel("Role-stage epoch")
            axis.set_xlim(1, 50)
            axis.grid(alpha=0.2)
        axes[index, 0].legend(frameon=False)
    for name in ("trajectories.png", "trajectories.svg"):
        figure.savefig(args.output_dir / name, dpi=180)
    plt.close(figure)
    report = {"schema": "slot-competition-fp32-six-end-complete-analysis-v2",
              "analyzed_at": datetime.now().astimezone().isoformat(),
              "source_sha256": sha(Path(__file__)), "accepted": 6,
              "rows": rows, "pairs": pairs, "slot_statistics": slot_rows,
              "registered_advancement_gate": "PASS" if all(pair["registered_gate_passed"] for pair in pairs) else "FAIL",
              "source_artifacts_sha256": sources,
              "reused_analysis_source_sha256": {name: sha(ROOT / "tools" / name) for name in (
                  "analyze_correspondence_distances.py", "audit_correspondence_context_identity_losses.py",
                  "diagnose_patch_memory_readout.py")},
              "figure_sha256": {name: sha(args.output_dir / name) for name in ("trajectories.png", "trajectories.svg")},
              "boundaries": ["Within-checkpoint readouts are not independently trained controls.",
                             "Slot statistics precede anchor addition and cross-role bridging; CNN spatial convolution has already run.",
                             "Low slot similarity is not physical correspondence or unique causal proof.",
                             "Loss scalars and gradient activity are not AdamW update shares.",
                             "R1 failure and valid R1 controls remain separate; no failed receipt is rewritten."]}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": 6, "gate": report["registered_advancement_gate"],
                      "paired_deltas": [{"dataset": pair["dataset"], "metrics": pair["paired_diagnosis"]["delta_metrics"]}
                                        for pair in pairs], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
