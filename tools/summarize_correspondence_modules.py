#!/usr/bin/env python3
"""Summarize every module context after all 24 registered endpoints pass."""

import argparse
from datetime import datetime
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))


def contrasts(values):
    assert set(values) == set(VARIANTS)
    result = {}
    for size in (1, 2, 3):
        for selected in combinations(range(3), size):
            name = "x".join(f"M{index + 1}" for index in selected)
            result[name] = sum(
                value * product_sign(bits, selected) for bits, value in values.items()
            ) / (2 ** (3 - size))
    return result


def product_sign(bits, selected):
    sign = 1
    for index in selected:
        sign *= 1 if bits[index] == "1" else -1
    return sign


def dataset_summary(rows, dataset):
    rows = {row["variant"]: row for row in rows if row["dataset"] == dataset}
    assert set(rows) == set(VARIANTS)
    assert all(row["status"] == "VERIFIED_COMPLETE" for row in rows.values())
    metrics = ("mAP", "Rank-1", "Rank-5", "Rank-10") if dataset == "RGBNT201" else ("mAP", "Rank-1")
    average = {}
    for metric in metrics:
        for name, value in contrasts({bits: row["metrics"][metric] for bits, row in rows.items()}).items():
            average.setdefault(name, {})[metric] = value
    conditional = []
    for module in range(3):
        other = [index for index in range(3) if index != module]
        for context in product(("0", "1"), repeat=2):
            control = ["0"] * 3
            for index, bit in zip(other, context):
                control[index] = bit
            candidate = control.copy()
            candidate[module] = "1"
            control, candidate = "".join(control), "".join(candidate)
            before, after = rows[control], rows[candidate]
            conditional.append({
                "module": f"M{module + 1}", "control": control, "candidate": candidate,
                "delta_metrics": {name: after["metrics"][name] - before["metrics"][name] for name in metrics},
                "trainable_parameters_delta": after["trainable_parameters"] - before["trainable_parameters"],
                "training_and_epoch_eval_seconds_delta": after["training_and_epoch_eval_seconds"] - before["training_and_epoch_eval_seconds"],
            })
    return {"conditional_differences": conditional, "average_contrasts": average}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from tools.collect_correspondence_roles import collect
    matrix = collect()
    assert matrix["expected_endpoints"] == matrix["verified_complete"] == len(matrix["rows"]) == 24
    assert matrix["seed"] == 42
    report = {
        "status": "FULL_24_ENDPOINT_DESCRIPTIVE_MODULE_CONTRASTS_COMPLETE",
        "at": datetime.now().astimezone().isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definitions": {
            "conditional": "Candidate minus control, toggling exactly one module in each of its four other-module contexts.",
            "main": "Mean of four conditional metric differences for a module.",
            "pair": "Difference of differences for the two modules, averaged over the third module being off/on.",
            "triple": "Difference between the two pair differences when the remaining module is off/on.",
        },
        "limits": "All endpoints use official mAP-best selection and one training seed. Capacity, prediction inputs, random consumption and concurrent load can differ. These are descriptive configuration contrasts, not multi-seed uncertainty, unseen-test estimates or isolated causal effects.",
        "matrix": matrix,
        "datasets": {dataset: dataset_summary(matrix["rows"], dataset)
                     for dataset in ("RGBNT201", "RGBNT100", "MSVR310")},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "average_contrasts":
                      {dataset: value["average_contrasts"] for dataset, value in report["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
