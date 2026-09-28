#!/usr/bin/env python3
"""Render extreme RGBNT201 flips from verified distances, without inference."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import sha, camera_scores


def triplet(root, record):
    assert len(record["paths"]) == 3
    canvas = Image.new("RGB", (288, 192), "white")
    for modal, path in enumerate(record["paths"]):
        with Image.open(root / path) as original:
            thumbnail = ImageOps.contain(original.convert("RGB"), (96, 192))
            canvas.paste(thumbnail, (modal * 96 + (96 - thumbnail.width) // 2,
                                     (192 - thumbnail.height) // 2))
    return canvas


def legal_order(data, query):
    order = np.argsort(data["fused"][query].numpy())
    excluded = ((data["gallery_ids"][order] == data["query_ids"][query]) &
                (data["gallery_cameras"][order] == data["query_cameras"][query]))
    return order[~excluded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows = {row["variant"]: row for row in matrix["rows"]
            if row["dataset"] == "RGBNT201" and row["status"] == "VERIFIED_COMPLETE"}
    protocol_path = ROOT / "logs/official_three_dataset_protocols_20260923/RGBNT201.json"
    protocol = json.loads(protocol_path.read_text())
    dataset_root = Path(protocol["dataset_root"])
    query_records, gallery_records = (protocol["records"][name] for name in ("query", "gallery"))
    arrays, scores, bindings = {}, {}, {}
    for bits in ("000", "111"):
        row = rows[bits]
        run = Path(row["run_dir"])
        assert sha(run / "official_metrics.json") == row["receipt_sha256"]
        assert sha(run / "official_distances.pt") == row["distance_sha256"]
        receipt = json.loads((run / "official_metrics.json").read_text())
        assert receipt["protocol_sha256"] == sha(protocol_path)
        data = torch.load(run / "official_distances.pt", map_location="cpu", weights_only=False)
        for split, records in (("query", query_records), ("gallery", gallery_records)):
            for column in ("identity", "camera"):
                name = "ids" if column == "identity" else "cameras"
                assert np.array_equal(data[f"{split}_{name}"], [record[column] for record in records])
        scores[bits] = camera_scores(data["fused"].numpy(), data["query_ids"], data["gallery_ids"],
                                     data["query_cameras"], data["gallery_cameras"])
        assert all(abs(scores[bits]["metrics"][name] - row["metrics"][name]) < 1e-5
                   for name in row["metrics"])
        arrays[bits] = data
        bindings[bits] = {name: row[name] for name in
                          ("best_epoch", "checkpoint_sha256", "distance_sha256", "receipt_sha256")}
    first = np.asarray(scores["000"]["first_match_rank"])
    second = np.asarray(scores["111"]["first_match_rank"])
    delta = np.asarray(scores["111"]["average_precision"]) - scores["000"]["average_precision"]
    pools = {"new_errors": np.flatnonzero((first == 1) & (second != 1)),
             "repairs": np.flatnonzero((first != 1) & (second == 1))}
    report = {"dataset": "RGBNT201", "bindings": bindings, "selection":
              "Two largest AP decreases among new R1 errors and two largest increases among repairs; extreme post-selection illustrations, not representative samples or training labels.",
              "pool_sizes": {key: len(value) for key, value in pools.items()}, "cases": {}}
    top_cameras = {bits: np.asarray([data["gallery_cameras"][legal_order(data, query)[0]]
                                    for query in range(len(first))]) for bits, data in arrays.items()}
    contexts = (("000_all_errors", "000", np.flatnonzero(first != 1)),
                ("111_all_errors", "111", np.flatnonzero(second != 1)),
                ("111_new_errors", "111", pools["new_errors"]),
                ("000_repaired_errors", "000", pools["repairs"]))
    report["top_negative_camera"] = {
        name: {"queries": len(indices), "same_camera_as_query":
               int((top_cameras[bits][indices] == arrays[bits]["query_cameras"][indices]).sum())}
        for name, bits, indices in contexts}
    for kind, pool in pools.items():
        chosen = sorted(pool.tolist(), key=lambda index: (delta[index] if kind == "new_errors"
                                                         else -delta[index], index))[:2]
        assert len(chosen) == 2
        fig, axes = plt.subplots(6, 3, figsize=(12, 12), layout="constrained")
        cases = []
        for case_number, query in enumerate(chosen):
            base_row = case_number * 3
            record = query_records[query]
            orders = {bits: legal_order(data, query) for bits, data in arrays.items()}
            true_positions = {}
            case = {"query_index": query, "identity": record["identity"], "camera": record["camera"],
                    "query_paths": record["paths"], "delta_ap_points": float(delta[query] * 100), "outputs": {}}
            for bits, order in orders.items():
                positions = np.flatnonzero(arrays[bits]["gallery_ids"][order] == record["identity"])
                assert positions[0] + 1 == scores[bits]["first_match_rank"][query]
                true_positions[bits] = int(order[positions[0]])
                case["outputs"][bits] = {"AP": scores[bits]["average_precision"][query],
                                        "first_match_rank": int(positions[0] + 1),
                                        "top3_gallery_indices": order[:3].tolist(),
                                        "first_positive_gallery_index": true_positions[bits]}
            axes[base_row, 0].imshow(triplet(dataset_root, record))
            axes[base_row, 0].set_title(f"Query q{query} | ID {record['identity']} | cam_idx {record['camera']}")
            axes[base_row, 1].text(0.02, 0.8,
                f"000: AP {scores['000']['average_precision'][query] * 100:.2f}% | first positive {first[query]}\n"
                f"111: AP {scores['111']['average_precision'][query] * 100:.2f}% | first positive {second[query]}\n"
                f"AP change: {delta[query] * 100:+.2f} pp\n\nTriplets: RGB / NIR / TIR\nGreen: same identity; red: different identity",
                transform=axes[base_row, 1].transAxes, va="top", fontsize=11)
            positive = gallery_records[true_positions["111"]]
            axes[base_row, 2].imshow(triplet(dataset_root, positive))
            axes[base_row, 2].set_title(f"111 first legal positive: rank {second[query]}")
            for row_offset, bits in enumerate(("000", "111"), 1):
                for column, gallery in enumerate(orders[bits][:3]):
                    ax = axes[base_row + row_offset, column]
                    item = gallery_records[int(gallery)]
                    correct = item["identity"] == record["identity"]
                    ax.imshow(triplet(dataset_root, item))
                    ax.set_title(f"{bits} rank {column + 1} | ID {item['identity']} | cam_idx {item['camera']}",
                                 color="#16803b" if correct else "#b52525", fontsize=10)
            for ax in axes[base_row:base_row + 3].flat:
                ax.set_axis_off()
            cases.append(case)
        fig.suptitle(f"RGBNT201 | 000 vs 111 | extreme {kind.replace('_', ' ')}\n"
                     "Same-ID same-camera exclusions applied; saved best checkpoints; no new inference", fontsize=13)
        path = args.output_dir / f"correspondence_roles_RGBNT201_000_vs111_{kind}_20260928.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        report["cases"][kind] = {"image": str(path), "image_sha256": sha(path), "rows": cases}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pool_sizes": report["pool_sizes"], "top_negative_camera": report["top_negative_camera"],
                      "cases": {kind: value["rows"] for kind, value in report["cases"].items()}}))


if __name__ == "__main__":
    main()
