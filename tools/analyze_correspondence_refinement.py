#!/usr/bin/env python3
"""Compare one completed five-cell M2 panel using verified gallery distances."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.queue_correspondence_refinement import DATASETS, VARIANTS


CONTRASTS = (
    ("single_pooled", "uniform_pooled", "candidate_neighborhood_uniform_average"),
    ("uniform_pooled", "query_pooled", "content_selection_with_same_candidates"),
    ("single_pooled", "query_pooled", "candidate_neighborhood_and_content_selection"),
    ("single_pooled", "single_regions", "structured_readout_with_single_sampling"),
    ("query_pooled", "query_regions", "structured_readout_with_content_selection"),
    ("single_regions", "query_regions", "candidate_neighborhood_and_content_selection_with_regions"),
    ("original_111", "single_pooled", "unchanged_forward_replication"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--original-matrix", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    original = json.loads(args.original_matrix.read_text())
    assert matrix["schema"] == "trifusion-correspondence-refinement-verification-v1"
    assert matrix["expected_endpoints"] == len(matrix["rows"]) == 18
    assert original["expected_endpoints"] == original["verified_complete"] == len(original["rows"]) == 24
    global_rows = [row for row in matrix["rows"] if row["phase"] == "global"]
    assert {row["dataset"] for row in global_rows} == set(DATASETS)
    assert all(row["status"] == "VERIFIED_COMPLETE" for row in global_rows)
    rows = [row for row in matrix["rows"]
            if row["phase"] == "m2" and row["dataset"] == args.dataset]
    assert len(rows) == len(VARIANTS) and {row["variant"] for row in rows} == set(VARIANTS)
    assert all(row["status"] == "VERIFIED_COMPLETE" for row in rows)
    original_row = next(row for row in original["rows"]
                        if row["dataset"] == args.dataset and row["variant"] == "111")
    rows.append({**original_row, "variant": "original_111"})
    trainings = [json.loads((Path(row["run_dir"]) / "training.json").read_text()) for row in rows]
    first = trainings[0]
    for training in trainings:
        for name in ("dataset", "protocol_sha256", "seed", "epochs", "checkpoint_policy"):
            assert training[name] == first[name]
        for name in ("author_checkpoint_sha256", "width", "fused_width", "m1", "m2", "m3",
                     "prediction_weight", "learning_rate", "weight_decay", "model_source_sha256",
                     "initial_model_state_sha256", "trainable_parameters"):
            assert training["initializer"][name] == first["initializer"][name]
    args.output_dir.mkdir()
    pairs = []
    for control, candidate, question in CONTRASTS:
        result = compare({"rows": rows}, args.dataset, control, candidate)
        result["question"] = question
        path = args.output_dir / f"{control}_to_{candidate}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        pairs.append({key: value for key, value in result.items() if key != "identity_changes"})
        pairs[-1].update(report=str(path), report_sha256=sha(path))
    summary = {
        "status": "COMPLETE_FIVE_CELL_M2_PAIRED_DIAGNOSIS",
        "at": datetime.now().astimezone().isoformat(), "dataset": args.dataset,
        "source_sha256": sha(Path(__file__)),
        "matrix": str(args.matrix), "matrix_sha256": sha(args.matrix),
        "original_matrix": str(args.original_matrix), "original_matrix_sha256": sha(args.original_matrix),
        "pairs": pairs,
        "boundary": "Only full-50-epoch/reload accepted endpoints; CPU full-gallery diagnosis. "
                    "Single-seed official best selection, not independent test or seed uncertainty. "
                    "Sampling and readout also change M3 inputs/targets; teacher/student addresses "
                    "remain independently predicted. Equal parameter initialization does not ensure "
                    "equal numerical or optimization paths. No training, reweighting or new inference.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
