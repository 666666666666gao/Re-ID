#!/usr/bin/env python3
"""Check every RGBNT100 training montage, identity fold, and camera query mask."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_rgbnt100_signal_oof import configure, read_montage, camera_scores, sha256


def run(args):
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg, binding = configure(config)
    from data.datasets.bases import read_image
    from utils.metrics import eval_func
    import numpy as np

    protocol_path = ROOT / config["protocol"]
    assert sha256(protocol_path) == config["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = protocol["records"]
    assert len(records) == 8675
    dimensions = Counter()
    for row in records:
        path = Path(config["dataset_root"]) / row["path"]
        assert sha256(path) == row["sha256"], row["path"]
        actual, original = read_montage(path), read_image(str(path))
        assert len(actual) == len(original) == 3
        for left, right in zip(actual, original, strict=True):
            assert left.mode == right.mode == "RGB"
            assert left.size == right.size == (256, 128)
            assert left.tobytes() == right.tobytes(), row["path"]
        dimensions["768x128"] += 1
    all_heldout, all_queries = [], []
    fold_checks = []
    for fold in protocol["folds"]:
        source, gallery = set(fold["source_record_indices"]), fold["gallery_record_indices"]
        assert source.isdisjoint(gallery)
        assert source | set(gallery) == set(range(8675))
        assert {records[i]["identity"] for i in source} == set(fold["source_ids"])
        assert {records[i]["identity"] for i in gallery} == set(fold["heldout_ids"])
        assert set(fold["source_ids"]).isdisjoint(fold["heldout_ids"])
        assert {records[i]["camera"] for i in source} == set(range(8))
        assert fold["source_label_map"] == {str(pid): i for i, pid in enumerate(fold["source_ids"])}
        counts = Counter(records[i]["identity"] for i in gallery)
        camera_counts = Counter((records[i]["identity"], records[i]["camera"]) for i in gallery)
        query_lookup = {q["record_index"]: q for q in fold["query_rows"]}
        assert len(query_lookup) == len(fold["query_rows"])
        expected_queries = []
        for position, index in enumerate(gallery):
            row = records[index]
            positive_count = counts[row["identity"]] - camera_counts[row["identity"], row["camera"]]
            if positive_count:
                q = query_lookup[index]
                assert q == {"record_index": index, "gallery_position": position,
                             "identity": row["identity"], "valid_positive_count": positive_count}
                expected_queries.append(index)
        assert expected_queries == [q["record_index"] for q in fold["query_rows"]]
        all_heldout.extend(gallery)
        all_queries.extend(expected_queries)
        fold_checks.append({"fold": fold["fold"], "source_records": len(source),
                            "gallery_records": len(gallery), "query_records": len(expected_queries),
                            "source_identities": len(fold["source_ids"]),
                            "heldout_identities": len(fold["heldout_ids"]),
                            "all_query_positive_counts_match": True, "identity_isolation": True})
    assert sorted(all_heldout) == sorted(all_queries) == list(range(8675))
    # Hand-computable fixtures: preserve same-camera negatives and every legal positive.
    fixtures = [
        {"d": [0., .1, .3, .2, .4], "ids": [1, 1, 1, 2, 3],
         "cams": [0, 0, 1, 0, 1], "ap": .5, "rank": 2},
        {"d": [0., .1, .2, .3], "ids": [1, 1, 2, 1],
         "cams": [0, 1, 0, 2], "ap": 5 / 6, "rank": 1},
    ]
    fixture_results = []
    for f in fixtures:
        d, ids, cams = np.asarray([f["d"]]), np.asarray(f["ids"]), np.asarray(f["cams"])
        scores = camera_scores(d, np.asarray([1]), ids, np.asarray([0]), cams)
        cmc, ap = eval_func(d, np.asarray([1]), ids, np.asarray([0]), cams, max_rank=3)
        assert abs(scores["average_precision"][0] - f["ap"]) < 1e-15
        assert scores["first_match_rank"] == [f["rank"]]
        assert abs(float(ap) - f["ap"]) < 1e-15
        assert int(np.flatnonzero(cmc)[0]) + 1 == f["rank"]
        fixture_results.append({"expected_ap": f["ap"], "actual_ap": float(ap),
                                "first_match_rank": f["rank"], "synthetic_not_retrieval_result": True})
    output = args.output.resolve()
    assert not output.exists()
    report = {"status": "PASS_FULL_TRAIN_PROTOCOL_AND_MONTAGE",
              "config_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path),
              "runner_sha256": sha256(ROOT / "tools/train_rgbnt100_signal_oof.py"),
              "verifier_sha256": sha256(__file__), **binding,
              "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "training_montages_verified": 8675, "modality_crops_compared": 26025,
              "image_decodes": 17350, "dimensions": dict(dimensions), "folds": fold_checks,
              "camera_ranking_synthetic_fixtures": fixture_results,
              "model_forwards": 0, "optimizer_steps": 0, "real_retrieval_metric_calls": 0,
              "official_query_gallery_image_access": 0,
              "elapsed_seconds": time.perf_counter() - started}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

