#!/usr/bin/env python3
"""Replay source-only camera-label support with the unchanged training sampler."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from trifusion.aligned_data import CrossCameraIdentitySampler


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    started = time.time()
    assert sha(args.geometry) == args.geometry_sha256
    assert sha(args.m0) == args.m0_sha256
    geometry = json.loads(args.geometry.read_bytes())
    m0 = json.loads(args.m0.read_bytes())
    assert len(geometry["folds"]) == len(m0["preflight"]) == 3
    sampler_path = Path(inspect.getfile(CrossCameraIdentitySampler))
    report = {
        "status": "SOURCE_CAMERA_METADATA_REPLAY_COMPLETE",
        "evaluation_type": "train_source_label_and_sampler_metadata_only_no_model_execution",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha(Path(__file__)),
        "sampler_sha256": sha(sampler_path),
        "geometry_sha256": args.geometry_sha256,
        "m0_sha256": args.m0_sha256,
        "seed": 42,
        "epochs_replayed": 20,
        "model_instantiations": 0,
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_tensor_loads": 0,
        "image_reads": 0,
        "heldout_metric_computations": 0,
        "dev_access_count": 0,
        "official_test_access_count": 0,
        "folds": [],
    }
    for fold in geometry["folds"]:
        number = fold["fold"]
        manifest = fold["endpoints"]["frozen_private_tail"]["source"]["gallery_manifest"]
        assert manifest == fold["endpoints"]["trained_private_tail"]["source"]["gallery_manifest"]
        original = m0["preflight"][number]["endpoints"][0]
        assert fold["source_identity_mapping"] == original["binding"]["fit_identity_ids"]
        records = [((r["file"],) * 3, r["identity"], r["camera"], 0) for r in manifest]
        by_identity = defaultdict(set)
        for row in manifest:
            by_identity[row["identity"]].add(row["camera"])
        source_camera_counts = dict(Counter(r["camera"] for r in manifest))
        sampler = CrossCameraIdentitySampler(records, batch_size=64, num_instances=8, seed=42)
        batches, prefix_equal = [], []
        for epoch in range(1, 21):
            indices = list(iter(sampler))
            assert len(indices) == len(sampler) and len(indices) % 64 == 0
            for step, offset in enumerate(range(0, len(indices), 64), 1):
                selected = indices[offset:offset + 64]
                rows = [manifest[index] for index in selected]
                identities = np.array([row["identity"] for row in rows])
                cameras = np.array([row["camera"] for row in rows])
                same_id = identities[:, None] == identities[None, :]
                same_camera = cameras[:, None] == cameras[None, :]
                positive = same_id & ~np.eye(64, dtype=bool)
                negative = ~same_id
                cross_positive_count = (positive & ~same_camera).sum(axis=1)
                same_negative_count = (negative & same_camera).sum(axis=1)
                other_negative_count = (negative & ~same_camera).sum(axis=1)
                assert np.all(positive.sum(axis=1) == 7)
                assert len(set(identities.tolist())) == 8
                group_cameras = [len(set(cameras[identities == identity].tolist())) for identity in sorted(set(identities.tolist()))]
                paths = [row["file"] for row in rows]
                if epoch == 1 and step <= 8:
                    expected = original["batch_receipts"][step - 1]
                    assert selected == expected["sampler_indices"]
                    assert paths == expected["paths"]
                    prefix_equal.append(True)
                batches.append({
                    "epoch": epoch, "step": step,
                    "sample_order_sha256": hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest(),
                    "camera_counts": dict(Counter(cameras.tolist())),
                    "cross_camera_identity_groups": sum(count > 1 for count in group_cameras),
                    "cross_camera_positive_rows": int((cross_positive_count > 0).sum()),
                    "directed_cross_camera_positive_pairs": int(cross_positive_count.sum()),
                    "all_directed_positive_pairs": int(positive.sum()),
                    "same_camera_negative_missing_rows": int((same_negative_count == 0).sum()),
                    "other_camera_negative_missing_rows": int((other_negative_count == 0).sum()),
                    "both_camera_negative_groups_available_rows": int(((same_negative_count > 0) & (other_negative_count > 0)).sum()),
                    "same_camera_negative_counts_per_row": same_negative_count.tolist(),
                    "other_camera_negative_counts_per_row": other_negative_count.tolist(),
                    "cross_camera_positive_counts_per_row": cross_positive_count.tolist(),
                })
        assert len(prefix_equal) == 8
        total_rows = len(batches) * 64
        sums = {key: sum(row[key] for row in batches) for key in (
            "cross_camera_positive_rows", "directed_cross_camera_positive_pairs",
            "all_directed_positive_pairs", "same_camera_negative_missing_rows",
            "other_camera_negative_missing_rows", "both_camera_negative_groups_available_rows",
        )}
        report["folds"].append({
            "fold": number, "source_records": len(manifest), "source_identities": len(by_identity),
            "identities_by_camera_count": dict(Counter(len(value) for value in by_identity.values())),
            "source_camera_counts": source_camera_counts,
            "first_8_batches_exactly_match_v21_m0_indices_and_paths": all(prefix_equal),
            "batch_count": len(batches), "sample_exposures": total_rows,
            "cross_camera_groups_histogram": dict(Counter(row["cross_camera_identity_groups"] for row in batches)),
            "sums": sums,
            "cross_camera_positive_row_fraction": sums["cross_camera_positive_rows"] / total_rows,
            "cross_camera_positive_pair_fraction": sums["directed_cross_camera_positive_pairs"] / sums["all_directed_positive_pairs"],
            "both_camera_negative_groups_available_fraction": sums["both_camera_negative_groups_available_rows"] / total_rows,
            "batches": batches,
        })
    report["total_replayed_batches"] = sum(f["batch_count"] for f in report["folds"])
    assert report["total_replayed_batches"] == 1680
    report["elapsed_seconds"] = time.time() - started
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in report.items() if key != "folds"}))
    for fold in report["folds"]:
        print(json.dumps({key: value for key, value in fold.items() if key != "batches"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--m0-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
