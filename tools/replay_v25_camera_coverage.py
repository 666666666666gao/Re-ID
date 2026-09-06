#!/usr/bin/env python3
"""Replay all registered V25 source batches; no images, tensors or retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(args):
    started = time.time()
    plan = json.loads(args.plan.read_bytes())
    assert plan["stage"] == "metadata_replay_only"
    assert plan["seed"] == 42 and plan["epochs"] == 20
    assert plan["batch_size"] == 64 and plan["num_instances"] == 8
    assert sha(__file__) == plan["script_sha256"]
    assert sha(args.new_sampler) == plan["candidate_sampler_sha256"]
    root = args.repository.resolve()
    for rel, expected in plan["bound_source_files"].items():
        assert sha(root / rel) == expected, rel
    sys.path.insert(0, str(root / "modeling"))
    from trifusion.aligned_data import CrossCameraIdentitySampler
    assert Path(inspect.getfile(CrossCameraIdentitySampler)).resolve() == (
        root / "modeling/trifusion/aligned_data.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "trifusion.camera_coverage_v25", args.new_sampler
    )
    candidate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate_module)
    geometry = json.loads((root / plan["geometry"]).read_bytes())
    previous = json.loads((root / plan["old_replay"]).read_bytes())
    old_folds = {row["fold"]: row for row in previous["folds"]}
    assert sorted(old_folds) == [0, 1, 2]
    report = {
        "status": "IN_PROGRESS",
        "started_at": datetime.now().astimezone().isoformat(),
        "stage": plan["stage"],
        "plan_sha256": sha(args.plan),
        "script_sha256": sha(__file__),
        "candidate_sampler_sha256": sha(args.new_sampler),
        "bound_source_files": plan["bound_source_files"],
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "seed": 42, "epochs": 20, "batch_size": 64, "num_instances": 8,
        "model_instantiations": 0, "model_forward_calls": 0,
        "optimizer_steps": 0, "checkpoint_tensor_loads": 0,
        "image_reads": 0, "retrieval_metric_computations": 0,
        "dev_access_count": 0, "official_test_access_count": 0,
        "folds": [],
    }
    for fold in geometry["folds"]:
        number = fold["fold"]
        manifest = fold["endpoints"]["frozen_private_tail"]["source"]["gallery_manifest"]
        assert manifest == fold["endpoints"]["trained_private_tail"]["source"]["gallery_manifest"]
        assert len(manifest) == old_folds[number]["source_records"]
        assert len({row["file"] for row in manifest}) == len(manifest)
        records = [((row["file"],) * 3, row["identity"], row["camera"], 0)
                   for row in manifest]
        by_id = defaultdict(list)
        for index, row in enumerate(manifest):
            by_id[row["identity"]].append(index)
        camera_sets = {
            identity: {manifest[index]["camera"] for index in indices}
            for identity, indices in by_id.items()
        }
        cross_ids = {identity for identity, cameras in camera_sets.items()
                     if len(cameras) > 1}
        assert len(by_id) == 94 and len(cross_ids) == 14
        expected_batches = {(row["epoch"], row["step"]): row
                            for row in old_folds[number]["batches"]}
        assert len(expected_batches) == old_folds[number]["batch_count"]
        fold_report = {
            "fold": number,
            "source_identity_mapping": fold["source_identity_mapping"],
            "source_records": len(manifest),
            "source_identities": len(by_id),
            "cross_camera_source_identities": sorted(cross_ids),
            "single_camera_source_identities": sorted(set(by_id) - cross_ids),
            "source_manifest": manifest,
            "arms": {},
        }
        for arm, sampler_type in (
            ("control", CrossCameraIdentitySampler),
            ("two_cross_camera", candidate_module.TwoCrossCameraIdentitySampler),
        ):
            sampler = sampler_type(records, batch_size=64, num_instances=8, seed=42)
            exposures = Counter()
            identity_groups = Counter()
            epoch_rows, batch_rows = [], []
            for epoch in range(1, 21):
                indices = list(iter(sampler))
                assert len(indices) == len(sampler) and len(indices) % 64 == 0
                epoch_exposures = Counter(indices)
                assert set(epoch_exposures) <= set(range(len(manifest)))
                epoch_ids = {manifest[index]["identity"] for index in indices}
                exposures.update(indices)
                for step, offset in enumerate(range(0, len(indices), 64), 1):
                    selected = indices[offset:offset + 64]
                    identities = [manifest[index]["identity"] for index in selected]
                    assert sorted(Counter(identities).values()) == [8] * 8
                    cross_group_count = 0
                    directed_cross_pairs = 0
                    group_records = []
                    for group_offset in range(0, 64, 8):
                        group = selected[group_offset:group_offset + 8]
                        identity = manifest[group[0]]["identity"]
                        assert all(manifest[index]["identity"] == identity for index in group)
                        cameras = Counter(manifest[index]["camera"] for index in group)
                        is_cross = len(cameras) > 1
                        assert is_cross == (identity in cross_ids)
                        cross_group_count += int(is_cross)
                        directed_cross_pairs += 64 - sum(n * n for n in cameras.values())
                        identity_groups[identity] += 1
                        group_records.append({
                            "identity": identity,
                            "camera_counts": dict(cameras),
                            "unique_records": len(set(group)),
                        })
                    if arm == "two_cross_camera":
                        assert cross_group_count == 2
                    order_hash = hashlib.sha256(json.dumps(
                        [manifest[index]["file"] for index in selected],
                        separators=(",", ":")
                    ).encode()).hexdigest()
                    batch = {
                        "epoch": epoch, "step": step,
                        "sampler_indices": selected,
                        "sample_order_sha256": order_hash,
                        "cross_camera_identity_groups": cross_group_count,
                        "cross_camera_positive_rows": 8 * cross_group_count,
                        "directed_cross_camera_positive_pairs": directed_cross_pairs,
                        "all_directed_positive_pairs": 448,
                        "groups": group_records,
                    }
                    if arm == "control":
                        expected = expected_batches[(epoch, step)]
                        for key in (
                            "sample_order_sha256", "cross_camera_identity_groups",
                            "cross_camera_positive_rows",
                            "directed_cross_camera_positive_pairs",
                            "all_directed_positive_pairs",
                        ):
                            assert batch[key] == expected[key], (number, epoch, step, key)
                    batch_rows.append(batch)
                epoch_rows.append({
                    "epoch": epoch, "batches": len(indices) // 64,
                    "sample_exposures": len(indices),
                    "unique_records": len(epoch_exposures),
                    "repeated_record_positions": len(indices) - len(epoch_exposures),
                    "source_identity_coverage": len(epoch_ids),
                    "cross_camera_identity_coverage": len(epoch_ids & cross_ids),
                    "single_camera_identity_coverage": len(epoch_ids - cross_ids),
                    "missing_source_identities": sorted(set(by_id) - epoch_ids),
                    "cross_camera_record_exposures": sum(
                        count for index, count in epoch_exposures.items()
                        if manifest[index]["identity"] in cross_ids
                    ),
                })
            per_identity = []
            for identity, record_indices in sorted(by_id.items()):
                counts = [exposures[index] for index in record_indices]
                per_identity.append({
                    "identity": identity, "source_records": len(record_indices),
                    "cameras": sorted(camera_sets[identity]),
                    "cross_camera_identity": identity in cross_ids,
                    "sample_exposures": sum(counts),
                    "identity_groups": identity_groups[identity],
                    "unique_records_seen": sum(value > 0 for value in counts),
                    "minimum_record_exposures": min(counts),
                    "median_record_exposures": statistics.median(counts),
                    "maximum_record_exposures": max(counts),
                    "record_indices": record_indices,
                    "record_exposures": counts,
                })
            cross_pairs = sum(row["directed_cross_camera_positive_pairs"] for row in batch_rows)
            positive_pairs = sum(row["all_directed_positive_pairs"] for row in batch_rows)
            arm_report = {
                "batches": batch_rows, "epochs": epoch_rows,
                "identities": per_identity,
                "batch_count": len(batch_rows),
                "sample_exposures": sum(exposures.values()),
                "unique_records_seen": len(exposures),
                "missing_record_indices": sorted(set(range(len(manifest))) - set(exposures)),
                "missing_source_identities": [
                    row["identity"] for row in per_identity if row["sample_exposures"] == 0
                ],
                "cross_camera_group_histogram": dict(Counter(
                    row["cross_camera_identity_groups"] for row in batch_rows
                )),
                "cross_camera_record_exposures": sum(
                    row["sample_exposures"] for row in per_identity
                    if row["cross_camera_identity"]
                ),
                "within_epoch_repeated_record_positions": sum(
                    row["repeated_record_positions"] for row in epoch_rows
                ),
                "directed_cross_camera_positive_pairs": cross_pairs,
                "all_directed_positive_pairs": positive_pairs,
                "cross_camera_positive_pair_fraction": cross_pairs / positive_pairs,
                "all_old_batch_orders_and_pair_counts_match":
                    arm == "control",
            }
            fold_report["arms"][arm] = arm_report
        control = fold_report["arms"]["control"]
        candidate = fold_report["arms"]["two_cross_camera"]
        assert control["batch_count"] == candidate["batch_count"] == len(expected_batches)
        assert control["sample_exposures"] == candidate["sample_exposures"]
        fold_report["gates"] = {
            "exact_all_control_batch_replay": control["all_old_batch_orders_and_pair_counts_match"],
            "all_source_identities_seen_in_candidate": not candidate["missing_source_identities"],
            "all_source_records_seen_in_candidate": not candidate["missing_record_indices"],
            "exactly_two_cross_camera_groups_each_batch":
                candidate["cross_camera_group_histogram"] == {2: len(expected_batches)},
            "cross_camera_positive_fraction_increases":
                candidate["cross_camera_positive_pair_fraction"]
                > control["cross_camera_positive_pair_fraction"],
            "equal_batch_and_sample_budget":
                control["sample_exposures"] == candidate["sample_exposures"],
        }
        fold_report["all_gates_pass"] = all(fold_report["gates"].values())
        report["folds"].append(fold_report)
        print(json.dumps({
            "fold": number, "gates": fold_report["gates"],
            "control_cross_positive_fraction": control["cross_camera_positive_pair_fraction"],
            "candidate_cross_positive_fraction": candidate["cross_camera_positive_pair_fraction"],
            "candidate_missing_identities": candidate["missing_source_identities"],
            "candidate_missing_records": len(candidate["missing_record_indices"]),
        }), flush=True)
    report["all_gates_pass"] = all(fold["all_gates_pass"] for fold in report["folds"])
    report["total_batches"] = sum(
        arm["batch_count"] for fold in report["folds"] for arm in fold["arms"].values()
    )
    assert report["total_batches"] == 3360
    report["status"] = (
        "COMPLETE_METADATA_REPLAY_PASS" if report["all_gates_pass"]
        else "COMPLETE_METADATA_REPLAY_FAIL"
    )
    report["completed_at"] = datetime.now().astimezone().isoformat()
    report["elapsed_seconds"] = time.time() - started
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "folds"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--new-sampler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
