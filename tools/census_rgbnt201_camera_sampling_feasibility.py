#!/usr/bin/env python3
"""Count every source identity's camera-group capacity from frozen text evidence."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    root = args.project_root
    assert sha(Path(__file__)) == plan["script_sha256"]
    data = {}
    for name, entry in plan["inputs"].items():
        path = root / entry["path"]
        assert sha(path) == entry["sha256"], name
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    census = next(d for d in data["environment"]["datasets"] if d["dataset"] == "RGBNT201")
    sampled = data["sampler"]
    assert sampled["epochs_replayed"] == 20 and sampled["seed"] == 42
    folds = []
    for f, old in zip(census["folds"], sampled["folds"], strict=True):
        assert f["fold"] == old["fold"]
        rows = f["source_identity_support"]
        assert len(rows) == old["source_identities"] == 94
        assert sum(row["records"] for row in rows) == old["source_records"]
        members, identity_rows = Counter(), []
        for row in rows:
            counts = row["environment_counts"]
            assert row["environments"] == len(counts) in (1, 2)
            assert sum(counts.values()) == row["records"]
            members.update(counts.keys())
            identity_rows.append({
                "identity": row["identity"], "records": row["records"],
                "camera_counts": counts, "cross_camera": len(counts) == 2,
                "original_K8_groups": max(row["records"], 8) // 8,
                "disjoint_4plus4_groups": min(counts.values()) // 4 if len(counts) == 2 else 0,
            })
        cross = [row for row in identity_rows if row["cross_camera"]]
        assert len(cross) == 14
        all_groups = sum(row["original_K8_groups"] for row in identity_rows)
        batches = all_groups // 8
        assert batches * 20 == old["batch_count"]
        available = sum(row["original_K8_groups"] for row in cross)
        observed = sum(int(k) * v for k, v in old["cross_camera_groups_histogram"].items())
        assert observed % 20 == 0
        target = 2 * batches
        cross_records = sum(row["records"] for row in cross)
        folds.append({
            "fold": f["fold"], "source_identities": 94, "single_camera_identities": 80,
            "cross_camera_identities": 14, "source_records": old["source_records"],
            "formal_batches_per_epoch": batches, "all_original_groups": all_groups,
            "available_original_cross_groups": available,
            "actually_used_cross_groups_per_old_epoch": observed // 20,
            "target_two_cross_groups_per_batch": target,
            "shortfall_using_original_nonrepeated_groups": max(0, target - available),
            "two_cross_per_batch_possible_without_extra_group_draws": target <= available,
            "all_cross_source_records": cross_records,
            "aggregate_minimum_repeated_cross_record_positions": max(0, target * 8 - cross_records),
            "maximum_disjoint_4plus4_groups": sum(row["disjoint_4plus4_groups"] for row in cross),
            "available_single_camera_groups": all_groups - available,
            "single_camera_groups_required_for_two_cross_plus_six_single": batches * 6,
            "source_camera_identity_membership": dict(sorted(members.items())),
            "all94_source_identity_rows": identity_rows,
        })
    assert len(folds) == 3 and sampled["total_replayed_batches"] == sum(x["formal_batches_per_epoch"] * 20 for x in folds) == 1680
    result = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_ALL_SOURCE_LABEL_CAPACITY_CENSUS",
        "plan_sha256": sha(args.plan), "script_sha256": sha(Path(__file__)),
        "inputs": plan["inputs"], "folds": folds,
        "source_identity_fold_memberships_checked": 282,
        "source_record_fold_memberships_checked": sum(x["source_records"] for x in folds),
        "all_three_two_cross_targets_require_extra_draws": all(not x["two_cross_per_batch_possible_without_extra_group_draws"] for x in folds),
        "scope": "Necessary label-count constraints for the fixed two-cross-ID idea; not an implemented sampler or an efficacy experiment",
        "interpretation_limits": [
            "Changing the order of original groups alone cannot ensure two cross-camera identities in every batch.",
            "Additional group draws change identity exposure and may repeat records; all80 single-camera source identities must remain represented.",
            "4+4 within-identity camera balancing is a separate intervention and is not implied by selecting two cross-camera identities.",
            "These are aggregate lower bounds and necessary counts, not a proof that a particular batch scheduler satisfies every condition.",
            "No sampler, model, loss or active RGBNT100 training input was changed."
        ],
        "model_tensor_image_calls": 0, "optimizer_updates": 0, "retrieval_metric_computations": 0,
    }
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k:v for k,v in result.items() if k != "folds"},
                     "folds": [{k:v for k,v in f.items() if k != "all94_source_identity_rows"} for f in folds]}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("project-root", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
