#!/usr/bin/env python3
"""Full label-only census of the three frozen identity-OOF evaluation environments."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity_support(records, environment):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["identity"]].append(row)
    result = []
    for identity, rows in sorted(grouped.items()):
        counts = Counter(r[environment] for r in rows)
        n = len(rows)
        assert n >= 2
        pairs = n * (n - 1) // 2
        within = sum(c * (c - 1) // 2 for c in counts.values())
        result.append({"identity": identity, "records": n, "environment_counts": dict(sorted(counts.items())),
                       "environments": len(counts), "distinct_record_positive_pairs": pairs,
                       "cross_environment_positive_pairs": pairs - within,
                       "cross_environment_pair_fraction": (pairs - within) / pairs})
    return result

def group_summary(rows):
    total = sum(x["distinct_record_positive_pairs"] for x in rows)
    cross = sum(x["cross_environment_positive_pairs"] for x in rows)
    return {"identities": len(rows), "records": sum(x["records"] for x in rows),
            "identities_by_environment_count": dict(sorted(Counter(x["environments"] for x in rows).items())),
            "cross_environment_identities": sum(x["environments"] >= 2 for x in rows),
            "distinct_record_positive_pairs": total, "cross_environment_positive_pairs": cross,
            "record_pair_weighted_cross_environment_fraction": cross / total,
            "identity_macro_cross_environment_fraction": statistics.fmean(x["cross_environment_pair_fraction"] for x in rows)}

def census(dataset, environment, folds, expected):
    output, all_queries, all_gallery, source_membership = [], [], {}, Counter()
    for fold in folds:
        source, gallery = fold["source"], fold["gallery"]
        assert len({r["key"] for r in source}) == len(source)
        assert len({r["key"] for r in gallery}) == len(gallery)
        assert not {r["identity"] for r in source} & {r["identity"] for r in gallery}
        source_env = set(r[environment] for r in source)
        source_camera = set(r["camera"] for r in source)
        ids = Counter(r["identity"] for r in gallery)
        envs = Counter(r[environment] for r in gallery)
        cams = Counter(r["camera"] for r in gallery)
        id_env = Counter((r["identity"], r[environment]) for r in gallery)
        id_cam = Counter((r["identity"], r["camera"]) for r in gallery)
        queries = []
        for r in gallery:
            positive = ids[r["identity"]] - id_env[(r["identity"], r[environment])]
            if positive == 0:
                continue
            negative = len(gallery) - ids[r["identity"]]
            same_env_negative = envs[r[environment]] - id_env[(r["identity"], r[environment])]
            same_camera_negative = cams[r["camera"]] - id_cam[(r["identity"], r["camera"])]
            assert negative > 0 and 0 <= same_env_negative <= negative and 0 <= same_camera_negative <= negative
            queries.append({"fold": fold["fold"], "record_key": r["key"], "identity": r["identity"],
                            "environment": r[environment], "camera": r["camera"],
                            "valid_positive_records": positive, "negative_records": negative,
                            "same_environment_negative_records": same_env_negative,
                            "same_camera_negative_records": same_camera_negative,
                            "same_environment_negative_fraction": same_env_negative / negative,
                            "same_camera_negative_fraction": same_camera_negative / negative,
                            "environment_absent_from_source": r[environment] not in source_env,
                            "camera_absent_from_source": r["camera"] not in source_camera})
        assert len(queries) == fold["expected_queries"]
        eligible_ids = {q["identity"] for q in queries}
        support = identity_support(source, environment)
        row = {"fold": fold["fold"], "source_summary": group_summary(support),
               "source_identity_support": support, "source_environments": sorted(source_env),
               "source_camera_values": sorted(source_camera), "gallery_records": len(gallery),
               "gallery_identities": len(ids), "query_records": len(queries), "query_identities": len(eligible_ids),
               "gallery_only_distractor_identities": len(set(ids) - eligible_ids),
               "gallery_only_distractor_records": sum(n for identity, n in ids.items() if identity not in eligible_ids),
               "queries_in_environment_absent_from_source": sum(q["environment_absent_from_source"] for q in queries),
               "queries_in_camera_absent_from_source": sum(q["camera_absent_from_source"] for q in queries),
               "valid_positive_count_min": min(q["valid_positive_records"] for q in queries),
               "valid_positive_count_median": statistics.median(q["valid_positive_records"] for q in queries),
               "valid_positive_count_max": max(q["valid_positive_records"] for q in queries),
               "negative_count_mean": statistics.fmean(q["negative_records"] for q in queries),
               "same_environment_negative_fraction_query_mean": statistics.fmean(q["same_environment_negative_fraction"] for q in queries),
               "same_camera_negative_fraction_query_mean": statistics.fmean(q["same_camera_negative_fraction"] for q in queries)}
        output.append(row)
        all_queries.extend(queries)
        for r in gallery:
            assert r["key"] not in all_gallery
            all_gallery[r["key"]] = r
        for r in source:
            source_membership[r["key"]] += 1
    assert len(all_gallery) == expected["records"] and len(all_queries) == expected["queries"]
    assert len({q["identity"] for q in all_queries}) == expected["query_ids"]
    assert set(source_membership) == set(all_gallery) and set(source_membership.values()) == {2}
    global_support = identity_support(list(all_gallery.values()), environment)
    summary = group_summary(global_support)
    assert summary["identities"] == expected["identities"]
    summary.update({"query_records": len(all_queries), "query_identities": expected["query_ids"],
                    "gallery_only_distractor_identities": sum(r["gallery_only_distractor_identities"] for r in output),
                    "gallery_only_distractor_records": sum(r["gallery_only_distractor_records"] for r in output),
                    "queries_in_environment_absent_from_source": sum(q["environment_absent_from_source"] for q in all_queries),
                    "queries_in_camera_absent_from_source": sum(q["camera_absent_from_source"] for q in all_queries),
                    "negative_count_mean": statistics.fmean(q["negative_records"] for q in all_queries),
                    "same_environment_negative_fraction_query_mean": statistics.fmean(q["same_environment_negative_fraction"] for q in all_queries),
                    "same_camera_negative_fraction_query_mean": statistics.fmean(q["same_camera_negative_fraction"] for q in all_queries)})
    return {"dataset": dataset, "evaluation_environment": environment, "global_labels": summary,
            "folds": output, "all_identity_support": global_support, "all_query_label_rows": all_queries,
            "each_record_heldout_once_and_source_twice": True}

def rgb201_folds(geometry):
    result = []
    for f in geometry["folds"]:
        a, b = f["endpoints"]["frozen_private_tail"], f["endpoints"]["trained_private_tail"]
        sets = {}
        for part in ("source", "heldout"):
            assert a[part]["gallery_manifest"] == b[part]["gallery_manifest"]
            sets[part] = [{"key": r["file"], "identity": r["file"].split("_")[0],
                           "camera": r["camera"]} for r in a[part]["gallery_manifest"]]
            assert len(sets[part]) == a[part]["gallery"]
        result.append({"fold": f["fold"], "source": sets["source"], "gallery": sets["heldout"],
                       "expected_queries": a["heldout"]["eligible_queries"]})
    return result

def vehicle_folds(protocol, dataset):
    records = []
    for r in protocol["records"]:
        if dataset == "MSVR310":
            records.append({"key": r["paths"][0], "identity": str(r["identity"]),
                            "camera": r["camera"], "scene": r["scene"]})
        else:
            assert dataset == "RGBNT100"
            records.append({"key": r["path"], "identity": str(r["identity"]), "camera": r["camera"]})
    result = []
    for f in protocol["folds"]:
        source = [records[i] for i in f["source_record_indices"]]
        gallery = [records[i] for i in f["gallery_record_indices"]]
        env = "scene" if dataset == "MSVR310" else "camera"
        for q in f["query_rows"]:
            r = records[q["record_index"]]
            count = sum(g["identity"] == r["identity"] and g[env] != r[env] for g in gallery)
            key = "valid_positives" if dataset == "MSVR310" else "valid_positive_count"
            assert count == q[key]
        result.append({"fold": f["fold"], "source": source, "gallery": gallery,
                       "expected_queries": len(f["query_rows"])})
    return result

def main(args):
    start = time.perf_counter()
    root = args.project_root.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    assert sha(Path(__file__)) == plan["script_sha256"]
    data = {}
    for name, row in plan["inputs"].items():
        path = root / row["path"]
        assert sha(path) == row["sha256"], name
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    datasets = [
        census("RGBNT201", "camera", rgb201_folds(data["rgbnt201_geometry"]),
               {"records": 3126, "queries": 571, "query_ids": 21, "identities": 141}),
        census("MSVR310", "scene", vehicle_folds(data["msvr310_protocol"], "MSVR310"),
               {"records": 1032, "queries": 600, "query_ids": 60, "identities": 155}),
        census("RGBNT100", "camera", vehicle_folds(data["rgbnt100_protocol"], "RGBNT100"),
               {"records": 8675, "queries": 8675, "query_ids": 50, "identities": 50}),
    ]
    sampled = []
    old = data["rgbnt201_sampler"]
    cross = sum(f["sums"]["directed_cross_camera_positive_pairs"] for f in old["folds"])
    total = sum(f["sums"]["all_directed_positive_pairs"] for f in old["folds"])
    sampled.append({"dataset": "RGBNT201", "execution": "Original sampler20epoch metadata replay; not V24 two-view training",
                    "environment": "camera", "pair_counting": "directed", "cross_pairs": cross,
                    "all_pairs": total, "fraction": cross / total, "batches": old["total_replayed_batches"]})
    for name, dataset, environment, cross_key, total_key, label in [
        ("msvr310_sampling", "MSVR310", "scene", "cross_scene_positive_pairs", "same_identity_positive_pairs", "Complete original roles20epoch"),
        ("rgbnt100_sampling", "RGBNT100", "camera", "cross_camera_pairs", "same_identity_pairs", "Complete Signal R2 B0 thirtyepochs; active role comparison not used"),
    ]:
        d = data[name]
        cross = sum(f[cross_key] for f in d["folds"]); total = sum(f[total_key] for f in d["folds"])
        sampled.append({"dataset": dataset, "execution": label, "environment": environment,
                        "pair_counting": "unordered batch positions; duplicate record positions retained",
                        "cross_pairs": cross, "all_pairs": total, "fraction": cross / total,
                        "batches": d["optimizer_steps"]})
    result = {"completed_at": datetime.now().astimezone().isoformat(), "status": "PASS_FULL_THREE_DATASET_LABEL_CENSUS",
              "datasets": datasets, "separate_historical_sampled_pair_facts": sampled,
              "counts": {"datasets": 3, "global_records_across_separate_datasets": 12833,
                         "query_label_rows": sum(len(d["all_query_label_rows"]) for d in datasets),
                         "global_identities_across_separate_datasets": 346},
              "plan_sha256": sha(args.plan), "script_sha256": sha(Path(__file__)), "inputs": plan["inputs"],
              "model_tensor_image_calls": 0, "optimizer_updates": 0, "retrieval_metric_computations": 0,
              "current_comparison_outputs_read": False, "elapsed_seconds": time.perf_counter() - start,
              "interpretation_limits": plan["interpretation_limits"]}
    assert result["counts"]["query_label_rows"] == 9846
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "datasets": [{k: v for k, v in d.items() if k in ("dataset", "evaluation_environment", "global_labels")} for d in datasets],
                      "sampled": sampled, "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project-root", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
