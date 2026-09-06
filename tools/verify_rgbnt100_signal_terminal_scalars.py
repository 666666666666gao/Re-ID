#!/usr/bin/env python3
"""Replay all RGBNT100 query ranks and source training scalars using JSON only."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(aps, ranks):
    return {"mAP": mean(aps) * 100,
            **{f"Rank-{k}": mean(r <= k for r in ranks) * 100 for k in (1, 5, 10)}}


def run(args):
    started = time.perf_counter()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    m0 = json.loads(args.preflight.read_text(encoding="utf-8"))
    remote = json.loads(args.remote_verification.read_text(encoding="utf-8"))
    assert not args.output.exists()
    assert summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION" and summary["mode"] == "train"
    assert summary["seed"] == 42 and summary["checkpoint_selection"] == "fixed_epoch_30"
    assert not summary["epochs_selected_by_heldout"] and summary["training_source_only"]
    assert summary["expert_training"] == summary["official_test_image_access"] == summary["fixed_rgbnt201_dev_image_access"] == 0
    assert summary["heldout_image_forwards"] == 8675
    assert summary["runner_sha256"] == m0["runner_sha256"]
    assert summary["config_sha256"] == m0["config_sha256"]
    assert sha(args.summary) == remote["summary_sha256"]
    assert remote["status"] == "PASS_COMPLETE_BASELINE_FILES_ARRAYS_RANKS_AND_AUTHOR_LR"
    records = protocol["records"]
    assert len(records) == 8675
    assert [r["index"] for r in records] == list(range(8675))
    all_ids = {r["identity"] for r in records}
    assert all_ids == set(range(501, 600, 2))
    assert all(r["path"].startswith("rgbir/bounding_box_train/") for r in records)
    all_aps, all_ranks, identities, folds, epochs, query_details = [], [], [], [], [], []
    metric_diffs, loss_diffs, mean_diffs, gallery_union, heldout_union = [], [], [], [], []
    for actual, fold, capacity in zip(summary["folds"], protocol["folds"], m0["folds"], strict=True):
        index = fold["fold"]
        assert actual["fold"] == capacity["fold"] == index
        assert actual["source_ids"] == fold["source_ids"] and actual["heldout_ids"] == fold["heldout_ids"]
        assert actual["counts"] == fold["counts"]
        assert set(fold["source_ids"]).isdisjoint(fold["heldout_ids"])
        assert set(fold["source_ids"]) | set(fold["heldout_ids"]) == all_ids
        source = set(fold["source_record_indices"])
        gallery_indices = fold["gallery_record_indices"]
        assert source.isdisjoint(gallery_indices) and source | set(gallery_indices) == set(range(8675))
        gallery = [records[i] for i in gallery_indices]
        assert {records[i]["identity"] for i in source} == set(fold["source_ids"])
        assert {r["identity"] for r in gallery} == set(fold["heldout_ids"])
        retrieval, training = actual["retrieval"], actual["training"]
        assert retrieval["gallery_manifest"] == gallery and retrieval["query_rows"] == fold["query_rows"]
        assert retrieval["feature_width"] == 3072 and actual["heldout_image_forwards"] == len(gallery)
        rank_path = args.rankings_root / f"fold_{index}" / "full_rankings.json.gz"
        assert sha(rank_path) == retrieval["full_rankings_sha256"]
        with gzip.open(rank_path, "rt", encoding="utf-8") as handle:
            ranked = json.load(handle)
        assert ranked["gallery_record_indices"] == gallery_indices
        assert ranked["query_gallery_positions"] == [q["gallery_position"] for q in fold["query_rows"]]
        assert len(fold["query_rows"]) == len(gallery)
        id_counts = Counter(r["identity"] for r in gallery)
        id_camera_counts = Counter((r["identity"], r["camera"]) for r in gallery)
        aps, ranks, identity_ap = [], [], defaultdict(list)
        for q, order, saved_ap, saved_rank in zip(fold["query_rows"], ranked["ordered_gallery_positions"],
                                                 retrieval["average_precision"], retrieval["first_match_rank"], strict=True):
            query_record = gallery[q["gallery_position"]]
            assert query_record["index"] == q["record_index"] and query_record["identity"] == q["identity"]
            count = id_counts[q["identity"]] - id_camera_counts[q["identity"], query_record["camera"]]
            assert count == q["valid_positive_count"] and count > 0
            assert sorted(order) == list(range(len(gallery)))
            kept = [gallery[i]["identity"] == q["identity"] for i in order
                    if not (gallery[i]["identity"] == q["identity"] and gallery[i]["camera"] == query_record["camera"])]
            positions = [i + 1 for i, positive in enumerate(kept) if positive]
            assert len(positions) == count and saved_rank == positions[0]
            ap = mean(j / r for j, r in enumerate(positions, 1))
            metric_diffs.append(abs(ap - saved_ap))
            aps.append(ap)
            ranks.append(positions[0])
            identity_ap[q["identity"]].append(ap)
            query_details.append({"fold": index, "record_index": q["record_index"], "identity": q["identity"],
                                  "valid_positive_count": count, "AP": ap, "first_match_rank": positions[0]})
        recalculated = metrics(aps, ranks)
        metric_diffs.extend(abs(v - retrieval["metrics"][k]) for k, v in recalculated.items())
        assert max(retrieval["upstream_metric_difference_pp"].values()) < 1e-5
        assert training["epochs"] == len(training["history"]) == 30
        assert training["optimizer_steps"] == len(training["steps"])
        assert training["initial_state_sha256"] == capacity["training"]["initial_state_sha256"]
        assert training["initial_state_sha256"] != capacity["training"]["final_state_sha256"]
        assert training["trainable_parameters"] == capacity["training"]["trainable_parameters"]
        assert training["trainable_tensors"] == training["gradient_tensors"] == len(training["optimizer_groups"]) == 195
        assert not training["trainable_without_gradient"] and training["overflow_events"] == 0
        assert training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"]
        assert training["frozen_token_selection_parameters"] == 787968
        for group in training["optimizer_groups"]:
            expected = .0007 * (2 if "bias" in group["name"] else 1)
            if "base" in group["name"] and "adapter" not in group["name"]:
                expected = .000005
            assert group["initial_lr"] == expected and group["lr"] == .1 * .0007
            assert not group["name"].startswith("SIM.token_selection.")
        losses, exposed = defaultdict(list), []
        same_identity_pairs = cross_camera_pairs = 0
        for step_number, step in enumerate(training["steps"], 1):
            assert step["step"] == step_number and 1 <= step["epoch"] <= 30
            assert step["amp_scale_after"] >= step["amp_scale_before"]
            sampled = step["sampled_record_indices"]
            assert len(sampled) == 64 and set(sampled) <= source
            assert sorted(Counter(records[i]["identity"] for i in sampled).values()) == [8] * 8
            assert len(step["id_triplet_head_losses"]) == 4
            composed = sum(step["id_triplet_head_losses"]) + .1 * step["gram_loss"] + .1 * step["patch_loss"]
            assert math.isfinite(composed) and math.isfinite(step["loss"])
            loss_diffs.append(abs(composed - step["loss"]))
            losses[step["epoch"]].append(step["loss"])
            exposed.extend(sampled)
            camera_groups = defaultdict(Counter)
            for i in sampled:
                r = records[i]
                camera_groups[r["identity"]][r["camera"]] += 1
            same_identity_pairs += 8 * 28
            cross_camera_pairs += sum(28 - sum(n * (n - 1) // 2 for n in counts.values()) for counts in camera_groups.values())
        for number, epoch in enumerate(training["history"], 1):
            assert epoch["epoch"] == number and epoch["optimizer_steps"] == len(losses[number])
            mean_diffs.append(abs(mean(losses[number]) - epoch["mean_loss"]))
            assert epoch["learning_rates"] == remote["author_lr_schedule"][number - 1]["learning_rates"]
            epochs.append(epoch)
        all_aps.extend(aps)
        all_ranks.extend(ranks)
        identities.extend({"fold": index, "identity": pid, "queries": len(values), "mAP": mean(values) * 100}
                          for pid, values in sorted(identity_ap.items()))
        folds.append({"fold": index, "metrics": recalculated, "source_identities": len(fold["source_ids"]),
                      "source_records": len(source), "gallery_records": len(gallery), "query_records": len(aps),
                      "optimizer_steps": training["optimizer_steps"], "source_exposures": len(exposed),
                      "source_identities_exposed": len({records[i]["identity"] for i in exposed}),
                      "unique_source_records_exposed": len(set(exposed)), "same_identity_pairs": same_identity_pairs,
                      "cross_camera_pairs": cross_camera_pairs,
                      "epoch_step_counts": dict(Counter(e["optimizer_steps"] for e in training["history"])),
                      "training_seconds": sum(e["elapsed_seconds"] for e in training["history"]),
                      "peak_allocated_mib": actual["peak_allocated_mib"]})
        gallery_union.extend(gallery_indices)
        heldout_union.extend(fold["heldout_ids"])
        del ranked
    assert sorted(gallery_union) == list(range(8675)) and sorted(heldout_union) == sorted(all_ids)
    assert len(all_aps) == len(query_details) == 8675 and len(identities) == 50 and len(epochs) == 90
    log_epochs = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()
                  if line.startswith('{"event": "signal_source_epoch"')]
    assert [{k: v for k, v in row.items() if k != "event"} for row in log_epochs] == epochs
    aggregate = metrics(all_aps, all_ranks)
    metric_diffs.extend(abs(v - summary["aggregate"][k]) for k, v in aggregate.items())
    assert max(metric_diffs) < 1e-10 and max(mean_diffs) < 1e-10
    assert summary["optimizer_steps"] == sum(f["optimizer_steps"] for f in folds)
    result = {"status": "PASS_COMPLETE8675_QUERY_AND_ALL_TRAINING_SCALARS",
              "aggregate": aggregate, "folds": folds, "identity_results": identities,
              "query_results": query_details, "optimizer_steps": summary["optimizer_steps"],
              "epoch_rows": 90, "query_records": 8675, "gallery_records": 8675, "query_identities": 50,
              "maximum_metric_difference": max(metric_diffs), "maximum_epoch_mean_difference": max(mean_diffs),
              "maximum_loss_composition_difference": max(loss_diffs),
              "loss_note": "Stored float components regrouped with Python doubles; AMP intermediate dtypes not retained; no new scientific tolerance gate.",
              "new_method_qualification": False, "official_test_access": 0, "local_model_tensor_image_calls": 0,
              "elapsed_seconds": time.perf_counter() - started,
              "input_sha256": {k: sha(getattr(args, k)) for k in ("summary", "protocol", "preflight", "log", "remote_verification")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("query_results", "identity_results")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("summary", "protocol", "preflight", "log", "rankings-root", "remote-verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())

