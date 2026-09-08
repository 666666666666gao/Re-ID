#!/usr/bin/env python3
"""Recompute terminal labels, complete rankings and training scalars from JSON."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time


def metrics(aps, first):
    return {"mAP": mean(aps) * 100,
            **{f"Rank-{k}": mean(rank <= k for rank in first) * 100 for k in (1, 5, 10)}}


def verify(summary, protocol, rankings, preflight, log):
    assert summary["mode"] == "train" and summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert summary["seed"] == 42 and summary["checkpoint_selection"] == "fixed_epoch_50"
    assert not summary["epochs_selected_by_heldout"] and summary["training_source_only"]
    assert summary["expert_training"] == summary["official_test_image_access"] == summary["fixed_rgbnt201_dev_image_access"] == 0
    assert summary["heldout_image_forwards"] == 1032
    assert len(summary["folds"]) == len(protocol["folds"]) == len(rankings["folds"]) == 3
    assert preflight["status"] == "PASS_ENGINEERING_ONLY"
    assert summary["runner_sha256"] == preflight["runner_sha256"]
    assert summary["config_sha256"] == preflight["config_sha256"]
    records = protocol["records"]
    assert len(records) == 1032 and [r["index"] for r in records] == list(range(1032))
    all_ids = {r["identity"] for r in records}
    assert len(all_ids) == 155
    rows, all_ap, all_first, gallery_union, heldout_union = [], [], [], [], []
    loss_errors, metric_errors, history_rows, per_identity = [], [], [], []
    for actual, fold, ranked, m0 in zip(summary["folds"], protocol["folds"], rankings["folds"], preflight["folds"], strict=True):
        assert actual["fold"] == fold["fold"] == ranked["fold"] == m0["fold"]
        assert actual["counts"] == fold["counts"]
        assert actual["source_ids"] == fold["source_ids"] and actual["heldout_ids"] == fold["heldout_ids"]
        assert not set(fold["source_ids"]) & set(fold["heldout_ids"])
        assert set(fold["source_ids"]) | set(fold["heldout_ids"]) == all_ids
        assert fold["source_label_map"] == {str(identity): label for label, identity in enumerate(fold["source_ids"])}
        source = set(fold["source_record_indices"])
        gallery_indices = fold["gallery_record_indices"]
        assert not source & set(gallery_indices) and source | set(gallery_indices) == set(range(1032))
        assert {records[i]["identity"] for i in source} == set(fold["source_ids"])
        gallery = [records[i] for i in gallery_indices]
        assert {r["identity"] for r in gallery} == set(fold["heldout_ids"])
        gallery_union.extend(gallery_indices)
        heldout_union.extend(fold["heldout_ids"])
        retrieval, training = actual["retrieval"], actual["training"]
        assert retrieval["gallery_manifest"] == gallery and retrieval["query_rows"] == fold["query_rows"]
        assert ranked["gallery_record_indices"] == gallery_indices
        assert ranked["source_array_sha256"] == retrieval["retrieval_arrays_sha256"]
        assert ranked["query_gallery_positions"] == [q["gallery_position"] for q in fold["query_rows"]]
        assert retrieval["feature_width"] == 3072 and actual["heldout_image_forwards"] == len(gallery)
        expected_queries = []
        for position, query in enumerate(gallery):
            positives = sum(r["identity"] == query["identity"] and r["scene"] != query["scene"] for r in gallery)
            if positives:
                expected_queries.append((position, query["index"], positives))
        assert expected_queries == [(q["gallery_position"], q["record_index"], q["valid_positives"]) for q in fold["query_rows"]]
        aps, first, identity_ap = [], [], defaultdict(list)
        for query, order, reported_ap, reported_first in zip(fold["query_rows"], ranked["sorted_gallery_positions"], retrieval["average_precision"], retrieval["first_match_rank"], strict=True):
            assert sorted(order) == list(range(len(gallery)))
            matches = [gallery[i]["identity"] == query["identity"] for i in order
                       if not (gallery[i]["identity"] == query["identity"] and gallery[i]["scene"] == query["scene"])]
            positions = [i + 1 for i, positive in enumerate(matches) if positive]
            assert len(positions) == query["valid_positives"]
            ap = mean(number / rank for number, rank in enumerate(positions, 1))
            assert 0 < reported_ap <= 1 and reported_first == positions[0]
            metric_errors.append(abs(ap - reported_ap))
            aps.append(ap)
            first.append(positions[0])
            identity_ap[query["identity"]].append(ap)
        recalculated = metrics(aps, first)
        for key, value in recalculated.items():
            metric_errors.append(abs(value - retrieval["metrics"][key]))
        assert max(retrieval["upstream_metric_difference_pp"].values()) < 1e-5
        assert training["epochs"] == len(training["history"]) == 50
        assert training["optimizer_steps"] == len(training["steps"])
        assert training["overflow_events"] == 0
        assert training["trainable_tensors"] == training["gradient_tensors"] == len(training["optimizer_groups"]) == 195
        assert not training["trainable_without_gradient"]
        assert training["initial_state_sha256"] == m0["training"]["initial_state_sha256"]
        assert training["initial_state_sha256"] != training["final_state_sha256"]
        assert training["trainable_parameters"] == m0["training"]["trainable_parameters"]
        assert training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"]
        assert training["frozen_token_selection_parameters"] == 787968
        for group in training["optimizer_groups"]:
            assert not group["name"].startswith("SIM.token_selection.")
            if "classifier" in group["name"]:
                assert group["lr"] == 0.0005
        losses_by_epoch, exposures = defaultdict(list), []
        for index, step in enumerate(training["steps"]):
            assert step["step"] == index + 1 and 1 <= step["epoch"] <= 50
            assert step["amp_scale_after"] >= step["amp_scale_before"]
            sampled = step["sampled_record_indices"]
            assert len(sampled) == 64 and set(sampled) <= source
            assert sorted(Counter(records[i]["identity"] for i in sampled).values()) == [8] * 8
            assert len(step["id_triplet_head_losses"]) == 4
            composed = sum(step["id_triplet_head_losses"]) + 0.2 * step["gram_loss"] + 0.01 * step["patch_loss"]
            assert math.isfinite(composed) and math.isfinite(step["loss"])
            loss_errors.append(abs(composed - step["loss"]))
            losses_by_epoch[step["epoch"]].append(step["loss"])
            exposures.extend(sampled)
        for epoch_number, epoch in enumerate(training["history"], 1):
            assert epoch["epoch"] == epoch_number
            values = losses_by_epoch[epoch_number]
            assert epoch["optimizer_steps"] == len(values)
            assert abs(mean(values) - epoch["mean_loss"]) < 1e-10
            multiplier = 0.1 ** (int(epoch_number >= 20) + int(epoch_number >= 40))
            expected_lr = [base * multiplier for base in (5e-6, 1e-5, 0.0005)]
            assert all(abs(a - b) < 1e-15 for a, b in zip(epoch["learning_rates"], expected_lr, strict=True))
            history_rows.append(epoch)
        all_ap.extend(aps)
        all_first.extend(first)
        per_identity.extend({"fold": fold["fold"], "identity": identity, "queries": len(values), "mAP": mean(values) * 100}
                            for identity, values in sorted(identity_ap.items()))
        rows.append({"fold": fold["fold"], "source_identities": len(fold["source_ids"]),
                     "source_records": len(source), "heldout_identities": len(fold["heldout_ids"]),
                     "gallery_records": len(gallery), "query_records": len(aps), "query_identities": len(identity_ap),
                     "metrics": recalculated, "optimizer_steps": training["optimizer_steps"],
                     "source_training_exposures": len(exposures), "unique_source_records_exposed": len(set(exposures)),
                     "source_identities_exposed": len({records[i]["identity"] for i in exposures}),
                     "epoch_step_counts": dict(Counter(e["optimizer_steps"] for e in training["history"])),
                     "training_elapsed_seconds": sum(e["elapsed_seconds"] for e in training["history"]),
                     "peak_allocated_mib": actual["peak_allocated_mib"]})
    assert sorted(gallery_union) == list(range(1032)) and sorted(heldout_union) == sorted(all_ids)
    assert len(all_ap) == 600 and len(per_identity) == 60 and len(history_rows) == 150
    log_epochs = [json.loads(line) for line in log.splitlines() if line.startswith('{"event": "signal_source_epoch"')]
    assert [{k: v for k, v in row.items() if k != "event"} for row in log_epochs] == history_rows
    aggregate = metrics(all_ap, all_first)
    metric_errors.extend(abs(value - summary["aggregate"][key]) for key, value in aggregate.items())
    assert max(metric_errors) < 1e-10
    assert summary["optimizer_steps"] == sum(r["optimizer_steps"] for r in rows)
    return {"scope": "Full categorical rankings and dataset-label arithmetic plus every logged training step; no local model/tensor/image runtime",
            "folds": rows, "aggregate": aggregate, "identity_results": per_identity,
            "maximum_metric_absolute_difference": max(metric_errors),
            "maximum_loss_composition_absolute_discrepancy": max(loss_errors),
            "loss_note": "AMP intermediate operation dtypes were not retained; discrepancy is reported without changing training or qualification thresholds.",
            "optimizer_steps": summary["optimizer_steps"], "epoch_rows": 150, "query_records": 600,
            "gallery_records": 1032, "query_identities": 60, "all_heldout_identities": 155,
            "official_test_image_access": 0, "fixed_rgbnt201_dev_image_access": 0,
            "new_method_qualification": False, "arithmetic_checks": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("summary", "protocol", "rankings", "preflight", "log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.perf_counter()
    result = verify(*(json.loads(getattr(args, key).read_text(encoding="utf-8")) for key in
                      ("summary", "protocol", "rankings", "preflight")), args.log.read_text(encoding="utf-8"))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["inputs_sha256"] = {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                               for key in ("summary", "protocol", "rankings", "preflight", "log")}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "identity_results"}))


if __name__ == "__main__":
    main()
