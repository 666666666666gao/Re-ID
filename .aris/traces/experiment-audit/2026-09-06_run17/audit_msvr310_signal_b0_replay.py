from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import time
from collections import Counter, defaultdict


RUN_DIR = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(r"C:\Users\gb\.trifusion_github_publish_22c3bee")
MANIFEST = RUN_DIR / "input_manifest.json"
OUT = RUN_DIR / "independent_replay_output.json"


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    manifest = read_json(MANIFEST)
    checked = []
    mismatches = []
    for index, row in enumerate(manifest["files"], 1):
        path = pathlib.Path(row["path"])
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_bytes(path)
        ok = actual_bytes == row["bytes"] and actual_sha256 == row["sha256"]
        checked.append(
            {
                "index": index,
                "path": str(path),
                "expected_bytes": row["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": row["sha256"],
                "actual_sha256": actual_sha256,
                "ok": ok,
            }
        )
        if not ok:
            mismatches.append(checked[-1])
    return {
        "manifest_path": str(MANIFEST),
        "created_at": manifest["created_at"],
        "input_count": len(manifest["files"]),
        "all_ok": not mismatches,
        "checked": checked,
        "mismatches": mismatches,
        "signal_source_path_mapping_count": len(manifest["signal_source_path_mapping"]),
    }


def metric_summary(ap_values, first_ranks):
    return {
        "mAP": float(statistics.mean(ap_values) * 100.0),
        "Rank-1": float(sum(1 for rank in first_ranks if rank <= 1) / len(first_ranks) * 100.0),
        "Rank-5": float(sum(1 for rank in first_ranks if rank <= 5) / len(first_ranks) * 100.0),
        "Rank-10": float(sum(1 for rank in first_ranks if rank <= 10) / len(first_ranks) * 100.0),
    }


def recompute_query(order, gallery_rows, query_row):
    q_identity = query_row["identity"]
    q_scene = query_row["scene"]
    kept = []
    for position in order:
        gallery_row = gallery_rows[position]
        remove = gallery_row["identity"] == q_identity and gallery_row["scene"] == q_scene
        if not remove:
            kept.append(position)

    positives = [gallery_rows[position]["identity"] == q_identity for position in kept]
    positive_total = sum(1 for value in positives if value)
    assert positive_total > 0
    cumulative = 0
    precision_sum = 0.0
    first_match_rank = None
    for rank, positive in enumerate(positives, 1):
        if positive:
            cumulative += 1
            precision_sum += cumulative / rank
            if first_match_rank is None:
                first_match_rank = rank
    average_precision = float(precision_sum / positive_total)
    assert first_match_rank is not None
    return average_precision, first_match_rank, kept


def replay_rankings(protocol, summary, rankings):
    records = protocol["records"]
    record_by_index = {row["index"]: row for row in records}
    ranking_by_fold = {row["fold"]: row for row in rankings["folds"]}
    summary_by_fold = {row["fold"]: row for row in summary["folds"]}
    all_identity_values = sorted({row["identity"] for row in records})

    gallery_union = []
    heldout_identity_counter = Counter()
    all_ap = []
    all_first = []
    identity_ap = defaultdict(list)
    folds = []

    for fold in protocol["folds"]:
        fold_id = fold["fold"]
        ranked = ranking_by_fold[fold_id]
        actual = summary_by_fold[fold_id]
        source_ids = set(fold["source_ids"])
        heldout_ids = set(fold["heldout_ids"])
        source_record_indices = list(fold["source_record_indices"])
        gallery_indices = list(fold["gallery_record_indices"])
        gallery_rows = [record_by_index[index] for index in gallery_indices]

        assert source_ids.isdisjoint(heldout_ids)
        assert source_ids | heldout_ids == set(all_identity_values)
        assert all(record_by_index[index]["identity"] in source_ids for index in source_record_indices)
        assert all(row["identity"] in heldout_ids for row in gallery_rows)
        assert gallery_indices == [row["index"] for row in actual["retrieval"]["gallery_manifest"]]
        assert ranked["gallery_record_indices"] == gallery_indices
        assert ranked["query_gallery_positions"] == [row["gallery_position"] for row in fold["query_rows"]]
        assert actual["retrieval"]["query_rows"] == fold["query_rows"]
        assert actual["retrieval"]["feature_width"] == 3072
        assert actual["heldout_image_forwards"] == len(gallery_indices)

        complete_gallery = sorted(gallery_indices) == sorted(
            row["index"] for row in records if row["identity"] in heldout_ids
        )
        assert complete_gallery

        query_indices = {row["record_index"] for row in fold["query_rows"]}
        excluded_indices = set(fold["excluded_query_record_indices"])
        assert query_indices.isdisjoint(excluded_indices)
        assert query_indices | excluded_indices == set(gallery_indices)

        scene_vs_camera_changed = 0
        ap_values = []
        first_values = []
        retained_lengths = []
        single_scene_distractor_ids = set()
        for query, order, reported_ap, reported_first in zip(
            fold["query_rows"],
            ranked["sorted_gallery_positions"],
            actual["retrieval"]["average_precision"],
            actual["retrieval"]["first_match_rank"],
            strict=True,
        ):
            assert sorted(order) == list(range(len(gallery_indices)))
            q_position = query["gallery_position"]
            q_record = gallery_rows[q_position]
            assert q_record["index"] == query["record_index"]
            assert q_record["identity"] == query["identity"]
            assert q_record["scene"] == query["scene"]

            same_identity = [row for row in gallery_rows if row["identity"] == query["identity"]]
            valid_positives = sum(row["scene"] != query["scene"] for row in same_identity)
            removed_same_scene = sum(row["scene"] == query["scene"] for row in same_identity)
            retained_gallery = len(gallery_rows) - removed_same_scene
            negative_distractors = retained_gallery - valid_positives
            camera_positives = sum(
                row["camera"] != q_record["camera"] for row in same_identity
            )
            if camera_positives != valid_positives:
                scene_vs_camera_changed += 1

            assert valid_positives == query["valid_positives"]
            assert removed_same_scene == query["removed_same_identity_same_scene"]
            assert retained_gallery == query["retained_gallery"]
            assert negative_distractors == query["negative_identity_distractors"]

            computed_ap, computed_first, kept = recompute_query(order, gallery_rows, query)
            retained_lengths.append(len(kept))
            assert abs(computed_ap - reported_ap) < 1e-12
            assert computed_first == reported_first
            ap_values.append(computed_ap)
            first_values.append(computed_first)
            all_ap.append(computed_ap)
            all_first.append(computed_first)
            identity_ap[(fold_id, query["identity"])].append(computed_ap)

        for identity in heldout_ids:
            scenes = {row["scene"] for row in gallery_rows if row["identity"] == identity}
            if len(scenes) == 1:
                single_scene_distractor_ids.add(identity)

        recalculated = metric_summary(ap_values, first_values)
        for key, value in recalculated.items():
            assert abs(value - actual["retrieval"]["metrics"][key]) < 1e-10

        gallery_union.extend(gallery_indices)
        heldout_identity_counter.update(heldout_ids)
        folds.append(
            {
                "fold": fold_id,
                "source_identities": len(source_ids),
                "heldout_identities": len(heldout_ids),
                "source_heldout_disjoint": True,
                "source_records": len(source_record_indices),
                "gallery_records": len(gallery_indices),
                "query_records": len(ap_values),
                "query_identities": len({row["identity"] for row in fold["query_rows"]}),
                "excluded_records_retained_in_gallery": len(excluded_indices),
                "single_scene_distractor_identities": len(single_scene_distractor_ids),
                "single_scene_distractor_records": sum(
                    1 for row in gallery_rows if row["identity"] in single_scene_distractor_ids
                ),
                "scene_vs_camera_positive_count_differences": scene_vs_camera_changed,
                "retained_gallery_min": min(retained_lengths),
                "retained_gallery_max": max(retained_lengths),
                "metrics": recalculated,
                "reported_metrics": actual["retrieval"]["metrics"],
                "max_ap_abs_diff": max(
                    abs(a - b)
                    for a, b in zip(
                        ap_values,
                        actual["retrieval"]["average_precision"],
                        strict=True,
                    )
                ),
            }
        )

    aggregate = metric_summary(all_ap, all_first)
    for key, value in aggregate.items():
        assert abs(value - summary["aggregate"][key]) < 1e-10

    per_identity = [
        {
            "fold": fold_id,
            "identity": identity,
            "queries": len(values),
            "mAP": float(statistics.mean(values) * 100.0),
        }
        for (fold_id, identity), values in sorted(identity_ap.items())
    ]

    return {
        "folds": folds,
        "aggregate": aggregate,
        "reported_aggregate": summary["aggregate"],
        "query_records": len(all_ap),
        "query_identities": len(per_identity),
        "gallery_union_records": len(gallery_union),
        "gallery_union_covers_all_records_once": sorted(gallery_union) == list(range(len(records))),
        "heldout_identity_counter_values": sorted(heldout_identity_counter.values()),
        "per_identity_count": len(per_identity),
        "per_identity_results": per_identity,
    }


def verify_training(protocol, summary, preflight):
    log = (REPO / "evidence/trifusion_msvr310_signal_v1_baseline_run_20260906.log").read_text(
        encoding="utf-8"
    )
    log_epochs = [
        json.loads(line)
        for line in log.splitlines()
        if line.startswith('{"event": "signal_source_epoch"')
    ]
    protocol_by_fold = {row["fold"]: row for row in protocol["folds"]}
    preflight_by_fold = {row["fold"]: row for row in preflight["folds"]}
    log_cursor = 0
    folds = []
    all_loss_diffs = []
    for actual in summary["folds"]:
        fold_id = actual["fold"]
        training = actual["training"]
        training_file = REPO / "evidence" / "msvr310_signal_v1_baseline_receipts" / f"fold_{fold_id}_training.json"
        receipt_file = REPO / "evidence" / "msvr310_signal_v1_baseline_receipts" / f"fold_{fold_id}_receipt.json"
        assert read_json(training_file) == training
        assert read_json(receipt_file) == actual

        source_indices = set(protocol_by_fold[fold_id]["source_record_indices"])
        assert training["epochs"] == len(training["history"]) == 50
        assert training["optimizer_steps"] == len(training["steps"]) == 650
        assert training["trainable_tensors"] == training["gradient_tensors"] == len(training["optimizer_groups"])
        assert training["trainable_without_gradient"] == []
        assert training["initial_state_sha256"] == preflight_by_fold[fold_id]["training"]["initial_state_sha256"]
        assert training["initial_state_sha256"] != training["final_state_sha256"]
        assert (
            training["frozen_token_selection_initial_sha256"]
            == training["frozen_token_selection_final_sha256"]
        )

        losses_by_epoch = defaultdict(list)
        exposed_records = []
        amp_scales = set()
        for step_number, step in enumerate(training["steps"], 1):
            assert step["step"] == step_number
            assert 1 <= step["epoch"] <= 50
            assert len(step["id_triplet_head_losses"]) == 4
            assert len(step["sampled_record_indices"]) == 64
            assert set(step["sampled_record_indices"]) <= source_indices
            assert math.isfinite(step["loss"])
            component_loss = (
                sum(step["id_triplet_head_losses"])
                + 0.2 * step["gram_loss"]
                + 0.01 * step["patch_loss"]
            )
            all_loss_diffs.append(abs(component_loss - step["loss"]))
            losses_by_epoch[step["epoch"]].append(step["loss"])
            exposed_records.extend(step["sampled_record_indices"])
            amp_scales.add((step["amp_scale_before"], step["amp_scale_after"]))

        for epoch_number, epoch in enumerate(training["history"], 1):
            assert epoch["epoch"] == epoch_number
            losses = losses_by_epoch[epoch_number]
            assert epoch["optimizer_steps"] == len(losses) == 13
            assert abs(statistics.mean(losses) - epoch["mean_loss"]) < 1e-10
            multiplier = 0.1 ** (int(epoch_number >= 20) + int(epoch_number >= 40))
            expected_lr = [base * multiplier for base in (5e-6, 1e-5, 0.0005)]
            assert all(
                abs(observed - expected) < 1e-15
                for observed, expected in zip(epoch["learning_rates"], expected_lr, strict=True)
            )

        rows = [{key: value for key, value in row.items() if key != "event"} for row in log_epochs[log_cursor:log_cursor + 50]]
        assert rows == training["history"]
        log_cursor += 50

        folds.append(
            {
                "fold": fold_id,
                "epochs": training["epochs"],
                "optimizer_steps": training["optimizer_steps"],
                "epoch_step_counts": dict(Counter(row["optimizer_steps"] for row in training["history"])),
                "trainable_parameters": training["trainable_parameters"],
                "trainable_tensors": training["trainable_tensors"],
                "gradient_tensors": training["gradient_tensors"],
                "trainable_without_gradient": training["trainable_without_gradient"],
                "initial_state_sha256": training["initial_state_sha256"],
                "final_state_sha256": training["final_state_sha256"],
                "frozen_token_selection_sha256_unchanged": True,
                "source_training_exposures": len(exposed_records),
                "unique_source_records_exposed": len(set(exposed_records)),
                "sampled_records_all_source_only": True,
                "amp_scale_pairs": sorted(amp_scales),
            }
        )

    assert log_cursor == len(log_epochs) == 150
    return {
        "folds": folds,
        "optimizer_steps": sum(row["optimizer_steps"] for row in folds),
        "summary_optimizer_steps": summary["optimizer_steps"],
        "log_epoch_rows": len(log_epochs),
        "history_matches_log": True,
        "maximum_loss_composition_absolute_discrepancy": max(all_loss_diffs),
        "checkpoint_selection": summary["checkpoint_selection"],
        "epochs_selected_by_heldout": summary["epochs_selected_by_heldout"],
        "training_source_only": summary["training_source_only"],
        "heldout_image_forwards": summary["heldout_image_forwards"],
    }


def verify_bindings(config, manifest_result):
    manifest_paths = {pathlib.Path(row["path"]): row for row in manifest_result["checked"]}
    signal_matches = []
    for relative, expected_sha256 in config["signal_source_file_sha256"].items():
        path = pathlib.Path(
            read_json(MANIFEST)["signal_source_path_mapping"][relative]
        )
        assert path in manifest_paths
        assert manifest_paths[path]["actual_sha256"] == expected_sha256
        signal_matches.append(relative)

    project_matches = []
    for relative, expected_sha256 in config["project_source_file_sha256"].items():
        path = REPO / relative
        assert path in manifest_paths
        assert manifest_paths[path]["actual_sha256"] == expected_sha256
        project_matches.append(relative)

    return {
        "signal_source_files_bound": len(signal_matches),
        "project_source_files_bound": len(project_matches),
        "signal_source_files": signal_matches,
        "project_source_files": project_matches,
    }


def main():
    started = time.perf_counter()
    manifest = verify_manifest()
    config = read_json(REPO / "configs/MSVR310/Signal-source-oof-v1.json")
    protocol = read_json(REPO / "protocols/msvr310_train_oof_v1.json")
    summary = read_json(REPO / "evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json")
    rankings = read_json(REPO / "evidence/trifusion_msvr310_signal_v1_baseline_saved_rankings_20260906.json")
    preflight = read_json(REPO / "evidence/trifusion_msvr310_signal_v1_m0_r2_20260906.json")
    protocol_verification = read_json(REPO / "evidence/trifusion_msvr310_train_oof_protocol_verification_20260906.json")
    label_support = read_json(REPO / "evidence/trifusion_msvr310_source_label_support_20260906.json")

    assert manifest["all_ok"]
    assert protocol["aggregate_counts"]["valid_queries"] == 600
    assert protocol["aggregate_counts"]["unique_gallery_records"] == 1032
    assert protocol["aggregate_counts"]["unique_query_identities"] == 60
    assert protocol_verification["all_1032_records_match_original_training_labels"]
    assert protocol_verification["all_source_heldout_disjoint"]
    assert protocol_verification["heldout_union_covers_training_records_once"]
    assert label_support["train_records"] == 1032
    assert label_support["train_identities"] == 155
    assert label_support["identities_with_cross_scene_positives"] == 60
    assert label_support["eligible_train_as_query_records_under_scene_filter"] == 600

    replay = replay_rankings(protocol, summary, rankings)
    training = verify_training(protocol, summary, preflight)
    assert training["optimizer_steps"] == training["summary_optimizer_steps"] == 1950
    assert summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert summary["evaluation_type"] == "train_internal_identity_oof_baseline"
    assert summary["official_test_image_access"] == 0
    assert summary["fixed_rgbnt201_dev_image_access"] == 0
    assert summary["expert_training"] == 0

    output = {
        "script": str(pathlib.Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "numpy_imported": False,
        "input_manifest": {
            "manifest_path": manifest["manifest_path"],
            "created_at": manifest["created_at"],
            "input_count": manifest["input_count"],
            "all_ok": manifest["all_ok"],
            "mismatch_count": len(manifest["mismatches"]),
            "signal_source_path_mapping_count": manifest["signal_source_path_mapping_count"],
        },
        "source_binding": verify_bindings(config, manifest),
        "protocol_summary": {
            "records": len(protocol["records"]),
            "identities": len(protocol["identity_scene_membership"]),
            "aggregate_counts": protocol["aggregate_counts"],
            "evaluation": protocol["evaluation"],
        },
        "rank_replay": replay,
        "training_replay": training,
        "reported_status": {
            "summary_status": summary["status"],
            "evaluation_type": summary["evaluation_type"],
            "checkpoint_selection": summary["checkpoint_selection"],
            "epochs_selected_by_heldout": summary["epochs_selected_by_heldout"],
            "official_test_image_access": summary["official_test_image_access"],
            "fixed_rgbnt201_dev_image_access": summary["fixed_rgbnt201_dev_image_access"],
            "expert_training": summary["expert_training"],
        },
        "independent_arithmetic_result": "PASS",
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "output": str(OUT),
            "elapsed_seconds": output["elapsed_seconds"],
            "input_count": output["input_manifest"]["input_count"],
            "aggregate": output["rank_replay"]["aggregate"],
            "query_records": output["rank_replay"]["query_records"],
            "optimizer_steps": output["training_replay"]["optimizer_steps"],
            "log_epoch_rows": output["training_replay"]["log_epoch_rows"],
            "max_loss_discrepancy": output["training_replay"]["maximum_loss_composition_absolute_discrepancy"],
            "result": output["independent_arithmetic_result"],
        },
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
