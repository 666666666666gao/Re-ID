#!/usr/bin/env python3
"""Recompute MSVR310 Signal M0 scalar and source-label evidence from JSON."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def audit(summary, protocol):
    assert summary["mode"] == "preflight" and summary["status"] == "PASS_ENGINEERING_ONLY"
    assert summary["seed"] == 42 and summary["optimizer_steps"] == 24
    assert summary["heldout_image_forwards"] == 0
    assert summary["official_test_image_access"] == summary["fixed_rgbnt201_dev_image_access"] == 0
    assert summary["expert_training"] == 0 and not summary["epochs_selected_by_heldout"]
    assert len(summary["folds"]) == 3
    rows, deviations = [], []
    for actual, fold in zip(summary["folds"], protocol["folds"], strict=True):
        assert actual["fold"] == fold["fold"]
        assert actual["source_ids"] == fold["source_ids"]
        assert actual["heldout_ids"] == fold["heldout_ids"]
        assert actual["counts"] == fold["counts"]
        assert actual["strict_reload_exact_feature_parity"]
        assert actual["feature_width"] == 3072 and actual["clean_source_feature_forwards"] == 16
        assert actual["heldout_image_forwards"] == 0
        training = actual["training"]
        assert training["epochs"] == 1 and training["optimizer_steps"] == 8
        assert training["overflow_events"] == 0 and len(training["steps"]) == 8
        assert training["trainable_tensors"] == training["gradient_tensors"] == 195
        assert not training["trainable_without_gradient"]
        assert training["initial_state_sha256"] != training["final_state_sha256"]
        assert training["frozen_token_selection_initial_sha256"] == training["frozen_token_selection_final_sha256"]
        assert training["frozen_token_selection_parameters"] == 787968
        source_indices = set(fold["source_record_indices"])
        assert len(training["optimizer_groups"]) == 195
        for group in training["optimizer_groups"]:
            assert not group["name"].startswith("SIM.token_selection.")
            if "classifier" in group["name"]:
                assert group["lr"] == 0.0005
        loss_values = []
        source_exposures = []
        for index, step in enumerate(training["steps"]):
            assert step["step"] == index + 1 and step["epoch"] == 1
            assert step["amp_scale_after"] >= step["amp_scale_before"]
            sampled = step["sampled_record_indices"]
            assert len(sampled) == 64 and set(sampled) <= source_indices
            identity_counts = Counter(protocol["records"][i]["identity"] for i in sampled)
            assert sorted(identity_counts.values()) == [8] * 8
            assert len(step["id_triplet_head_losses"]) == 4
            reconstructed = (sum(step["id_triplet_head_losses"]) +
                             0.2 * step["gram_loss"] + 0.01 * step["patch_loss"])
            deviations.append(abs(reconstructed - step["loss"]))
            loss_values.append(step["loss"])
            source_exposures.extend(sampled)
        epoch = training["history"]
        assert len(epoch) == 1 and epoch[0]["optimizer_steps"] == 8
        mean_error = abs(sum(loss_values) / 8 - epoch[0]["mean_loss"])
        assert mean_error < 1e-12
        assert epoch[0]["learning_rates"] == [5e-6, 1e-5, 0.0005]
        rows.append({"fold": fold["fold"], "source_identities": len(fold["source_ids"]),
                     "source_training_exposures": len(source_exposures),
                     "unique_source_records_exposed": len(set(source_exposures)),
                     "source_identities_exposed": len({protocol["records"][i]["identity"] for i in source_exposures}),
                     "mean_loss_recomputation_error": mean_error,
                     "trainable_parameters": training["trainable_parameters"],
                     "trainable_and_gradient_tensors": 195,
                     "optimizer_steps": 8, "clean_source_feature_forwards": 16,
                     "frozen_selector_unchanged": True})
    return {"scope": "Supplied JSON scalar/label arithmetic; no model, tensor, feature or checkpoint replay",
            "folds": rows, "optimizer_steps": 24, "source_training_exposures": 1536,
            "clean_source_feature_forwards": 48, "heldout_dev_official_image_forwards": 0,
            "maximum_loss_composition_absolute_discrepancy": max(deviations),
            "loss_note": "Recorded scalar components came from AMP; raw and weighted operation dtypes were not saved. Discrepancy is reported, not retroactively treated as a changed M0 qualification threshold.",
            "labels_counts_optimizer_and_state_metadata_checks": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.perf_counter()
    result = audit(json.loads(args.summary.read_text(encoding="utf-8")),
                   json.loads(args.protocol.read_text(encoding="utf-8")))
    result.update({"elapsed_seconds": time.perf_counter() - started,
                   "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                   "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest()})
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
