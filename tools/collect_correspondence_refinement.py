#!/usr/bin/env python3
"""Bind the 18 queued controls to full training, weights and gallery scores."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_correspondence_roles import METRICS, read, sha
from tools.queue_correspondence_refinement import DATASETS, VARIANTS, child_campaign
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, WEIGHTS


def verify(run, m0_run, dataset, phase, variant, protocol, protocol_sha, baseline_sha):
    import numpy as np
    import torch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    training, official, m0 = (read(path) for path in
                             (run / "training.json", run / "official_metrics.json",
                              m0_run / "training.json"))
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert official["status"] == "COMPLETE" and m0["status"] == "M0_PASS"
    for receipt in (training, m0):
        assert receipt["dataset"] == dataset and receipt["seed"] == 42
        assert receipt["epochs"] == 50 and receipt["checkpoint_policy"] == "best_official_map"
        assert receipt["protocol_sha256"] == protocol_sha
        assert receipt["initializer"]["author_checkpoint_sha256"] == baseline_sha
    binding = training["initializer"]
    assert binding == m0["initializer"]
    assert binding["width"] == 128 and binding["fused_width"] == 1536
    if phase == "global":
        assert variant is None and binding["architecture"] == "correspondence_m1_global_only_v1"
        assert not binding["role_path"] and binding["common_initializer_variant"] == "100"
        assert binding["m1"] and not binding["m2"] and not binding["m3"]
        assert binding["prediction_weight"] == 0.0
        source_fields = {
            "entry_sha256": "tools/run_correspondence_global_only.py",
            "global_model_source_sha256": "modeling/trifusion/correspondence_global_only.py",
            "initializer_entry_sha256": "tools/run_correspondence_roles.py",
        }
        schema = "trifusion-correspondence-global-only-v1"
    else:
        assert phase == "m2" and binding["architecture"] == "correspondence_evidence_readout_v1"
        selection, structured = VARIANTS[variant]
        refinement = {"selection": selection, "structured_readout": structured}
        assert binding["refinement"] == refinement
        assert binding["m1"] and binding["m2"] and binding["m3"]
        assert binding["prediction_weight"] == 0.1
        source_fields = {
            "entry_sha256": "tools/run_correspondence_evidence_readout.py",
            "evidence_source_sha256": "modeling/trifusion/correspondence_evidence_readout.py",
            "reused_entry_sha256": "tools/run_correspondence_roles.py",
        }
        schema = "trifusion-correspondence-evidence-readout-v1"
    assert binding["model_source_sha256"] == sha(ROOT / "modeling/trifusion/correspondence_roles.py")
    assert all(binding[name] == sha(ROOT / path) for name, path in source_fields.items())
    assert m0["m0"]["frozen_signal_unchanged"]
    assert m0["m0"]["nonzero_gradient_parameters"] == m0["m0"]["trainable_parameters"]
    assert m0["m0"]["reload_max_abs_difference"] <= 1e-5
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    best = max(training["history"], key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    checkpoint = run / "best_map.pth"
    distances_path = run / "official_distances.pt"
    assert str(checkpoint) == training["checkpoint"]
    assert sha(checkpoint) == official["checkpoint_sha256"]
    assert sha(distances_path) == official["distance_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == schema
    assert payload["dataset"] == official["dataset"] == dataset
    assert payload["seed"] == official["seed"] == 42
    assert payload["protocol_sha256"] == official["protocol_sha256"] == protocol_sha
    assert payload["baseline_sha256"] == official["baseline_sha256"] == baseline_sha
    assert payload["epoch"] == training["best_epoch"] == official["selected_epoch"] == best["epoch"]
    assert official["training_epochs"] == 50 and not official["reranking"]
    assert official["independent_upstream_metrics_equal"]
    if phase == "m2":
        assert payload["variants"] == {"m1": True, "m2": True, "m3": True}
        assert payload["refinement"] == refinement
    for name in METRICS:
        assert abs(payload["metrics"][name] - official["metrics"][name]) < 1e-5
        assert abs(best["official_fused"][name] - official["metrics"][name]) < 1e-5

    # This is a trusted local artifact containing NumPy arrays, not only tensors.
    arrays = torch.load(distances_path, map_location="cpu", weights_only=False)
    for split in ("query", "gallery"):
        records = protocol["records"][split]
        assert len(records) == protocol["counts"][split]
        for key, field in (("ids", "identity"), ("cameras", "camera"), ("scenes", "scene")):
            assert np.array_equal(arrays[f"{split}_{key}"], np.asarray([row[field] for row in records]))
    distances = arrays["fused"].numpy()
    assert distances.shape == (protocol["counts"]["query"], protocol["counts"]["gallery"])
    assert np.isfinite(distances).all()
    field = "scenes" if dataset == "MSVR310" else "cameras"
    scores = (scene_scores if dataset == "MSVR310" else camera_scores)(
        distances, arrays["query_ids"], arrays["gallery_ids"],
        arrays[f"query_{field}"], arrays[f"gallery_{field}"])
    differences = {name: abs(scores["metrics"][name] - official["metrics"][name]) for name in METRICS}
    assert max(differences.values()) < 1e-5
    return {
        "status": "VERIFIED_COMPLETE", "best_epoch": best["epoch"], "metrics": official["metrics"],
        "trainable_parameters": binding["trainable_parameters"],
        "training_and_epoch_eval_seconds":
            (datetime.fromisoformat(training["completed_at"]) -
             datetime.fromisoformat(training["started_at"])).total_seconds(),
        "checkpoint_sha256": official["checkpoint_sha256"],
        "distance_sha256": official["distance_sha256"],
        "receipt_sha256": sha(run / "official_metrics.json"),
        "full_gallery_recomputed_metric_difference_pp": differences,
    }


def collect(campaign):
    manifest = read(campaign / "manifest.json")
    expected = [("global", dataset, None) for dataset in DATASETS]
    expected += [("m2", dataset, variant) for variant in VARIANTS for dataset in DATASETS]
    assert [(row["phase"], row["dataset"], row["variant"]) for row in manifest["jobs"]] == expected
    assert manifest["seed"] == 42 and manifest["epochs"] == 50
    protocols = {dataset: read(PROTOCOLS / f"{dataset}.json") for dataset in DATASETS}
    protocol_shas = {dataset: sha(PROTOCOLS / f"{dataset}.json") for dataset in DATASETS}
    for dataset, (weight, digest) in BASELINES.items():
        assert sha(WEIGHTS / weight) == digest
    rows = []
    for phase, dataset, variant in expected:
        directory = child_campaign(campaign, phase, dataset, variant)
        row = {"phase": phase, "dataset": dataset, "variant": variant, "campaign_dir": str(directory)}
        if not (directory / "campaign.json").exists():
            row["status"] = "PENDING"
        else:
            state = read(directory / "campaign.json")
            assert state["dataset"] == dataset
            if state["status"] == "COMPLETE":
                assert [job["mode"] for job in state["jobs"]] == ["m0", "train", "evaluate"]
                assert all(job["status"] == "COMPLETE" and job["exit_code"] == 0 for job in state["jobs"])
                m0_run, run = (Path(state["jobs"][index]["output_dir"]) for index in (0, 2))
                assert state["jobs"][1]["output_dir"] == str(run)
                row["run_dir"] = str(run)
                row.update(verify(run, m0_run, dataset, phase, variant, protocols[dataset],
                                  protocol_shas[dataset], BASELINES[dataset][1]))
            else:
                row.update(status="UNACCEPTED", campaign_status=state["status"])
        rows.append(row)
    return {
        "schema": "trifusion-correspondence-refinement-verification-v1",
        "collected_at": datetime.now().astimezone().isoformat(), "seed": 42,
        "expected_endpoints": 18,
        "verified_complete": sum(row["status"] == "VERIFIED_COMPLETE" for row in rows),
        "scope": "full training, checkpoint/protocol binding and CPU full-gallery scoring; no new inference",
        "rows": rows,
    }


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.campaign.resolve())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("collected_at", "expected_endpoints", "verified_complete")}))
