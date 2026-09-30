#!/usr/bin/env python3
"""Accept only complete Patch-memory runs with bound checkpoints and full-gallery scores."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_correspondence_context_identity_losses import audit
from tools.collect_correspondence_roles import METRICS, read, sha
from tools.queue_correspondence_refinement import DATASETS, child_campaign
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, WEIGHTS
from tools.queue_slot_competition_roles import CONDITIONS


def verify(run, m0_run, dataset, variant, manifest):
    import numpy as np
    import torch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    training, official, m0 = (read(path) for path in (
        run / "training.json", run / "official_metrics.json", m0_run / "training.json"))
    assert training["status"] == "BEST_OFFICIAL_MAP_TRAINING_COMPLETE"
    assert official["status"] == "COMPLETE" and m0["status"] == "M0_PASS"
    protocol_path = PROTOCOLS / f"{dataset}.json"
    protocol = read(protocol_path)
    _, baseline_sha = BASELINES[dataset]
    normalization = CONDITIONS[variant]
    memory_mode = "full"
    condition = {"query_mode": "context", "auxiliary_target": "none"}
    for receipt in (training, m0):
        assert receipt["schema"] == "trifusion-correspondence-context-identity-training-v1"
        assert receipt["dataset"] == dataset and receipt["seed"] == 42
        assert receipt["epochs"] == 50 and receipt["checkpoint_policy"] == "best_official_map"
        assert receipt["protocol_sha256"] == sha(protocol_path)
        assert receipt["initializer"]["author_checkpoint_sha256"] == baseline_sha
    binding = training["initializer"]
    assert binding == m0["initializer"]
    assert binding["architecture"] == "slot_competition_roles_v1"
    assert binding["attention_normalization"] == normalization
    assert binding["memory_mode"] == memory_mode
    assert binding["slot_count"] == 16 and binding["patch_count"] == 128
    assert binding["memory_candidates"] == 128 and binding["fixed_slot_positions"]
    assert binding["condition"] == condition and binding["query_context_width"] == 512
    assert binding["width"] == 128 and binding["fused_width"] == 1536
    assert binding["m1"] and binding["m2"] and not binding["m3"]
    for field, path in (("model_source_sha256", "modeling/trifusion/correspondence_roles.py"),
                        ("patch_memory_source_sha256", "modeling/trifusion/patch_memory_roles.py"),
                        ("slot_competition_source_sha256", "modeling/trifusion/slot_competition_roles.py"),
                        ("entry_sha256", "tools/run_slot_competition_roles.py"),
                        ("reused_context_entry_sha256", "tools/run_correspondence_context_identity.py"),
                        ("reused_entry_sha256", "tools/run_correspondence_roles.py"),
                        ("context_source_sha256", "modeling/trifusion/correspondence_context_identity.py"),
                        ("evidence_source_sha256", "modeling/trifusion/correspondence_evidence_readout.py")):
        assert binding[field] == manifest["source_sha256"][path] == sha(ROOT / path)
    assert m0["m0"]["frozen_signal_unchanged"]
    assert m0["m0"]["nonzero_gradient_parameters"] == m0["m0"]["trainable_parameters"]
    assert m0["m0"]["reload_max_abs_difference"] <= 1e-5
    assert len(m0["history"]) == 1 and m0["history"][0]["steps"] == 8
    m0_steps = [json.loads(line) for line in (m0_run / "training_steps.jsonl").read_text().splitlines()]
    assert [row["batch"] for row in m0_steps] == list(range(8))
    assert all(row["epoch"] == 1 and row["auxiliary_target"] == "none"
               and row["auxiliary_id"] == 0 for row in m0_steps)
    probe = m0_run / "m0_reload_probe.pth"
    assert sha(probe) == m0["m0"]["reload_probe_sha256"]
    probe_payload = torch.load(probe, map_location="cpu", weights_only=True)
    assert probe_payload["schema"] == "trifusion-slot-competition-roles-v1"
    assert probe_payload["memory_mode"] == memory_mode
    assert probe_payload["attention_normalization"] == normalization
    assert [row["epoch"] for row in training["history"]] == list(range(1, 51))
    best = max(training["history"], key=lambda row: (row["official_fused"]["mAP"], row["epoch"]))
    checkpoint, distance_path = run / "best_map.pth", run / "official_distances.pt"
    assert training["checkpoint"] == str(checkpoint)
    assert sha(checkpoint) == official["checkpoint_sha256"]
    assert sha(distance_path) == official["distance_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-slot-competition-roles-v1"
    assert payload["condition"] == official["condition"] == condition
    assert payload["memory_mode"] == official["memory_mode"] == memory_mode
    assert official["architecture"] == "slot_competition_roles_v1"
    assert payload["attention_normalization"] == official["attention_normalization"] == normalization
    assert payload["variants"] == {"m1": True, "m2": True, "m3": False}
    assert payload["dataset"] == official["dataset"] == dataset
    assert payload["seed"] == official["seed"] == 42
    assert payload["protocol_sha256"] == official["protocol_sha256"] == sha(protocol_path)
    assert payload["baseline_sha256"] == official["baseline_sha256"] == baseline_sha
    assert payload["epoch"] == official["selected_epoch"] == training["best_epoch"] == best["epoch"]
    assert official["schema"] == "trifusion-correspondence-context-identity-retrieval-v1"
    assert official["training_epochs"] == 50 and not official["reranking"]
    assert official["independent_upstream_metrics_equal"]
    for name in METRICS:
        assert abs(payload["metrics"][name] - official["metrics"][name]) < 1e-5
        assert abs(best["official_fused"][name] - official["metrics"][name]) < 1e-5
    arrays = torch.load(distance_path, map_location="cpu", weights_only=False)
    for split in ("query", "gallery"):
        records = protocol["records"][split]
        assert len(records) == protocol["counts"][split]
        for key, field in (("ids", "identity"), ("cameras", "camera"), ("scenes", "scene")):
            assert np.array_equal(arrays[f"{split}_{key}"], np.asarray([row[field] for row in records]))
    differences = {}
    for path in ("fused", "shared_global", "joint_local"):
        distances = arrays[path].numpy()
        assert distances.shape == (protocol["counts"]["query"], protocol["counts"]["gallery"])
        assert np.isfinite(distances).all()
        field = "scenes" if dataset == "MSVR310" else "cameras"
        scorer = scene_scores if dataset == "MSVR310" else camera_scores
        scores = scorer(distances, arrays["query_ids"], arrays["gallery_ids"],
                        arrays[f"query_{field}"], arrays[f"gallery_{field}"])
        metrics = official["metrics"] if path == "fused" else official["diagnostic_metrics"][path]
        differences[path] = {name: abs(scores["metrics"][name] - metrics[name]) for name in METRICS}
        assert max(differences[path].values()) < 1e-5
    losses = audit(run, "none")
    return {"status": "VERIFIED_COMPLETE", "best_epoch": best["epoch"],
            "metrics": official["metrics"], "diagnostic_metrics": official["diagnostic_metrics"],
            "condition": condition, "memory_mode": memory_mode, "attention_normalization": normalization,
            "initial_model_state_sha256": binding["initial_model_state_sha256"],
            "trainable_parameters": binding["trainable_parameters"],
            "training_and_epoch_eval_seconds": (datetime.fromisoformat(training["completed_at"]) -
                                                datetime.fromisoformat(training["started_at"])).total_seconds(),
            "checkpoint_sha256": official["checkpoint_sha256"],
            "distance_sha256": official["distance_sha256"],
            "receipt_sha256": sha(run / "official_metrics.json"),
            "full_gallery_recomputed_metric_difference_pp": differences,
            "task_scalars": {key: losses[key] for key in (
                "logged_steps", "maximum_loss_reconstruction_error", "nonzero_triplet_steps",
                "nonzero_auxiliary_id_steps", "first", "best", "last")}}


def collect(campaign):
    manifest = read(campaign / "manifest.json")
    assert manifest["schema"] == "trifusion-slot-competition-roles-panel-v1"
    expected = [(dataset, variant) for variant in CONDITIONS for dataset in DATASETS]
    assert [(row["dataset"], row["variant"]) for row in manifest["jobs"]] == expected
    assert manifest["seed"] == 42 and manifest["epochs"] == 50
    assert all(sha(ROOT / name) == digest for name, digest in manifest["source_sha256"].items())
    for weight, digest in BASELINES.values():
        assert sha(WEIGHTS / weight) == digest
    rows = []
    for dataset, variant in expected:
        folder = child_campaign(campaign, "slot_competition", dataset, variant)
        row = {"phase": "slot_competition", "dataset": dataset, "variant": variant, "campaign_dir": str(folder)}
        if not (folder / "campaign.json").exists():
            row["status"] = "PENDING"
        else:
            state = read(folder / "campaign.json")
            if state["status"] == "COMPLETE":
                assert state["dataset"] == dataset and state["variant"] == variant
                assert [job["mode"] for job in state["jobs"]] == ["m0", "train", "evaluate"]
                assert all(job["status"] == "COMPLETE" and job["exit_code"] == 0 for job in state["jobs"])
                m0_run, run = (Path(state["jobs"][index]["output_dir"]) for index in (0, 2))
                assert state["jobs"][1]["output_dir"] == str(run)
                row["run_dir"] = str(run)
                row.update(verify(run, m0_run, dataset, variant, manifest))
            else:
                row.update(status="UNACCEPTED", campaign_status=state["status"])
        rows.append(row)
    for dataset in DATASETS:
        complete = [row for row in rows if row["dataset"] == dataset and row["status"] == "VERIFIED_COMPLETE"]
        assert len({row["initial_model_state_sha256"] for row in complete}) <= 1
        assert len({row["trainable_parameters"] for row in complete}) <= 1
    return {"schema": "trifusion-slot-competition-roles-verification-v1",
            "collected_at": datetime.now().astimezone().isoformat(), "seed": 42,
            "expected_endpoints": 6,
            "verified_complete": sum(row["status"] == "VERIFIED_COMPLETE" for row in rows),
            "scope": "Full50, one mAP best, strict reload and complete gallery for fused/global/local",
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = collect(args.campaign.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "collected_at", "expected_endpoints", "verified_complete")}))
