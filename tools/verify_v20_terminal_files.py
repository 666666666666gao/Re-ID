#!/usr/bin/env python3
"""Verify remote V20 terminal file bytes and receipts without loading weights."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo, directory):
    summary_path = directory / "run_summary.json"
    summary = json.loads(summary_path.read_bytes())
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL") and len(summary["folds"]) == 3
    assert summary["repository_commit"] == "3cea5bfc17e214b1829c020527699d939efa221d"
    config_path = repo / "configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml"
    config = yaml.safe_load(config_path.read_text())
    expected = {repo / path: value for path, value in summary["source_file_sha256"].items()}
    expected[repo / "tools/train_signal_preserving_v20.py"] = summary["runner_sha256"]
    expected[config_path] = summary["config_sha256"]
    expected[repo / "refine-logs/v20/EXPERIMENT_PLAN.md"] = summary["plan_sha256"]
    expected[Path(config["SIGNAL"]["CLIP_WEIGHT"])] = config["SIGNAL"]["CLIP_WEIGHT_SHA256"]
    initial = config["INITIALIZATION"]
    expected[Path(initial["V12_RUN_SUMMARY"])] = initial["V12_RUN_SUMMARY_SHA256"]
    for paths in initial["V12_FOLDS"]:
        for kind in ("SIGNAL", "EXPERT"):
            expected[Path(paths[f"{kind}_CHECKPOINT"])] = paths[f"{kind}_CHECKPOINT_SHA256"]
    endpoints = []
    for fold in summary["folds"]:
        for name, endpoint in fold["endpoints"].items():
            expected[Path(endpoint["checkpoint"])] = endpoint["checkpoint_sha256"]
            receipt_path = directory / f"fold_{fold['fold']}_{name}_receipt.json"
            assert json.loads(receipt_path.read_bytes()) == endpoint
            endpoints.append({"fold": fold["fold"], "endpoint": name,
                              "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path),
                              "receipt_equals_summary": True})
    assert len(endpoints) == 6
    files = []
    for path, value in expected.items():
        actual = sha256(path)
        assert actual == value, (str(path), value, actual)
        files.append({"path": str(path), "bytes": path.stat().st_size,
                      "sha256": actual, "expected_sha256": value, "matches": True})
    signal = config["SIGNAL"]["SOURCE"]
    signal_head = subprocess.check_output(["git", "-C", signal, "rev-parse", "HEAD"], text=True).strip()
    signal_diff_sha = hashlib.sha256(subprocess.check_output(["git", "-C", signal, "diff", "--binary"])).hexdigest()
    assert signal_head == summary["signal_commit"] and signal_diff_sha == summary["signal_diff_sha256"]
    bootstrap_path = "modeling/trifusion/signal_preserving_v13.py"
    bootstrap_source = subprocess.check_output(["git", "-C", str(repo), "show",
                                              summary["repository_commit"] + ":" + bootstrap_path])
    assert (repo / bootstrap_path).read_bytes() == bootstrap_source
    return {"verified": True, "observed_at": datetime.datetime.now().astimezone().isoformat(),
            "scientific_status": summary["status"], "execution_source_commit": summary["repository_commit"],
            "observed_repository_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "run_summary_sha256": sha256(summary_path), "run_summary_bytes": summary_path.stat().st_size,
            "log_sha256": sha256(directory.with_suffix(".log")),
            "files": files, "endpoint_receipts": endpoints,
            "signal_commit": signal_head, "signal_diff_sha256": signal_diff_sha,
            "bootstrap_source_matches_execution_commit": True,
            "bootstrap_source_sha256": hashlib.sha256(bootstrap_source).hexdigest(),
            "verifier_sha256": sha256(Path(__file__)),
            "scope": "Whole-file SHA and receipt consistency only; no model or checkpoint tensor loading",
            "optimizer_steps": 0, "retrieval_evaluation_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.repo, args.run_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": report["verified"], "files": len(report["files"]),
                      "endpoints": len(report["endpoint_receipts"]),
                      "summary_sha256": report["run_summary_sha256"]}))
