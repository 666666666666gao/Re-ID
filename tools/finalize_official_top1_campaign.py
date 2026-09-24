#!/usr/bin/env python3
"""Run read-only diagnostics after the six fixed official endpoints complete."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("RGBNT201", "RGBNT100", "MSVR310")
METHODS = ("R2", "R2_TOP1")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    assert campaign["status"] == "COMPLETE" and campaign["fixed_epoch"] == 20
    assert campaign["seed"] == 42
    jobs = {(row["dataset"], row["method"]): row for row in campaign["jobs"]}
    assert set(jobs) == {(dataset, method) for dataset in DATASETS for method in METHODS}
    assert all(row["status"] == "COMPLETE" for row in jobs.values())
    output = args.campaign.parent / "diagnostics"
    output.mkdir(exist_ok=False)
    reports = {}
    for dataset in DATASETS:
        receipts = {}
        for method in METHODS:
            receipt = Path(jobs[dataset, method]["metrics_path"])
            assert receipt.is_file()
            diagnostic = output / f"{dataset}_{method}_signal_diagnosis.json"
            subprocess.run([sys.executable, str(ROOT / "tools/diagnose_official_retrieval.py"),
                            "--receipt", str(receipt), "--output", str(diagnostic)],
                           cwd=ROOT, check=True)
            receipts[method] = receipt
            reports[f"{dataset}_{method}"] = dict(path=str(diagnostic), sha256=sha256(diagnostic))
        paired = output / f"{dataset}_R2_vs_R2_TOP1.json"
        subprocess.run([sys.executable, str(ROOT / "tools/report_official_top1_pair.py"),
                        "--control", str(receipts["R2"]),
                        "--candidate", str(receipts["R2_TOP1"]),
                        "--output", str(paired)], cwd=ROOT, check=True)
        reports[f"{dataset}_paired"] = dict(path=str(paired), sha256=sha256(paired))
    result = dict(status="COMPLETE", campaign=str(args.campaign),
                  campaign_sha256=sha256(args.campaign), reports=reports)
    path = output / "finalization.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
