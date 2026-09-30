#!/usr/bin/env python3
"""Observe the current panel only, beginning at the registered completion estimate."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--due", required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    due = datetime.fromisoformat(args.due).timestamp()
    while time.time() < due:
        time.sleep(min(240, due - time.time()))
    observations = []
    while True:
        state = json.loads((args.campaign / "campaign.json").read_text())
        row = {"observed_at": datetime.now().astimezone().isoformat(), "status": state["status"],
               "jobs": [{key: job.get(key) for key in ("dataset", "variant", "status", "pid", "exit_code")}
                        for job in state["jobs"]]}
        observations.append(row)
        print(json.dumps(row), flush=True)
        if state["status"] in ("COMPLETE", "FAILED"):
            break
        time.sleep(240)
    report = {"status": state["status"], "observations": observations,
              "scope": "Observation only; no retry or runtime changes"}
    if state["status"] == "COMPLETE":
        matrix_path = args.campaign / "accepted_matrix.json"
        accepted = json.loads(matrix_path.read_text())
        assert accepted["verified_complete"] == accepted["expected_endpoints"] == 6
        report["accepted_count"] = 6
        report["accepted_matrix_sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "accepted_count": report.get("accepted_count")}), flush=True)


if __name__ == "__main__":
    main()
