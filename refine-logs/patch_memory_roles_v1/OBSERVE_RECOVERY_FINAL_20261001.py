#!/usr/bin/env python3
"""Observe the existing recovery panel at its estimated completion milestone."""

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
        row = {
            "observed_at": datetime.now().astimezone().isoformat(),
            "controller_status": state["status"],
            "jobs": [{key: job.get(key) for key in
                      ("dataset", "variant", "status", "pid", "exit_code")}
                     for job in state["jobs"]],
        }
        observations.append(row)
        print(json.dumps(row), flush=True)
        if state["status"] in ("COMPLETE", "FAILED"):
            break
        time.sleep(240)
    report = {"status": state["status"], "observations": observations,
              "campaign": str(args.campaign),
              "scope": "Observe only; no launch, retry or training changes"}
    if state["status"] == "COMPLETE":
        accepted_path = args.campaign / "accepted_matrix.json"
        accepted = json.loads(accepted_path.read_text())
        assert accepted["verified_complete"] == accepted["expected_endpoints"] == 6
        report["accepted_matrix_sha256"] = hashlib.sha256(accepted_path.read_bytes()).hexdigest()
        report["accepted_count"] = accepted["verified_complete"]
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "accepted_count": report.get("accepted_count")}), flush=True)


if __name__ == "__main__":
    main()
