#!/usr/bin/env python3
"""Wait for the original fixed pipeline, then render its fully verified results."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--reporter-sha256", required=True)
parser.add_argument("--expected-wrapper-pid", type=int, required=True)
parser.add_argument("--expected-original-pid", type=int, required=True)
args = parser.parse_args()
repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = args.run_dir
reporter = repo / "tools/report_v27_source_style_relations.py"
assert hashlib.sha256(reporter.read_bytes()).hexdigest() == args.reporter_sha256
launch = json.loads(Path(str(run) + ".launch.json").read_bytes())
assert launch["wrapper_pid"] == args.expected_wrapper_pid
assert launch["original_pid"] == args.expected_original_pid
process = Path("/proc", str(args.expected_wrapper_pid))
assert process.exists()
cmdline = (process / "cmdline").read_bytes().split(b"\0")
assert str(run).encode() in cmdline
assert str(repo / "tools/run_v27_style_relation_pipeline.py").encode() in cmdline
state_path = run / "report_queue.json"
assert not state_path.exists()
state = {
    "stage": "WAITING_FOR_ORIGINAL_DIAGNOSTIC_AND_CPU_VERIFIER",
    "queue_pid": os.getpid(), "original_pid": args.expected_original_pid,
    "wrapper_pid": args.expected_wrapper_pid, "poll_seconds": 180,
    "reporter_sha256": args.reporter_sha256,
    "queue_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "started_at": datetime.now().astimezone().isoformat(),
    "new_model_forwards": 0, "new_optimizer_updates": 0, "automatic_retries": 0,
}
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
while process.exists():
    time.sleep(180)
assert int(Path(str(run) + ".exit").read_text()) == 0
assert int((run / "verification.exit").read_text()) == 0
assert hashlib.sha256(reporter.read_bytes()).hexdigest() == args.reporter_sha256
argv = [sys.executable, "-u", str(reporter), "--run-dir", str(run),
        "--output-dir", str(run / "complete_report")]
state.update(stage="REPORTING_ALL_VERIFIED_CONDITIONS", report_started_at=datetime.now().astimezone().isoformat(), argv=argv)
with (run / "report.log").open("xb") as log:
    child = subprocess.Popen(argv, cwd=repo, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                             stdout=log, stderr=subprocess.STDOUT)
    state["reporter_pid"] = child.pid
    state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
    code = child.wait()
(run / "report.exit").write_bytes((str(code) + "\n").encode())
state.update(stage="COMPLETE" if code == 0 else "REPORT_FAILED",
             reporter_exit_code=code, completed_at=datetime.now().astimezone().isoformat())
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
raise SystemExit(code)
