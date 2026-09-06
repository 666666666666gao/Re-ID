from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v27_source_style_seed42_c225652")
reporter = repo / "tools/report_v27_complete_comparison.py"
expected_sha = "233f1458bed89986f5712fe08d4a6946da194a1b31941fc3c70c5e8b0f301616"
assert hashlib.sha256(reporter.read_bytes()).hexdigest() == expected_sha
assert json.loads((run / "run_summary.json").read_bytes())["m0"]["passed"]
assert not (run / "complete_comparison.json").exists()
state = {
    "status": "WAITING_FOR_FULL_TERMINAL_VERIFICATION",
    "queued_at": datetime.now().astimezone().isoformat(),
    "wrapper_pid": os.getpid(),
    "training_original_pid": 126981,
    "training_execution_commit": "c225652a031ffcc524a48db8f49c31c70098287c",
    "reporter_sha256": expected_sha,
    "run_dir": str(run),
    "poll_interval_seconds": 180,
    "automatic_retries": 0,
    "requires_all_six_endpoints": True,
}
state_path = run / "complete_report_queue.json"
assert not state_path.exists()
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
while not (run / "terminal_verification.exit").exists():
    time.sleep(180)
assert int((run / "terminal_verification.exit").read_text()) == 0
verified = json.loads((run / "terminal_verification.json").read_bytes())
assert verified["status"] == "PASS_COMPLETE_V27_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES"
assert verified["checked_training_steps"] == 3360 and verified["eligible_queries"] == 571
assert hashlib.sha256(reporter.read_bytes()).hexdigest() == expected_sha
state["status"] = "RENDERING_COMPLETE_VERIFIED_COMPARISON"
state["started_at"] = datetime.now().astimezone().isoformat()
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
argv = [
    "/root/miniconda3/envs/tri_reid/bin/python", "-u", str(reporter),
    "--summary", str(run / "run_summary.json"),
    "--verification", str(run / "terminal_verification.json"),
    "--training-dir", str(run),
    "--output-json", str(run / "complete_comparison.json"),
    "--output-md", str(run / "complete_comparison.md"),
    "--query-csv", str(run / "all2855_query_output_rows.csv"),
    "--identity-csv", str(run / "all105_identity_output_rows.csv"),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
with (run / "complete_report.log").open("xb") as log:
    result = subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
state.update({
    "status": "REPORT_PROCESS_COMPLETE",
    "exit_code": result.returncode,
    "completed_at": datetime.now().astimezone().isoformat(),
})
(run / "complete_report.exit").write_bytes((str(result.returncode) + "\n").encode())
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
raise SystemExit(result.returncode)
