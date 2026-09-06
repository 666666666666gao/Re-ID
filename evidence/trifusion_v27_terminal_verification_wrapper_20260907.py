from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v27_source_style_seed42_c225652")
verifier = repo / "tools/verify_v27_complete_terminal.py"
expected_sha = "46fa03d799e9556970880b9435c940af8d413a18b677929379dd40fe45a0179e"
assert hashlib.sha256(verifier.read_bytes()).hexdigest() == expected_sha
assert not (run / "terminal_verification.json").exists()
assert json.loads((run / "run_summary.json").read_bytes())["m0"]["passed"]
state = {
    "status": "WAITING_FOR_ORIGINAL_TRAINING_TERMINAL",
    "queued_at": datetime.now().astimezone().isoformat(),
    "wrapper_pid": os.getpid(), "training_original_pid": 126981,
    "training_execution_commit": "c225652a031ffcc524a48db8f49c31c70098287c",
    "verifier_sha256": expected_sha, "run_dir": str(run),
    "poll_interval_seconds": 180, "automatic_retries": 0,
}
state_path = run / "terminal_verification_queue.json"
assert not state_path.exists()
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
while not Path(str(run) + ".terminal.json").exists():
    time.sleep(180)
assert hashlib.sha256(verifier.read_bytes()).hexdigest() == expected_sha
state["status"] = "RUNNING_FULL_TERMINAL_VERIFICATION"
state["verification_started_at"] = datetime.now().astimezone().isoformat()
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())
argv = [
    "/root/miniconda3/envs/tri_reid/bin/python", "-u", str(verifier),
    "--repo", str(repo), "--run-dir", str(run),
    "--output", str(run / "terminal_verification.json"),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
started = time.time()
with (run / "terminal_verification.log").open("xb") as log:
    result = subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
state.update({
    "status": "VERIFICATION_PROCESS_COMPLETE",
    "verification_exit_code": result.returncode,
    "verification_completed_at": datetime.now().astimezone().isoformat(),
    "verification_elapsed_seconds": time.time() - started,
})
(run / "terminal_verification.exit").write_bytes((str(result.returncode) + "\n").encode())
state_path.write_bytes((json.dumps(state, indent=2) + "\n").encode())

raise SystemExit(result.returncode)
