from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd")
verifier = repo / "tools/verify_v25_complete_terminal.py"
expected_sha = "cc1cf0c1db52dc3b9bf770cd9cdc427d9142f751b62afda78bb5a73fa4e5a277"
assert hashlib.sha256(verifier.read_bytes()).hexdigest() == expected_sha
assert not (run / "terminal_verification.json").exists()
state = {
    "status": "WAITING_FOR_ORIGINAL_TRAINING_TERMINAL",
    "queued_at": datetime.now().astimezone().isoformat(),
    "wrapper_pid": os.getpid(), "training_original_pid": 112550,
    "training_execution_commit": "97468dd01bcef55d90455b35b1373aa03a2e335f",
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
