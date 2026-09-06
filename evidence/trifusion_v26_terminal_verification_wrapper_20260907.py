from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v26_role_modal_responsibility_seed42_ff18e40")
verifier = repo / "tools/verify_v26_complete_terminal.py"
expected_sha = "ac7b37ee389ddbea2eb48f94ffbc00fab885f02d248d803d7f04f6d23dff819e"
assert hashlib.sha256(verifier.read_bytes()).hexdigest() == expected_sha
assert not (run / "terminal_verification.json").exists()
state = {
    "status": "WAITING_FOR_ORIGINAL_TRAINING_TERMINAL",
    "queued_at": datetime.now().astimezone().isoformat(),
    "wrapper_pid": os.getpid(), "training_original_pid": 118939,
    "training_execution_commit": "ff18e40b12a62ddee25e36c5bdfe5fe4de0f7d5a",
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
