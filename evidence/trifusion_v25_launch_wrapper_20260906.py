from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd")
commit = "97468dd01bcef55d90455b35b1373aa03a2e335f"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == commit
argv = [
    "/root/miniconda3/envs/tri_reid/bin/python", "-u",
    "tools/train_signal_preserving_v25.py",
    "--config", "configs/RGBNT201/TriFusion-signal-preserving-v25-camera-coverage-rtx3090.json",
    "--config-sha256", "cc8178de4173e01115a39505115bbc3e67ce322c08cd59cafff952e965116e4b",
    "--plan", "refine-logs/trifusion_v25_camera_coverage/EXPERIMENT_PLAN.md",
    "--plan-sha256", "0399da7522f92c152bb453b3e3a6b5ebd6674e80c1fe02ba93e2e97aea47cc9f",
    "--output-dir", str(run),
]
env = dict(
    os.environ,
    PYTHONPATH=str(repo) + ":" + str(repo / "modeling") + ":" + str(repo / "tests"),
    CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
)
started = time.time()
meta = {
    "execution_commit": commit, "screen": "v25_camera_coverage_97468dd",
    "argv": argv, "run_dir": str(run), "log": str(run) + ".log",
    "exit_file": str(run) + ".exit",
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "wrapper_pid": os.getpid(), "planned_m0_optimizer_steps": 116,
    "planned_q1_optimizer_steps": 3360, "planned_q1_endpoint_epochs": 120,
    "same_original_process_m0_then_complete_q1": True,
    "dev_access_count": 0, "official_test_access_count": 0,
}
with Path(meta["log"]).open("xb") as handle:
    process = subprocess.Popen(argv, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
    meta.update({
        "launched_at": datetime.now().astimezone().isoformat(),
        "original_pid": process.pid,
    })
    Path(str(run) + ".launch.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode())
    code = process.wait()
Path(meta["exit_file"]).write_bytes((str(code) + "\n").encode())
terminal = {
    **meta, "completed_at": datetime.now().astimezone().isoformat(),
    "exit_code": code, "elapsed_seconds": time.time() - started,
}
Path(str(run) + ".terminal.json").write_bytes((json.dumps(terminal, indent=2) + "\n").encode())
