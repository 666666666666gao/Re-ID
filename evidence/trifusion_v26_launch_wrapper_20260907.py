from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v26_role_modal_responsibility_seed42_ff18e40")
commit = "ff18e40b12a62ddee25e36c5bdfe5fe4de0f7d5a"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == commit
argv = [
    "/root/miniconda3/envs/tri_reid/bin/python", "-u",
    "tools/train_signal_preserving_v26.py",
    "--config", "configs/RGBNT201/TriFusion-signal-preserving-v26-role-modal-responsibility-rtx3090.json",
    "--config-sha256", "01cc6e8d80d8ae68df3d2d858ed9102b9faf1b46c3021bb1ca993a0dbc775f13",
    "--plan", "refine-logs/trifusion_v26_role_modal_responsibility/EXPERIMENT_PLAN.md",
    "--plan-sha256", "0f23442fbc9b0b5e4fe9ddbef168338c529a5490b34db04920522c3f5bac842c",
    "--output-dir", str(run),
]
env = dict(
    os.environ,
    PYTHONPATH=str(repo) + ":" + str(repo / "modeling") + ":" + str(repo / "tests"),
    CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
)
started = time.time()
meta = {
    "execution_commit": commit, "screen": "v26_responsibility_ff18e40",
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

raise SystemExit(code)
