from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

repo = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
run = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v27_source_style_seed42_849b608")
commit = "849b608e4ce3bc522a8bf10870163e6a42d21b79"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == commit
argv = [
    "/root/miniconda3/envs/tri_reid/bin/python", "-u",
    "tools/train_signal_preserving_v27.py",
    "--config", "configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json",
    "--config-sha256", "dd1f3eed0816f4a8d00a76044cc5ad8b40786e597018afad7ea0bde90d29a3c6",
    "--plan", "refine-logs/trifusion_v27_source_style/EXPERIMENT_PLAN.md",
    "--plan-sha256", "305f89d0fe58257c76f9536d2260ab0ae6668b1fa42e81a6e766845649e94d3c",
    "--output-dir", str(run),
]
env = dict(
    os.environ,
    PYTHONPATH=str(repo) + ":" + str(repo / "modeling") + ":" + str(repo / "tests"),
    CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
)
started = time.time()
meta = {
    "execution_commit": commit, "screen": "v27_source_style_849b608",
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
