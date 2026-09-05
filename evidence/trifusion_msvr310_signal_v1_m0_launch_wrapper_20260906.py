import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
runroot = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_signal_source_oof_v1_seed42_2dcbe85")
assert json.loads((runroot / "t0_receipt.json").read_text())["returncode"] == 0
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == "2dcbe8541329653fd57ebed40f6ed2349ff4467b"
command = [
    "/root/miniconda3/envs/tri_reid/bin/python",
    str(root / "tools/train_msvr310_signal_oof.py"),
    "--config", str(root / "configs/MSVR310/Signal-source-oof-v1.json"),
    "--output-dir", str(runroot / "m0"),
    "--mode", "preflight",
]
with (runroot / "m0.log").open("w") as log:
    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    launch = {
        "launched_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "wrapper_pid": os.getpid(), "child_pid": child.pid, "command": command,
        "scope": "Three source-only8-step engineering runs; no heldout/dev/official",
    }
    (runroot / "m0_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    result = child.wait()
(runroot / "m0_exit.txt").write_text(str(result) + "\n")
sys.exit(result)
