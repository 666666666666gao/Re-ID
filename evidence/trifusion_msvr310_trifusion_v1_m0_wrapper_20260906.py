from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
runroot = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_trifusion_source_oof_v1_seed42_148f5a7")
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == "148f5a7f6436830a909f7b709faa1fc2a81dbbec"
command = ["/root/miniconda3/envs/tri_reid/bin/python",
           str(root / "tools/train_msvr310_trifusion_oof.py"),
           "--config", str(root / "configs/MSVR310/TriFusion-source-oof-v1.json"),
           "--output-dir", str(runroot / "m0"), "--mode", "m0"]
with (runroot / "m0.log").open("w") as log:
    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    launch = {"launched_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "wrapper_pid": os.getpid(), "child_pid": child.pid, "command": command,
              "scope": "Original three-role MSVR310 source-only M0, three8-step capacities plus fresh fixed100-step overfit; no heldout"}
    (runroot / "m0_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    result = child.wait()
(runroot / "m0_exit.txt").write_text(str(result) + "\n")
sys.exit(result)
