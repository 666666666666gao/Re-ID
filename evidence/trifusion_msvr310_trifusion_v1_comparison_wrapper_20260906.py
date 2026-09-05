from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
runroot = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_trifusion_source_oof_v1_seed42_1c444cd")
receipt = runroot / "m0/summary.json"
assert hashlib.sha256(receipt.read_bytes()).hexdigest() == "e021303b51d6af0b8bc49717016744e0ad483af419496652e0525b188d3d646b"
assert json.loads(receipt.read_text())["status"] == "PASS_ENGINEERING_ONLY"
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == "1c444cdf72e13fd041afd0c641dc8f522faa5844"
command = ["/root/miniconda3/envs/tri_reid/bin/python",
           str(root / "tools/train_msvr310_trifusion_oof.py"),
           "--config", str(root / "configs/MSVR310/TriFusion-source-oof-v1.json"),
           "--output-dir", str(runroot / "comparison"), "--mode", "comparison",
           "--m0-receipt", str(receipt)]
with (runroot / "comparison.log").open("w") as log:
    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    launch = {"launched_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "wrapper_pid": os.getpid(), "child_pid": child.pid, "command": command,
              "scope": "Three fresh original-role models over frozen vehicle B0, fixed20epochs each; final-only complete scene-filtered five-output comparison"}
    (runroot / "comparison_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    result = child.wait()
(runroot / "comparison_exit.txt").write_text(str(result) + "\n")
sys.exit(result)
