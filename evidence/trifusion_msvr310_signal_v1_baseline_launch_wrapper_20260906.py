import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
runroot = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_signal_source_oof_v1_seed42_bb01d60")
receipt = runroot / "m0/summary.json"
assert hashlib.sha256(receipt.read_bytes()).hexdigest() == "79c0e2b1c981c4bb10548f0249dca684c113feb43151cc4bdcc2a2e9bf2887ae"
assert json.loads(receipt.read_text())["status"] == "PASS_ENGINEERING_ONLY"
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == "bb01d60b6e1517ee6f5dc9120faefd17d75401e5"
command = [
    "/root/miniconda3/envs/tri_reid/bin/python",
    str(root / "tools/train_msvr310_signal_oof.py"),
    "--config", str(root / "configs/MSVR310/Signal-source-oof-v1.json"),
    "--output-dir", str(runroot / "baseline"),
    "--mode", "train",
    "--preflight-receipt", str(receipt),
]
with (runroot / "baseline.log").open("w") as log:
    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    launch = {"launched_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "wrapper_pid": os.getpid(), "child_pid": child.pid, "command": command,
              "scope": "Three freshly initialized source-only Signal models,50epochs each, final-only complete-gallery internal evaluation"}
    (runroot / "baseline_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    result = child.wait()
(runroot / "baseline_exit.txt").write_text(str(result) + "\n")
sys.exit(result)
