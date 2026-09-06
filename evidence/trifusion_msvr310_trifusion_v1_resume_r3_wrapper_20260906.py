from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
runroot = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_trifusion_source_oof_v1_seed42_1c444cd")
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == "1ff7e2d8ed2ed56668f814e229aca347f57b9b5d"
assert hashlib.sha256((root / "tools/resume_msvr310_trifusion_exact_inference.py").read_bytes()).hexdigest() == "50e898b3ce993e7071bc46ec5df12a96e525d0fe744f3a87e20f274a3d7b290c"
assert hashlib.sha256((root / "evidence/trifusion_msvr310_trifusion_v1_resume_r3_plan_20260906.json").read_bytes()).hexdigest() == "a9a0bae91c8ffc2b2314ca8e00a3c09cbdaacf29eb4e7d44ad3bd42dfbaf0ff4"
assert not (runroot / "comparison_resume_r3").exists()
command = [sys.executable, str(root / "tools/resume_msvr310_trifusion_exact_inference.py"), "--run-root", str(runroot)]
with (runroot / "comparison_resume_r3.log").open("x") as log:
    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    launch = {"launched_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "wrapper_pid": os.getpid(), "child_pid": child.pid, "command": command,
              "scope": "Original study continuation: reuse original fold0 training/verified360 features; train only folds1/2,520 new updates; fixed full600-query comparison"}
    (runroot / "comparison_resume_r3_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    result = child.wait()
(runroot / "comparison_resume_r3_exit.txt").write_text(str(result) + "\n")
sys.exit(result)
