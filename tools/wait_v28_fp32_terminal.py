#!/usr/bin/env python3
"""Wait for the registered V28 R2 terminal, then verify all saved evidence once."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import time

REPO = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN = Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95")
PYTHON = "/root/miniconda3/envs/tri_reid/bin/python"


def main():
    launch = json.loads(Path(str(RUN) + "_launch.json").read_bytes())
    assert launch["repository_commit"] == "bf8de956e685311dd70631395009a2c06a2c8591"
    sources = {
        "tools/verify_v28_fp32_complete_terminal.py": "6562d69d3f028801c7eddda33b0b4ed2a40aeef3df0d63fd04a304a423a108bd",
        "tools/report_v28_fp32_complete_comparison.py": "b7ea7c85ea9da0c48b0750b80b8d73570a3b9eef1ae39187823f4953c32b9f05",
    }
    for name, expected in sources.items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected
    exit_path = Path(str(RUN) + "_exit.json")
    print(json.dumps({"stage": "WAITING_FOR_COMPLETE_TRAINING_TERMINAL",
                      "at": datetime.now().astimezone().isoformat(), "poll_seconds": 240}), flush=True)
    while not exit_path.exists():
        time.sleep(240)
    terminal = json.loads(exit_path.read_bytes())
    assert terminal["exit_code"] == 0, terminal
    assert not (Path("/proc") / str(terminal["original_pid"])).exists()
    summary = json.loads((RUN / "run_summary.json").read_bytes())
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL") and len(summary["folds"]) == 3
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(REPO / "modeling") + ":" + str(REPO),
                "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
    verified = RUN / "complete_terminal_verification.json"
    subprocess.run([PYTHON, "tools/verify_v28_fp32_complete_terminal.py",
                    "--repo", str(REPO), "--run-dir", str(RUN), "--output", str(verified)],
                   cwd=REPO, env=env, check=True)
    subprocess.run([PYTHON, "tools/report_v28_fp32_complete_comparison.py",
                    "--summary", str(RUN / "run_summary.json"), "--training-dir", str(RUN),
                    "--verification", str(verified),
                    "--output-json", str(RUN / "complete_comparison.json"),
                    "--output-md", str(RUN / "complete_comparison.md"),
                    "--query-csv", str(RUN / "all_query_comparison.csv"),
                    "--identity-csv", str(RUN / "all_identity_comparison.csv")],
                   cwd=REPO, env=env, check=True)
    print(json.dumps({"stage": "COMPLETE_VERIFIED_AND_REPORTED", "scientific_status": summary["status"],
                      "at": datetime.now().astimezone().isoformat()}), flush=True)


if __name__ == "__main__":
    main()
