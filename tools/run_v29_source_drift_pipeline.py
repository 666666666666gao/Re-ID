#!/usr/bin/env python3
"""One durable source-diagnostic process followed by complete CPU verification."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import os
import subprocess
import time

PYTHON = "/root/miniconda3/envs/tri_reid/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    repo, run = Path.cwd().resolve(), args.run_dir.resolve()
    prefix = str(run)
    assert repo == Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
    assert run.parent == Path("/root/autodl-tmp/trifusion-v2/artifacts")
    assert not run.exists()
    assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip() == args.code_commit
    assert sha(args.contract) == args.contract_sha256 and sha(args.plan) == args.plan_sha256
    contract = json.loads(args.contract.read_bytes())
    assert sha(__file__) == contract["pipeline_sha256"]
    for path,expected in contract["source_file_sha256"].items():
        assert sha(path) == expected, path
    started = time.time()
    def write(suffix, row):
        with Path(prefix+suffix).open("x",encoding="utf-8") as handle:
            json.dump(row,handle,indent=2)
            handle.write("\n")
    write("_launcher.json",{"wrapper_pid":os.getpid(),"repository_commit":args.code_commit,
                            "contract_sha256":args.contract_sha256,"plan_sha256":args.plan_sha256,
                            "pipeline_sha256":sha(__file__),"started_at":datetime.now().astimezone().isoformat()})
    env = dict(os.environ)
    env.update({"PYTHONPATH":str(repo/"modeling")+":"+str(repo),"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"4"})
    commands = [
        ("math",[PYTHON,"-m","tools.check_v29_source_role_drift","--cuda","--output",prefix+"_math.json"]),
        ("diagnostic",[PYTHON,"-u","-m","tools.diagnose_v29_source_role_drift",
                       "--contract",str(args.contract),"--contract-sha256",args.contract_sha256,
                       "--output-dir",str(run)]),
        ("verification",[PYTHON,"-u","-m","tools.verify_v29_source_role_drift",
                         "--run-dir",str(run),"--contract",str(args.contract),
                         "--output",str(run/"complete_source_drift_verification.json")]),
    ]
    for stage,command in commands:
        if stage == "verification": env["CUDA_VISIBLE_DEVICES"] = ""
        stage_start = time.time()
        with Path(prefix+"_"+stage+".log").open("x") as log:
            child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write("_"+stage+"_launch.json",{"original_pid":child.pid,"wrapper_pid":os.getpid(),"command":command,
                                           "started_at":datetime.now().astimezone().isoformat()})
            code = child.wait()
        write("_"+stage+"_exit.json",{"original_pid":child.pid,"wrapper_pid":os.getpid(),"exit_code":code,
                                     "ended_at":datetime.now().astimezone().isoformat(),"elapsed_seconds":time.time()-stage_start})
        if code:
            write("_pipeline_exit.json",{"stage":stage.upper()+"_FAILED","exit_code":code,
                                        "wrapper_pid":os.getpid(),"ended_at":datetime.now().astimezone().isoformat()})
            return code
    summary = json.loads((run/"source_role_drift.json").read_bytes())
    proof = json.loads((run/"complete_source_drift_verification.json").read_bytes())
    assert summary["model_forwards"] == 10080
    assert proof["status"] == "PASS_COMPLETE_V29_SOURCE_ROLE_DRIFT_AND_RELATIONS"
    write("_pipeline_exit.json",{"stage":"COMPLETE_VERIFIED_SOURCE_ROLE_DRIFT","exit_code":0,
                                "wrapper_pid":os.getpid(),"ended_at":datetime.now().astimezone().isoformat(),
                                "elapsed_seconds":time.time()-started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--code-commit",required=True)
    raise SystemExit(main(parser.parse_args()))
