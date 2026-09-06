#!/usr/bin/env python3
"""Detached original diagnostic process followed by full CPU verification."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir",type=Path,required=True)
parser.add_argument("--contract",type=Path,required=True)
parser.add_argument("--contract-sha256",required=True)
parser.add_argument("--expected-commit",required=True)
args=parser.parse_args()
repo=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()==args.expected_commit
assert hashlib.sha256(args.contract.read_bytes()).hexdigest()==args.contract_sha256
assert not args.run_dir.exists()
run=args.run_dir
env=dict(os.environ,PYTHONPATH=str(repo)+":"+str(repo/"modeling")+":"+str(repo/"tests"),
         CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
argv=[sys.executable,"-u",str(repo/"tools/diagnose_v27_source_style_relations.py"),
      "--contract",str(args.contract),"--contract-sha256",args.contract_sha256,"--output-dir",str(run)]
started=time.time()
meta={"execution_commit":args.expected_commit,"contract_sha256":args.contract_sha256,
      "run_dir":str(run),"wrapper_pid":os.getpid(),"wrapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "argv":argv,"planned_model_forwards":10080,"planned_optimizer_updates":0,"automatic_retries":0}
with Path(str(run)+".log").open("xb") as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    meta.update({"launched_at":datetime.now().astimezone().isoformat(),"original_pid":child.pid})
    Path(str(run)+".launch.json").write_bytes((json.dumps(meta,indent=2)+"\n").encode())
    exit_code=child.wait()
Path(str(run)+".exit").write_bytes((str(exit_code)+"\n").encode())
terminal={**meta,"exit_code":exit_code,"completed_at":datetime.now().astimezone().isoformat(),
          "elapsed_seconds":time.time()-started}
Path(str(run)+".terminal.json").write_bytes((json.dumps(terminal,indent=2)+"\n").encode())
if exit_code != 0:
    raise SystemExit(exit_code)
verify=[sys.executable,"-u",str(repo/"tools/verify_v27_source_style_relations.py"),
        "--run-dir",str(run),"--contract",str(args.contract),"--output",str(run/"verification.json")]
state={"stage":"FULL_CPU_VERIFICATION","wrapper_pid":os.getpid(),"original_pid":child.pid,
       "original_exit_code":0,"started_at":datetime.now().astimezone().isoformat(),"argv":verify}
with (run/"verification.log").open("xb") as log:
    process=subprocess.Popen(verify,cwd=repo,env=dict(env,CUDA_VISIBLE_DEVICES=""),stdout=log,stderr=subprocess.STDOUT)
    state["verification_pid"]=process.pid
    (run/"pipeline.json").write_bytes((json.dumps(state,indent=2)+"\n").encode())
    code=process.wait()
(run/"verification.exit").write_bytes((str(code)+"\n").encode())
state.update({"stage":"PIPELINE_COMPLETE","verification_exit_code":code,
              "completed_at":datetime.now().astimezone().isoformat()})
(run/"pipeline.json").write_bytes((json.dumps(state,indent=2)+"\n").encode())
raise SystemExit(code)
