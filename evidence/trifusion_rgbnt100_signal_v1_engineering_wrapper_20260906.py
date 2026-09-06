from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_seed42_1157f0d")
PY="/root/miniconda3/envs/tri_reid/bin/python"
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="1157f0de6063bbf51855279ceb8b1822ef8c98d1"
assert sha(ROOT/"configs/RGBNT100/Signal-source-oof-v1.json")=="7270e2bf95c5f5a60e1dc6d6b047f043dce667d508783b36bc4734aecbc4c15b"
assert sha(ROOT/"tools/train_rgbnt100_signal_oof.py")=="514a2c86634b61ef8b6de32b6ff1990cdf18e208a91673fb192534b52e5a1357"
assert sha(ROOT/"tools/verify_rgbnt100_signal_protocol.py")=="4555014620f3adadf1fcdfe99700d843e2db3229ce0b8c5dc3bb2610aef1ee6f"
config=ROOT/"configs/RGBNT100/Signal-source-oof-v1.json"
start=time.perf_counter()
t0cmd=[PY,"-u",str(ROOT/"tools/verify_rgbnt100_signal_protocol.py"),"--config",str(config),"--output",str(OUT/"t0.json")]
dump(OUT/"t0_launch.json",{"launched_at":now(),"command":t0cmd,"wrapper_pid":os.getpid(),"project_commit":"1157f0de6063bbf51855279ceb8b1822ef8c98d1"})
with (OUT/"t0.log").open("w") as log:
    code=subprocess.run(t0cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"t0_exit.txt").write_text(str(code)+"\n")
if code!=0:
    (OUT/"engineering_exit.txt").write_text(str(code)+"\n")
    sys.exit(code)
t0=json.loads((OUT/"t0.json").read_text())
assert t0["status"]=="PASS_FULL_TRAIN_PROTOCOL_AND_MONTAGE"
free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert free>=22000,free
cmd=[PY,"-u",str(ROOT/"tools/train_rgbnt100_signal_oof.py"),"--config",str(config),
"--output-dir",str(OUT/"m0"),"--mode","preflight","--protocol-receipt",str(OUT/"t0.json")]
dump(OUT/"m0_launch.json",{"launched_at":now(),"command":cmd,"wrapper_pid":os.getpid(),"gpu_free_mib":free,
"project_commit":"1157f0de6063bbf51855279ceb8b1822ef8c98d1","t0_receipt_sha256":sha(OUT/"t0.json"),
"scope":"three fresh fold capacities8updates each;24updates total;48clean source strict reload forwards;0heldout model forwards"})
with (OUT/"m0.log").open("w") as log:
    code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"m0_exit.txt").write_text(str(code)+"\n")
(OUT/"engineering_exit.txt").write_text(str(code)+"\n")
dump(OUT/"engineering_terminal.json",{"completed_at":now(),"exit_code":code,"elapsed_seconds":time.perf_counter()-start})
sys.exit(code)
