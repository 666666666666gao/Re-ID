from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert head=="ed9c300ae04718e2a6f5113b0617394dc9ddcd28"
reg=json.loads((ROOT/"evidence/trifusion_rgbnt100_signal_v1_r2_storage_retry_registration_20260906.json").read_text())
CFG=ROOT/reg["config_path"];T0=Path(reg["t0_receipt"])
assert sha(CFG)==reg["config_sha256"]
assert sha(ROOT/"tools/train_rgbnt100_signal_oof.py")==reg["runner_sha256"]
assert sha(ROOT/"tools/signal_gram_stable.py")==reg["helper_sha256"]
assert sha(T0)==reg["t0_sha256"]
assert OUT.resolve()==OUT and OUT.stat().st_dev==Path("/root").stat().st_dev
space=os.statvfs(OUT);free_bytes=space.f_bavail*space.f_frsize
assert free_bytes>=reg["storage_prelaunch_minimum_free_bytes"],free_bytes
gpu_free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert gpu_free>=22000,gpu_free
command=[PY,"-u",str(ROOT/"tools/train_rgbnt100_signal_oof.py"),"--config",str(CFG),"--mode","preflight",
"--protocol-receipt",str(T0),"--output-dir",str(OUT/"m0")]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"project_commit":head,"command":command,"config_sha256":sha(CFG),
"runner_sha256":reg["runner_sha256"],"t0_receipt_sha256":sha(T0),"t0_reexecuted":False,
"output_free_bytes":free_bytes,"gpu_free_mib":gpu_free,
"scope":"sameR2 fresh3fold complete-first-epoch M0 after verified storage relocation;0heldout;no formal baseline",
"first_observation":"about4minutes"}
(OUT/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
started=time.perf_counter()
with (OUT/"m0.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"m0_exit.txt").write_text(str(code)+"\n")
(OUT/"terminal.json").write_text(json.dumps({**launch,
"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
sys.exit(code)
