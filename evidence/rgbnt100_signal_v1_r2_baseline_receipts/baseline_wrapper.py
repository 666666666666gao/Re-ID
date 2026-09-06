from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert head=="def7b9b7ecd9e7e37821716a13fdb2580b2d955c"
regpath=ROOT/"evidence/trifusion_rgbnt100_signal_v1_r2_baseline_registration_20260906.json"
reg=json.loads(regpath.read_text())
CFG=ROOT/reg["config"];T0=Path(reg["t0_receipt"]);M0=OUT/"m0/summary.json"
assert sha(CFG)==reg["config_sha256"] and sha(T0)==reg["t0_receipt_sha256"]
assert sha(M0)==reg["m0_receipt_sha256"]
assert sha(OUT/"m0_files_verification.json")==reg["m0_file_verification_sha256"]
assert sha(ROOT/"evidence/trifusion_rgbnt100_signal_v1_r2_m0_scalar_verification_20260906.json")==reg["m0_scalar_verification_sha256"]
for path,expected in reg["source_and_verifier_sha256"].items():assert sha(ROOT/path)==expected,path
assert json.loads(M0.read_text())["status"]=="PASS_ENGINEERING_ONLY"
assert not (OUT/"baseline").exists() and not (OUT/"baseline_launch.json").exists()
assert OUT.stat().st_dev==Path("/root").stat().st_dev
space=os.statvfs(OUT);free_bytes=space.f_bavail*space.f_frsize
assert free_bytes>=reg["storage_required_free_bytes"],free_bytes
gpu_free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert gpu_free>=22000,gpu_free
command=[PY,"-u",str(ROOT/"tools/train_rgbnt100_signal_oof.py"),"--config",str(CFG),"--mode","train",
"--protocol-receipt",str(T0),"--preflight-receipt",str(M0),"--output-dir",str(OUT/"baseline")]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"project_commit":head,"command":command,"config_sha256":sha(CFG),
"runner_sha256":reg["source_and_verifier_sha256"]["tools/train_rgbnt100_signal_oof.py"],
"registration_sha256":sha(regpath),"m0_sha256":sha(M0),"t0_sha256":sha(T0),
"m0_weights_used":False,"m0_reexecuted":False,"t0_reexecuted":False,"output_free_bytes":free_bytes,"gpu_free_mib":gpu_free,
"scope":"R2 Signal3fresh sourcefolds seed42 fixed30epochs; full8675internal heldout only at eachfold uniqueendpoint;0official",
"first_observation_minutes":15,"estimated_total_minutes":[70,90]}
(OUT/"baseline_launch.json").write_text(json.dumps(launch,indent=2)+"\n")
started=time.perf_counter()
with (OUT/"baseline.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"baseline_exit.txt").write_text(str(code)+"\n")
(OUT/"baseline_terminal.json").write_text(json.dumps({**launch,
"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
sys.exit(code)
