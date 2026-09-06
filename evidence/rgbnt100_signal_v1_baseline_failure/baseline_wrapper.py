from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_seed42_1157f0d")
PY="/root/miniconda3/envs/tri_reid/bin/python"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="60a3d0e99646ce1f48edfab698ac132b79146a9e"
plan=json.loads((ROOT/"evidence/trifusion_rgbnt100_signal_v1_baseline_launch_plan_20260906.json").read_text())
assert sha(ROOT/"configs/RGBNT100/Signal-source-oof-v1.json")==plan["config_sha256"]
assert sha(ROOT/"tools/train_rgbnt100_signal_oof.py")==plan["runner_sha256"]
assert sha(OUT/"t0.json")==plan["t0_receipt_sha256"]
assert sha(OUT/"m0/summary.json")==plan["m0_receipt_sha256"]
assert sha(OUT/"m0_files_verification.json")==plan["m0_files_verification_sha256"]
assert json.loads((OUT/"m0/summary.json").read_text())["status"]=="PASS_ENGINEERING_ONLY"
assert not (OUT/"baseline").exists()
free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert free>=22000,free
command=[PY,"-u",str(ROOT/"tools/train_rgbnt100_signal_oof.py"),"--config",str(ROOT/"configs/RGBNT100/Signal-source-oof-v1.json"),
"--output-dir",str(OUT/"baseline"),"--mode","train","--preflight-receipt",str(OUT/"m0/summary.json"),
"--protocol-receipt",str(OUT/"t0.json")]
receipt={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"command":command,"project_commit":"60a3d0e99646ce1f48edfab698ac132b79146a9e",
"gpu_free_mib":free,"runner_sha256":plan["runner_sha256"],"config_sha256":plan["config_sha256"],
"m0_receipt_sha256":plan["m0_receipt_sha256"],"t0_receipt_sha256":plan["t0_receipt_sha256"],
"scope":"fresh threefold fixed30epoch Signal source baseline, all8675query/gallery,0official, no M0 weight reuse"}
(OUT/"baseline_launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
start=time.perf_counter()
with (OUT/"baseline.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"baseline_exit.txt").write_text(str(code)+"\n")
(OUT/"baseline_terminal.json").write_text(json.dumps({"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n")
sys.exit(code)
