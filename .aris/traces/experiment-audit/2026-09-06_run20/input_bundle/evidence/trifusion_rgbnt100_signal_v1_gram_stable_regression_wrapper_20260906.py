from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_gram_stable_regression_v1_6c741b8")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="6c741b8b56b6a14f83c0e8a9452388aa23e5b301"
plan_path=ROOT/"evidence/trifusion_rgbnt100_signal_v1_gram_stable_regression_plan_20260906.json"
plan=json.loads(plan_path.read_text())
for name,digest in plan["project_files"].items(): assert sha(ROOT/name)==digest
free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert free>=22000,free
command=[PY,"-u",str(ROOT/"tools/verify_rgbnt100_signal_gram_stable.py"),"--plan",str(plan_path),
"--fixture",plan["fixture"],"--volume-inputs",plan["volume_inputs"],"--output-dir",str(OUT/"regression")]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"command":command,"gpu_free_mib":free,"project_commit":"6c741b8b56b6a14f83c0e8a9452388aa23e5b301",
"plan_sha256":sha(plan_path),"scope":"real Gram operator regression then full batch if operator passes;0updates"}
(OUT/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
start=time.perf_counter()
with (OUT/"regression.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"exit.txt").write_text(str(code)+"\n")
(OUT/"terminal.json").write_text(json.dumps({**launch,"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n")
sys.exit(code)
