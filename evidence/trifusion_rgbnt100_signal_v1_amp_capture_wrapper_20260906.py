from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_amp_capture_v1_seed42_8b412d0")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="8b412d072dedd1365419b7ff256aadae7202d436"
plan=json.loads((ROOT/"evidence/trifusion_rgbnt100_signal_v1_amp_capture_plan_20260906.json").read_text())
assert sha(ROOT/plan["diagnostic"])==plan["diagnostic_sha256"]
assert sha(ROOT/plan["runner"])==plan["runner_sha256"]
assert sha(ROOT/plan["config"])==plan["config_sha256"]
free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert free>=22000,free
assert not (OUT/"capture").exists()
command=[PY,"-u",str(ROOT/plan["diagnostic"]),"--config",str(ROOT/plan["config"]),
"--output-dir",str(OUT/"capture"),"--m0-receipt",
"/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_seed42_1157f0d/m0/summary.json"]
(OUT/"launch.json").write_text(json.dumps({"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"command":command,"gpu_free_mib":free,
"project_commit":"8b412d072dedd1365419b7ff256aadae7202d436","diagnostic_sha256":plan["diagnostic_sha256"],
"scope":plan["scope"],"expected_exit_on_reproduced_assertion":1},indent=2)+"\n")
start=time.perf_counter()
with (OUT/"capture.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(OUT/"exit.txt").write_text(str(code)+"\n")
(OUT/"terminal.json").write_text(json.dumps({"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n")
sys.exit(code)
