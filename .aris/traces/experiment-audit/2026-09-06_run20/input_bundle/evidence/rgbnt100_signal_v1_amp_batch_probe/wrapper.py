from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_amp_batch_probe_v1_seed42_56f094f")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="56f094f82ad03c7140ed844f452d91346ad977cf"
plan_path=ROOT/"evidence/trifusion_rgbnt100_signal_v1_amp_batch_probe_plan_20260906.json"
plan=json.loads(plan_path.read_text())
assert sha(ROOT/plan["probe"])==plan["probe_sha256"]
assert sha(ROOT/plan["config"])==plan["config_sha256"]
assert sha(Path(plan["fixture"]))==plan["fixture_sha256"]
start=time.perf_counter()
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"project_commit":"56f094f82ad03c7140ed844f452d91346ad977cf",
"plan_sha256":sha(plan_path),"scope":plan["scope"],"modes":plan["fixed_modes"]}
(OUT/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
rows=[]
for mode in plan["fixed_modes"]:
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>=22000,free
    command=[PY,"-u",str(ROOT/plan["probe"]),"--plan",str(plan_path),"--fixture",plan["fixture"],
             "--output-dir",str(OUT/mode),"--mode",mode]
    t=time.perf_counter()
    with (OUT/(mode+".log")).open("w") as log:
        code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    rows.append({"mode":mode,"command":command,"exit_code":code,"gpu_free_mib_before":free,
                 "elapsed_seconds":time.perf_counter()-t})
    (OUT/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
(OUT/"terminal.json").write_text(json.dumps({**launch,
"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"status":"FIXED_THREE_PROBES_FINISHED_INSPECT_ACTUAL_RESULTS","probes":rows,
"elapsed_seconds":time.perf_counter()-start,"optimizer_steps":0},indent=2)+"\n")
