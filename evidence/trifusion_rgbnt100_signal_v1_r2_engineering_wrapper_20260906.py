from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
OUT=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_r2_seed42_e699eac")
PY="/root/miniconda3/envs/tri_reid/bin/python"
CFG=ROOT/"configs/RGBNT100/Signal-source-oof-v1-r2.json"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()=="e699eaca888061145a402e6b91fe90ebf96a7989"
registration=json.loads((ROOT/"evidence/trifusion_rgbnt100_signal_v1_r2_registration_20260906.json").read_text())
assert sha(CFG)==registration["config_sha256"]
assert sha(ROOT/"tools/train_rgbnt100_signal_oof.py")==registration["runner_sha256"]
assert sha(ROOT/"tools/signal_gram_stable.py")==registration["helper_sha256"]
free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
assert free>=22000,free
commands=[
("t0",[PY,"-u",str(ROOT/"tools/verify_rgbnt100_signal_protocol.py"),"--config",str(CFG),"--output",str(OUT/"t0.json")]),
("m0",[PY,"-u",str(ROOT/"tools/train_rgbnt100_signal_oof.py"),"--config",str(CFG),"--mode","preflight",
       "--protocol-receipt",str(OUT/"t0.json"),"--output-dir",str(OUT/"m0")])]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"commands":commands,"gpu_free_mib":free,"project_commit":"e699eaca888061145a402e6b91fe90ebf96a7989",
"config_sha256":sha(CFG),"runner_sha256":registration["runner_sha256"],
"scope":"R2 fullT0 then three fresh complete-first-epoch source M0;0heldout; no baseline restart","first_observation":"about4minutes"}
(OUT/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
start=time.perf_counter()
rows=[]
for name,command in commands:
    t=time.perf_counter()
    with (OUT/(name+".log")).open("w") as log:
        code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    rows.append({"stage":name,"exit_code":code,"elapsed_seconds":time.perf_counter()-t})
    (OUT/(name+"_exit.txt")).write_text(str(code)+"\n")
    (OUT/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
    if code!=0: break
(OUT/"terminal.json").write_text(json.dumps({**launch,"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"stages":rows,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n")
sys.exit(code)
