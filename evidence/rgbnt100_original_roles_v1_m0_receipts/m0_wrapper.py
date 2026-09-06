from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN=Path("/root/trifusion-storage/artifacts/rgbnt100_trifusion_source_oof_v1_seed42_20260906")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
registration=ROOT/"evidence/trifusion_rgbnt100_original_roles_registration_20260906.json"
reg=json.loads(registration.read_text())
config=ROOT/reg["config"]
assert sha(config)==reg["config_sha256"]
assert not (RUN/"m0").exists() and not (RUN/"m0_launch.json").exists()
assert shutil.disk_usage(RUN).free>=8*1024**3
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,memory.total","--format=csv,noheader,nounits"],text=True).strip()
gpu_row=[int(x.strip()) for x in gpu.split(",")]
assert len(gpu_row)==4 and gpu_row[1]<500 and gpu_row[2]>=22000,gpu
command=[PY,"-u",str(ROOT/"tools/train_rgbnt100_trifusion_oof.py"),"--config",str(config),"--mode","m0","--output-dir",str(RUN/"m0")]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
"command":command,"config_sha256":sha(config),"registration_sha256":sha(registration),
"wrapper_sha256":sha(Path(__file__)),"source_and_verifier_sha256":reg["source_and_verifier_sha256"],
"gpu_prelaunch":gpu,"output_volume_free_bytes":shutil.disk_usage(RUN).free,
"fixed_m0":reg["fixed_m0"],"estimated_minutes":[3,8],"first_observation_seconds":300,
"baseline_training_reexecuted":False,"official_test_access":0,"rgbnt201_dev_access":0}
(RUN/"m0_launch.json").write_text(json.dumps(launch,indent=2)+"\n")
started=time.perf_counter()
with (RUN/"m0.log").open("w") as log:
 code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(RUN/"m0_exit.txt").write_text(str(code)+"\n")
(RUN/"m0_terminal.json").write_text(json.dumps({**launch,
"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
sys.exit(code)
