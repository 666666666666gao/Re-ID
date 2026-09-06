from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN=Path("/root/trifusion-storage/artifacts/rgbnt100_trifusion_source_oof_v1_seed42_20260906")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
registration=ROOT/"evidence/trifusion_rgbnt100_original_roles_comparison_registration_20260906.json"
reg=json.loads(registration.read_text())
config=ROOT/reg["config"]
assert sha(config)==reg["config_sha256"]
m0=Path(reg["m0_summary"])
assert sha(m0)==reg["m0_summary_sha256"]
assert json.loads(m0.read_text())["status"]=="PASS_ENGINEERING_ONLY"
assert (RUN/"m0_exit.txt").read_text().strip()=="0"
for key in ("m0_executor_closure","m0_file_verification","m0_scalar_verification"):
 assert sha(ROOT/reg[key])==reg[key+"_sha256"]
assert json.loads((ROOT/reg["m0_executor_closure"]).read_text())["status"]=="PASS_COMPLETE_M0_EXECUTOR_CLOSURE"
scalar=json.loads((ROOT/reg["m0_scalar_verification"]).read_text())
assert scalar["engineering_gate_passed"] and scalar["exact_fp32_loss_recompositions"]==124
assert not (RUN/"comparison").exists() and not (RUN/"comparison_launch.json").exists()
assert shutil.disk_usage(RUN).free>=8*1024**3
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,memory.total","--format=csv,noheader,nounits"],text=True).strip()
row=[int(x.strip()) for x in gpu.split(",")]
assert len(row)==4 and row[1]<500 and row[2]>=22000,gpu
command=[PY,"-u",str(ROOT/"tools/train_rgbnt100_trifusion_oof.py"),"--config",str(config),
"--mode","comparison","--output-dir",str(RUN/"comparison"),"--m0-receipt",str(m0)]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"wrapper_pid":os.getpid(),"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
"command":command,"config_sha256":sha(config),"registration_sha256":sha(registration),"wrapper_sha256":sha(Path(__file__)),
"m0_summary_sha256":sha(m0),"m0_executor_closure_sha256":reg["m0_executor_closure_sha256"],
"m0_scalar_verification_sha256":reg["m0_scalar_verification_sha256"],
"source_and_verifier_sha256":reg["source_and_verifier_sha256"],
"initialization_state_sha256_by_fold":reg["initialization_state_sha256_by_fold"],
"gpu_prelaunch":gpu,"output_volume_free_bytes":shutil.disk_usage(RUN).free,
"epochs_per_fold":20,"seed":42,"expected_optimizer_steps_approximate":reg["expected_optimizer_steps_approximate"],
"query_output_records":43375,"query_identities":50,"estimated_minutes":reg["estimated_minutes"],
"first_observation_seconds":reg["first_observation_seconds"],
"baseline_training_reexecuted":False,"m0_trained_weights_loaded":False,"official_test_access":0,"rgbnt201_dev_access":0}
(RUN/"comparison_launch.json").write_text(json.dumps(launch,indent=2)+"\n")
started=time.perf_counter()
with (RUN/"comparison.log").open("w") as log:
 code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(RUN/"comparison_exit.txt").write_text(str(code)+"\n")
(RUN/"comparison_terminal.json").write_text(json.dumps({**launch,
"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"exit_code":code,"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
sys.exit(code)
