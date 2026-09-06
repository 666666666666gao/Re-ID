from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN=Path("/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906")
PY="/root/miniconda3/envs/tri_reid/bin/python"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (RUN/"baseline_exit.txt").read_text().strip()=="0"
assert json.loads((RUN/"baseline_terminal.json").read_text())["exit_code"]==0
summary=RUN/"baseline/summary.json"
assert json.loads(summary.read_text())["status"]=="COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
assert not (RUN/"baseline_verification_launch.json").exists()
assert not (RUN/"baseline_terminal_file_verification.json").exists()
reg=json.loads((ROOT/"evidence/trifusion_rgbnt100_signal_v1_r2_baseline_registration_20260906.json").read_text())
verifier=ROOT/"tools/verify_rgbnt100_signal_terminal_files.py"
assert sha(verifier)==reg["source_and_verifier_sha256"]["tools/verify_rgbnt100_signal_terminal_files.py"]
command=[PY,"-u",str(verifier),"--run-root",str(RUN),"--project-root",str(ROOT),"--protocol-receipt",reg["t0_receipt"]]
launch={"launched_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "wrapper_pid":os.getpid(),"verification_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "command":command,"verifier_sha256":sha(verifier),"baseline_summary_sha256":sha(summary),
        "model_forwards":0,"optimizer_updates":0,"scope":"All3stored checkpoint and feature-distance-ranking files; actualauthor30epochLR CPU scalar replay",
        "estimated_minutes":[1,3],"first_observation_seconds":180}
(RUN/"baseline_verification_launch.json").write_text(json.dumps(launch,indent=2)+"\n")
started=time.perf_counter()
with (RUN/"baseline_verification.log").open("w") as log:
    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
(RUN/"baseline_verification_exit.txt").write_text(str(code)+"\n")
(RUN/"baseline_verification_terminal.json").write_text(json.dumps({**launch,
    "completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    "exit_code":code,"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
sys.exit(code)
