from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,yaml
repo=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
base=Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v23_spectral_adapter_seed42_9f4a10b")
snap=Path(str(base)+".m0_snapshot_2207.json")
raw=snap.read_bytes();summary=json.loads(raw)
assert summary["m0"]["passed"] and len(summary["preflight"])==3
assert not summary["folds"] and summary["repository_commit"]=="9f4a10b6162b9658ba103cd92466411ebb6ccd8f"
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024**2),b""):h.update(chunk)
    return h.hexdigest()
config_path=repo/"configs/RGBNT201/TriFusion-signal-preserving-v23-spectral-adapter-rtx3090.yml"
cfg=yaml.safe_load(config_path.read_text())
expected={repo/p:h for p,h in summary["source_file_sha256"].items()}
expected[repo/"tools/train_signal_preserving_v23.py"]=summary["runner_sha256"]
expected[config_path]=summary["config_sha256"]
expected[repo/"refine-logs/v23/EXPERIMENT_PLAN.md"]=summary["plan_sha256"]
expected[Path(cfg["SIGNAL"]["CLIP_WEIGHT"])]=cfg["SIGNAL"]["CLIP_WEIGHT_SHA256"]
expected[Path(cfg["INITIALIZATION"]["V12_RUN_SUMMARY"])]=cfg["INITIALIZATION"]["V12_RUN_SUMMARY_SHA256"]
for source in cfg["INITIALIZATION"]["V12_FOLDS"]:
    for kind in ["SIGNAL","EXPERT"]:expected[Path(source[kind+"_CHECKPOINT"])]=source[kind+"_CHECKPOINT_SHA256"]
rows=[]
for path,h in expected.items():
    actual=sha(path);assert actual==h,str(path)
    rows.append({"path":str(path),"bytes":path.stat().st_size,"sha256":actual,"expected_sha256":h,"matches":True})
prereg=repo/"evidence/trifusion_v23_preregistration_20260905.json"
assert prereg.read_bytes()==subprocess.check_output(["git","show",summary["repository_commit"]+":evidence/trifusion_v23_preregistration_20260905.json"],cwd=repo)
preregistered=json.loads(prereg.read_bytes())
for path,h in preregistered["paths_sha256"].items():
    assert hashlib.sha256(subprocess.check_output(["git","show",summary["repository_commit"]+":"+path],cwd=repo)).hexdigest()==h,path
signal=cfg["SIGNAL"]["SOURCE"]
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=signal,text=True).strip()==summary["signal_commit"]
assert hashlib.sha256(subprocess.check_output(["git","diff","--binary"],cwd=signal)).hexdigest()==summary["signal_diff_sha256"]
record={"observed_at":datetime.now().astimezone().isoformat(),"verified":True,"files":rows,
"execution_commit":summary["repository_commit"],"observed_repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
"m0_snapshot_sha256":sha(snap),"m0_snapshot_bytes":len(raw),"m0_snapshot_log_sha256":sha(Path(str(base)+".m0_log_2207.log")),
"preregistration_sha256":sha(prereg),"all_preregistered_execution_blobs_match":True,"preregistered_execution_blobs":preregistered["paths_sha256"],
"mutable_latest_tracker_excluded_from_current_prereg_hash_claim":True,"signal_source_and_diff_unchanged":True,
"original_pid":44684,"original_process_live":Path("/proc/44684/cmdline").exists(),"optimizer_steps":0,"retrieval_evaluation_runs":0,
"scope":"Whole-file SHA on existing source/checkpoint bytes and immutable M0 snapshot; no tensor/image/model operations",
"m0_snapshot_contains_q1_retrieval":False}
out=Path(str(base)+".m0_file_verification_2210.json");assert not out.exists()
out.write_bytes((json.dumps(record,indent=2)+"\n").encode())
print(json.dumps({"output":str(out),"verified_files":len(rows),"observed_at":record["observed_at"],"snapshot_sha256":record["m0_snapshot_sha256"]}))
