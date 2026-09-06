from pathlib import Path
import datetime,hashlib,json,subprocess,time
import torch
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN=Path('/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906')
T0=Path('/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_r2_seed42_e699eac/t0.json')
def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):d.update(chunk)
    return d.hexdigest()
started=time.perf_counter()
assert not (RUN/"m0_files_verification.json").exists()
assert (RUN/"m0_exit.txt").read_text().strip()=="0"
assert json.loads((RUN/"terminal.json").read_text())["exit_code"]==0
config=json.loads((ROOT/"configs/RGBNT100/Signal-source-oof-v1-r2.json").read_text())
protocol=json.loads((ROOT/config["protocol"]).read_text())
summary=json.loads((RUN/"m0/summary.json").read_text())
assert summary["status"]=="PASS_ENGINEERING_ONLY" and summary["optimizer_steps"]>24
assert summary["config_sha256"]==sha(ROOT/"configs/RGBNT100/Signal-source-oof-v1-r2.json")
assert summary["protocol_sha256"]==sha(ROOT/config["protocol"])
assert summary["protocol_receipt_sha256"]==sha(T0)
assert summary["runner_sha256"]==sha(ROOT/"tools/train_rgbnt100_signal_oof.py")
assert summary["engineering_revision"]==2 and summary["preflight_contract"]=="one complete source epoch per fold"
t0=json.loads(T0.read_text())
assert t0["runner_sha256"]==summary["runner_sha256"] and t0["config_sha256"]==summary["config_sha256"]
assert sha(T0)=="e54c826d1cfba2eca6526d4e4bdecb6758a19ff3b0ada4a498adfb3f11883ba5"
r1=json.loads(Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_seed42_1157f0d/m0/summary.json").read_text())
assert len(summary["folds"])==len(protocol["folds"])==3
for path,digest in config["project_source_file_sha256"].items():assert sha(ROOT/path)==digest,path
for path,digest in config["signal_source_file_sha256"].items():assert sha(Path(config["signal_source"])/path)==digest,path
files={p.relative_to(RUN).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(RUN.rglob("*")) if p.is_file()}
checks=[]
for actual,fold in zip(summary["folds"],protocol["folds"],strict=True):
    directory=RUN/"m0"/("fold_"+str(fold["fold"]))
    assert actual==json.loads((directory/"receipt.json").read_text())
    assert actual["training"]==json.loads((directory/"training.json").read_text())
    training=actual["training"]
    assert training["steps"]==[json.loads(line) for line in (directory/"steps.jsonl").read_text().splitlines()]
    assert training["epochs"]==len(training["history"])==1
    assert training["optimizer_steps"]==len(training["steps"])==training["history"][0]["optimizer_steps"]>8
    assert training["initial_state_sha256"]==r1["folds"][fold["fold"]]["training"]["initial_state_sha256"]
    assert training["initial_state_sha256"]!=training["final_state_sha256"]
    assert training["trainable_tensors"]==training["gradient_tensors"]==195 and not training["trainable_without_gradient"]
    assert training["overflow_events"]==0 and all(row["optimizer_update_applied"] for row in training["steps"])
    assert training["frozen_token_selection_initial_sha256"]==training["frozen_token_selection_final_sha256"]
    assert actual["strict_reload_exact_feature_parity"] and actual["feature_width"]==3072
    assert actual["clean_source_feature_forwards"]==16 and actual["heldout_image_forwards"]==0
    assert training["history"][0]["learning_rates"]==r1["folds"][fold["fold"]]["training"]["history"][0]["learning_rates"]
    cp=Path(actual["checkpoint"])
    assert cp==directory/"signal_m0.pth"
    assert sha(cp)==actual["checkpoint_sha256"]
    payload=torch.load(cp,map_location="cpu",weights_only=True)
    assert payload["fold"]==fold["fold"]
    assert payload["source_ids"]==fold["source_ids"] and payload["heldout_ids"]==fold["heldout_ids"]
    assert payload["config_sha256"]==summary["config_sha256"] and payload["protocol_sha256"]==summary["protocol_sha256"]
    digest=hashlib.sha256()
    for name,tensor in sorted(payload["model_state_dict"].items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    assert digest.hexdigest()==actual["training"]["final_state_sha256"]
    checks.append({"fold":fold["fold"],"checkpoint_sha256":sha(cp),"checkpoint_content_state_sha256":digest.hexdigest(),
                   "fold_source_heldout_binding":True,"receipt_training_summary_equal":True})
    del payload
assert summary["optimizer_steps"]==sum(row["training"]["optimizer_steps"] for row in summary["folds"])
log_epochs=[json.loads(line) for line in (RUN/"m0.log").read_text().splitlines() if line.startswith('{"event": "signal_source_epoch"')]
assert [{k:v for k,v in row.items() if k!="event"} for row in log_epochs]==[row["training"]["history"][0] for row in summary["folds"]]
report={"verified_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"status":"PASS_COMPLETE_M0_FILES_AND_CHECKPOINT_CONTENTS","files":files,"folds":checks,
"summary_sha256":sha(RUN/"m0/summary.json"),"t0_receipt_sha256":sha(T0),
"successful_training_steps_verified":summary["optimizer_steps"],"saved_per_step_logs_equal_training":True,"verifier_sha256":sha(__file__),
"project_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
"source_bindings":{"project":len(config["project_source_file_sha256"]),"signal":len(config["signal_source_file_sha256"])},
"model_forwards":0,"image_decodes":0,"optimizer_steps":0,"elapsed_seconds":time.perf_counter()-started}
(RUN/"m0_files_verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="files"}))
