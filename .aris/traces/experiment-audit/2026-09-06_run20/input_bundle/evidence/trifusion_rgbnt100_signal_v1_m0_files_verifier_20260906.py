from pathlib import Path
import datetime,hashlib,json,subprocess,time
import torch
ROOT=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
RUN=Path("/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_seed42_1157f0d")
def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):d.update(chunk)
    return d.hexdigest()
started=time.perf_counter()
assert not (RUN/"m0_files_verification.json").exists()
assert (RUN/"engineering_exit.txt").read_text().strip()=="0"
assert not Path("/proc/79272").exists()
config=json.loads((ROOT/"configs/RGBNT100/Signal-source-oof-v1.json").read_text())
protocol=json.loads((ROOT/config["protocol"]).read_text())
summary=json.loads((RUN/"m0/summary.json").read_text())
assert summary["status"]=="PASS_ENGINEERING_ONLY" and summary["optimizer_steps"]==24
assert summary["config_sha256"]==sha(ROOT/"configs/RGBNT100/Signal-source-oof-v1.json")
assert summary["protocol_sha256"]==sha(ROOT/config["protocol"])
assert summary["protocol_receipt_sha256"]==sha(RUN/"t0.json")
for path,digest in config["project_source_file_sha256"].items():assert sha(ROOT/path)==digest,path
for path,digest in config["signal_source_file_sha256"].items():assert sha(Path(config["signal_source"])/path)==digest,path
files={p.relative_to(RUN).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(RUN.rglob("*")) if p.is_file()}
checks=[]
for actual,fold in zip(summary["folds"],protocol["folds"],strict=True):
    directory=RUN/"m0"/("fold_"+str(fold["fold"]))
    assert actual==json.loads((directory/"receipt.json").read_text())
    assert actual["training"]==json.loads((directory/"training.json").read_text())
    cp=Path(actual["checkpoint"])
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
report={"verified_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
"status":"PASS_COMPLETE_M0_FILES_AND_CHECKPOINT_CONTENTS","files":files,"folds":checks,
"summary_sha256":sha(RUN/"m0/summary.json"),"verifier_sha256":sha(__file__),
"project_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
"source_bindings":{"project":len(config["project_source_file_sha256"]),"signal":len(config["signal_source_file_sha256"])},
"model_forwards":0,"image_decodes":0,"optimizer_steps":0,"elapsed_seconds":time.perf_counter()-started}
(RUN/"m0_files_verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="files"}))
