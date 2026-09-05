import datetime,hashlib,json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
p=Path.cwd()
summary_path=p/"evidence/trifusion_v22_q1_seed42_5ae096b.json"
d=json.loads(summary_path.read_bytes())
log_path=p/"evidence/trifusion_v22_complete_run_20260905.log"
events=[json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith('{"')]
epochs=[e for e in events if "epoch" in e and "endpoint" in e]
histories=[row for fold in d["folds"] for e in ("batch_hard_residual","camera_negative_residual") for row in fold["endpoints"][e]["training"]["history"]]
assert epochs==histories and len(epochs)==120
assert sum(row["batches"] for row in epochs)==3360
assert d["m0"]==json.loads((p/"evidence/trifusion_v22_m0_seed42_5ae096b.json").read_bytes())["m0"]
assert all(not v for v in d["scientific_checks"].values()) and not d["next_phase_qualified"]
terminal_files=json.loads((p/"evidence/trifusion_v22_terminal_file_verification_20260905.json").read_bytes())
assert terminal_files["verified"] and len(terminal_files["files"])==36
assert len(terminal_files["endpoint_receipts"])==6
per_endpoint=[]
support_total={e:{} for e in ("batch_hard_residual","camera_negative_residual")}
for f in d["folds"]:
    for endpoint,r in f["endpoints"].items():
        tr=r["training"]
        first,last=tr["history"][0],tr["history"][-1]
        keys=("mean_total","mean_common_identity_and_branch_triplet","mean_ordinary_residual_triplet","mean_camera_residual_metric")
        row={"fold":f["fold"],"endpoint":endpoint,"optimizer_steps":tr["optimizer_steps"],
             "first_epoch":{k:first[k] for k in keys},"final_epoch":{k:last[k] for k in keys},
             "final_mcnl_diagnostics":{k:v for k,v in last.items() if k.startswith("mean_mcnl_")},
             "camera_support_sums":tr["camera_support_sums"]}
        per_endpoint.append(row)
        for k,v in tr["camera_support_sums"].items():
            support_total[endpoint][k]=support_total[endpoint].get(k,0)+v
assert support_total["batch_hard_residual"]==support_total["camera_negative_residual"]
assert support_total["batch_hard_residual"]=={"valid_rows":98796,"same_negative_missing_rows":7852,"other_negative_missing_rows":872,"cross_camera_positive_rows":17600}
out=p/"evidence/trifusion_v22_terminal_log_and_loss_verification_20260905.json"
assert not out.exists()
r={"verified_at":datetime.datetime.now().astimezone().isoformat(),
   "status":"ALL_120_EPOCH_ROWS_EQUAL_ENDPOINT_HISTORIES","summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
   "log_sha256":hashlib.sha256(log_path.read_bytes()).hexdigest(),"epoch_rows":120,"optimizer_steps":3360,
   "m0_unchanged_from_immutable_snapshot":True,"remote_whole_file_sha_count":36,
   "standalone_receipts":6,"per_endpoint":per_endpoint,"per_arm_camera_support_sums":support_total,
   "optimizer_updates_executed_by_this_verifier":0,"new_retrieval_evaluations":0,"new_model_or_tensor_loads":0}
out.write_bytes((json.dumps(r,indent=2)+"\n").encode())
print(json.dumps(r,indent=2))
