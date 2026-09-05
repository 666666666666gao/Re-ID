from pathlib import Path
import json,hashlib,math,sys,time
import numpy as np
from datetime import datetime
sys.stdout.reconfigure(encoding="utf-8")
r=Path.cwd();tic=time.perf_counter()
raw=(r/"evidence/trifusion_v23_m0_seed42_9f4a10b.json").read_bytes();a=json.loads(raw)
m=a["m0"];o=m["overfit"]
assert a["status"]=="RUNNING" and not a["folds"] and m["passed"]
assert a["repository_commit"]=="9f4a10b6162b9658ba103cd92466411ebb6ccd8f"
assert a["dev_access_count"]==a["official_test_access_count"]==0
pairs=[]
for f in a["preflight"]:
    c,t=f["endpoints"]
    for k in ["initial_state_sha256","batch_receipts","all_output_sha256","legacy_zero_adapter_first_batch_exact"]:
        assert c[k]==t[k],(f["fold"],k)
    assert len(c["batch_receipts"])==len(c["all_output_sha256"])==8
    for k in ["source","source_model_state_sha256","fit_identity_ids","heldout_identity_ids","new_inference_parameters","total_parameters"]:
        assert c["binding"][k]==t["binding"][k]
    assert c["state_unchanged"] and t["state_unchanged"] and c["legacy_zero_adapter_first_batch_exact"]
    cb,tb=c["binding"],t["binding"]
    assert (cb["trainable_parameters"],tb["trainable_parameters"],cb["trainable_tensors"],tb["trainable_tensors"])==(7841292,9618828,203,239)
    assert cb["total_parameters"]==tb["total_parameters"]==100577677
    assert tb["trainable_parameters"]-cb["trainable_parameters"]==9*(768*128+128+128*768+768)==1777536
    assert len(cb["fit_identity_ids"])==94 and len(cb["heldout_identity_ids"])==47
    assert set(cb["fit_identity_ids"]).isdisjoint(cb["heldout_identity_ids"])
    pairs.append({"fold":f["fold"],"initial_state_sha256":c["initial_state_sha256"],"source_model_state_sha256":cb["source_model_state_sha256"],"eight_batch_pairing_exact":True,"legacy_zero_output_exact":True,"source_heldout_disjoint":True,"control_trainable":7841292,"candidate_trainable":9618828,"control_tensors":203,"candidate_tensors":239})
component_errors=[];rows=[]
for label,c in [("control_capacity",m["capacities"][0]),("candidate_capacity",m["capacities"][1]),("candidate_fixed100",o)]:
    assert c["steps"]==len(c["losses"])==len(c["components"])
    assert c["overflow_events"]==0 and c["frozen_state_unchanged"] and not c["missing_nonzero_gradients"]
    assert c["trainable_tensors"]==c["nonzero_gradient_tensors"]==(203 if label=="control_capacity" else 239)
    arr=np.asarray([[row[k] for k in row] for row in c["components"]])
    assert np.isfinite(arr).all() and (arr>=0).all()
    for loss,row in zip(c["losses"],c["components"],strict=True):
        assert loss==row["total"]
        ordinary=.25*sum(row["triplet_residual_"+e] for e in ["cnn","transformer","mamba"])
        common=.25*row["id_fused"]+row["triplet_fused"]
        for e in ["cnn","transformer","mamba"]:
            common+=row["id_"+e]/12+.25*row["triplet_"+e]+row["id_residual_"+e]/12
        component_errors.extend([abs(ordinary-row["ordinary_residual_triplet"]),abs(common-row["common_identity_and_branch_triplet"]),abs(common+ordinary-row["total"])])
    rows.append({"stage":label,"steps":c["steps"],"first_loss":c["losses"][0],"last_loss":c["losses"][-1],
    "all_component_ranges":{key:{"min":min(row[key] for row in c["components"]),"max":max(row[key] for row in c["components"])} for key in c["components"][0]},
    "first_components":c["components"][0],"last_components":c["components"][-1],"all_gradients_live":True,"frozen_unchanged":True})
assert sum(row["steps"] for row in rows)==116
correct=1-.1+.1/94;other=.1/94
floor=.75*(-correct*math.log(correct)-93*other*math.log(other))
ratio=(o["losses"][99]-floor)/(o["losses"][0]-floor)
assert floor==o["combined_loss_floor"] and ratio==o["excess_loss_ratio"] and ratio<=.1
assert max(component_errors)<1e-6
log=(r/"evidence/trifusion_v23_m0_run_snapshot_20260905.log").read_text(encoding="utf-8")
parsed=[json.loads(line) for line in log.splitlines() if line.startswith("{")]
logged=[row for row in parsed if row.get("stage")=="M0"]
assert len(logged)==1 and {k:logged[0][k] for k in m}==m
epoch=[row for row in parsed if "epoch" in row and "endpoint" in row]
assert len(epoch)==2 and [row["epoch"] for row in epoch]==[1,2]
report={"verified_at":datetime.now().astimezone().isoformat(),"python":sys.version,"numpy":np.__version__,
"scope":"All six M0 paired preflights, all 116 update component rows and fixed first/100th gate; JSON/NumPy only",
"m0_passed":True,"input_summary_sha256":hashlib.sha256(raw).hexdigest(),"pairing":pairs,"stages":rows,
"loss_component_max_roundoff":max(component_errors),"floor":floor,"fixed100_excess_ratio":ratio,
"log_m0_object_matches":True,"captured_nonterminal_q1_epoch_rows":2,"captured_nonterminal_q1_optimizer_steps":58,
"terminal_q1_retrieval_available":False,"new_optimizer_steps":0,"new_retrieval_evaluations":0,
"dev_access_count":0,"official_test_access_count":0,"runtime_seconds":time.perf_counter()-tic}
out=r/"evidence/trifusion_v23_m0_array_verification_20260905.json";assert not out.exists()
out.write_bytes((json.dumps(report,indent=2)+"\n").encode())
print(json.dumps({k:report[k] for k in ["m0_passed","floor","fixed100_excess_ratio","loss_component_max_roundoff","runtime_seconds"]}))
