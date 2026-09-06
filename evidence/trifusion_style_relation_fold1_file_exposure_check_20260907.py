
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,numpy as np,collections
run=Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v27_style_relation_support_r2_seed42_d5bc048")
repo=Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
summary_bytes=(run/"style_relation_support.json").read_bytes()
summary=json.loads(summary_bytes)
fold=summary["folds"][1]
assert fold["fold"]==1 and fold["batches"]==560 and fold["model_forwards"]==3360
assert fold["all_model_states_unchanged"] and fold["all_gradients_absent"]
assert fold["first_eight_augmentation_receipts_equal_original_v27"]
assert fold["all_sample_orders_and_record_exposures_equal_registered"]
assert all(Path("/proc",str(pid)).exists() for pid in (134209,134211,134960))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
ap=Path(fold["similarities"]["path"]);rp=Path(fold["batch_receipts"]["path"])
assert ap.parent==rp.parent==run
assert ap.stat().st_size==fold["similarities"]["bytes"] and sha(ap)==fold["similarities"]["sha256"]
assert sha(rp)==fold["batch_receipts"]["sha256"]
array=np.load(ap,mmap_mode="r");assert array.shape==(560,3,2,18,64,64) and array.dtype==np.float32
shape=list(array.shape);values=int(array.size);del array
records=[json.loads(x) for x in rp.read_text().splitlines()];assert len(records)==560
config=json.loads((repo/"configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json").read_bytes())
replay=json.loads((repo/config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
expected=replay["folds"][1]["arms"]["control"]
exposure=collections.Counter()
for i,(row,wanted) in enumerate(zip(records,expected["batches"],strict=True)):
 assert row["case_index"]==i
 for k in ("epoch","step","sampler_indices","sample_order_sha256","directed_cross_camera_positive_pairs"):assert row[k]==wanted[k]
 exposure.update(row["sampler_indices"])
for identity in expected["identities"]:
 assert [exposure[i] for i in identity["record_indices"]]==identity["record_exposures"]
out={"observed_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),"status":"FOLD1_COMPLETE_FILE_AND_EXPOSURE_MILESTONE_NOT_DIAGNOSTIC_TERMINAL","run_dir":str(run),"source_summary_snapshot_sha256":hashlib.sha256(summary_bytes).hexdigest(),"fold":1,"source_records":fold["source_records"],"source_identities":len(expected["identities"]),"batches":560,"model_forwards":3360,"raw_sample_exposures":sum(exposure.values()),"actual_registered_order_and_exposure_replay_pass":True,"first_eight_original_augmentation_match_receipted":True,"all_three_fixed_model_states_unchanged_receipted":True,"all_parameter_gradients_absent_receipted":True,"maximum_decomposition_error":fold["maximum_decomposition_error"],"array":{"shape":shape,"values":values,**fold["similarities"],"current_full_sha_match":True,"downloaded":False},"batch_receipts":fold["batch_receipts"],"model_bindings":fold["bindings"],"current_total_persisted_batches":summary["completed_batches"],"current_total_model_forwards":summary["model_forwards"],"all_three_original_processes_still_alive":True,"full_cpu_relation_verification_complete":False,"scientific_conclusion":None,"new_model_forwards":0,"optimizer_updates":0,"image_reads":0,"torch_imports":0,"independent_audit":False}
print(json.dumps(out))
