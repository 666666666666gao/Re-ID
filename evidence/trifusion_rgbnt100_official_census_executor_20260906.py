from pathlib import Path
from collections import Counter
from datetime import datetime
import json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
op=ROOT/"evidence/trifusion_rgbnt100_official_observation_20260906_205449.json"
o=json.loads(op.read_text());s=o["summary"];v=o["verification"]
base=ROOT/"evidence/rgbnt100_official_main_receipts"
sp=base/"official/summary.json";vp=base/"official_verification.json";cp=base/"official/query_error_census.json"
assert o["terminal"]["exit_code"]==o["terminal"]["evaluation_exit_code"]==o["terminal"]["verification_exit_code"]==0
assert json.loads(sp.read_text())==s and json.loads(vp.read_text())==v
assert sha(sp)==v["summary_sha256"] and sha(cp)==v["error_census_sha256"]
assert v["status"]=="PASS_COMPLETE_OFFICIAL_FILES_ARRAYS_RANKINGS_AND_SCORES"
assert v["distance_entries_verified"]==v["complete_ranking_positions_verified"]==73530625 and v["query_output_scores_verified"]==8575
assert v["signal_features_and_distances_bitwise_equal"] and v["all5_saved_distance_matrices_bitwise_recomputed"]
assert s["signal_official_record_forwards"]==s["roles_official_record_forwards"]==10290 and s["training_updates"]==0
for p,x in o["files"].items():assert (base/p).stat().st_size==x["bytes"],p
c=json.loads(cp.read_text());assert c["summary_sha256"]==sha(sp)
assert len(c["queries"])==1715 and len(c["per_identity"])==50
protocol=json.loads((ROOT/"protocols/rgbnt100_official_main_v1.json").read_text())
assert sha(ROOT/"protocols/rgbnt100_official_main_v1.json")==s["protocol_sha256"]
for i,(q,r) in enumerate(zip(c["queries"],protocol["records"]["query"],strict=True)):
 assert q["query_index"]==i and (q["identity"],q["camera"],q["path"])==(r["identity"],r["camera"],r["path"])
summary={}
for name in ("baseline_only","fused","cnn","transformer","mamba"):
 qs=c["queries"];rows=[q["outputs"][name] for q in qs]
 errors=[q for q in qs if q["outputs"][name]["first_match_rank"]>1]
 ids=[{"identity":x["identity"],"query_count":x["query_count"],"delta_mAP":x["map_by_output"][name]-x["map_by_output"]["baseline_only"],"weighted_contribution_pp":x["query_count"]*(x["map_by_output"][name]-x["map_by_output"]["baseline_only"])/1715} for x in c["per_identity"]]
 item={"rank1_errors":len(errors),"same_camera_rank1_errors":sum(q["outputs"][name]["top_same_camera"] for q in errors),
 "identity_changes":{"improved":sum(x["delta_mAP"]>0 for x in ids),"declined":sum(x["delta_mAP"]<0 for x in ids),"unchanged":sum(x["delta_mAP"]==0 for x in ids)},
 "mean_nearest_positive_distance":statistics.fmean(x["nearest_positive_distance"] for x in rows),"mean_nearest_negative_distance":statistics.fmean(x["nearest_negative_distance"] for x in rows),
 "mean_negative_minus_positive_margin":statistics.fmean(x["negative_minus_positive_distance"] for x in rows),"all_identity_deltas":ids}
 if name!="baseline_only":
  new=[q for q in qs if q["outputs"][name]["rank1_new_error"]];repaired=[q for q in qs if q["outputs"][name]["rank1_repaired"]]
  item["new_errors"]=[{"query_index":q["query_index"],"identity":q["identity"],"camera":q["camera"],"top_identity":q["outputs"][name]["top_identity"],"top_camera":q["outputs"][name]["top_camera"],"ap_delta_pp":q["outputs"][name]["ap_delta_over_signal_pp"]} for q in new]
  item["new_error_same_camera_count"]=sum(q["outputs"][name]["top_same_camera"] for q in new)
  item["new_error_count"]=len(new);item["repaired_count"]=len(repaired)
  item["new_rank1_errors_with_ap_increase"]=sum(q["outputs"][name]["ap_delta_over_signal_pp"]>0 for q in new)
  item["rank1_repairs_with_ap_decline"]=sum(q["outputs"][name]["ap_delta_over_signal_pp"]<0 for q in repaired)
 summary[name]=item
fm=[]
for q in c["queries"]:
 a=q["outputs"]["fused"];b=q["outputs"]["mamba"]
 fm.append({"query_index":q["query_index"],"identity":q["identity"],"delta_ap_pp":100*(a["average_precision"]-b["average_precision"]),"mamba_error_fused_fixed":b["first_match_rank"]>1 and a["first_match_rank"]==1,"mamba_correct_fused_error":b["first_match_rank"]==1 and a["first_match_rank"]>1})
fused=s["comparison"]["metrics"]["fused"];signal=s["comparison"]["metrics"]["baseline_only"];mamba=s["comparison"]["metrics"]["mamba"]
fmi=[]
for pid in sorted({q["identity"] for q in fm}):
 rows=[q for q in fm if q["identity"]==pid]
 fmi.append({"identity":pid,"query_count":len(rows),"fused_minus_mamba_mAP_pp":statistics.fmean(q["delta_ap_pp"] for q in rows),"rank1_repairs":sum(q["mamba_error_fused_fixed"] for q in rows),"rank1_new_errors":sum(q["mamba_correct_fused_error"] for q in rows)})
assert abs(statistics.fmean(q["delta_ap_pp"] for q in fm)-(fused["mAP"]-mamba["mAP"]))<1e-10
cams=[]
for cam in sorted({q["camera"] for q in c["queries"]}):
 qs=[q for q in c["queries"] if q["camera"]==cam]
 cams.append({"camera":cam,"query_count":len(qs),"metrics":{n:{"mAP":100*statistics.fmean(q["outputs"][n]["average_precision"] for q in qs),"rank1_errors":sum(q["outputs"][n]["first_match_rank"]>1 for q in qs)} for n in summary}})
gain=fused["mAP"]-signal["mAP"];top=sorted(summary["fused"]["all_identity_deltas"],key=lambda x:x["weighted_contribution_pp"],reverse=True)
analysis={"analyzed_at":datetime.now().astimezone().isoformat(),"status":"COMPLETE_ALL_OFFICIAL_QUERY_CENSUS_DESCRIPTIVE_ANALYSIS",
"scope":"All1715 queries, five outputs and50 identities; no new selection criterion, official optimization or model execution",
"summary_sha256":sha(sp),"census_sha256":sha(cp),"analysis_script_sha256":sha(Path(__file__)),
"outputs":summary,"per_camera":cams,"fused_top2_identities_share_of_net_gain":sum(x["weighted_contribution_pp"] for x in top[:2])/gain,
"fused_vs_mamba":{"mAP_delta_pp":fused["mAP"]-mamba["mAP"],"R1_delta_pp":fused["Rank-1"]-mamba["Rank-1"],"query_ap_improved":sum(q["delta_ap_pp"]>0 for q in fm),"query_ap_declined":sum(q["delta_ap_pp"]<0 for q in fm),"query_ap_unchanged":sum(q["delta_ap_pp"]==0 for q in fm),"rank1_repaired":sum(q["mamba_error_fused_fixed"] for q in fm),"rank1_new_errors":sum(q["mamba_correct_fused_error"] for q in fm),"per_identity":fmi,"all_query_changes":fm},
"literature_mAP_gap_pp":{"fused_to_signal_reported86_3":86.3-fused["mAP"],"baseline_to_signal_reported86_3":86.3-signal["mAP"],"fused_to_rodi_clip88_5":88.5-fused["mAP"],"fused_to_PMKD91_6":91.6-fused["mAP"]},
"limits":["Literature gaps are to author-reported values, not matched reproductions","Nearest-distance means span different embedding metrics and do not establish a causal explanation","Official result does not authorize tuning weights/checkpoints on official queries","Single seed42; no across-seed stability claim"],"model_tensor_image_calls":0,"training_updates":0}
ap=ROOT/"evidence/trifusion_rgbnt100_official_census_analysis_20260906.json";assert not ap.exists();ap.write_text(json.dumps(analysis,indent=2)+"\n")
closure={"verified_at":analysis["analyzed_at"],"status":"COMPLETE_RGBNT100_OFFICIAL_EXECUTOR_VERIFIED_SCIENTIFIC_SUPPORT_FAIL","execution_commit":s["execution_commit"],
"summary_sha256":sha(sp),"verification_sha256":sha(vp),"error_census_sha256":sha(cp),"observation_sha256":sha(op),
"complete_remote_arrays_rankings_endpoints_and_metrics":"PASS","full_intake_files":len(o["files"]),"full_intake_bytes":sum(x["bytes"] for x in o["files"].values()),"intake_whole_sha256_verified_by_completed_transfer":True,
"query_output_scores_verified":8575,"distance_and_ranking_positions_verified":73530625,"metrics":s["comparison"]["metrics"],"gains_over_signal_pp":s["comparison"]["gains_over_signal_pp"],"scientific_checks":v["scientific_checks"],"scientific_passed":v["scientific_passed"],
"bootstrap_lower_pp":v["identity_bootstrap_lower_pp"],"all50_identity_analysis":str(ap.relative_to(ROOT)),"official_model_record_forwards":20580,"new_optimizer_updates_in_evaluation":0,
"local_model_tensor_image_calls":0,"independent_audit":"UNAVAILABLE_SERVICE_LIMIT; executor verification is not independent"}
p=ROOT/"evidence/trifusion_rgbnt100_official_executor_closure_20260906.json";assert not p.exists();p.write_text(json.dumps(closure,indent=2)+"\n")
print(json.dumps({"closure":closure,"fused_top2_share":analysis["fused_top2_identities_share_of_net_gain"],"camera_rows":len(cams)}))

