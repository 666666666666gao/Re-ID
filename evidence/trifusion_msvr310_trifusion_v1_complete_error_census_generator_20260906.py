from pathlib import Path
import datetime,hashlib,json
root=Path(r"C:/Users/gb/.trifusion_github_publish_22c3bee")
sp=root/"evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json"
vp=root/"evidence/trifusion_msvr310_trifusion_v1_terminal_scalar_verification_20260906.json"
s=json.loads(sp.read_text());v=json.loads(vp.read_text())
outputs=("fused","cnn","transformer","mamba")
census={k:[] for k in outputs}; baseline_wrong=[];rank_sha={}
for f in s["folds"]:
 path=root/f"evidence/msvr310_trifusion_v1_complete/fold_{f['fold']}_rankings.json"
 ranks=json.loads(path.read_text());rank_sha[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
 gallery=f["retrieval"]["gallery_manifest"];queries=f["retrieval"]["query_rows"]
 for n,q in enumerate(queries):
  def first_valid(name):
   return next(gallery[i] for i in ranks[name][n] if not (gallery[i]["identity"]==q["identity"] and gallery[i]["scene"]==q["scene"]))
  b=first_valid("baseline_only")
  if b["identity"]!=q["identity"]:baseline_wrong.append({"same_camera":b["camera"]==gallery[q["gallery_position"]]["camera"],"same_scene":b["scene"]==q["scene"]})
  for name in outputs:
   a=first_valid(name)
   br=int(f["retrieval"]["outputs"]["baseline_only"]["first_match_rank"][n]);ar=int(f["retrieval"]["outputs"][name]["first_match_rank"][n])
   census[name].append({"fold":f["fold"],"query_record_index":q["record_index"],"identity":q["identity"],"camera":gallery[q["gallery_position"]]["camera"],"scene":q["scene"],
    "query_gallery_position":q["gallery_position"],"baseline_first_match_rank":br,"candidate_first_match_rank":ar,
    "baseline_first_identity":b["identity"],"candidate_first_identity":a["identity"],"candidate_first_camera":a["camera"],"candidate_first_scene":a["scene"],
    "ap_delta_pp":100*(f["retrieval"]["outputs"][name]["average_precision"][n]-f["retrieval"]["outputs"]["baseline_only"]["average_precision"][n]),
    "rank1_repaired":br>1 and ar==1,"rank1_new_error":br==1 and ar>1,
    "wrong_first_same_camera":a["identity"]!=q["identity"] and a["camera"]==gallery[q["gallery_position"]]["camera"],
    "wrong_first_same_scene":a["identity"]!=q["identity"] and a["scene"]==q["scene"]})
stats={}
for name,rows in census.items():
 new=[x for x in rows if x["rank1_new_error"]];wrong=[x for x in rows if x["candidate_first_match_rank"]>1]
 stats[name]={"all_queries":len(rows),"rank1_repaired":sum(x["rank1_repaired"] for x in rows),"rank1_new_errors":len(new),
  "new_errors_same_camera":sum(x["wrong_first_same_camera"] for x in new),"new_errors_same_scene":sum(x["wrong_first_same_scene"] for x in new),
  "all_rank1_errors":len(wrong),"all_rank1_errors_same_camera":sum(x["wrong_first_same_camera"] for x in wrong),
  "all_rank1_errors_same_scene":sum(x["wrong_first_same_scene"] for x in wrong)}
 assert stats[name]["rank1_repaired"]==s["comparison"]["query_changes"][name]["rank1_repaired"]
 assert stats[name]["rank1_new_errors"]==s["comparison"]["query_changes"][name]["rank1_new_errors"]
ids=[{"identity":r["identity"],"queries":r["query_count"],"fused_delta_pp":r["map_by_output"]["fused"]-r["map_by_output"]["baseline_only"]} for r in s["comparison"]["per_identity"]]
cross=sum(f["cross_scene_positive_pairs"] for f in v["folds"]);total=sum(f["same_identity_positive_pairs"] for f in v["folds"])
report={"observed_at":datetime.datetime.now().astimezone().isoformat(),"scope":"Post-hoc descriptive census of all600 queries/four candidates and60 identities from already verified rankings, no image/model/optimization; no new gate or selection",
"input_summary_sha256":hashlib.sha256(sp.read_bytes()).hexdigest(),"ranking_sha256":rank_sha,"summary":stats,
"baseline_rank1_errors":{"total":len(baseline_wrong),"same_camera":sum(x["same_camera"] for x in baseline_wrong),"same_scene":sum(x["same_scene"] for x in baseline_wrong)},
"identity_directions":{"improved":sum(x["fused_delta_pp"]>0 for x in ids),"declined":sum(x["fused_delta_pp"]<0 for x in ids),"unchanged":sum(x["fused_delta_pp"]==0 for x in ids)},
"training_positive_pairs":{"cross_scene":cross,"all_same_identity":total,"cross_scene_percent":100*cross/total,"scope":"MSVR310 scene relation; not interchangeable with RGBNT201 same-camera definitions"},
"per_identity":ids,"query_rows":census,"execution_note":"Initial local JSON-only census used camera on query metadata; camera belongs to its indexed gallery record. It stopped before writing output. This full census uses the actual gallery schema.",
"causal_limit":"Camera/scene coincidence describes rankings; it does not identify an exclusive causal mechanism"}
p=root/"evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json";assert not p.exists();p.write_bytes((json.dumps(report,indent=2)+"\n").encode())
print(json.dumps({k:v for k,v in report.items() if k not in ("query_rows","per_identity","ranking_sha256")}))
