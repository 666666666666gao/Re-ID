from collections import Counter
from datetime import datetime
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
p=Path.cwd()
source_path=p/"evidence/trifusion_v22_initialization_full_gallery_20260905.json"
terminal_path=p/"evidence/trifusion_v22_q1_seed42_5ae096b.json"
source=json.loads(source_path.read_bytes());terminal=json.loads(terminal_path.read_bytes())
assert source["status"]=="COMPLETE_READONLY_DIAGNOSTIC"
assert source["terminal_summary_sha256"]==hashlib.sha256(terminal_path.read_bytes()).hexdigest()
assert source["optimizer_steps"]==source["checkpoint_writes"]==source["dev_access_count"]==source["official_test_access_count"]==0
names=("baseline_only","fused","cnn","transformer","mamba")
ends=("batch_hard_residual","camera_negative_residual")
stages=("initialization",*ends)
metric_names=("mAP","Rank-1","Rank-5","Rank-10")
def metrics(ap,ranks):
    a=np.asarray(ap,dtype=np.float64);r=np.asarray(ranks)
    return {"mAP":float(a.mean()*100),**{f"Rank-{k}":float(np.mean(r<=k)*100) for k in (1,5,10)}}
ap_all={stage:{name:[] for name in names} for stage in stages}
rank_all={stage:{name:[] for name in names} for stage in stages}
errors=[];fold_rows=[];identity_rows=[];scopes=[];heldout=set();ids_all=[]
for init,final in zip(source["folds"],terminal["folds"],strict=True):
    fold=init["fold"];assert fold==final["fold"]
    gallery=init["gallery_manifest"];assert gallery==final["gallery_manifest"]
    counts=Counter(x["identity"] for x in gallery);cams=Counter((x["identity"],x["camera"]) for x in gallery)
    assert len(counts)==47 and heldout.isdisjoint(counts);heldout.update(counts)
    query=[i for i,x in enumerate(gallery) if counts[x["identity"]]>cams[x["identity"],x["camera"]]]
    excluded=sorted(set(range(len(gallery)))-set(query))
    ids=np.array([gallery[i]["identity"] for i in query]);ids_all.extend(ids.tolist())
    scopes.append({"fold":fold,"gallery":len(gallery),"queries":len(query),"excluded_only_from_query":len(excluded)})
    assert init["matches_both_terminal_initial_states"] and init["model_state_unchanged"] and init["no_parameter_gradients"]
    assert init["forward_counts"]=={"forward_calls":(len(gallery)+127)//128,"triplets":len(gallery)}
    results={"initialization":init["outputs"],**{e:final["endpoints"][e]["outputs"] for e in ends}}
    for e in ends:
        assert init["binding"]==final["endpoints"][e]["binding"]
        assert init["initial_state_sha256"]==final["endpoints"][e]["training"]["initial_state_sha256"]
        assert results["initialization"]["baseline_only"]==results[e]["baseline_only"]
    for stage in stages:
        for name in names:
            result=results[stage][name]
            assert result["query_indices"]==query and result["excluded_no_cross_camera_positive"]==excluded
            ap=np.array(result["average_precision"]);ranks=np.array(result["first_match_rank"])
            assert ap.shape==ranks.shape==(len(query),) and np.isfinite(ap).all()
            assert ((ap>=0)&(ap<=1)).all() and ((ranks>=1)&(ranks<=len(gallery))).all()
            m=metrics(ap,ranks);errors.extend(abs(m[k]-result["metrics_percent"][k]) for k in metric_names)
            ap_all[stage][name].extend(ap.tolist());rank_all[stage][name].extend(ranks.tolist())
            fold_rows.append({"fold":fold,"stage":stage,"output":name,**m})
    for e in ends:
        for name in names:
            delta=metrics(results[e][name]["average_precision"],results[e][name]["first_match_rank"])["mAP"]-metrics(results["initialization"][name]["average_precision"],results["initialization"][name]["first_match_rank"])["mAP"]
            errors.append(abs(delta-init["terminal_minus_initial_mAP"][e][name]))
    for identity in np.unique(ids):
        positions=np.flatnonzero(ids==identity)
        original={gallery[query[i]]["file"].split("_",1)[0] for i in positions};assert len(original)==1
        measurements={stage:{name:metrics(np.asarray(results[stage][name]["average_precision"])[positions],
                                         np.asarray(results[stage][name]["first_match_rank"])[positions])
                             for name in names} for stage in stages}
        gains={e:{name:{k:measurements[e][name][k]-measurements["initialization"][name][k] for k in metric_names} for name in names} for e in ends}
        identity_rows.append({"fold":fold,"encoded_identity":int(identity),"original_identity":original.pop(),"queries":len(positions),
                              "metrics_percent":measurements,"terminal_minus_initial_percentage_points":gains})
assert len(heldout)==141 and len(ids_all)==571 and len(set(ids_all))==len(identity_rows)==21
assert len(fold_rows)==45 and source["total_model_forward_calls"]==26 and source["total_triplet_forwards"]==3126
aggregate={stage:{name:metrics(ap_all[stage][name],rank_all[stage][name]) for name in names} for stage in stages}
gains={e:{name:aggregate[e][name]["mAP"]-aggregate["initialization"][name]["mAP"] for name in names} for e in ends}
for stage in stages:
    recorded=source["initialization_aggregate"] if stage=="initialization" else terminal["aggregate"][stage]
    for name in names:
        errors.extend(abs(aggregate[stage][name][k]-recorded[name][k]) for k in metric_names)
for e in ends:
    errors.extend(abs(gains[e][name]-source["terminal_minus_initial_mAP"][e][name]) for name in names)
assert max(errors)<1e-9,max(errors)
changes={}
for e in ends:
    changes[e]={}
    for name in names:
        ap=np.asarray(ap_all[e][name])-np.asarray(ap_all["initialization"][name])
        ri,re=np.asarray(rank_all["initialization"][name]),np.asarray(rank_all[e][name])
        changes[e][name]={"ap_improved":int((ap>0).sum()),"ap_declined":int((ap<0).sum()),"ap_equal":int((ap==0).sum()),
                          "rank1_repaired":int(((ri>1)&(re==1)).sum()),"rank1_broken":int(((ri==1)&(re>1)).sum())}
record={"status":"COMPLETE_ARRAY_AND_BINDING_VERIFICATION","verified_at":datetime.now().astimezone().isoformat(),
        "evaluation_type":source["evaluation_type"],"source_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "terminal_sha256":hashlib.sha256(terminal_path.read_bytes()).hexdigest(),"numpy_version":np.__version__,
        "max_arithmetic_difference_percent":max(errors),"scope":scopes,"all_five_aggregate":aggregate,
        "terminal_minus_initial_mAP":gains,"all_45_fold_metric_rows":fold_rows,"all_21_identity_rows":identity_rows,
        "all_query_paired_changes":changes,"diagnostic_model_calls":26,"diagnostic_triplet_forwards":3126,
        "diagnostic_optimizer_steps":0,"diagnostic_checkpoint_writes":0,"local_model_or_tensor_loads":0,
        "changes_q1_qualification":False,"new_dev_or_official_result":False}
out=p/"evidence/trifusion_v22_initialization_full_comparison_20260905.json";assert not out.exists()
out.write_bytes((json.dumps(record,indent=2)+"\n").encode())
lines=["# V22 共同初始化与全部终态的完整图库诊断","",
       "三个固定初始化只读评价已完成，74.088494秒、26次模型调用、3126次triplet前向；优化步和checkpoint写入均0。",
       "同一完整图库上的初始化fused为80.590328 mAP，普通继续训练终态80.640677（+0.050348），MCNL终态78.984454（-1.605874）。",
       "这组数据不支持融合mAP在普通继续训练后整体下降；三个专家方向不同，不能把其中一支替代完整结论。",
       "旧V12约88mAP使用eligible-only gallery及residual/bank输出，两项都与当前fused完整图库不同，不能直接相减作退化证据。",
       "本结果仅作复用OOF诊断，不选择初始化checkpoint、不改变V22 Q1_FAIL，D1/dev/official没有执行。","",
       "| 输出 | 初始化mAP | 普通终态mAP | MCNL终态mAP | 普通差 | MCNL差 | 初始化R1 | 普通R1 | MCNL R1 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name in names:
    a,b,c=[aggregate[s][name] for s in stages]
    vals=[a["mAP"],b["mAP"],c["mAP"],gains[ends[0]][name],gains[ends[1]][name],a["Rank-1"],b["Rank-1"],c["Rank-1"]]
    lines.append("| "+name+" | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
lines+=["","## 全三折、三阶段、五输出","","| 折 | 阶段 | 输出 | mAP | R1 | R5 | R10 |","|---|---|---|---:|---:|---:|---:|"]
for row in fold_rows:
    lines.append(f"| {row['fold']} | {row['stage']} | {row['output']} | "+" | ".join(f"{row[k]:.6f}" for k in metric_names)+" |")
for e in ends:
    lines+=["",f"## 全部21身份：{e}相对初始化的mAP差","",
            "| 折 | 原始身份 | query | baseline | fused | CNN | Transformer | Mamba |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in identity_rows:
        lines.append(f"| {row['fold']} | {row['original_identity']} | {row['queries']} | "+
                     " | ".join(f"{row['terminal_minus_initial_percentage_points'][e][n]['mAP']:+.6f}" for n in names)+" |")
lines+=["","## 验证与边界","",
        "全3126gallery/571query/21身份、2555个query-only排除项、五类AP/Rank数组、45行fold指标全部重算，最大差0。",
        "每个初始化的binding/state SHA与两个终态各自保存的初始state相同；评价前后完整模型state不变，全部梯度仍为None。",
        "三fold baseline-only与六个终态的对应全部数组完全一致；远端额外只读账本核对30项完整文件SHA。",
        "完整模型/图像/张量操作在远端；本地仅JSON/NumPy指标算术，不重建检索距离。",
        f"原始诊断：{source_path.relative_to(p).as_posix()}，SHA {record['source_sha256']}。",
        f"全部比较及逐查询变化：{out.relative_to(p).as_posix()}，SHA {hashlib.sha256(out.read_bytes()).hexdigest()}。",
        "只读观察及源文件账本：evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json。",
        "本执行器复核不替代独立诊断审计；独立审计待完成。不能据此主张camera唯一因果或初始化具新部署资格。",""]
doc=p/"results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md";assert not doc.exists()
doc.write_bytes("\n".join(lines).encode())
print(json.dumps({"comparison":str(out),"report":str(doc),"max_difference":max(errors),"all_five_aggregate":aggregate,
                  "terminal_minus_initial_mAP":gains,"fold_rows":45,"identity_rows":21},indent=2))
