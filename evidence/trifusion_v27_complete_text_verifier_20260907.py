#!/usr/bin/env python3
"""Recompute every V27 query, identity, gate and training stratum from terminal text."""
from pathlib import Path
import argparse,csv,hashlib,json,math
from datetime import datetime
import numpy as np

ENDS=("control","source_style")
OUTPUTS=("baseline_only","fused","cnn","transformer","mamba")
METRICS=("mAP","Rank-1","Rank-5","Rank-10")

def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a)-float(b))<1e-10,(a,b)

def score(aps,ranks):
    return {"mAP":math.fsum(aps)/len(aps)*100,
            **{f"Rank-{k}":sum(r<=k for r in ranks)/len(ranks)*100 for k in (1,5,10)}}

def main(args):
    directory=args.directory
    raw=(directory/"run_summary.json").read_bytes()
    summary=json.loads(raw)
    audit_raw=(directory/"terminal_verification.json").read_bytes()
    audit=json.loads(audit_raw)
    report=json.loads((directory/"complete_comparison.json").read_bytes())
    assert summary["status"] in ("Q1_PASS","Q1_FAIL") and len(summary["folds"])==3
    assert audit["status"]=="PASS_COMPLETE_V27_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES"
    assert audit["run_summary_sha256"]==report["source_summary_sha256"]==hashlib.sha256(raw).hexdigest()
    assert report["array_audit_sha256"]==hashlib.sha256(audit_raw).hexdigest()
    assert summary["repository_commit"]=="c225652a031ffcc524a48db8f49c31c70098287c"
    assert summary["dev_access_count"]==summary["official_test_access_count"]==0
    assert not summary["d1_executed"]
    audit_queries={(r["fold"],r["query_index"]):r for r in audit["all_queries"]}
    assert len(audit_queries)==571
    query_rows=list(csv.DictReader((directory/"all2855_query_output_rows.csv").open(encoding="utf-8",newline="")))
    identity_rows=list(csv.DictReader((directory/"all105_identity_output_rows.csv").open(encoding="utf-8",newline="")))
    assert len(query_rows)==2855 and len(identity_rows)==105
    query_csv={(int(r["fold"]),int(r["query_index"]),r["output"]):r for r in query_rows}
    identity_csv={(int(r["fold"]),int(r["encoded_identity"]),r["output"]):r for r in identity_rows}
    assert len(query_csv)==2855 and len(identity_csv)==105
    for key,filename in (("query_csv","all2855_query_output_rows.csv"),("identity_csv","all105_identity_output_rows.csv")):
        assert report[key]["sha256"]==hashlib.sha256((directory/filename).read_bytes()).hexdigest()
    pairs={e:{name:([],[]) for name in OUTPUTS} for e in ENDS}
    cluster_data={}
    all_gallery_ids=set()
    fold_gains=[]
    checked=0
    for fold in summary["folds"]:
        number=fold["fold"]
        gallery=fold["gallery_manifest"]
        all_gallery_ids.update(r["identity"] for r in gallery)
        eligible=[i for i,row in enumerate(gallery) if any(
            other["identity"]==row["identity"] and other["camera"]!=row["camera"] for other in gallery)]
        assert len(gallery)==(1000,1051,1075)[number]
        assert len(eligible)==(190,179,202)[number]
        assert {r["query_index"] for r in audit["all_queries"] if r["fold"]==number}==set(eligible)
        for endpoint in ENDS:
            receipt=fold["endpoints"][endpoint]
            assert json.loads((directory/f"fold_{number}_{endpoint}_receipt.json").read_bytes())==receipt
            for name in OUTPUTS:
                result=receipt["outputs"][name]
                assert result["query_indices"]==eligible
                aps=result["average_precision"]
                ranks=result["first_match_rank"]
                assert len(aps)==len(ranks)==len(eligible)
                assert all(0<=a<=1 and 1<=r<=len(gallery) for a,r in zip(aps,ranks))
                for metric,value in score(aps,ranks).items(): close(value,result["metrics_percent"][metric])
                pairs[endpoint][name][0].extend(aps)
                pairs[endpoint][name][1].extend(ranks)
                for position,index in enumerate(eligible):
                    row=audit_queries[number,index]
                    assert row["file"]==gallery[index]["file"] and row["identity"]==gallery[index]["identity"]
                    saved=row["outputs"][name]
                    prefix="candidate" if endpoint=="source_style" else "control"
                    close(aps[position],saved[prefix+"_ap"])
                    assert ranks[position]==saved[prefix+"_first_match_rank"]
                    csv_row=query_csv[number,index,name]
                    assert csv_row["file"]==gallery[index]["file"]
                    close(aps[position]*100,csv_row[prefix+"_ap_percent"])
                    assert ranks[position]==int(csv_row[prefix+"_first_match_rank"])
                for identity in sorted({gallery[i]["identity"] for i in eligible}):
                    positions=[p for p,i in enumerate(eligible) if gallery[i]["identity"]==identity]
                    row=identity_csv[number,identity,name]
                    assert int(row["queries"])==len(positions)
                    assert row["original_identity"]==gallery[eligible[positions[0]]]["file"].split("_",1)[0]
                    identity_metrics=score([aps[p] for p in positions],[ranks[p] for p in positions])
                    prefix="candidate" if endpoint=="source_style" else "control"
                    for metric,value in identity_metrics.items(): close(value,row[prefix+"_"+metric])
            training=receipt["training"]
            path=directory/Path(training["all_training_steps_path"]).name
            training_raw=path.read_bytes()
            assert hashlib.sha256(training_raw).hexdigest()==training["all_training_steps_sha256"]
            steps=[json.loads(line) for line in training_raw.splitlines()]
            assert len(steps)==training["optimizer_steps"]==(580,560,540)[number]
            assert training["epochs"]==len(training["history"])==20
            assert all(not r["overflow"] and all(math.isfinite(v) for v in r["losses"].values()) for r in steps)
            assert [r["optimizer_step"] for r in steps]==list(range(1,len(steps)+1))
            for history in training["history"]:
                selected=[r for r in steps if r["epoch"]==history["epoch"]]
                assert len(selected)==history["batches"]
                for key in selected[0]["losses"]:
                    close(math.fsum(r["losses"][key] for r in selected)/len(selected),history["mean_"+key])
            for stratum in report["training_loss_strata"]:
                if (stratum["fold"],stratum["endpoint"])!=(number,endpoint): continue
                selected=[r for r in steps if stratum["stratum"]=="all" or
                          r["style_plan"]["active"]==(stratum["stratum"]=="plan_active")]
                assert len(selected)==stratum["steps"]
                assert sum(r["losses"]["style_active"] for r in selected)==stratum["actual_style_steps"]
                for key,values in stratum["losses"].items():
                    data=[r["losses"][key] for r in selected]
                    close(math.fsum(data)/len(data),values["mean"])
                    close(min(data),values["minimum"])
                    close(max(data),values["maximum"])
                    assert sum(v>0 for v in data)==values["positive_batches"]
            checked+=len(steps)
        a,b=[fold["endpoints"][e]["outputs"]["fused"]["average_precision"] for e in ENDS]
        fold_gains.append((math.fsum(b)-math.fsum(a))/len(a)*100)
        for position,index in enumerate(eligible):
            identity=gallery[index]["identity"]
            cluster_data.setdefault(identity,[]).append(b[position]-a[position])
    assert checked==3360 and len(all_gallery_ids)==141 and len(cluster_data)==21
    assert len(report["training_loss_strata"])==18
    aggregate={e:{name:score(*pairs[e][name]) for name in OUTPUTS} for e in ENDS}
    for endpoint in ENDS:
        for name in OUTPUTS:
            assert len(pairs[endpoint][name][0])==571
            for metric,value in aggregate[endpoint][name].items():
                close(value,summary["aggregate"][endpoint][name][metric])
                close(value,audit["aggregate"][endpoint][name][metric])
                close(value,report["aggregate"][endpoint][name][metric])
    gains={name:aggregate["source_style"][name]["mAP"]-aggregate["control"][name]["mAP"] for name in OUTPUTS}
    for name,value in gains.items(): close(value,summary["matched_gains_mAP"][name])
    clusters=sorted(cluster_data)
    sums=np.array([math.fsum(cluster_data[i]) for i in clusters])
    counts=np.array([len(cluster_data[i]) for i in clusters])
    sampled=np.random.default_rng(42).integers(0,21,size=(10000,21))
    lower=float(np.quantile(sums[sampled].sum(1)/counts[sampled].sum(1),.025)*100)
    close(lower,summary["bootstrap"]["lower_bound_95_mAP"])
    for value,wanted in zip(fold_gains,summary["fold_fused_gains_mAP"],strict=True): close(value,wanted)
    checks={
        "aggregate_fused_gain_at_least_1pp":gains["fused"]>=1,
        "all_fold_fused_nonnegative":all(v>=0 for v in fold_gains),
        "all_expert_aggregate_nonnegative":all(gains[n]>=0 for n in OUTPUTS[2:]),
        "fused_bootstrap_lower_positive":lower>0,
        "fused_beats_baseline_and_experts":all(aggregate["source_style"]["fused"]["mAP"]>aggregate["source_style"][n]["mAP"] for n in ("baseline_only",*OUTPUTS[2:]))}
    assert checks==summary["scientific_checks"]==audit["scientific_checks"]==report["scientific_checks"]
    for name in OUTPUTS:
        rows=[r for r in query_rows if r["output"]==name]
        for row in rows:
            close(float(row["candidate_ap_percent"])-float(row["control_ap_percent"]),row["ap_gain_pp"])
            assert (int(row["control_first_match_rank"])>1 and int(row["candidate_first_match_rank"])==1)==(row["rank1_repaired"]=="True")
            assert (int(row["control_first_match_rank"])==1 and int(row["candidate_first_match_rank"])>1)==(row["rank1_new_error"]=="True")
        for key,column in (("rank1_repaired","rank1_repaired"),("rank1_broken","rank1_new_error")):
            assert sum(r[column]=="True" for r in rows)==report["all_query_paired_changes"][name][key]
    for row in identity_rows:
        for metric in METRICS:
            close(float(row["candidate_"+metric])-float(row["control_"+metric]),row["gain_"+metric+"_pp"])
    proof={"status":"PASS_COMPLETE_V27_TERMINAL_TEXT_RECOMPUTATION","verified_at":datetime.now().astimezone().isoformat(),
           "summary_sha256":hashlib.sha256(raw).hexdigest(),"array_verification_sha256":hashlib.sha256(audit_raw).hexdigest(),
           "text_verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "scientific_status":summary["status"],"scientific_checks":checks,"aggregate":aggregate,"matched_gains_mAP":gains,
           "fold_fused_gains_mAP":fold_gains,"bootstrap_lower_95_mAP":lower,"checked_training_updates":checked,
           "checked_endpoint_epochs":120,"checked_query_output_csv_rows":2855,"checked_identity_output_csv_rows":105,
           "eligible_queries":571,"query_identities":21,"gallery_records":3126,"training_loss_strata":18,
           "local_model_forwards":0,"local_optimizer_steps":0,"local_torch_imports":0,"independent_audit":False}
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(proof,indent=2)+"\n").encode())
    print(json.dumps(proof))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
