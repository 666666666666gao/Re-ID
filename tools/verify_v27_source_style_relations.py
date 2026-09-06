#!/usr/bin/env python3
"""Independent NumPy replay of every saved source-style diagnostic relation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda:handle.read(4*1024**2),b""): digest.update(data)
    return digest.hexdigest()


def numpy_statistics(matrix, labels, cameras, protocol):
    a=np.asarray(matrix,dtype=np.float64)
    labels,cameras=np.asarray(labels),np.asarray(cameras)
    positive=(labels[:,None]==labels[None,:]) & ~np.eye(len(labels),dtype=bool)
    if protocol=="cross_camera": positive &= cameras[:,None]!=cameras[None,:]
    negative=labels[:,None]!=labels[None,:]
    q,p,n=np.nonzero(positive[:,:,None] & negative[:,None,:])
    margins=a[:,q,p]-a[:,q,n]
    distance=np.sqrt(np.maximum(0,2-2*a))
    hinge=np.maximum(0,distance[:,q,p]-distance[:,q,n]+.3)
    mv=np.stack([margins.mean(1),margins.min(1),margins.max(1),(margins<=0).sum(1),
                 hinge.mean(1),(hinge>0).sum(1)],axis=1)
    slot=margins[5:14]
    fused=margins[1]
    weight=1/(1+np.exp(fused[None]/.1))/(1+np.exp(slot/.1))
    penalty=.1*np.logaddexp(0,-slot/.1)
    unweighted=-1/(1+np.exp(slot/.1))/slot.size
    derivative=weight*unweighted
    wv=np.stack([weight.mean(1),weight.min(1),weight.max(1),
                 weight.sum(1)**2/(weight*weight).sum(1),(weight*penalty).mean(1),
                 np.linalg.norm(derivative,axis=1),np.linalg.norm(unweighted,axis=1),
                 ((slot<=0)&(fused[None]<=0)).sum(1)],axis=1)
    patterns=[]
    for i in range(9):
        bits=(margins[0]<=0).astype(np.int64)*8+(margins[14]<=0).astype(np.int64)*4
        bits+=(margins[15+i//3]<=0).astype(np.int64)*2+(fused<=0).astype(np.int64)
        patterns.append(np.bincount(bits[slot[i]<=0],minlength=16).tolist())
    values={"triplets":len(q),"slot_triplets":slot.size,"eligible_anchor_rows":int(positive.any(1).sum()),
            "margin_values":mv,"weight_values":wv,"slot_nonpositive_support_patterns":patterns,
            "v26_auxiliary_loss":float((weight*penalty).mean()),
            "v26_to_uniform_margin_derivative_norm_ratio":float(np.linalg.norm(derivative)/np.linalg.norm(unweighted))}
    return values,Counter(int(labels[x]) for x in q[fused<=0])


def verify(args):
    started=time.time()
    run=args.run_dir
    assert int(Path(str(run)+".exit").read_text())==0
    terminal=json.loads(Path(str(run)+".terminal.json").read_bytes())
    assert terminal["exit_code"]==0 and not Path("/proc",str(terminal["original_pid"])).exists()
    raw=(run/"style_relation_support.json").read_bytes()
    summary=json.loads(raw)
    assert summary["status"]=="COMPLETE_FIXED_SOURCE_STYLE_RELATION_DIAGNOSTIC_PENDING_CPU_VERIFICATION"
    assert summary["completed_batches"]==1680 and summary["model_forwards"]==10080 and len(summary["folds"])==3
    contract=json.loads(args.contract.read_bytes())
    assert sha(args.contract)==summary["contract_sha256"]
    assert contract["verifier_sha256"]==sha(__file__)
    assert summary["runner_sha256"]==contract["runner_sha256"]
    for file,expected in contract["source_file_sha256"].items():
        assert sha(file)==expected,file
    assert summary["new_optimizer_updates"]==0 and not summary["encoder_gradients_computed"]
    assert summary["dev_image_reads"]==summary["official_image_reads"]==summary["heldout_image_reads"]==0
    states,views,protocols=summary["states"],summary["views"],summary["protocols"]
    assert states==contract["states"] and views==contract["views"] and protocols==contract["protocols"]
    assert summary["outputs"]==contract["outputs"]
    prior=json.loads((Path(contract["v27_run"])/"run_summary.json").read_bytes())
    config=json.loads(Path(contract["v27_config"]).read_bytes())
    replay=json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    assert sha(summary["cases"]["path"])==summary["cases"]["sha256"]
    groups={}
    checked=0
    checked_triplets=0
    maximum_error=0.0
    with gzip.open(summary["cases"]["path"],"rt",encoding="utf-8") as handle:
        for fold in summary["folds"]:
            number=fold["fold"]
            assert fold["all_model_states_unchanged"] and fold["all_gradients_absent"]
            assert sha(fold["similarities"]["path"])==fold["similarities"]["sha256"]
            assert sha(fold["batch_receipts"]["path"])==fold["batch_receipts"]["sha256"]
            matrices=np.load(fold["similarities"]["path"],mmap_mode="r")
            count=(580,560,540)[number]
            assert matrices.shape==(count,3,2,18,64,64) and matrices.dtype==np.float32
            receipts=[json.loads(x) for x in Path(fold["batch_receipts"]["path"]).read_text().splitlines()]
            assert len(receipts)==count and fold["batches"]==count
            exposure=Counter()
            manifest=replay["folds"][number]["source_manifest"]
            for step_index,(receipt,wanted) in enumerate(zip(receipts,replay["folds"][number]["arms"]["control"]["batches"],strict=True)):
                assert receipt["case_index"]==step_index and receipt["sampler_indices"]==wanted["sampler_indices"]
                for key in ("epoch","step","sample_order_sha256","directed_cross_camera_positive_pairs"):
                    assert receipt[key]==wanted[key]
                indices=receipt["sampler_indices"]
                exposure.update(indices)
                labels=np.array([manifest[i]["identity"] for i in indices])
                cameras=np.array([manifest[i]["camera"] for i in indices])
                assert sorted(Counter(labels).values())==[8]*8
                plan=receipt["style_plan"]
                assert all(cameras[d]!=cameras[i] for i,d in enumerate(plan["donors"]))
                rng=np.random.default_rng(np.random.SeedSequence([42,number,step_index]))
                active=bool(rng.uniform()<.5)
                scores=rng.uniform(size=(64,64))
                donors=[int(np.flatnonzero(cameras!=c)[np.argmin(scores[i,cameras!=c])]) for i,c in enumerate(cameras)]
                coefficients=rng.beta(.1,.1,size=64).tolist()
                assert plan=={"fold":number,"step":step_index,"active":active,"forced_active":False,
                              "donors":donors,"coefficients":coefficients,"all_donors_cross_camera":True}
                if step_index<8:
                    assert receipt["raw_receipt"]==prior["folds"][number]["endpoints"]["control"]["training"]["first_eight_batch_receipts"][step_index]
                for state_index,state in enumerate(states):
                    for view_index,view in enumerate(views):
                        row=json.loads(next(handle))
                        assert (row["fold"],row["case_index"],row["state"],row["view"])==(number,step_index,state,view)
                        assert row["plan_active"]==active and row["actual_style_active"]==bool(view_index and active)
                        matrix=matrices[step_index,state_index,view_index]
                        assert np.isfinite(matrix).all()
                        assert np.array_equal(matrix[0],matrices[step_index,0,0,0])
                        if not active: assert np.array_equal(matrix,matrices[step_index,state_index,0])
                        error=max(float(np.abs(matrix[1]-(.5*matrix[0]+matrix[5:14].sum(0)/18)).max()),
                                  float(np.abs(matrix[1]-matrix[2:5].mean(0)).max()),
                                  float(np.abs(matrix[14]-matrix[15:18].mean(0)).max()))
                        assert error<.005
                        for protocol in protocols:
                            actual,identity_errors=numpy_statistics(matrix,labels,cameras,protocol)
                            saved=row["protocols"][protocol]
                            for key in ("triplets","slot_triplets","eligible_anchor_rows","slot_nonpositive_support_patterns"):
                                assert actual[key]==saved[key],(number,step_index,state,view,protocol,key)
                            for key in ("margin_values","weight_values","v26_auxiliary_loss","v26_to_uniform_margin_derivative_norm_ratio"):
                                x,y=np.asarray(actual[key]),np.asarray(saved[key])
                                assert np.allclose(x,y,rtol=1e-9,atol=1e-9),(number,step_index,state,view,protocol,key,float(np.max(np.abs(x-y))))
                                maximum_error=max(maximum_error,float(np.max(np.abs(x-y))))
                            assert np.array_equal(actual["margin_values"][:,[3,5]],np.asarray(saved["margin_values"])[:,[3,5]])
                            assert np.array_equal(actual["weight_values"][:,-1],np.asarray(saved["weight_values"])[:,-1])
                            triplets=actual["triplets"]
                            if protocol=="identity": assert triplets==25088
                            else: assert triplets==wanted["directed_cross_camera_positive_pairs"]*56
                            checked_triplets+=triplets
                            key=(number,state,view,protocol,"active" if active else "inactive")
                            if key not in groups:
                                groups[key]={"cases":0,"triplets":0,"nonpositive":np.zeros(18,dtype=np.int64),
                                             "hinge_positive":np.zeros(18,dtype=np.int64),
                                             "margin_sums":np.zeros(18),"hinge_sums":np.zeros(18),
                                             "weight_sums":np.zeros(9),"auxiliary_sum":0.,
                                             "derivative_ratio_sum":0.,"fused_nonpositive_by_source_identity":Counter()}
                            group=groups[key]
                            group["cases"]+=1; group["triplets"]+=triplets
                            group["nonpositive"]+=actual["margin_values"][:,3].astype(np.int64)
                            group["hinge_positive"]+=actual["margin_values"][:,5].astype(np.int64)
                            group["margin_sums"]+=actual["margin_values"][:,0]*triplets
                            group["hinge_sums"]+=actual["margin_values"][:,4]*triplets
                            group["weight_sums"]+=actual["weight_values"][:,0]*triplets
                            group["auxiliary_sum"]+=actual["v26_auxiliary_loss"]*triplets
                            group["derivative_ratio_sum"]+=actual["v26_to_uniform_margin_derivative_norm_ratio"]
                            group["fused_nonpositive_by_source_identity"].update(identity_errors)
                        checked+=1
            for identity in replay["folds"][number]["arms"]["control"]["identities"]:
                assert [exposure[i] for i in identity["record_indices"]]==identity["record_exposures"]
            del matrices
        assert handle.readline()==""
    assert checked==10080
    assert checked_triplets==contract["planned_protocol_triplets"]==273297024
    aggregates=[]
    for key,group in groups.items():
        number,state,view,protocol,stratum=key
        original_ids={r["identity"]:r["file"].split("_",1)[0] for r in replay["folds"][number]["source_manifest"]}
        aggregates.append({"fold":number,"state":state,"view":view,"protocol":protocol,"plan_stratum":stratum,
            "cases":group["cases"],"triplets":group["triplets"],
            "nonpositive_per_output":group["nonpositive"].tolist(),"hinge_positive_per_output":group["hinge_positive"].tolist(),
            "mean_margin_per_output":(group["margin_sums"]/group["triplets"]).tolist(),
            "mean_euclidean_hinge_per_output":(group["hinge_sums"]/group["triplets"]).tolist(),
            "mean_slot_weight":(group["weight_sums"]/group["triplets"]).tolist(),
            "mean_auxiliary_loss":group["auxiliary_sum"]/group["triplets"],
            "mean_margin_derivative_ratio_per_batch":group["derivative_ratio_sum"]/group["cases"],
            "fused_nonpositive_by_source_identity":{original_ids[i]:count for i,count in sorted(group["fused_nonpositive_by_source_identity"].items())}})
    assert len(aggregates)==72
    proof={"status":"PASS_ALL_FIXED_SOURCE_STYLE_RELATIONS_AND_AGGREGATES",
           "verified_at":datetime.now().astimezone().isoformat(),"source_summary_sha256":hashlib.sha256(raw).hexdigest(),
           "verifier_sha256":sha(__file__),"checked_model_forward_cases":checked,"checked_protocol_triplets":checked_triplets,
           "checked_similarity_values":10080*18*64*64,"maximum_numeric_error":maximum_error,
           "all_source_exposures_match":True,"all_model_states_unchanged_receipted":True,
           "all_reported_count_fields_exact":True,"aggregates":aggregates,"aggregate_cells":72,
           "new_model_forwards":0,"new_optimizer_updates":0,"image_reads":0,"torch_imports":0,"independent_audit":False,
           "elapsed_seconds":time.time()-started}
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(proof,indent=2)+"\n").encode())
    print(json.dumps({k:v for k,v in proof.items() if k!="aggregates"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    verify(parser.parse_args())
