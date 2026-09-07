#!/usr/bin/env python3
"""Independently recompute all saved V28 source-slot geometry using NumPy only."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

EXPERTS=("cnn","transformer","mamba")
MODALITIES=("RGB","NI","TI")
VIEWS=("original","registered_style")
FIELDS=("h2","c2","hc","sum2","y2","yh","yc","correction_abs_mean",
        "h_norm","c_norm","sum_norm","c_to_h_norm","cos_y_h","cos_y_c","cos_h_c","distance_y_h_unit")
METRICS=FIELDS[8:]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def geometry(data):
    a,b,ab,total,z,za,zb=np.moveaxis(data[...,:7],-1,0)
    an,bn,tn,zn=(np.sqrt(value) for value in (a,b,total,z))
    assert np.all(an>0)
    return np.stack((an,bn,tn,bn/an,
        za/(np.clip(zn,1e-8,None)*np.clip(an,1e-8,None)),
        zb/(np.clip(zn,1e-8,None)*np.clip(bn,1e-8,None)),
        ab/(np.clip(an,1e-8,None)*np.clip(bn,1e-8,None)),
        np.sqrt(np.clip(z+1-2*za/an,0,None))),axis=-1)


def style_plan(cameras,fold,step):
    cameras=np.asarray(cameras,dtype=np.int64)
    allowed=cameras[:,None]!=cameras[None,:]
    assert allowed.any(axis=1).all()
    rng=np.random.default_rng(np.random.SeedSequence([42,fold,step]))
    active=bool(rng.random()<.5)
    choices=rng.random((64,64))
    choices[~allowed]=np.inf
    return {"fold":fold,"step":step,"active":active,"forced_active":False,
            "donors":choices.argmin(axis=1).tolist(),
            "coefficients":rng.beta(.1,.1,size=64).tolist(),"all_donors_cross_camera":True}


def statistics(values):
    values=values.reshape(-1,len(FIELDS))
    assert len(values)>0
    result={"observations":len(values),
        "zero_c":int((values[:,1]==0).sum()),"zero_sum":int((values[:,3]==0).sum()),
        "ratio_gt1":int((values[:,11]>1).sum()),"ratio_gt10":int((values[:,11]>10).sum()),
        "ratio_gt100":int((values[:,11]>100).sum()),
        "cos_y_h_negative":int((values[:,12]<0).sum()),
        "cos_y_h_lt_half":int((values[:,12]<.5).sum()),
        "cos_y_c_ge_099":int(((values[:,1]>0)&(values[:,13]>=.99)).sum()),
        "closer_to_c":int(((values[:,1]>0)&(values[:,13]>values[:,12])).sum()),
        "correction_abs_mean":float(values[:,7].mean())}
    for index,name in enumerate(METRICS,8):
        x=values[:,index]
        quantiles=np.quantile(x,[.05,.5,.95])
        result.update({name+"_mean":float(x.mean()),name+"_min":float(x.min()),
                       name+"_p05":float(quantiles[0]),name+"_median":float(quantiles[1]),
                       name+"_p95":float(quantiles[2]),name+"_max":float(x.max())})
    return result


def write_csv(path,rows):
    with path.open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def verify(args):
    started=time.time()
    contract=json.loads(args.contract.read_bytes())
    assert sha(args.contract)==args.contract_sha256
    assert sha(__file__)==contract["verifier_sha256"]
    for path,expected in contract["source_file_sha256"].items(): assert sha(path)==expected,path
    run=args.run_dir
    summary_path=run/"source_joint_scale.json"
    summary=json.loads(summary_path.read_bytes())
    assert summary["status"]=="COMPLETE_FIXED_SOURCE_JOINT_SCALE_DIAGNOSTIC_PENDING_CPU_VERIFICATION"
    assert summary["contract_sha256"]==args.contract_sha256
    assert summary["runner_sha256"]==contract["runner_sha256"]
    assert summary["fields"]==list(FIELDS) and summary["views"]==list(VIEWS)
    assert summary["completed_batches"]==1680 and summary["model_forwards"]==3360
    assert summary["slot_observations"]==1935360 and len(summary["folds"])==3
    assert summary["new_optimizer_updates"]==summary["dev_image_reads"]==summary["official_image_reads"]==summary["heldout_image_reads"]==0
    assert not summary["encoder_gradients_computed"] and not summary["retrieval_evaluation"]
    prior_dir=Path(contract["v28_run"])
    assert sha(prior_dir/"run_summary.json")==contract["v28_summary_sha256"]
    prior=json.loads((prior_dir/"run_summary.json").read_bytes())
    config=json.loads(Path(contract["v28_config"]).read_bytes())
    assert sha(contract["v28_config"])==contract["v28_config_sha256"]
    metadata=Path(config["SUPERVISION_METADATA"]["PATH"])
    assert sha(metadata)==config["SUPERVISION_METADATA"]["SHA256"]
    replay=json.loads(metadata.read_bytes())
    cells,identities=[],[]
    checked_batches,checked_slots,active_batches=0,0,0
    maximum_error=0.
    fold_proofs=[]
    for fold in summary["folds"]:
        number=fold["fold"]
        assert number==len(fold_proofs)
        source=fold["source_manifest"]
        assert source==replay["folds"][number]["source_manifest"]
        expected=replay["folds"][number]["arms"]["control"]
        saved=prior["folds"][number]["endpoints"]["joint_tokens"]
        binding=fold["binding"]
        assert binding["fixed_state_sha256"]==saved["training"]["final_state_sha256"]
        assert binding["checkpoint_sha256"]==saved["checkpoint_sha256"]==sha(binding["checkpoint"])
        assert fold["all_model_states_unchanged"] and fold["all_gradients_absent"]
        assert fold["first_eight_augmentation_receipts_equal_original_v28"]
        assert fold["all_sample_orders_and_record_exposures_equal_registered"]
        array=Path(fold["statistics"]["path"])
        receipt=Path(fold["batch_receipts"]["path"])
        assert array.parent.resolve()==receipt.parent.resolve()==run.resolve()
        for p,key in [(array,"statistics"),(receipt,"batch_receipts")]:
            assert sha(p)==fold[key]["sha256"] and p.stat().st_size==fold[key]["bytes"]
        data=np.load(array,mmap_mode="r")
        assert data.dtype==np.float64 and data.shape==((580,560,540)[number],2,64,3,3,16)
        assert np.isfinite(data).all()
        assert np.all(data[...,0]>0) and np.all(data[...,[1,3,4,7]]>=0)
        wanted_geometry=geometry(data)
        assert np.allclose(wanted_geometry,data[...,8:],rtol=1e-5,atol=1e-5)
        error=float(np.max(np.abs(wanted_geometry-data[...,8:])/(1+np.abs(wanted_geometry))))
        maximum_error=max(maximum_error,error)
        assert np.allclose(data[...,3],data[...,0]+data[...,1]+2*data[...,2],rtol=1e-10,atol=1e-10)
        assert np.max(np.abs(wanted_geometry[...,0]-1))<2e-5
        assert np.max(np.abs(data[...,4]-1))<2e-5
        rows=[json.loads(line) for line in receipt.read_bytes().splitlines()]
        assert len(rows)==fold["batches"]==expected["batch_count"]
        labels,active=[],[]
        exposures=Counter()
        for index,row in enumerate(rows):
            wanted=expected["batches"][index]
            assert row["fold"]==number and row["case_index"]==index
            assert (row["epoch"],row["step"])==(wanted["epoch"],wanted["step"])
            assert row["sampler_indices"]==wanted["sampler_indices"]
            assert row["sample_order_sha256"]==wanted["sample_order_sha256"]
            assert hashlib.sha256(json.dumps(row["paths"],separators=(",",":")).encode()).hexdigest()==row["sample_order_sha256"]
            assert row["raw_receipt"]["paths"]==row["paths"] and row["raw_receipt"]["sampler_indices"]==row["sampler_indices"]
            if index<8: assert row["raw_receipt"]==saved["training"]["first_eight_batch_receipts"][index]
            ids=[source[i]["identity"] for i in row["sampler_indices"]]
            cameras=[source[i]["camera"] for i in row["sampler_indices"]]
            assert sorted(Counter(ids).values())==[8]*8
            assert row["style_plan"]==style_plan(cameras,number,index)
            if not row["style_plan"]["active"]: assert np.array_equal(data[index,0],data[index,1])
            labels.append(ids);active.append(row["style_plan"]["active"])
            exposures.update(row["sampler_indices"])
        for identity in expected["identities"]:
            assert [exposures[i] for i in identity["record_indices"]]==identity["record_exposures"]
        labels=np.asarray(labels)
        active=np.asarray(active,dtype=bool)
        active_batches+=int(active.sum())
        assert len(np.unique(labels))==94
        for vi,view in enumerate(VIEWS):
            for ei,expert in enumerate(EXPERTS):
                for mi,modality in enumerate(MODALITIES):
                    base={"fold":number,"view":view,"role":expert,"modality":modality}
                    for stratum,mask in [("all",np.ones(len(rows),dtype=bool)),("plan_active",active),("plan_inactive",~active)]:
                        cells.append({**base,"stratum":stratum,**statistics(data[mask,vi,:,ei,mi])})
                    for identity in np.unique(labels):
                        mask=labels==identity
                        values=data[:,vi,:,ei,mi][mask]
                        original_ids={source[i]["file"].split("_",1)[0] for i in exposures if source[i]["identity"]==int(identity)}
                        assert len(original_ids)==1
                        identities.append({**base,"encoded_identity":int(identity),"original_identity":original_ids.pop(),
                                           **statistics(values)})
        slots=data.shape[0]*2*64*9
        checked_batches+=len(rows);checked_slots+=slots
        fold_proofs.append({"fold":number,"all_arrays_and_receipts_match":True,"batches":len(rows),
                            "slot_observations":slots,"maximum_scaled_error":error,
                            "statistics_sha256":sha(array),"batch_receipts_sha256":sha(receipt)})
        del data,wanted_geometry
    assert checked_batches==1680 and checked_slots==1935360 and active_batches==819
    assert len(cells)==162 and len(identities)==5076
    overall={}
    count_keys=("zero_c","zero_sum","ratio_gt1","ratio_gt10","ratio_gt100",
                "cos_y_h_negative","cos_y_h_lt_half","cos_y_c_ge_099","closer_to_c")
    mean_keys=("correction_abs_mean",*(n+"_mean" for n in METRICS))
    for view in VIEWS:
        rows=[r for r in cells if r["view"]==view and r["stratum"]=="all"]
        count=sum(r["observations"] for r in rows)
        assert count==967680
        overall[view]={"observations":count,
                       **{key:sum(r[key] for r in rows) for key in count_keys},
                       **{key:sum(r[key]*r["observations"] for r in rows)/count for key in mean_keys}}
    cells_path=run/"all_fold_view_role_modality_strata.csv"
    identities_path=run/"all_source_identity_role_modality.csv"
    write_csv(cells_path,cells);write_csv(identities_path,identities)
    result={"status":"PASS_COMPLETE_SOURCE_JOINT_SCALE_ARRAYS_AND_GEOMETRY",
            "verified_at":datetime.now().astimezone().isoformat(),
            "execution_commit":summary["execution_commit"],"source_summary_sha256":sha(summary_path),
            "verifier_sha256":sha(__file__),"contract_sha256":args.contract_sha256,
            "checked_batches":checked_batches,"checked_slots":checked_slots,
            "checked_scalar_values":checked_slots*16,"registered_active_batches":active_batches,
            "maximum_cpu_gpu_scaled_error":maximum_error,"fold_proofs":fold_proofs,
            "overall_by_view":overall,"all_cells":cells,"all_identity_rows":identities,
            "cells_csv_sha256":sha(cells_path),"identity_csv_sha256":sha(identities_path),
            "new_model_forwards":0,"new_optimizer_updates":0,"image_reads":0,
            "raw_vectors_retained":False,"independent_external_review":False,
            "boundary":"All raw h/c/y vectors were checked on CPU against GPU in the original diagnostic forward, then discarded. This verifier recomputes all persisted scalar statistics, not raw vector dot products or images.",
            "elapsed_seconds":time.time()-started}
    output=run/"complete_source_scale_verification.json"
    assert not output.exists()
    output.write_bytes((json.dumps(result,indent=2)+chr(10)).encode())
    lines=["# V28固定来源联合修正尺度：完整诊断","",
           "全部3fold、1680batch、3360固定模型前向、1935360槽位观测已完整核验。优化更新0，未执行检索或官方评估。",
           "两个输入条件共享每个实际batch；原输入仍含原SharedGeometry训练增强，不能称为干净原图。",
           "全部162个fold/view/role/modality/计划分层及5076个来源身份输出行见配套CSV。","",
           "| 输入 | 观测数 | 修正/原槽位范数>1 | >10 | >100 | 更接近修正方向 | cos(y,c)>=0.99 | cos(y,h)<0 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for view,row in overall.items():
        lines.append("| "+view+" | "+" | ".join(str(row[k]) for k in
                     ("observations","ratio_gt1","ratio_gt10","ratio_gt100","closer_to_c","cos_y_c_ge_099","cos_y_h_negative"))+" |")
    lines += ["","重复批次按观测曝光统计，不当作独立身份、图片或摄像头；跨fold只汇总标量，不混合特征距离。",
              "原始向量仅在服务器内存中逐批独立核对并丢弃，保存全部FP64充分统计及GPU测量值。后验复算覆盖所有保存标量，但不重读图像或重算原始向量内积。",
              "修正尺度和方向主导是表示诊断；不构成检索因果干预，也不改变V28 Q1_FAIL及原科学条件。",
              "零修正的余弦采用现有cosine_similarity的epsilon约定为0，零范数单独计数，不能解释成真实正交。",
              f"原诊断摘要SHA256：{sha(summary_path)}。全量CPU核验SHA256：{sha(output)}。",
              "没有新的模型训练、official/dev、权重选择或外部独立审稿结论。",""]
    (run/"complete_source_scale_report.md").write_bytes((chr(10).join(lines)).encode())
    print(json.dumps({k:v for k,v in result.items() if k not in ("all_cells","all_identity_rows","fold_proofs")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    verify(parser.parse_args())
