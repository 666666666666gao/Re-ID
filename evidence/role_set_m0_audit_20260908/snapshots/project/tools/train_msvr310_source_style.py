#!/usr/bin/env python3
"""Fixed paired MSVR310 source-style comparison using the original role trainer."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from tools.train_msvr310_signal_oof import configure, loader_for, new_model, records_for, sha256, write_json
from tools.train_msvr310_trifusion_oof import (
    build_model as original_model, train_roles, engineering_checks, output_mapping,
    evaluate, comparison_summary, OUTPUT_WIDTHS, EXPERTS,
)

ROOT=Path(__file__).resolve().parents[1]
ENDPOINTS=('control','source_style')


def context(path):
    config=json.loads(path.read_bytes())
    assert config['EXPERIMENT']['SEED']==42
    assert config['MODEL']['GRID_SIZE']==[8,16]
    assert config['OPTIMIZATION']['MAX_EPOCHS']==20
    assert config['STYLE']==dict(PROBABILITY=.5,BETA_ALPHA=.1,VARIANCE_EPSILON=1e-6,
                                 CROSS_CAMERA_DONOR=True,SHARED_MODAL_PLAN=True)
    for name,expected in config['project_source_file_sha256'].items():
        assert sha256(ROOT/name)==expected,name
    assert sha256(config['BASELINE']['SUMMARY'])==config['BASELINE']['SUMMARY_SHA256']
    base_path=ROOT/config['BASELINE']['CONFIG']
    assert sha256(base_path)==config['BASELINE']['CONFIG_SHA256']
    base=json.loads(base_path.read_bytes())
    cfg,binding=configure(base)
    assert sha256(ROOT/base['protocol'])==base['protocol_sha256']
    protocol=json.loads((ROOT/base['protocol']).read_bytes())
    baseline=json.loads(Path(config['BASELINE']['SUMMARY']).read_bytes())
    meta_path=ROOT/config['SOURCE_METADATA']['PATH']
    assert sha256(meta_path)==config['SOURCE_METADATA']['SHA256']
    meta=json.loads(meta_path.read_bytes())
    assert meta['status']=='PASS_EXISTING_MSVR310_SOURCE_EXPOSURE_AND_CAMERA_DONOR_SUPPORT'
    for fold,b0,md in zip(protocol['folds'],baseline['folds'],meta['folds'],strict=True):
        assert fold['fold']==b0['fold']==md['fold']
        assert not set(fold['source_ids']) & set(fold['heldout_ids'])
        assert sha256(b0['checkpoint'])==b0['checkpoint_sha256']
        assert sha256(Path(b0['checkpoint']).parent/'retrieval_arrays.pt')==b0['retrieval']['retrieval_arrays_sha256']
    return config,base,cfg,binding,protocol,baseline,meta


def build(config,cfg,fold,b0):
    from tools.run_signal_preserving_v5 import _module_state_sha256
    from trifusion.source_style_msvr import SourceStyleMSVRBackbone
    model,binding=original_model(config,cfg,fold,b0)
    before=_module_state_sha256(model)
    model.baseline=SourceStyleMSVRBackbone(model.baseline.signal,fold=fold['fold'])
    assert _module_state_sha256(model)==before==binding['initial_state_sha256']
    return model,binding


def pixels(batch):
    return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name,value in batch['images'].items()}


def fit(model,records,fold,config,metadata,*,endpoint,mode,directory):
    from trifusion.source_style_v27 import make_style_plan
    enabled=endpoint=='source_style'
    styles=[]
    pending={}
    path=directory/'style_steps.jsonl'
    def before(_model,args):
        batch=args[0]
        index=0 if mode=='overfit' else len(styles)
        plan=make_style_plan(batch['camera_ids'].detach().cpu().numpy(),fold=fold['fold'],
                             step=index,force_active=mode=='overfit')
        if mode!='overfit':
            assert plan==metadata['batches'][index]['style_plan']
        model.baseline.style_enabled=enabled
        model.baseline.style_plan=plan
        pending.update(plan=plan,pixel_sha256=pixels(batch))
    with path.open('x',encoding='utf-8') as stream:
        def after(_model,_args,_output):
            row=dict(step=len(styles)+1,plan=pending['plan'],pixel_sha256=pending['pixel_sha256'],
                     statistics=dict(model.baseline.last_style_stats))
            assert row['statistics']['additional_visual_passes']==3
            assert row['statistics']['style_active']==int(enabled and row['plan']['active'])
            styles.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
        with model.register_forward_pre_hook(before),model.register_forward_hook(after):
            training=train_roles(model,records,fold['source_record_indices'],config,fold=fold['fold'],
                                 mode=mode,directory=directory)
    assert len(styles)==training['optimizer_steps']
    for index,row in enumerate(training['steps']):
        expected=metadata['batches'][0 if mode=='overfit' else index]
        assert row['sampled_record_indices']==expected['record_indices']
    if mode=='comparison':
        assert len(styles)==260
    if mode=='overfit':
        assert all(r['pixel_sha256']==styles[0]['pixel_sha256'] and r['plan']==styles[0]['plan'] for r in styles)
    training.update(style_steps_path=str(path),style_steps_sha256=sha256(path),
                    active_style_steps=sum(r['statistics']['style_active'] for r in styles),
                    additional_visual_passes=3*len(styles),all_registered_sample_orders_match=True)
    write_json(directory/'training.json',training)
    model.baseline.style_plan=None
    return training,styles


def extract(model,records):
    import torch
    from tools.run_signal_preserving_v5 import _training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    model.eval();model.baseline.style_plan=None
    parts={name:[] for name in OUTPUT_WIDTHS}
    for raw in loader_for(records,False):
        batch,_=_training_batch(raw)
        values=output_mapping(exact_signal_forward(model,batch))
        for name,value in values.items():parts[name].append(value.float().cpu())
    return {name:torch.cat(values) for name,values in parts.items()}


def preflight(model,records,cfg,fold,b0):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256,_training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from trifusion.signal_preserving_v8 import HierarchicalFrozenSignalBackbone
    from trifusion.source_style_v27 import make_style_plan
    model.eval()
    before=_module_state_sha256(model)
    raw=next(iter(loader_for(records[:8],False)))
    batch,_=_training_batch(raw)
    direct=new_model(cfg,fold)
    payload=torch.load(b0['checkpoint'],map_location='cpu',weights_only=True)
    direct.load_state_dict(payload['model_state_dict'],strict=True);direct.eval()
    with torch.no_grad():
        target=direct(batch['images'],cam_label=batch['camera_ids'],view_label=raw[3].cuda(),
                      training=False,sge=cfg.MODEL.stageName)
        output=exact_signal_forward(model,batch)
        output_mapping(output)
        assert torch.equal(target,output.baseline_embedding)
        del direct,payload,target,output
        original=HierarchicalFrozenSignalBackbone.forward(model.baseline,batch)
        plan=make_style_plan(raw[2].numpy(),fold=fold['fold'],step=0,force_active=True)
        model.baseline.train(True)
        rows=[]
        for enabled,active in ((False,True),(True,True),(True,False)):
            model.baseline.style_enabled=enabled
            model.baseline.style_plan={**plan,'active':active}
            actual=model.baseline(batch)
            assert torch.equal(original.baseline_embedding,actual.baseline_embedding)
            assert torch.equal(original.direct_modal,actual.direct_modal)
            assert tuple(actual.anchor_sequence.shape)==(8,3,129,768)
            assert tuple(actual.reference_sequence.shape)==(8,3,129,768)
            rows.append(dict(enabled=enabled,active=active,**model.baseline.last_style_stats))
    model.eval();model.baseline.style_plan=None
    assert _module_state_sha256(model)==before
    return dict(standalone_source_signal_bitwise_equal=True,original_state_unchanged=True,
                source_records=8,heldout_image_reads=0,field_cases=rows,pixel_sha256=pixels(batch))


def checkpoint(model,binding,fold,config_hash,path):
    import torch
    state=model.state_dict()
    source=model.baseline.signal.state_dict()
    aliases={(t.data_ptr(),tuple(t.shape),str(t.dtype)):k for k,t in source.items()}
    mapping={k:aliases[t.data_ptr(),tuple(t.shape),str(t.dtype)] for k,t in state.items() if k.startswith('baseline.')}
    assert not path.exists()
    torch.save(dict(role_state_dict={k:t.detach().cpu() for k,t in state.items() if not k.startswith('baseline.')},
                    baseline_aliases=mapping,binding=binding,fold=fold['fold'],source_ids=fold['source_ids'],
                    heldout_ids=fold['heldout_ids'],config_sha256=config_hash),path)
    return sha256(path)


def reload_model(config,cfg,fold,b0,path,binding,final_state,config_hash):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256
    model,actual=build(config,cfg,fold,b0)
    assert actual==binding
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['config_sha256']==config_hash
    assert payload['binding']==binding and payload['fold']==fold['fold']
    assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids']
    state=model.state_dict()
    assert set(payload['role_state_dict'])=={k for k in state if not k.startswith('baseline.')}
    assert set(payload['baseline_aliases'])=={k for k in state if k.startswith('baseline.')}
    source=model.baseline.signal.state_dict()
    assert all(torch.equal(state[k],source[v]) for k,v in payload['baseline_aliases'].items())
    state.update(payload['role_state_dict']);model.load_state_dict(state,strict=True)
    assert _module_state_sha256(model)==final_state
    return model


def paired_summary(folds):
    import torch
    from trifusion.signal_preserving_v13 import identity_cluster_bootstrap_lower_bound
    metrics={end:comparison_summary([f['endpoints'][end] for f in folds]) for end in ENDPOINTS}
    identities=np.array([q['identity'] for f in folds for q in f['endpoints']['control']['retrieval']['query_rows']])
    assert len(identities)==600 and len(np.unique(identities))==60
    aps={end:{name:np.array([v for f in folds for v in f['endpoints'][end]['retrieval']['outputs'][name]['average_precision']])
              for name in OUTPUT_WIDTHS} for end in ENDPOINTS}
    gains={name:float((aps['source_style'][name]-aps['control'][name]).mean()*100) for name in OUTPUT_WIDTHS}
    difference=(aps['source_style']['fused']-aps['control']['fused'])*100
    boot=identity_cluster_bootstrap_lower_bound(torch.from_numpy(difference),torch.from_numpy(identities),seed=42,resamples=10000)
    fold_gains=[f['endpoints']['source_style']['retrieval']['outputs']['fused']['metrics']['mAP']
                -f['endpoints']['control']['retrieval']['outputs']['fused']['metrics']['mAP'] for f in folds]
    candidate=metrics['source_style']['metrics']
    checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,
                all_fold_fused_nonnegative=all(g>=0 for g in fold_gains),
                all_role_gains_nonnegative=all(gains[e]>=0 for e in EXPERTS),
                paired_identity_bootstrap_lower_positive=boot.lower_bound>0,
                candidate_fused_strictly_best=all(candidate['fused']['mAP']>candidate[e]['mAP'] for e in ('baseline_only',*EXPERTS)))
    return dict(endpoints=metrics,matched_gains_mAP=gains,fold_fused_gains_mAP=fold_gains,
                paired_bootstrap_lower_pp=boot.lower_bound,paired_checks=checks,
                paired_pass=all(checks.values()),vehicle_baseline_pass=metrics['source_style']['scientific_passed'],
                next_phase_qualified=all(checks.values()) and metrics['source_style']['scientific_passed'],
                paired_per_identity=[dict(identity=int(i),query_count=int((identities==i).sum()),
                                          gains_mAP={name:float((aps['source_style'][name][identities==i]-aps['control'][name][identities==i]).mean()*100)
                                                      for name in OUTPUT_WIDTHS}) for i in np.unique(identities)])


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256,evaluate_overfit_gate,overfit_loss_floor
    started=time.time()
    config,base,cfg,environment,protocol,baseline,meta=context(args.config)
    assert not args.output_dir.exists();args.output_dir.mkdir()
    m0=args.mode=='m0'
    old=None if m0 else json.loads(args.m0_receipt.read_bytes())
    if not m0:
        assert old['status']=='PASS_ENGINEERING_ONLY' and old['config_sha256']==sha256(args.config)
        proof=json.loads(args.m0_verification.read_bytes())
        assert proof['status']=='PASS_COMPLETE_MSVR_STYLE_M0'
        assert proof['summary_sha256']==sha256(args.m0_receipt)
    summary=dict(status='RUNNING',mode=args.mode,seed=42,config_sha256=sha256(args.config),
                 project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 runner_sha256=sha256(__file__),environment=environment,folds=[],overfit={},
                 official_image_reads=0,rgbnt201_dev_image_reads=0,heldout_record_forwards=0,
                 m0_weights_reused=False,baseline_retrained=False)
    def save():
        summary['elapsed_seconds']=time.time()-started
        write_json(args.output_dir/'summary.json',summary)
    save()
    for fold,b0,md in zip(protocol['folds'],baseline['folds'],meta['folds'],strict=True):
        records=records_for(base,protocol,fold,True)
        result=dict(fold=fold['fold'],endpoints={})
        summary['folds'].append(result)
        paired_styles=[]
        for end in ENDPOINTS:
            directory=args.output_dir/f"fold_{fold['fold']}_{end}"
            directory.mkdir()
            model,binding=build(config,cfg,fold,b0)
            if not m0:assert binding==old['folds'][fold['fold']]['endpoints'][end]['initialization']
            check=preflight(model,records,cfg,fold,b0) if m0 else None
            training,styles=fit(model,records,fold,config,md,endpoint=end,
                                mode='capacity' if m0 else 'comparison',directory=directory)
            checks=engineering_checks(training)
            checks['fixed_training_length']=training['optimizer_steps']==(8 if m0 else 260)
            row=dict(fold=fold['fold'],initialization=binding,preflight=check,training=training,engineering_checks=checks)
            result['endpoints'][end]=row;save()
            assert all(checks.values()),checks
            path=directory/('roles_m0.pth' if m0 else 'roles_epoch20.pth')
            cksha=checkpoint(model,binding,fold,sha256(args.config),path)
            before=extract(model,records[:8]) if m0 else None
            del model;torch.cuda.empty_cache()
            model=reload_model(config,cfg,fold,b0,path,binding,training['final_state_sha256'],sha256(args.config))
            row.update(checkpoint=str(path),checkpoint_sha256=cksha,strict_reload_state_sha256=_module_state_sha256(model))
            if m0:
                after=extract(model,records[:8])
                assert all(torch.equal(before[k],after[k]) for k in OUTPUT_WIDTHS)
                row['strict_reload_all_outputs_bitwise_equal']=True
            else:
                gallery=records_for(base,protocol,fold,False)
                features=extract(model,gallery)
                row['retrieval']=evaluate(features,protocol,fold,directory,b0)
                summary['heldout_record_forwards']+=len(gallery)
            assert _module_state_sha256(model)==training['final_state_sha256']
            assert sha256(path)==cksha
            paired_styles.append([{k:r[k] for k in ('plan','pixel_sha256')} for r in styles])
            write_json(directory/'receipt.json',row);save()
            del model;torch.cuda.empty_cache()
        assert paired_styles[0]==paired_styles[1]
        assert result['endpoints']['control']['initialization']==result['endpoints']['source_style']['initialization']
        result['all_paired_inputs_and_style_plans_exact']=True;save()
    if m0:
        fold=protocol['folds'][0]
        records=records_for(base,protocol,fold,True)
        paired=[]
        for end in ENDPOINTS:
            directory=args.output_dir/('overfit_'+end);directory.mkdir()
            model,binding=build(config,cfg,fold,baseline['folds'][0])
            training,styles=fit(model,records,fold,config,meta['folds'][0],endpoint=end,mode='overfit',directory=directory)
            gate=evaluate_overfit_gate([r['loss'] for r in training['steps']],max_ratio=.1,
                                       minimum_loss=overfit_loss_floor(config,num_classes=len(fold['source_ids'])))
            checks=engineering_checks(training)
            checks.update(fixed_100_steps=training['optimizer_steps']==100,overfit_excess_ratio_at_most_point1=gate['passed'])
            summary['overfit'][end]=dict(initialization=binding,training=training,gate=gate,checks=checks)
            save();assert all(checks.values()),checks
            paired.append([{k:r[k] for k in ('plan','pixel_sha256')} for r in styles])
            del model;torch.cuda.empty_cache()
        assert paired[0]==paired[1]
        summary.update(status='PASS_ENGINEERING_ONLY',optimizer_steps=248,paired_overfit_inputs_exact=True)
    else:
        result=paired_summary(summary['folds'])
        summary.update(status='Q1_PASS' if result['next_phase_qualified'] else 'Q1_FAIL',comparison=result,
                       optimizer_steps=1560,m0_receipt_sha256=sha256(args.m0_receipt),
                       m0_verification_sha256=sha256(args.m0_verification))
        assert summary['heldout_record_forwards']==2064
    summary['completed_at']=datetime.now().astimezone().isoformat();save()
    print(json.dumps(dict(status=summary['status'],mode=args.mode,elapsed_seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('m0','q1'),required=True)
    p.add_argument('--m0-receipt',type=Path)
    p.add_argument('--m0-verification',type=Path)
    args=p.parse_args()
    assert args.mode=='m0' or (args.m0_receipt is not None and args.m0_verification is not None)
    args.config=args.config.resolve();args.output_dir=args.output_dir.resolve()
    run(args)
