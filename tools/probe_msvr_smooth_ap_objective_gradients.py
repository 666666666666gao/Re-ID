"""Fixed-state source gradients: fused ranking versus the other thirteen losses.

No optimizer exists in this diagnostic. All comparisons use the same parameters.
The fixed scale and restored buffers do not reconstruct the original trajectory.
"""
import argparse
from datetime import datetime
import gc
import itertools
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tools.train_msvr310_signal_oof import sha256,write_json,loader_for,records_for
from tools.train_msvr_smooth_ap import context as training_context,reload_model
from tools.msvr_freshness_probe import ViewFields
from tools.msvr_instance_memory import InstanceMemory
from tools.msvr_role_set_relations import fused_distances
from tools.msvr_smooth_ap import paired_objectives
from tools.probe_msvr_history_candidate_gradients import gradients,compare,candidate_vjp,encode_graph,SCALE
from tools.run_signal_preserving_v5 import _module_state_sha256,_set_seed,_training_batch,weighted_training_loss
from tools.train_msvr310_source_style import pixels

ROOT=Path(__file__).resolve().parents[1]
ENDS=('control','smooth_ap')
EXPERTS=('cnn','transformer','mamba')


def objective(features,history,ids,mids,endpoint):
    dc,dh=fused_distances(features,history)
    hard,smooth,ap=paired_objectives(dc,dh,ids,mids)
    return (hard if endpoint=='control' else smooth),dc,dh


def decomposition_error(first,second,total):
    summed=[a+b for a,b in zip(first,second,strict=True)]
    agreement=compare(summed,total)
    sizes=compare(first,second)
    denominator=sizes['first_norm']+sizes['second_norm']
    relative=agreement['difference_norm']/denominator if denominator else 0.
    assert relative<=.005,agreement
    return dict(**agreement,relative_to_sum_of_component_norms=relative,tolerance=.005)


def math_check():
    _set_seed(42)
    parameter=torch.randn(5,4,requires_grad=True)
    current=torch.randn(8,5)@parameter
    historical=torch.randn(4,5)@parameter
    ids=[0,0,1,1,2,2,3,3];mids=[0,1,4,5]
    reports={}
    for end in ENDS:
        h=F.normalize(historical,dim=1)
        fused,_,_=objective(current,h,ids,mids,end)
        other=(current**2).mean()
        a=gradients(fused,[parameter]);b=gradients(other,[parameter])
        total=gradients(fused+other,[parameter])
        reports[end]=decomposition_error(a,b,total)
    return dict(status='PASS_SYNTHETIC_OBJECTIVE_DECOMPOSITION',scale=SCALE,checks=reports,
                model_forwards=0,optimizer_updates=0)


def context(path):
    spec=json.loads(path.read_bytes())
    assert spec['schema']=='msvr310-smooth-ap-objective-gradient-v1'
    assert spec['seed']==42 and spec['optimizer_updates']==0 and spec['scale']==SCALE==256.
    assert spec['preflight_steps_per_end']==8 and spec['source_steps_per_end']==260
    for name,digest in spec['project_file_sha256'].items():assert sha256(ROOT/name)==digest,name
    assert sha256(spec['q1_summary'])==spec['q1_summary_sha256']
    q1=json.loads(Path(spec['q1_summary']).read_bytes())
    assert q1['status']=='Q1_FAIL' and q1['optimizer_steps']==1560
    audit=json.loads((ROOT/spec['q1_audit']).read_bytes())
    assert audit['deterministic_checks_status']=='pass'
    for fold in q1['folds']:
        for end in ENDS:
            row=fold['endpoints'][end]
            assert sha256(row['checkpoint'])==row['checkpoint_sha256']
    return spec,training_context(ROOT/spec['training_config'])[1],q1


def probe(model,records,fold,protocol,config,md,saved,endpoint,mode,directory):
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    _set_seed(42);model.train()
    fixed=_module_state_sha256(model)
    buffers={n:v.clone() for n,v in model.named_buffers()}
    selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    names=[n for n,p in selected];parameters=[p for n,p in selected]
    assert len(parameters)==189
    criterion=ExpertFormationV8Criterion(triplet_margin=.3,label_smoothing=.1).cuda()
    weight=float(config['LOSS']['TRIPLET_FUSED']);assert weight==1.
    source=[protocol['records'][i] for i in fold['source_record_indices']]
    memory=InstanceMemory(source,capacity=512,maximum_age=8);fields=ViewFields()
    lookup={Path(r[0][0]).name:i for r,i in zip(records,fold['source_record_indices'],strict=True)}
    reference_path=Path(saved['checkpoint']).parent/'memory_steps.jsonl'
    assert sha256(reference_path)==saved['training']['audit_files']['memory_steps.jsonl']['sha256']
    reference=[json.loads(line) for line in reference_path.read_text().splitlines()]
    loader=loader_for(records,True);steps=[];seen=set();direct_proof=None
    warmup=2 if mode=='preflight' else 65
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (directory/'steps.jsonl').open('x') as log,(directory/'distances.f32').open('xb') as matrices:
        for epoch in range(1,(1 if mode=='preflight' else 20)+1):
            for raw in (itertools.islice(loader,8) if mode=='preflight' else loader):
                step=len(steps);indices=[lookup[n] for n in raw[-1]];seen.update(indices)
                assert indices==md['batches'][step]['record_indices']==reference[step]['record_indices']
                batch,labels=_training_batch(raw);pixel=pixels(batch)
                assert pixel==reference[step]['pixel_sha256']
                ids=[protocol['records'][i]['identity'] for i in indices]
                stored,metadata=memory.read(step,indices,torch.empty((0,7680),device='cuda'))
                fresh,refresh_count=fields.refresh(model,metadata,stored,step)
                assert torch.equal(fresh,stored) # Same fixed parameters and original historical view.
                with torch.autocast('cuda',dtype=torch.float16):
                    output,item=fields.capture(model,batch,indices)
                    components=criterion(output,labels)
                unit=F.normalize(output.fused_embedding.float(),dim=1)
                if step==0:assert torch.equal(unit.detach(),fields.encode(model,item))
                active=step>=warmup
                mids=[r['identity'] for r in metadata]
                fused,dc,dh=objective(output.fused_embedding,fresh,ids,mids,endpoint if active else 'control')
                if not active:assert torch.equal(fused,components['triplet_fused'])
                components['triplet_fused']=fused
                total=weighted_training_loss(components,config)
                others=dict(components);others['triplet_fused']=torch.zeros_like(fused)
                other=weighted_training_loss(others,config)
                g_current=gradients(weight*fused,parameters)
                repeated=gradients(weight*fused,parameters)
                g_other=gradients(other,parameters)
                g_total=gradients(total,parameters)
                current_identity=decomposition_error(g_current,g_other,g_total)
                historical=[torch.zeros_like(p,dtype=torch.float32) for p in parameters]
                history_repeat=[torch.zeros_like(p,dtype=torch.float32) for p in parameters]
                history_count=0;groups=[];extra_direct=0
                before_cpu=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state()
                if metadata:
                    leaf=fresh.detach().requires_grad_(True)
                    partial=objective(output.fused_embedding.detach(),leaf,ids,mids,endpoint)[0]
                    assert torch.equal(partial.detach(),fused.detach())
                    upstream=torch.autograd.grad(partial*(SCALE*weight),leaf)[0].float()/SCALE
                    assert bool(torch.isfinite(upstream).all())
                    historical,history_repeat,history_count,groups=candidate_vjp(model,fields,metadata,fresh,upstream,parameters)
                    if direct_proof is None:
                        keys={r['stored_step'] for r in metadata};assert len(keys)==1
                        key=next(iter(keys));encoded=encode_graph(model,fields.fields[key])
                        positions=[fields.fields[key]['positions'][r['record_index']] for r in metadata]
                        assert torch.equal(encoded[positions].detach(),fresh)
                        direct_fused=objective(output.fused_embedding,encoded[positions],ids,mids,endpoint)[0]
                        direct=gradients(weight*direct_fused+other,parameters)
                        combined=[a+b for a,b in zip(g_total,historical,strict=True)]
                        comparison=compare(direct,combined)
                        denominator=max(comparison['first_norm'],comparison['second_norm'])
                        relative=comparison['difference_norm']/denominator if denominator else 0.
                        assert relative<=.005
                        direct_proof=dict(step=step+1,history_groups=1,comparison=comparison,relative_error=relative,tolerance=.005)
                        extra_direct=64
                        del encoded,direct_fused,direct,combined
                    del leaf,partial,upstream
                assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
                gf=[a+b for a,b in zip(g_current,historical,strict=True)]
                gt=[a+b for a,b in zip(g_total,historical,strict=True)]
                full_identity=decomposition_error(gf,g_other,gt)
                if mode=='preflight' and direct_proof and step+1==direct_proof['step']:
                    torch.save(dict(names=names,step=step+1,scale=SCALE,
                        fused=[v.cpu() for v in gf],other=[v.cpu() for v in g_other],
                        full=[v.cpu() for v in gt]),directory/'first_history_gradients.pt')
                roles={}
                for expert in EXPERTS:
                    ii=[i for i,n in enumerate(names) if n.startswith('encoder.'+expert+'_')];assert ii
                    take=lambda values:[values[i] for i in ii]
                    roles[expert]=dict(fused_vs_other=compare(take(gf),take(g_other)),
                        fused_vs_full=compare(take(gf),take(gt)),
                        current_repeat=compare(take(g_current),take(repeated)),
                        history_repeat=compare(take(historical),take(history_repeat)))
                offset=matrices.tell()
                for values in (dc,dh):matrices.write(values.detach().cpu().contiguous().numpy().tobytes())
                row=dict(step=step+1,epoch=epoch,record_indices=indices,pixel_sha256=pixel,identities=ids,
                    memory=metadata,active_fused_metric=endpoint if active else 'control',
                    components={k:float(v.detach()) for k,v in components.items()},
                    fused_loss=float(fused.detach()),other_loss=float(other.detach()),total_loss=float(total.detach()),
                    current_decomposition=current_identity,full_decomposition=full_identity,roles=roles,
                    refresh_record_forwards=refresh_count,history_vjp_record_forwards=history_count,
                    direct_check_record_forwards=extra_direct,history_groups=groups,
                    distance_offset_bytes=offset,distance_float_count=dc.numel()+dh.numel(),scale=SCALE)
                log.write(json.dumps(row)+'\n');log.flush();steps.append(row)
                if active:memory.update(step,indices,unit);fields.fields[step]=item
                for name,value in model.named_buffers():value.copy_(buffers[name])
                assert all(p.grad is None for p in model.parameters())
                del output,components,others,unit,fused,other,total,dc,dh,g_current,repeated,g_other,g_total,historical,history_repeat,gf,gt,item
            print(json.dumps(dict(event='objective_gradient_epoch',endpoint=endpoint,epoch=epoch,
                batches=len(steps),elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(steps)==(8 if mode=='preflight' else 260) and direct_proof is not None
    assert fixed==_module_state_sha256(model)
    if mode=='source':assert seen==set(fold['source_record_indices'])
    receipt=dict(status='PASS_FIXED_STATE_OBJECTIVE_GRADIENTS',steps=len(steps),model_state_sha256=fixed,
        model_state_unchanged=True,gradients_absent=True,optimizer_updates=0,heldout_record_forwards=0,
        official_image_reads=0,observed_records=sorted(seen),parameters=names,scale=SCALE,
        direct_history_proof=direct_proof,elapsed_seconds=time.perf_counter()-started,
        extra_record_forwards=sum(r['refresh_record_forwards']+r['history_vjp_record_forwards']+r['direct_check_record_forwards'] for r in steps)+64,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in directory.iterdir()},
        parameter_gradient_scope='runtime full-vector calculations; scalar norms and dots saved, full per-step vectors not persisted')
    write_json(directory/'receipt.json',receipt);return receipt


def run(args):
    spec,(config,base,cfg,environment,protocol,baseline,meta),q1=context(args.contract)
    torch.set_num_threads(4)
    if args.mode=='source':
        previous=json.loads(args.preflight.read_bytes())
        assert previous['status']=='PASS_PREFLIGHT' and previous['contract_sha256']==sha256(args.contract)
    args.root.mkdir()
    summary=dict(status='RUNNING',contract_sha256=sha256(args.contract),mode=args.mode,conditions=[],
        started_at=datetime.now().astimezone().isoformat(),math=math_check(),optimizer_updates=0,environment=environment)
    write_json(args.root/'summary.json',summary)
    for fold,b0,md,saved in zip(protocol['folds'],baseline['folds'],meta['folds'],q1['folds'],strict=True):
        records=records_for(base,protocol,fold,True)
        for end in ENDS:
            row=saved['endpoints'][end]
            model=reload_model(config,cfg,fold,b0,row['checkpoint'],row['initialization'],row['training']['final_state_sha256'],sha256(ROOT/spec['training_config']))
            directory=args.root/f'fold_{fold["fold"]}_{end}';directory.mkdir()
            receipt=probe(model,records,fold,protocol,config,md,row,end,args.mode,directory)
            summary['conditions'].append(dict(directory=directory.name,**receipt));write_json(args.root/'summary.json',summary)
            del model;gc.collect();torch.cuda.empty_cache()
    assert len(summary['conditions'])==6
    context(args.contract)
    summary.update(status='PASS_PREFLIGHT' if args.mode=='preflight' else 'COMPLETE_SOURCE_OBJECTIVE_GRADIENTS',
        completed_at=datetime.now().astimezone().isoformat())
    write_json(args.root/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--mode',choices=('preflight','source'),required=True)
    parser.add_argument('--preflight',type=Path)
    run(parser.parse_args())
