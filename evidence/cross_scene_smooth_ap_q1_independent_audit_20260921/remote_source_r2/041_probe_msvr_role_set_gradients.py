"""Three-source-fold, fixed-initial-state checks of role-set parameter gradients.

Eight source batches per fold, no optimizer or checkpoint writes. This is an
engineering probe, not a full source census or a retrieval experiment.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from tools.train_msvr_history_gradient import context as previous_context
from tools.msvr_freshness_probe import ViewFields
from tools.msvr_instance_memory import InstanceMemory
from tools.msvr_role_set_relations import relation_objectives, fused_distances
from tools.probe_msvr_history_candidate_gradients import gradients, compare, encode_graph
from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch
from tools.train_msvr310_signal_oof import loader_for, records_for, sha256, write_json
from tools.train_msvr310_trifusion_oof import build_model, EXPERTS

ROOT=Path(__file__).resolve().parents[1]


def encode_all(model,item):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.set_rng_state(item['cpu_rng']);torch.cuda.set_rng_state(item['cuda_rng'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            representations=model.encoder(item['anchor'].cuda(),item['reference'].cuda())
            fusion=model.fusion(item['baseline'].cuda(),representations)
            values={'fused':fusion.fused_embedding,**fusion.branch_embeddings}
            return {k:F.normalize(v.float(),dim=1) for k,v in values.items()}


def refresh_all(model,fields,metadata,step):
    for key in [k for k in fields.fields if step-k>8]:del fields.fields[key]
    before={n:v.clone() for n,v in model.named_buffers()}
    cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
    groups=sorted({r['stored_step'] for r in metadata})
    encoded={k:encode_all(model,fields.fields[k]) for k in groups}
    values={name:torch.stack([encoded[r['stored_step']][name][fields.fields[r['stored_step']]['positions'][r['record_index']]] for r in metadata])
            if metadata else torch.empty((0,7680 if name=='fused' else 4608),device='cuda')
            for name in ('fused',*EXPERTS)}
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(v,before[n]) for n,v in model.named_buffers())
    return values,64*len(groups)


def historical_vjps(model,fields,metadata,fresh,upstreams,parameters):
    totals=[[torch.zeros_like(p,dtype=torch.float32) for p in parameters] for _ in upstreams]
    groups=[]
    for key in sorted({r['stored_step'] for r in metadata}):
        rows=[i for i,r in enumerate(metadata) if r['stored_step']==key]
        if not any(bool(u[rows].abs().sum()>0) for u in upstreams):continue
        item=fields.fields[key];unit=encode_graph(model,item)
        positions=[item['positions'][metadata[i]['record_index']] for i in rows]
        assert torch.equal(unit[positions].detach(),fresh[rows])
        for target,u in zip(totals,upstreams,strict=True):
            coefficients=torch.zeros_like(unit);coefficients[positions]=u[rows]
            values=gradients((unit*coefficients.detach()).sum(),parameters)
            for out,value in zip(target,values,strict=True):out.add_(value)
        groups.append(key)
        del unit,values,coefficients
    return totals,groups


def probe(model,records,fold,protocol,md,directory):
    _set_seed(42);model.train()
    before=_module_state_sha256(model)
    buffers={n:v.clone() for n,v in model.named_buffers()}
    selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    names,parameters=zip(*selected)
    role_indexes={e:[i for i,n in enumerate(names) if n.startswith('encoder.'+e+'_')] for e in EXPERTS}
    source=[protocol['records'][i] for i in fold['source_record_indices']]
    memory=InstanceMemory(source,capacity=512,maximum_age=8);fields=ViewFields()
    lookup={Path(r[0][0]).name:i for r,i in zip(records,fold['source_record_indices'],strict=True)}
    rows=[];seen=set();fresh_count=history_count=direct_count=0;direct_proof=None
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (directory/'steps.jsonl').open('x') as log,(directory/'distances.f32').open('xb') as arrays:
        for step,raw in enumerate(itertools.islice(loader_for(records,True),8)):
            indices=[lookup[n] for n in raw[-1]];seen.update(indices)
            assert indices==md['batches'][step]['record_indices']
            batch,_labels=_training_batch(raw)
            ids=[protocol['records'][i]['identity'] for i in indices]
            stale,metadata=memory.read(step,indices,torch.empty((0,7680),device='cuda'))
            fresh,count=refresh_all(model,fields,metadata,step);fresh_count+=count
            with torch.autocast('cuda',dtype=torch.float16):output,item=fields.capture(model,batch,indices)
            unit=F.normalize(output.fused_embedding.float(),dim=1)
            if step==0:
                reproduced=encode_all(model,item);fresh_count+=64
                assert torch.equal(unit.detach(),reproduced['fused'])
                assert all(torch.equal(F.normalize(output.branch_embeddings[e].float(),dim=1),reproduced[e]) for e in EXPERTS)
                del reproduced
            mids=[r['identity'] for r in metadata]
            role_distances=[]
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                for e in EXPERTS:
                    value=F.normalize(output.branch_embeddings[e].float(),dim=1)
                    role_distances.append(torch.cdist(value,torch.cat((value,fresh[e]),dim=0)))
            leaf=fresh['fused'].detach().requires_grad_(True)
            dc,dh=fused_distances(output.fused_embedding,leaf)
            hard,candidate,selection=relation_objectives(dc,dh,ids,mids,role_distances)
            current_hard=gradients(hard,parameters)
            current_candidate=gradients(candidate,parameters)
            repeated=gradients(candidate,parameters)
            upstream=([torch.autograd.grad(loss*256.,leaf,retain_graph=True)[0].float()/256.
                       for loss in (hard,candidate,candidate)] if metadata
                      else [torch.zeros_like(leaf) for _ in range(3)])
            cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
            hparts,groups=historical_vjps(model,fields,metadata,fresh['fused'],upstream,parameters)
            history_count+=64*len(groups)
            full_hard=[a+b for a,b in zip(current_hard,hparts[0],strict=True)]
            full_candidate=[a+b for a,b in zip(current_candidate,hparts[1],strict=True)]
            full_repeated=[a+b for a,b in zip(repeated,hparts[2],strict=True)]
            if metadata and direct_proof is None:
                keys={r['stored_step'] for r in metadata};assert len(keys)==1
                old_item=fields.fields[next(iter(keys))]
                encoded=encode_graph(model,old_item)
                positions=[old_item['positions'][r['record_index']] for r in metadata]
                assert torch.equal(encoded[positions].detach(),fresh['fused'])
                ec,eh=fused_distances(output.fused_embedding,encoded[positions])
                _,full_loss,_=relation_objectives(ec,eh,ids,mids,role_distances)
                assert torch.equal(full_loss,candidate)
                direct=gradients(full_loss,parameters)
                check=compare(direct,full_candidate)
                denominator=max(check['first_norm'],check['second_norm'])
                relative=check['difference_norm']/denominator if denominator>0 else 0.
                assert relative<=.005
                direct_proof=dict(step=step+1,comparison=check,relative_l2_error=relative,tolerance=.005)
                direct_count+=64
                del encoded,ec,eh,full_loss,direct
            assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
            metrics={e:dict(hard_vs_role_set=compare([full_hard[i] for i in ii],[full_candidate[i] for i in ii]),
                            role_set_repeat_noise=compare([full_candidate[i] for i in ii],[full_repeated[i] for i in ii]))
                     for e,ii in role_indexes.items()}
            offset=arrays.tell()
            for value in (torch.cat((dc,dh),dim=1),*role_distances):
                arrays.write(value.detach().cpu().contiguous().numpy().tobytes())
            row=dict(step=step+1,record_indices=indices,identities=ids,memory=metadata,
                     hard_loss=float(hard.detach()),candidate_loss=float(candidate.detach()),
                     proposals=selection['proposals'].cpu().tolist(),negative_counts=selection['counts'].cpu().tolist(),
                     extra_active_counts=selection['extra_active'].sum(1).cpu().tolist(),roles=metrics,
                     distance_offset_bytes=offset,distance_float_count=4*64*(64+len(metadata)),
                     fresh_role_record_forwards=count,history_vjp_groups=groups,
                     history_role_record_forwards=64*len(groups),optimizer_updates=0,
                     history_age_semantics='batch_recency_not_parameter_updates')
            log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
            if step>=2:memory.update(step,indices,unit.detach());fields.fields[step]=item
            for n,v in model.named_buffers():v.copy_(buffers[n])
            assert all(p.grad is None for p in model.parameters())
            del output,unit,item,stale,fresh,leaf,dc,dh,hard,candidate,selection,role_distances
            del current_hard,current_candidate,repeated,upstream,hparts,full_hard,full_candidate,full_repeated
            print(json.dumps(dict(fold=fold['fold'],step=step+1,elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(rows)==8 and direct_proof is not None
    after=_module_state_sha256(model);assert after==before
    support={e:sum(r['roles'][e]['hard_vs_role_set']['difference_norm']>
                   r['roles'][e]['role_set_repeat_noise']['difference_norm'] for r in rows) for e in EXPERTS}
    history_support={e:sum(bool(r['memory']) and r['roles'][e]['hard_vs_role_set']['difference_norm']>
                           r['roles'][e]['role_set_repeat_noise']['difference_norm'] for r in rows) for e in EXPERTS}
    report=dict(status='COMPLETE_FIXED_INITIAL_GRADIENT_CHECK',fold=fold['fold'],batches=8,
                unique_source_records=sorted(seen),initial_state_sha256=before,final_state_sha256=after,
                direct_history_chain_rule=direct_proof,changed_gradient_batches_above_repeat_noise=support,
                changed_gradient_history_batches_above_repeat_noise=history_support,
                active_extra_negative_exposures=sum(sum(r['extra_active_counts']) for r in rows),
                current_role_record_forwards=512,extra_fresh_role_record_forwards=fresh_count,
                history_role_record_forwards=history_count,extra_direct_role_record_forwards=direct_count,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                elapsed_seconds=time.perf_counter()-started,optimizer_updates=0,checkpoint_writes=0,
                files={n:dict(bytes=(directory/n).stat().st_size,sha256=sha256(directory/n)) for n in ('steps.jsonl','distances.f32')})
    write_json(directory/'receipt.json',report)
    return report


def run(args):
    torch.set_num_threads(4)
    spec=json.loads(args.config.read_bytes())
    assert spec['schema']=='msvr310-role-set-gradient-check-v1' and spec['seed']==42
    for name,digest in spec['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    math=json.loads(Path(spec['math_receipt']).read_bytes())
    assert sha256(spec['math_receipt'])==spec['math_sha256'] and math['status']=='PASS_ROLE_SET_MATHEMATICS'
    _,(config,base,cfg,environment,protocol,baseline,meta)=previous_context(ROOT/spec['previous_config'])
    assert sha256(spec['protocol'])==spec['protocol_sha256']
    assert protocol==json.loads(Path(spec['protocol']).read_bytes())
    args.output_dir.mkdir()
    result=dict(status='RUNNING',seed=42,config_sha256=sha256(args.config),
                project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                environment=environment,folds=[],optimizer_updates=0,heldout_record_forwards=0,official_image_reads=0,
                started_at=datetime.now().astimezone().isoformat())
    write_json(args.output_dir/'summary.json',result)
    for fold,b0,md in zip(protocol['folds'],baseline['folds'],meta['folds'],strict=True):
        records=records_for(base,protocol,fold,True);model,binding=build_model(config,cfg,fold,b0)
        directory=args.output_dir/f"fold_{fold['fold']}";directory.mkdir()
        report=probe(model,records,fold,protocol,md,directory)
        result['folds'].append(dict(initialization=binding,**report))
        write_json(args.output_dir/'summary.json',result)
        del model;torch.cuda.empty_cache()
    result.update(status='COMPLETE_ROLE_SET_PARAMETER_GRADIENT_CHECK',completed_at=datetime.now().astimezone().isoformat(),
                  scope='Three fixed initial states, 8 source batches per fold. No training updates or retrieval result.')
    write_json(args.output_dir/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())
