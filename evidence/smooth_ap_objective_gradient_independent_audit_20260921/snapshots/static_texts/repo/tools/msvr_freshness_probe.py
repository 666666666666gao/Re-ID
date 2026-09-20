"""Direct stale/current-coordinate diagnostics on source-only training paths."""
import argparse
from collections import OrderedDict
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools import train_msvr_instance_memory as old
from tools.msvr_instance_memory import InstanceMemory, expanded_triplet
from tools.train_msvr310_trifusion_oof import build_model, frozen_state_sha, EXPERTS
from tools.train_msvr310_signal_oof import loader_for, records_for, sha256, write_json
from tools.run_signal_preserving_v5 import (
    _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier,
    weighted_training_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def context(path):
    spec = json.loads(path.read_bytes())
    for name, digest in spec['project_files'].items():
        assert sha256(ROOT/name) == digest, name
    for name, digest in spec['fixed_files'].items():
        assert sha256(name) == digest, name
    assert spec['seed'] == 42 and spec['source_steps_per_end'] == 260
    assert spec['preflight_steps_per_end'] == 12
    previous = json.loads(Path(spec['old_summary']).read_bytes())
    cpu = json.loads(Path(spec['old_cpu']).read_bytes())
    assert previous['status'] == 'Q1_FAIL'
    assert cpu['status'] == 'PASS_COMPLETE_INSTANCE_MEMORY_Q1'
    assert cpu['summary_sha256'] == sha256(spec['old_summary'])
    return spec, old.context(ROOT/spec['memory_config']), previous


class ViewFields:
    """Keep the original frozen field and role RNG for at most eight updates."""
    def __init__(self):
        self.fields = {}

    def capture(self, model, batch, indices):
        holder = {}
        def baseline_hook(_module, _args, field):
            holder['anchor'] = field.anchor_sequence.detach().cpu().clone()
            holder['reference'] = field.reference_sequence.detach().cpu().clone()
            holder['baseline'] = field.baseline_embedding.detach().cpu().clone()
        def encoder_hook(_module, _args):
            holder['cpu_rng'] = torch.get_rng_state()
            holder['cuda_rng'] = torch.cuda.get_rng_state()
        with model.baseline.register_forward_hook(baseline_hook), model.encoder.register_forward_pre_hook(encoder_hook):
            output = model(batch, return_aux=True)
        holder['positions'] = {int(index): i for i, index in enumerate(indices)}
        return output, holder

    @staticmethod
    def encode(model, item):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(item['cpu_rng'])
            torch.cuda.set_rng_state(item['cuda_rng'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                representations = model.encoder(item['anchor'].cuda(), item['reference'].cuda())
                fusion = model.fusion(item['baseline'].cuda(), representations)
                unit = F.normalize(fusion.fused_embedding.float(), dim=1)
        return unit.detach()

    def refresh(self, model, metadata, prototype, step):
        for key in [k for k in self.fields if step-k > 8]:
            del self.fields[key]
        groups = sorted(set(row['stored_step'] for row in metadata))
        before = {n: v.clone() for n, v in model.named_buffers()}
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        encoded = {k: self.encode(model, self.fields[k]) for k in groups}
        assert all(torch.equal(v, before[n]) for n, v in model.named_buffers())
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        values = [encoded[r['stored_step']][self.fields[r['stored_step']]['positions'][r['record_index']]] for r in metadata]
        result = torch.stack(values) if values else prototype.new_empty((0, 7680))
        assert not result.requires_grad and torch.isfinite(result).all()
        return result, 64*len(groups)


def grad_comparison(a, b):
    x = torch.cat([v.float().reshape(-1) for v in a])
    y = torch.cat([v.float().reshape(-1) for v in b])
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    nx, ny = float(x.norm()), float(y.norm())
    return dict(stale_norm=nx, comparison_norm=ny, difference_norm=float((x-y).norm()),
                cosine=float(torch.dot(x, y)/(x.norm()*y.norm())) if nx > 0 and ny > 0 else None)


def parameter_probe(model, stale, fresh, scale):
    selected = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    def gradient(loss):
        values = torch.autograd.grad(loss*scale, [p for _,p in selected], retain_graph=True, allow_unused=True)
        return {name: value.float()/scale for (name,_), value in zip(selected,values,strict=True) if value is not None}
    first, repeated, current = gradient(stale), gradient(stale), gradient(fresh)
    assert first.keys() == repeated.keys() == current.keys()
    result = {}
    for expert in EXPERTS:
        names = [n for n in first if n.startswith('encoder.'+expert+'_')]
        assert names
        result[expert] = dict(
            fresh=grad_comparison([first[n] for n in names], [current[n] for n in names]),
            duplicate_noise=grad_comparison([first[n] for n in names], [repeated[n] for n in names]))
    assert all(p.grad is None for _,p in selected)
    return result


def fit(model, records, fold, protocol, config, md, previous, *, endpoint, mode, directory):
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    _set_seed(42);model.train()
    initial = _module_state_sha256(model)
    frozen = frozen_state_sha(model)
    signal = _module_state_sha256(model.baseline.signal)
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['OPTIMIZATION']['NEW_MODULE_LR'],
                                 weight_decay=config['OPTIMIZATION']['WEIGHT_DECAY'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256.)
    criterion = ExpertFormationV8Criterion(triplet_margin=.3, label_smoothing=.1).cuda()
    source = [protocol['records'][i] for i in fold['source_record_indices']]
    memory = InstanceMemory(source, capacity=512, maximum_age=8)
    fields = ViewFields()
    warmup = 2 if mode == 'preflight' else 65
    lookup = {Path(r[0][0]).name:i for r,i in zip(records,fold['source_record_indices'],strict=True)}
    loader = loader_for(records,True)
    history, steps, live = [], [], set()
    extra_records = overflow = 0
    torch.cuda.reset_peak_memory_stats()
    before_sealed = [r['loss'] for r in previous['training']['steps']]
    original_audits = [json.loads(line) for line in (Path(previous['checkpoint']).parent/'memory_steps.jsonl').read_text().splitlines()]
    with (directory/'audit.jsonl').open('x') as log, (directory/'distances.f32').open('xb') as matrices:
        for epoch in range(1, (20 if mode=='source' else 1)+1):
            started = time.perf_counter()
            lr = config['OPTIMIZATION']['NEW_MODULE_LR']
            if mode=='source': lr *= learning_rate_multiplier(epoch,max_epochs=20,warmup_epochs=5)
            for group in optimizer.param_groups: group['lr']=lr
            batches=loader if mode=='source' else itertools.islice(loader,12)
            epoch_rows=[]
            for raw in batches:
                step=len(steps)
                indices=[lookup[n] for n in raw[-1]]
                assert indices==md['batches'][step]['record_indices']
                batch, labels=_training_batch(raw)
                pixels=old.prior.pixels(batch)
                assert indices==original_audits[step]['record_indices'] and pixels==original_audits[step]['pixel_sha256']
                ids=[protocol['records'][i]['identity'] for i in indices]
                scenes=[protocol['records'][i]['scene'] for i in indices]
                optimizer.zero_grad(set_to_none=True)
                stale,metadata=memory.read(step,indices,torch.empty((0,7680),device='cuda'))
                fresh, count=fields.refresh(model,metadata,stale,step)
                extra_records+=count
                with torch.autocast('cuda',dtype=torch.float16):
                    if step>=warmup or step==0:
                        output,item=fields.capture(model,batch,indices)
                    else:
                        output=model(batch,return_aux=True);item=None
                    old.output_mapping(output)
                    components=criterion(output,labels)
                unit=F.normalize(output.fused_embedding.float(),dim=1)
                if step==0:
                    buffers={n:v.clone() for n,v in model.named_buffers()}
                    zero=fields.encode(model,item);extra_records+=64
                    assert torch.equal(unit,zero), float((unit-zero).abs().max())
                    assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
                ls,basic,_,dc,ds,stats=expanded_triplet(output.fused_embedding,ids,scenes,stale,metadata)
                lf,fresh_basic,_,fc,df,fstats=expanded_triplet(output.fused_embedding,ids,scenes,fresh,metadata)
                assert torch.equal(basic,components['triplet_fused']) and torch.equal(basic,fresh_basic)
                assert torch.equal(dc,fc)
                diagnostic=parameter_probe(model,ls,lf,scaler.get_scale()) if metadata else {}
                if endpoint=='instance_memory' and step>=warmup: components['triplet_fused']=ls
                with torch.autocast('cuda',dtype=torch.float16): loss=weighted_training_loss(components,config)
                assert torch.isfinite(loss)
                scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                for n,p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        assert torch.isfinite(p.grad).all(),n
                        if p.grad.abs().sum()>0: live.add(n)
                scaler.step(optimizer);scaler.update();overflow+=int(scaler.get_scale()<scale)
                offset=matrices.tell()
                for values in (dc,ds,df): matrices.write(values.detach().cpu().contiguous().numpy().tobytes())
                row=dict(step=step+1,epoch=epoch,loss=float(loss.detach()),
                         components={k:float(v.detach()) for k,v in components.items()},
                         record_indices=indices,pixel_sha256=pixels,identities=ids,scenes=scenes,
                         memory=metadata,stale_statistics=stats,fresh_statistics=fstats,
                         cache_l2_drift=(stale-fresh).norm(dim=1).cpu().tolist(),
                         parameter_gradients=diagnostic,distance_offset_bytes=offset,
                         distance_float_count=dc.numel()+ds.numel()+df.numel(),
                         fresh_role_record_forwards=count,amp_scale_before=scale,amp_scale_after=scaler.get_scale(),
                         original_trajectory_loss_difference=float(loss.detach())-before_sealed[step] if mode=='source' else None,
                         refreshed_loss_used_for_update=False)
                log.write(json.dumps(row)+'\n');log.flush()
                if step>=warmup:
                    memory.update(step,indices,unit)
                    fields.fields[step]=item
                steps.append(row);epoch_rows.append(row)
                del output,components,loss,unit,stale,fresh,ls,lf,basic,fresh_basic,dc,ds,df,fc,item
            h=dict(epoch=epoch,updates=len(epoch_rows),learning_rate=lr,mean_loss=float(np.mean([r['loss'] for r in epoch_rows])),
                   elapsed_seconds=time.perf_counter()-started)
            history.append(h)
            print(json.dumps(dict(event='freshness_epoch',fold=fold['fold'],endpoint=endpoint,mode=mode,**h)),flush=True)
    assert len(steps)==(12 if mode=='preflight' else 260)
    assert names==live and overflow==0
    assert frozen_state_sha(model)==frozen and _module_state_sha256(model.baseline.signal)==signal
    report=dict(steps=len(steps),history=history,initial_state_sha256=initial,final_state_sha256=_module_state_sha256(model),
                frozen_state_sha256=frozen,signal_state_sha256=signal,all_trainable_gradients=True,
                trainable_tensor_count=len(names),overflow_events=overflow,zero_age_reencoding_bitwise=True,
                extra_role_record_forwards=extra_records,source_input_record_forwards=64*len(steps),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                audit_sha256=sha256(directory/'audit.jsonl'),distances_sha256=sha256(directory/'distances.f32'),
                exact_reproduction_of_original_training_not_claimed=True)
    write_json(directory/'training.json',report)
    return report


def run(args):
    spec,(memory_spec,(config,base,cfg,environment,protocol,baseline,meta)),previous=context(args.config)
    torch.set_num_threads(4)
    args.output.mkdir()
    summary=dict(status='RUNNING',mode=args.mode,config_sha256=sha256(args.config),source_sha256=sha256(__file__),
                 project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 started_at=datetime.now().astimezone().isoformat(),endpoints=[],official_reads=0,heldout_image_reads=0,
                 updated_loss_rules=['original_batch','original_stale_memory'],fresh_loss_updates=0)
    def save(): write_json(args.output/'summary.json',summary)
    save()
    for fold,b0,md in zip(protocol['folds'],baseline['folds'],meta['folds'],strict=True):
        records=records_for(base,protocol,fold,True)
        for end in old.ENDPOINTS:
            directory=args.output/f"fold_{fold['fold']}_{end}";directory.mkdir()
            model,binding=build_model(config,cfg,fold,b0)
            old_end=previous['folds'][fold['fold']]['endpoints'][end]
            assert binding==old_end['initialization']
            training=fit(model,records,fold,protocol,config,md,old_end,
                         endpoint=end,mode=args.mode,directory=directory)
            path=directory/'roles_terminal.pth'
            digest=old.prior.checkpoint(model,binding,fold,sha256(args.config),path)
            state=training['final_state_sha256']
            del model;torch.cuda.empty_cache()
            model=old.reload_model(config,cfg,fold,b0,path,binding,state,sha256(args.config))
            assert _module_state_sha256(model)==state and sha256(path)==digest
            row=dict(fold=fold['fold'],endpoint=end,initialization=binding,training=training,
                     checkpoint=str(path),checkpoint_sha256=digest,strict_state_reload=True)
            summary['endpoints'].append(row);save()
            del model;torch.cuda.empty_cache()
    summary.update(status='COMPLETE_SOURCE_FRESHNESS_MEASUREMENT',completed_at=datetime.now().astimezone().isoformat())
    save()


def verify(args):
    from tools.verify_msvr310_source_style import state_sha
    spec,(memory_spec,(config,_,_,_,protocol,baseline,meta)),previous=context(args.config)
    s=json.loads((args.output/'summary.json').read_bytes())
    assert s['status']=='COMPLETE_SOURCE_FRESHNESS_MEASUREMENT'
    assert s['config_sha256']==sha256(args.config) and len(s['endpoints'])==6
    rows_out=[];count=elements=0;paired={}
    for end in s['endpoints']:
        directory=args.output/f"fold_{end['fold']}_{end['endpoint']}"
        tr=end['training']
        assert tr==json.loads((directory/'training.json').read_bytes())
        assert sha256(directory/'audit.jsonl')==tr['audit_sha256']
        assert sha256(directory/'distances.f32')==tr['distances_sha256']
        assert sha256(end['checkpoint'])==end['checkpoint_sha256'] and end['strict_state_reload']
        payload=torch.load(end['checkpoint'],map_location='cpu',weights_only=True)
        original=torch.load(baseline['folds'][end['fold']]['checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
        state={name:original[key] for name,key in payload['baseline_aliases'].items()}
        state.update(payload['role_state_dict'])
        assert state_sha(state)==tr['final_state_sha256']
        assert payload['binding']==end['initialization'] and payload['config_sha256']==s['config_sha256']
        del payload,original,state
        audits=[json.loads(x) for x in (directory/'audit.jsonl').read_text().splitlines()]
        assert len(audits)==tr['steps']==(12 if s['mode']=='preflight' else 260)
        paired[(end['fold'],end['endpoint'])]=[(r['record_indices'],r['pixel_sha256']) for r in audits]
        cache=OrderedDict()
        warmup=2 if s['mode']=='preflight' else 65
        with (directory/'distances.f32').open('rb') as f:
            for step,a in enumerate(audits):
                assert a['step']==step+1 and a['distance_offset_bytes']==f.tell()
                assert a['record_indices']==meta['folds'][end['fold']]['batches'][step]['record_indices']
                indices=a['record_indices'];fold=protocol['folds'][end['fold']]
                assert set(indices)<=set(fold['source_record_indices'])
                assert a['identities']==[protocol['records'][i]['identity'] for i in indices]
                for i in [i for i,k in cache.items() if step-k>8]: del cache[i]
                expected=[dict(record_index=i,identity=protocol['records'][i]['identity'],
                               scene=protocol['records'][i]['scene'],age=step-k,stored_step=k)
                          for i,k in cache.items() if i not in set(indices)]
                assert expected==a['memory']
                m=len(a['memory'])
                raw=np.fromfile(f,dtype=np.float32,count=a['distance_float_count'])
                assert len(raw)==4096+128*m and np.isfinite(raw).all()
                elements+=len(raw);count+=1
                dc=raw[:4096].reshape(64,64)
                ds=raw[4096:4096+64*m].reshape(64,m);df=raw[4096+64*m:].reshape(64,m)
                ids=np.array(a['identities']);p=(ids[:,None]==ids[None,:]) & ~np.eye(64,dtype=bool);n=ids[:,None]!=ids[None,:]
                hp=np.where(p,dc,-np.inf).max(1);hn=np.where(n,dc,np.inf).min(1)
                base_loss=float(np.maximum(0,hp-hn+np.float32(.3)).mean())
                fresh_results=[]
                for key,dm in [('stale',ds),('fresh',df)]:
                    mp=ids[:,None]==np.array([r['identity'] for r in a['memory']])[None,:]
                    allp=np.concatenate([np.where(p,dc,-np.inf),np.where(mp,dm,-np.inf)],axis=1)
                    alln=np.concatenate([np.where(n,dc,np.inf),np.where(~mp,dm,np.inf)],axis=1)
                    up=allp.max(1);un=alln.min(1)
                    loss=float(np.maximum(0,up-un+np.float32(.3)).mean())
                    assert abs(loss-a[key+'_statistics']['expanded_triplet'])<2e-6
                    assert int((up>=un).sum())==a[key+'_statistics']['expanded_wrong_order_anchors']
                    fresh_results.append((loss,up,un,allp.argmax(1),alln.argmin(1),
                                          int(((allp==up[:,None]).sum(1)>1).sum()),
                                          int(((alln==un[:,None]).sum(1)>1).sum())))
                st,fr=fresh_results
                assert not a['refreshed_loss_used_for_update']
                expected_loss=st[0] if end['endpoint']=='instance_memory' and step>=warmup else base_loss
                assert abs(a['components']['triplet_fused']-expected_loss)<2e-6
                c=a['components'];w=config['LOSS']
                total=w['ID_FUSED']*c['id_fused']+w['TRIPLET_FUSED']*c['triplet_fused']
                total+=sum(w['ID_BRANCH']*c['id_'+e]+w['TRIPLET_BRANCH']*c['triplet_'+e]
                           +w['ID_RESIDUAL']*c['id_residual_'+e]+w['TRIPLET_RESIDUAL']*c['triplet_residual_'+e] for e in EXPERTS)
                assert abs(total-a['loss'])<1e-5 and a['amp_scale_after']>=a['amp_scale_before']
                row=dict(fold=end['fold'],endpoint=end['endpoint'],step=step+1,epoch=a['epoch'],memory_records=m,
                         stale_loss=st[0],fresh_loss=fr[0],positive_winner_changes=int((st[3]!=fr[3]).sum()),
                         negative_winner_changes=int((st[4]!=fr[4]).sum()),
                         stale_wrong_fresh_correct=int(((st[1]>=st[2])&(fr[1]<fr[2])).sum()),
                         stale_correct_fresh_wrong=int(((st[1]<st[2])&(fr[1]>=fr[2])).sum()),
                         stale_positive_ties=st[5],stale_negative_ties=st[6],
                         fresh_positive_ties=fr[5],fresh_negative_ties=fr[6],
                         historical_ages=[r['age'] for r in a['memory']],
                         drift=a['cache_l2_drift'],parameter_gradients=a['parameter_gradients'])
                if m:
                    assert len(a['cache_l2_drift'])==m and set(a['parameter_gradients'])==set(EXPERTS)
                    for expert in a['parameter_gradients'].values():
                        for gradient in expert.values():
                            assert all(np.isfinite(v) for v in gradient.values() if v is not None)
                            assert gradient['stale_norm']>=0 and gradient['comparison_norm']>=0 and gradient['difference_norm']>=0
                            assert gradient['cosine'] is None or -1.00001<=gradient['cosine']<=1.00001
                    drift=np.array(a['cache_l2_drift'])
                    assert np.all(np.abs(ds-df)<=drift[None,:]+2e-6)
                    assert all(r['record_index'] not in set(indices) and 1<=r['age']<=8 for r in a['memory'])
                    row['distance_error_mean']=float(np.abs(ds-df).mean())
                    row['distance_error_max']=float(np.abs(ds-df).max())
                rows_out.append(row)
                if step>=warmup:
                    for i in indices:
                        cache.pop(i,None);cache[i]=step
                    while len(cache)>512: cache.popitem(last=False)
            assert f.read(1)==b''
        assert tr['all_trainable_gradients'] and tr['trainable_tensor_count']==203 and tr['overflow_events']==0
        assert tr['extra_role_record_forwards']==64+sum(a['fresh_role_record_forwards'] for a in audits)
    for fold in range(3): assert paired[(fold,'control')]==paired[(fold,'instance_memory')]
    assert count==(72 if s['mode']=='preflight' else 1560)
    write_json(args.output/'cpu_verification.json',dict(status='PASS_COMPLETE_FRESHNESS_'+s['mode'].upper(),
               checked_at=datetime.now().astimezone().isoformat(),summary_sha256=sha256(args.output/'summary.json'),
               all_steps=count,distance_elements=elements,all_rows=rows_out,heldout_image_reads=0,official_reads=0,
               limitations='Runtime parameter-gradient witnesses and strict reload are not independent gradient recomputation; source trajectories are new diagnostic runs, not claimed bitwise copies of the sealed run.'))
    print(json.dumps(dict(status='PASS_COMPLETE_FRESHNESS_'+s['mode'].upper(),steps=count,distance_elements=elements)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','source','verify'],required=True)
    args=parser.parse_args()
    verify(args) if args.mode=='verify' else run(args)
