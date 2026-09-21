import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys
sys.dont_write_bytecode = True
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
START = time.monotonic()
REPO = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN = Path('/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3')
ROLES = ('cnn','transformer','mamba')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def js(p): return json.loads(Path(p).read_bytes())
def state_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()
spec=js(REPO/'configs/MSVR310/TriFusion-supported-task-state-paired-v1.json')
config_sha=sha(REPO/'configs/MSVR310/TriFusion-supported-task-state-paired-v1.json')
assert config_sha=='4ebedda4a10d5e535d6820aa05684b7f707f7a0b8308d454b2316e9b3a61f3c3'
cfg=js(REPO/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=js(REPO/cfg['BASELINE']['CONFIG'])
protocol=js(REPO/base['protocol'])
assert sha(REPO/base['protocol'])==base['protocol_sha256']
metadata=js(REPO/cfg['SOURCE_METADATA']['PATH'])
assert sha(REPO/cfg['SOURCE_METADATA']['PATH'])==cfg['SOURCE_METADATA']['SHA256']
baseline=js(cfg['BASELINE']['SUMMARY'])
assert sha(cfg['BASELINE']['SUMMARY'])==cfg['BASELINE']['SUMMARY_SHA256']
source_files={}; configs={};todo=['configs/MSVR310/TriFusion-supported-task-state-paired-v1.json']
def strings(x):
    if isinstance(x,dict):
        for value in x.values(): yield from strings(value)
    elif isinstance(x,list):
        for value in x: yield from strings(value)
    elif isinstance(x,str): yield x
while todo:
    rel=todo.pop()
    if rel in configs:continue
    configs[rel]=js(REPO/rel)
    source_files[str(REPO/rel)]={'sha256':sha(REPO/rel),'bytes':(REPO/rel).stat().st_size}
    for key in ('project_file_sha256','project_source_file_sha256'):
        for path,expected in configs[rel].get(key,{}).items():
            p=REPO/path; actual=sha(p);assert actual==expected,(path,actual,expected)
            source_files[str(p)]={'sha256':actual,'bytes':p.stat().st_size}
    for value in strings(configs[rel]):
        if value.startswith('configs/') and value.endswith('.json') and (REPO/value).is_file():todo.append(value)
for rel,expected in base['signal_source_file_sha256'].items():
    p=Path(base['signal_source'])/rel;assert sha(p)==expected
    source_files[str(p)]={'sha256':expected,'bytes':p.stat().st_size}
assert sha(base['clip_weight'])==base['clip_weight_sha256']
support=js(REPO/spec['source_support'])
for rel,expected in support['input_bindings'].items():assert sha(REPO/rel)==expected
for rel in (base['protocol'],cfg['SOURCE_METADATA']['PATH'],spec['source_support']):
    p=REPO/rel;source_files[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
summary=js(RUN/'m0/summary.json');cpu=js(RUN/'m0_cpu.json');t0=js(RUN/'t0.json')
assert summary['config_sha256']==t0['config_sha256']==config_sha
assert summary['status']=='PASS_ENGINEERING_ONLY' and summary['optimizer_steps']==248
assert summary['heldout_record_forwards']==summary['official_image_reads']==0
assert cpu['summary_sha256']==sha(RUN/'m0/summary.json')
files={}
for p,expected in cpu['files'].items():
    q=Path(p);actual=dict(bytes=q.stat().st_size,sha256=sha(q));assert actual==expected,p;files[p]=actual
for name in ('m0_cpu.json','t0.json','m0.log','m0_cpu.log'):
    p=RUN/name;files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
record_labels=[]
for i,r in enumerate(protocol['records']):
    assert r['index']==i
    for path in r['paths']:
        assert path.startswith('bounding_box_train/')
        match=re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg',Path(path).name)
        assert match and tuple(map(int,match.groups()[:3]))==(r['identity'],r['scene'],r['camera'])
    assert len({Path(p).name for p in r['paths']})==1
fold_scope=[]; t0_steps=0
for fold,md,t in zip(protocol['folds'],metadata['folds'],t0['folds'],strict=True):
    source=set(fold['source_record_indices']);heldout=set(fold['gallery_record_indices'])
    assert not set(fold['source_ids']) & set(fold['heldout_ids'])
    assert source.isdisjoint(heldout)
    assert source=={r['index'] for r in protocol['records'] if r['identity'] in fold['source_ids']}
    assert heldout=={r['index'] for r in protocol['records'] if r['identity'] in fold['heldout_ids']}
    queue=collections.OrderedDict()
    for index,(batch,recorded) in enumerate(zip(md['batches'],t['steps'],strict=True)):
        ids=batch['record_indices'];assert set(ids)<=source and len(ids)==64
        assert sorted(collections.Counter(protocol['records'][i]['identity'] for i in ids).values())==[8]*8
        for i in list(queue):
            if index-queue[i]>8:del queue[i]
        hist=[i for i in queue if i not in ids]
        pairs=[protocol['records'][i] for i in ids+hist]
        counts=[sum(q['identity']==p['identity'] and q['scene']!=p['scene'] for p in pairs) for q in pairs[:64]]
        assert recorded['step']==index+1 and recorded['cross_scene_positive_counts']==counts
        assert recorded['historical_records']==len(hist) and recorded['ages']==[index-queue[i] for i in hist]
        assert recorded['eligible_anchors']==sum(n>0 for n in counts)
        if index>=65:
            for i in ids:queue.pop(i,None);queue[i]=index
            while len(queue)>512:queue.popitem(last=False)
        t0_steps+=1
    fold_scope.append(dict(fold=fold['fold'],source_records=len(source),source_ids=len(fold['source_ids']),heldout_records=len(heldout),heldout_ids=len(fold['heldout_ids']),planned_queries=len(fold['query_rows'])))
assert t0_steps==780
def pair(p):
    a,b,d=p['first_norm'],p['second_norm'],p['difference_norm'];c=p['cosine']
    assert all(math.isfinite(v) and v>=0 for v in (a,b,d))
    assert (c is None)==(a==0 or b==0)
    rhs=a*a+b*b if c is None else a*a+b*b-2*a*b*c
    assert c is None or math.isfinite(c) and abs(c)<=1.00001
    assert abs(d*d-rhs)<=1e-7*max(1,a*a+b*b)
def ap_replay(dist, ids,scenes, cross):
    score=1-dist.astype(np.float64)**2/2; aps=[];counts=[]
    for i in range(64):
        positive=(ids==ids[i]) & (scenes!=scenes[i]) if cross else ids==ids[i]
        valid=~((ids==ids[i]) & (scenes==scenes[i])) if cross else np.arange(len(ids))!=i
        if not cross:positive[i]=False
        pos=np.flatnonzero(positive);counts.append(len(pos))
        if len(pos)==0:aps.append(0.);continue
        comparisons=1/(1+np.exp(-(score[i][None,:]-score[i,pos][:,None])/.01))
        comparisons[:,~valid]=0;comparisons[np.arange(len(pos)),pos]=0
        aps.append(float(((1+comparisons[:,positive].sum(1))/(1+comparisons.sum(1))).mean()))
    eligible=np.array(counts)>0
    return (1-float(np.array(aps)[eligible].mean()) if eligible.any() else 0.),aps,counts
endpoints=[]; moments_total=0; references_total=0; baseline_hashes={}; checkpoint_checks=[]
model_shapes={}
for fold,b0,sfold in zip(protocol['folds'],baseline['folds'],summary['folds'],strict=True):
    assert sha(b0['checkpoint'])==b0['checkpoint_sha256']
    baseline_hashes[b0['checkpoint']]=b0['checkpoint_sha256']
    saved_b0=torch.load(b0['checkpoint'],map_location='cpu',weights_only=True)
    assert saved_b0['source_ids']==fold['source_ids'] and saved_b0['heldout_ids']==fold['heldout_ids']
    original=saved_b0['model_state_dict']
    for arm,row in sfold['endpoints'].items():
        p=Path(row['checkpoint']);payload=torch.load(p,map_location='cpu',weights_only=True)
        assert payload['config_sha256']==config_sha and payload['binding']==row['initialization']
        assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids']
        rebuilt={k:original[v] for k,v in payload['baseline_aliases'].items()};rebuilt.update(payload['role_state_dict'])
        assert state_sha(rebuilt)==row['training']['final_state_sha256']==row['strict_reload_state_sha256']
        assert state_sha(original)==row['training']['signal_state_after_sha256']==row['training']['signal_state_before_sha256']
        model_shapes[fold['fold']]={k:tuple(t.shape) for k,t in payload['role_state_dict'].items()}
        checkpoint_checks.append(dict(endpoint=f"fold_{fold['fold']}_{arm}",path=str(p),sha256=sha(p),reconstructed_state_sha256=state_sha(rebuilt),baseline_alias_tensors=len(payload['baseline_aliases'])))
    assert sfold['endpoints']['control']['initialization']==sfold['endpoints']['split']['initialization']
    del original,saved_b0,payload,rebuilt
for name in [f'fold_{f}_{a}' for f in range(3) for a in ('control','split')]+['overfit_control','overfit_split']:
    d=RUN/'m0'/name;tr=js(d/'training.json');audits=[json.loads(l) for l in (d/'memory_steps.jsonl').read_text().splitlines()]
    arm=name.split('_')[-1];fold_index=0 if name.startswith('overfit') else int(name.split('_')[1])
    fold=protocol['folds'][fold_index];md=metadata['folds'][fold_index];overfit=name.startswith('overfit')
    bound=summary['overfit'][arm] if overfit else summary['folds'][fold_index]['endpoints'][arm]
    assert bound['training']==tr
    if not overfit: assert js(d/'receipt.json')==bound
    assert len(audits)==len(tr['steps'])==tr['optimizer_steps']==(100 if overfit else 8)
    assert tr['nonzero_gradient_tensors']==tr['trainable_tensors']==203 and not tr['missing_nonzero_gradients']
    assert tr['overflow_events']==0 and tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
    queue=collections.OrderedDict(); observed=0;refs=0;max_error=0.;loss_error=0.; elements=0;vjp=0
    max_ref=0.;update_norms={r:[] for r in ROLES};unique_records=set();unique_ids=set();unique_scenes=set()
    with (d/'memory_distances.f32').open('rb') as stream:
        for index,(a,s) in enumerate(zip(audits,tr['steps'],strict=True)):
            assert a['step']==s['step']==index+1 and a['zero_based_step']==index
            indices=a['record_indices'];assert indices==s['sampled_record_indices']==md['batches'][0 if overfit else index]['record_indices']
            assert set(indices)<=set(fold['source_record_indices'])
            assert a['identities']==[protocol['records'][i]['identity'] for i in indices]
            assert a['scenes']==[protocol['records'][i]['scene'] for i in indices]
            unique_records.update(indices);unique_ids.update(a['identities']);unique_scenes.update(a['scenes'])
            for i in list(queue):
                if index-queue[i]>8:del queue[i]
            wanted=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],stored_step=step,age=index-step) for i,step in queue.items() if i not in indices]
            assert wanted==a['memory'];assert a['warmup_steps']==2 and a['replacement_active']==(index>=2)
            assert s['active_fused_metric']==('cross_scene_smooth_ap' if index>=2 else 'hard_triplet')
            m=len(wanted);n=4*64*(64+m);assert stream.tell()==a['distance_offset_bytes'] and n==a['distance_float_count']
            raw=np.fromfile(stream,dtype=np.float32,count=n);assert raw.size==n and np.isfinite(raw).all() and (raw>=0).all()
            distances=raw.reshape(4,64,64+m);dist=distances[0];elements+=n
            ids=np.array(a['identities']+[x['identity'] for x in wanted]);scenes=np.array(a['scenes']+[x['scene'] for x in wanted])
            cross,aps,counts=ap_replay(dist,ids,scenes,True);standard,stdaps,_=ap_replay(dist,ids,scenes,False)
            rel=a['relation_objective'];assert counts==rel['cross_scene_positive_counts']
            assert np.allclose(aps,rel['cross_scene_per_anchor_ap'],rtol=0,atol=2e-6)
            assert np.allclose(stdaps,rel['per_anchor_smoothed_ap'],rtol=0,atol=2e-6)
            current_ids=ids[:64];dc=dist[:,:64]
            positives=(current_ids[:,None]==current_ids[None,:]) & ~np.eye(64,dtype=bool)
            hp=np.where(positives,dc,-np.inf).max(1);hn=np.where(current_ids[:,None]!=current_ids[None,:],dc,np.inf).min(1)
            basic=float(np.maximum(hp-hn+np.float32(.3),0).mean())
            expected=cross if index>=2 else basic
            for actual,target in ((s['components']['triplet_fused'],expected),(rel['cross_scene_loss'],cross),(rel['smooth_ap_loss'],standard),(a['original_triplet'],basic)):
                err=abs(actual-target);max_error=max(err,max_error);assert err<2e-6
            c=s['components'];w=cfg['LOSS'];total=w['ID_FUSED']*c['id_fused']+w['TRIPLET_FUSED']*c['triplet_fused']
            total+=sum(w['ID_BRANCH']*c['id_'+r]+w['TRIPLET_BRANCH']*c['triplet_'+r]+w['ID_RESIDUAL']*c['id_residual_'+r]+w['TRIPLET_RESIDUAL']*c['triplet_residual_'+r] for r in ROLES)
            loss_error=max(loss_error,abs(total-s['loss']));assert abs(total-s['loss'])<1e-5
            supported=index<2 or any(counts);assert a['rank_observed']==supported;observed+=int(supported)
            expected_support=dict(eligible_anchors=sum(x>0 for x in counts),eligible_identities=len({y for y,n in zip(current_ids,counts) if n>0}),identity_directed_scene_relations=len({(int(y),int(sc),int(other)) for y,sc,n in zip(current_ids,scenes[:64],counts) if n>0 for yy,other in zip(ids,scenes) if y==yy and sc!=other}))
            assert a['support']==expected_support
            assert a['classification_head_gradients_unchanged'] and a['direct_ra_assembly']
            for key in ('selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged'):assert a[key]
            assert a['current_anchor_count']==64 and a['history_anchor_count']==0
            assert s['amp_scale_before']==s['amp_scale_after']==256.
            assert set(a['actual_parameter_updates'])==set(a['task_states'])==set(ROLES)
            for role in ROLES:
                pair(a['actual_parameter_updates'][role]);update_norms[role].append(a['actual_parameter_updates'][role]['difference_norm'])
                assert a['actual_parameter_updates'][role]['difference_norm']>0
                tasks=a['task_states'][role];assert set(tasks)==({'shared'} if arm=='control' else {'rank','auxiliary'})
                for task,value in tasks.items():assert value['step']==(observed if task=='rank' else index+1)
            has=bool(a['direct_single_group_check']);assert has==bool(a['rank_auxiliary_reference_checks'])
            assert a['direct_component_backward_calls']==(4 if has else 0)
            if has:
                for role in ROLES:
                    checks=a['rank_auxiliary_reference_checks'][role];assert set(checks)=={'current_rank','historical_rank','full_rank','auxiliary','assembled'}
                    for check in checks.values():
                        pair(check);assert check['passed'];err=check['difference_norm']/check['first_norm'] if check['first_norm'] else None
                        assert err==check['relative_l2_error'];assert err<=.005 if err is not None else check['difference_norm']<=1e-8
                        max_ref=max(max_ref,err or 0);refs+=1
            vjp+=a['history_vjp_record_forwards']
            if index>=2:
                for i in indices:queue.pop(i,None);queue[i]=index
                while len(queue)>512:queue.popitem(last=False)
        assert not stream.read()
    assert refs==(0 if overfit else 15)
    saved=torch.load(d/'optimizer_state.pt',map_location='cpu',weights_only=True)
    assert saved['endpoint']==arm
    names=saved['parameter_names'];assert len(names)==len(set(names))==203
    assert names==bound['initialization']['trainable_names']
    states=saved['optimizer']['state'];groups=saved['optimizer']['param_groups']
    assert len(groups)==1 and groups[0]['params']==list(range(203)) and set(states)==set(range(203))
    g=groups[0];assert g['lr']==.00035 and g['weight_decay']==.0001 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
    role_indices={r:[i for i,name in enumerate(names) if name.startswith('encoder.'+r+'_')] for r in ROLES}
    assert [len(role_indices[r]) for r in ROLES]==[42,54,93]
    moments=0
    for i,name in enumerate(names):
        role=[r for r in ROLES if i in role_indices[r]]
        assert set(states[i])==({'shared'} if arm=='control' else {'rank','auxiliary'}) if role else set(states[i])=={'head'}
        for task,st in states[i].items():
            assert st['step']==(observed if task=='rank' else len(audits))
            assert st['exp_avg'].dtype==st['exp_avg_sq'].dtype==torch.float32
            assert tuple(st['exp_avg'].shape)==tuple(st['exp_avg_sq'].shape)==model_shapes[fold_index][name]
            assert torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all() and (st['exp_avg_sq']>=0).all();moments+=2
    final=audits[-1]['task_states'];norm_error=0.
    for role,indices in role_indices.items():
        for task,value in final[role].items():
            for field,moment in [('first_moment_norm','exp_avg'),('second_moment_norm','exp_avg_sq')]:
                norm=math.sqrt(sum(float(states[i][task][moment].double().square().sum()) for i in indices))
                norm_error=max(norm_error,abs(norm-value[field]));assert abs(norm-value[field])<1e-9*max(1,norm)
    assert saved['scaler']==dict(scale=256.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=len(audits))
    assert moments==tr['optimizer_state']['moment_tensors_checked']
    if overfit:
        nclass=len(fold['source_ids']);smooth=.1;p=1-smooth+smooth/nclass;q=smooth/nclass
        floor=-(p*math.log(p)+(nclass-1)*q*math.log(q))*.75
        ratio=(tr['steps'][-1]['loss']-floor)/(tr['steps'][0]['loss']-floor)
        assert abs(floor-bound['gate']['minimum_loss'])<1e-12 and abs(ratio-bound['gate']['loss_ratio'])<1e-12 and ratio<=.1
    endpoints.append(dict(endpoint=d.name,directory=str(d),steps=len(audits),observed_rank_steps=observed,absent_rank_steps=len(audits)-observed,distance_elements=elements,max_objective_error=max_error,max_total_loss_error=loss_error,direct_references=refs,max_direct_reference_relative_l2=max_ref,history_vjp_record_forwards=vjp,moment_tensors=moments,max_final_moment_norm_error=norm_error,parameter_group={k:v for k,v in g.items() if k!='params'},scaler=saved['scaler'],actual_update_norm_ranges={r:[min(v),max(v)] for r,v in update_norms.items()},unique_sampled_records=len(unique_records),unique_sampled_identities=len(unique_ids),unique_sampled_scenes=sorted(unique_scenes)))
    moments_total+=moments;references_total+=refs
    del saved,states
for f in range(3):
    aa=[[json.loads(l) for l in (RUN/'m0'/f'fold_{f}_{a}'/'memory_steps.jsonl').read_text().splitlines()] for a in ('control','split')]
    assert [(a['record_indices'],a['pixel_sha256']) for a in aa[0]]==[(a['record_indices'],a['pixel_sha256']) for a in aa[1]]
oo=[[json.loads(l) for l in (RUN/'m0'/('overfit_'+a)/'memory_steps.jsonl').read_text().splitlines()] for a in ('control','split')]
assert [(a['record_indices'],a['pixel_sha256']) for a in oo[0]]==[(a['record_indices'],a['pixel_sha256']) for a in oo[1]]
assert sum(x['steps'] for x in endpoints)==248 and sum(x['distance_elements'] for x in endpoints)==cpu['checked_memory_distance_elements']
assert references_total==90 and moments_total==4760
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INDEPENDENT_READ_ONLY_CPU_M0_ARTIFACT_CHECKS',generated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START,remote_head_at_check=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),execution_commit=summary['project_commit'],config_sha256=config_sha,summary_sha256=sha(RUN/'m0/summary.json'),cpu_receipt_sha256=sha(RUN/'m0_cpu.json'),source_files=source_files,bound_files=files,baseline_checkpoint_hashes=baseline_hashes,checkpoint_checks=checkpoint_checks,fold_scope=fold_scope,protocol_record_count=len(protocol['records']),t0_metadata_steps_independently_replayed=t0_steps,endpoints=endpoints,total_steps=248,total_moment_tensors=moments_total,total_direct_references=references_total,model_forwards=0,optimizer_updates=0,image_reads=0,cuda_initialized=torch.cuda.is_initialized(),threads=2,scope='Independent saved-artifact arithmetic and structure checks. No regeneration of model gradients, training trajectory, pixels, or per-task update vectors.'),indent=2))
