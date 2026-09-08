"""Independent CPU-only replay: saved M0 arrays, labels, queue, checkpoint bytes.

The only imported experiment function is the deployed objective, exercised on
distance tensors. No model is instantiated and no model/GPU forward/backward,
image load, optimizer step, process mutation or file write occurs remotely.
"""
from collections import OrderedDict, Counter, defaultdict
import hashlib, importlib.util, json, math, random, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
R=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
M0=RUN/'m0';started=time.perf_counter()
def j(p):return json.loads(Path(p).read_bytes())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def state_digest(state):
    h=hashlib.sha256()
    for name in sorted(state):h.update(name.encode());h.update(state[name].contiguous().numpy().tobytes())
    return h.hexdigest()
spec=j(R/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json')
config=j(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
base=j(R/config['BASELINE']['CONFIG']);protocol=j(R/base['protocol']);records=protocol['records']
md=j(R/config['SOURCE_METADATA']['PATH']);b0=j(config['BASELINE']['SUMMARY']);summary=j(M0/'summary.json')
modspec=importlib.util.spec_from_file_location('audited_objective',R/'tools/msvr_smooth_ap.py')
objective=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(objective)
out={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'steps':[],'endpoints':[],'checkpoints':[],'sampler':[],'t0_queue':[],'overfit':[],'boundaries':{'model_forwards':0,'model_backwards':0,'gpu_calls':0,'optimizer_updates':0,'image_reads':0,'official_image_reads':0,'q1_files_read':0,'remote_file_writes':0}}

def reference_smooth(d,ids):
    # Derive the Jacobian from d(AP)/d(rank-comparison), then chain to distance.
    n=d.shape[0];scores=1-d.astype(np.float64)**2/2;aps=[];grad=np.zeros_like(scores)
    for anchor in range(n):
        positive=np.asarray(ids)==ids[anchor];positive[anchor]=False
        p=np.flatnonzero(positive);assert len(p)>0
        z=(scores[anchor][None,:]-scores[anchor,p,None])/.01
        g=1/(1+np.exp(-z));valid=np.ones_like(g,dtype=bool)
        valid[:,anchor]=False;valid[np.arange(len(p)),p]=False;g[~valid]=0
        numerator=1+g[:,positive].sum(1);denominator=1+g.sum(1)
        aps.append(float((numerator/denominator).mean()))
        coefficient=-(positive[None,:]*denominator[:,None]-numerator[:,None])/(denominator[:,None]**2*len(p)*n)
        dz=coefficient*g*(1-g)/.01
        ds=dz.sum(0);ds[p]-=dz.sum(1)
        grad[anchor]=-d[anchor]*ds
    return np.asarray(aps),grad

def reference_hard(d,ids):
    n=d.shape[0];g=np.zeros(d.shape,dtype=np.float64);loss=[]
    for i in range(n):
        p=[k for k in range(n) if k!=i and ids[k]==ids[i]];q=[k for k in range(n) if ids[k]!=ids[i]]
        pi=p[int(np.argmax(d[i,p]))];qi=q[int(np.argmin(d[i,q]))]
        pos=[pi];neg=[qi];hp=d[i,pi];hn=d[i,qi]
        ph=[k for k in range(n,d.shape[1]) if ids[k]==ids[i]];nh=[k for k in range(n,d.shape[1]) if ids[k]!=ids[i]]
        if ph:
            ix=ph[int(np.argmax(d[i,ph]))]
            if d[i,ix]>hp:hp=d[i,ix];pos=[ix]
            elif d[i,ix]==hp:pos.append(ix)
        if nh:
            ix=nh[int(np.argmin(d[i,nh]))]
            if d[i,ix]<hn:hn=d[i,ix];neg=[ix]
            elif d[i,ix]==hn:neg.append(ix)
        hinge=hp-hn+np.float32(.3)
        loss.append(max(0.,float(hinge)))
        if hinge>0:g[i,pos]=1/(len(pos)*n);g[i,neg]=-1/(len(neg)*n)
    return float(np.mean(loss)),g

def sampler(source):
    prng=random.Random(42);nrng=np.random.RandomState(42)
    groups=defaultdict(list)
    for ix in source:groups[records[ix]['identity']].append(ix)
    batches=[]
    for epoch in range(20):
        ready={}
        for identity,values in groups.items():
            selected=values.copy()
            if len(selected)<8:selected=nrng.choice(selected,size=8,replace=True)
            prng.shuffle(selected)
            ready[identity]=[list(map(int,selected[k:k+8])) for k in range(0,len(selected)-7,8)]
        available=list(groups)
        while len(available)>=8:
            chosen=prng.sample(available,8);batch=[]
            for identity in chosen:
                batch.extend(ready[identity].pop(0))
                if not ready[identity]:available.remove(identity)
            batches.append(batch)
    return batches

assert summary['mode']=='m0' and summary['optimizer_steps']==248 and summary['status']=='PASS_ENGINEERING_ONLY'
assert summary['heldout_record_forwards']==summary['official_image_reads']==0
assert summary['config_sha256']==digest(R/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json')
t0=j(RUN/'t0.json')
for fold,metadata in zip(protocol['folds'],md['folds'],strict=True):
    source=fold['source_record_indices'];hold=fold['gallery_record_indices']
    assert set(fold['source_ids']).isdisjoint(fold['heldout_ids'])
    assert set(source).isdisjoint(hold) and sorted(source+hold)==list(range(1032))
    assert {records[i]['identity'] for i in source}==set(fold['source_ids'])
    assert fold['source_label_map']=={str(identity):i for i,identity in enumerate(fold['source_ids'])}
    actual=sampler(source);registered=[r['record_indices'] for r in metadata['batches']]
    assert actual==registered and len(actual)==260
    out['sampler'].append({'fold':fold['fold'],'steps':len(actual),'source_ids':len(fold['source_ids']),'source_records':len(source),'heldout_ids':len(fold['heldout_ids']),'heldout_records':len(hold),'all_registered_indices_exact':True})
    cache=OrderedDict();qrows=[]
    for step,indices in enumerate(actual):
        cache=OrderedDict((i,s) for i,s in cache.items() if step-s<=8)
        values=[step-s for i,s in cache.items() if i not in indices]
        qrows.append({'step':step+1,'historical_records':len(values),'ages':values})
        if step>=65:
            for i in indices:cache.pop(i,None);cache[i]=step
            while len(cache)>512:cache.popitem(last=False)
    assert qrows==t0['folds'][fold['fold']]['steps']
    out['t0_queue'].append({'fold':fold['fold'],'steps':260,'candidates':sum(r['historical_records'] for r in qrows),'max_candidates':max(r['historical_records'] for r in qrows),'first_history_step':next(r['step'] for r in qrows if r['historical_records'])})

def replay(folder,fold,end,mode):
    tr=j(folder/'training.json');audits=[json.loads(l) for l in (folder/'memory_steps.jsonl').read_text().splitlines()]
    expected=100 if mode=='overfit' else 8
    assert len(audits)==len(tr['steps'])==tr['optimizer_steps']==expected
    assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and tr['missing_nonzero_gradients']==[]
    assert tr['overflow_events']==0 and tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256'] and tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
    for name,proof in tr['audit_files'].items():assert digest(folder/name)==proof['sha256'] and (folder/name).stat().st_size==proof['bytes']
    cache=OrderedDict();pixels=[];fresh=64;vjp=directs=elements=history_candidates=0;witnesses=[]
    with (folder/'memory_distances.f32').open('rb') as f:
        for ix,(r,a) in enumerate(zip(tr['steps'],audits,strict=True)):
            indices=r['sampled_record_indices'];ids=[records[i]['identity'] for i in indices]
            assert a['step']==r['step']==ix+1 and a['zero_based_step']==ix
            assert indices==a['record_indices']==md['folds'][fold['fold']]['batches'][0 if mode=='overfit' else ix]['record_indices']
            assert set(indices)<=set(fold['source_record_indices']) and sorted(Counter(ids).values())==[8]*8
            assert a['identities']==ids and a['scenes']==[records[i]['scene'] for i in indices]
            cache=OrderedDict((i,s) for i,s in cache.items() if ix-s<=8)
            expected_memory=[dict(record_index=i,identity=records[i]['identity'],scene=records[i]['scene'],age=ix-s,stored_step=s) for i,s in cache.items() if i not in indices]
            assert a['memory']==expected_memory
            assert a['warmup_steps']==2 and a['replacement_active']==(ix>=2)
            assert r['active_fused_metric']==('smooth_ap' if end=='smooth_ap' and ix>=2 else 'hard_triplet')
            assert a['saved_space_order']==['fused','cnn','transformer','mamba']
            ncols=64+len(expected_memory);size=4*64*ncols
            assert a['distance_offset_bytes']==f.tell() and a['distance_float_count']==size
            arrays=np.fromfile(f,dtype='<f4',count=size).reshape(4,64,ncols)
            assert np.isfinite(arrays).all() and arrays.min()>=0 and arrays.max()<=2.000001
            allids=ids+[r['identity'] for r in expected_memory]
            aps,ds=reference_smooth(arrays[0],allids);hard,dh=reference_hard(arrays[0],allids)
            basic,_=reference_hard(arrays[0,:,:64],ids);smooth=1-float(aps.mean())
            rel=a['relation_objective'];assert rel['positive_counts']==[allids.count(identity)-1 for identity in ids]
            ap_error=float(np.max(np.abs(aps-rel['per_anchor_smoothed_ap'])))
            assert ap_error<=2e-6
            scalar_error=max(abs(hard-rel['hard_loss']),abs(smooth-rel['smooth_ap_loss']),abs(basic-a['original_triplet']))
            assert scalar_error<=2e-6
            d=torch.tensor(arrays[0],dtype=torch.float64,requires_grad=True)
            th,ts,ta=objective.paired_objectives(d[:,:64],d[:,64:],ids,allids[64:])
            torch_ds=torch.autograd.grad(ts,d,retain_graph=True)[0].numpy()
            torch_dh=torch.autograd.grad(th,d)[0].numpy()
            grad64_error=float(np.max(np.abs(torch_ds-ds)))
            assert grad64_error<1e-10 and np.array_equal(torch_dh,dh)
            d32=torch.tensor(arrays[0],requires_grad=True)
            h32,s32,a32=objective.paired_objectives(d32[:,:64],d32[:,64:],ids,allids[64:])
            g32=torch.autograd.grad(s32,d32,retain_graph=True)[0].numpy()
            gh32=torch.autograd.grad(h32,d32)[0].numpy()
            grad32_error=float(np.max(np.abs(g32-ds)))
            assert grad32_error<2e-6 and np.array_equal(gh32,dh)
            assert float(np.max(np.abs(a32.detach().numpy()-rel['per_anchor_smoothed_ap'])))<2e-6
            assert np.count_nonzero(np.diag(ds[:,:64]))==0
            c=r['components'];weights=config['LOSS']
            assert len(c)==14 and all(math.isfinite(v) for v in c.values())
            target=(smooth if end=='smooth_ap' else hard) if ix>=2 else basic
            assert abs(c['triplet_fused']-target)<2e-6
            role_errors={}
            for ai,role in enumerate(('cnn','transformer','mamba'),1):
                loss,_=reference_hard(arrays[ai,:,:64],ids)
                role_errors[role]=abs(loss-c['triplet_'+role])
                assert role_errors[role]<2e-6
            ledger=weights['ID_FUSED']*c['id_fused']+weights['TRIPLET_FUSED']*c['triplet_fused']
            for role in ('cnn','transformer','mamba'):
                for term,weight in [('id','ID_BRANCH'),('triplet','TRIPLET_BRANCH'),('id_residual','ID_RESIDUAL'),('triplet_residual','TRIPLET_RESIDUAL')]:ledger+=weights[weight]*c[term+'_'+role]
            assert abs(ledger-r['loss'])<1e-5 and r['amp_scale_before']==r['amp_scale_after']
            cur=arrays[0,:,:64];hist=arrays[0,:,64:];label=np.asarray(ids);hid=np.asarray(allids[64:]);scene=np.asarray(a['scenes']);hs=np.asarray([m['scene'] for m in expected_memory])
            pm=(label[:,None]==label)&~np.eye(64,dtype=bool);nm=label[:,None]!=label
            hp=np.where(pm,cur,-np.inf).max(1);hn=np.where(nm,cur,np.inf).min(1)
            hm=label[:,None]==hid
            mph=np.where(hm,hist,-np.inf).max(1) if hist.shape[1] else np.full(64,-np.inf)
            mhn=np.where(~hm,hist,np.inf).min(1) if hist.shape[1] else np.full(64,np.inf)
            up=np.maximum(hp,mph);un=np.minimum(hn,mhn)
            stats=dict(memory_records=len(expected_memory),memory_positive_pairs=int(hm.sum()),memory_negative_pairs=int((~hm).sum()),memory_cross_scene_positive_pairs=int((hm&(scene[:,None]!=hs)).sum()),memory_negative_violations_against_batch_hard_positive=int(((~hm)&(hist<hp[:,None]+np.float32(.3))).sum()),harder_positive_anchors=int((mph>hp).sum()),harder_negative_anchors=int((mhn<hn).sum()),current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((up>=un).sum()),expanded_hinge_positive_anchors=int((up-un+np.float32(.3)>0).sum()),maximum_memory_age=max((m['age'] for m in expected_memory),default=0))
            assert all(a['statistics'][k]==v for k,v in stats.items())
            assert abs(a['statistics']['current_triplet']-basic)<2e-6 and abs(a['statistics']['expanded_triplet']-hard)<2e-6
            norms=a['historical_leaf_upstream_norms'];assert len(norms)==len(expected_memory) and all(math.isfinite(v) and v>=0 for v in norms)
            vgroups=sorted({m['stored_step'] for m,n in zip(expected_memory,norms,strict=True) if n>0})
            assert vgroups==a['history_vjp_groups'] and a['history_vjp_record_forwards']==64*len(vgroups)
            fg=64*len({m['stored_step'] for m in expected_memory});assert fg==a['fresh_role_record_forwards']
            fresh+=fg;vjp+=a['history_vjp_record_forwards'];history_candidates+=len(expected_memory)
            for role in ('cnn','transformer','mamba'):
                pair=a['roles'][role]['total_vs_history'];both=a['roles'][role]['total_vs_both'];applied=a['applied_gradients'][role]
                for comparison in (pair,both,applied):
                    u,v,diff,cos=[comparison[k] for k in ('first_norm','second_norm','difference_norm','cosine')]
                    assert all(math.isfinite(t) and t>=0 for t in (u,v,diff))
                    assert (cos is None)==(u==0 or v==0)
                    if cos is not None:assert abs(diff*diff-(u*u+v*v-2*u*v*cos))<1e-7*max(1,u*u+v*v)
                assert abs(both['difference_norm']-pair['second_norm'])<1e-6*max(1,pair['second_norm'])
                for key in both:
                    if both[key] is None:assert applied[key] is None
                    else:assert abs(both[key]-applied[key])<1e-6*max(1,abs(both[key]))
            direct=a['direct_single_group_check']
            if direct:
                directs+=1;assert len({m['stored_step'] for m in expected_memory})==1
                u,v,diff,cos=[direct[k] for k in ('first_norm','second_norm','difference_norm','cosine')]
                assert abs(diff*diff-(u*u+v*v-2*u*v*cos))<1e-7*max(1,u*u+v*v)
                assert abs(direct['relative_l2_error']-diff/u)<1e-12 and diff/u<=.005
            if all(a['roles'][role]['total_vs_history']['second_norm']>0 for role in ('cnn','transformer','mamba')):witnesses.append(ix+1)
            pixels.append((indices,a['pixel_sha256']))
            if ix>=2:
                for i in indices:cache.pop(i,None);cache[i]=ix
                while len(cache)>512:cache.popitem(last=False)
            elements+=size
            out['steps'].append({'endpoint':folder.name,'step':ix+1,'memory':len(expected_memory),'positive_counts':rel['positive_counts'],'independent_ap':aps.tolist(),'max_saved_ap_error':ap_error,'max_scalar_error':scalar_error,'smooth_distance_jacobian_max_error_float64':grad64_error,'smooth_distance_jacobian_max_error_float32':grad32_error,'hard_distance_jacobian_bitwise_equal':True,'role_triplet_errors':role_errors,'ledger_error':abs(ledger-r['loss']),'direct_relative_l2_error':direct.get('relative_l2_error'),'gradient_moment_consistency':True})
        assert f.read()==b''
    assert fresh==tr['extra_fresh_role_record_forwards'] and vjp==tr['extra_history_vjp_record_forwards'] and directs*64==tr['extra_direct_check_record_forwards']
    assert directs==(0 if mode=='overfit' else 1)
    if mode=='capacity':
        witness=tr['historical_parameter_gradient_witness'];assert witness['step']==witnesses[0] and witness['roles']==audits[witness['step']-1]['roles']
    else:
        assert history_candidates==vjp==0 and all(p==pixels[0] for p in pixels)
    assert abs(sum(r['loss'] for r in tr['steps'])/expected-tr['history'][0]['mean_loss'])<1e-12
    out['endpoints'].append({'name':folder.name,'steps':expected,'current_anchor_exposures':64*expected,'unique_records':len(set(i for inds,p in pixels for i in inds)),'fresh_record_forwards':fresh,'history_vjp_forwards':vjp,'direct_record_forwards':directs*64,'history_candidates':history_candidates,'matrix_elements':elements,'actual_fit_seconds':tr['history'][0]['elapsed_seconds'],'peak_allocated_mib':tr['peak_allocated_mib'],'peak_reserved_mib':tr['peak_reserved_mib']})
    return tr,pixels

for frow,fold,baseline in zip(summary['folds'],protocol['folds'],b0['folds'],strict=True):
    original=torch.load(baseline['checkpoint'],map_location='cpu',weights_only=True)
    assert original['source_ids']==fold['source_ids'] and original['heldout_ids']==fold['heldout_ids']
    source_state=original['model_state_dict'];source_hash=state_digest(source_state)
    assert source_hash==baseline['training']['final_state_sha256']
    source_steps=baseline['training']['steps']
    assert len(source_steps)==650 and all(set(r['sampled_record_indices'])<=set(fold['source_record_indices']) for r in source_steps)
    pair=[]
    for end in ('control','smooth_ap'):
        folder=M0/f"fold_{fold['fold']}_{end}";row=j(folder/'receipt.json')
        assert row==frow['endpoints'][end]
        tr,pixels=replay(folder,fold,end,'capacity');pair.append(pixels)
        assert tr==row['training']
        p=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        assert p['binding']==row['initialization'] and p['config_sha256']==summary['config_sha256']
        assert p['source_ids']==fold['source_ids'] and p['heldout_ids']==fold['heldout_ids'] and p['fold']==fold['fold']
        assert row['initialization']['signal_state_sha256']==source_hash and not row['initialization']['role_weights_loaded']
        aliases=p['baseline_aliases'];state={key:source_state[value] for key,value in aliases.items()};state.update(p['role_state_dict'])
        actual_hash=state_digest(state);assert actual_hash==tr['final_state_sha256']==row['strict_reload_state_sha256']
        trainable=row['initialization']['trainable_names'];assert len(trainable)==203 and len([n for n in trainable if n.startswith('encoder.')])==189
        assert set(trainable)<=set(p['role_state_dict'])
        frozen={k:v for k,v in state.items() if k.startswith('baseline.') or (k not in trainable and not k.endswith(('running_mean','running_var','num_batches_tracked')))}
        assert state_digest(frozen)==tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
        assert sum(state[n].numel() for n in trainable)==row['initialization']['trainable_parameters']
        assert all(torch.isfinite(v).all() for v in state.values())
        out['checkpoints'].append({'fold':fold['fold'],'endpoint':end,'path':row['checkpoint'],'sha256':digest(row['checkpoint']),'full_state_sha256':actual_hash,'baseline_state_sha256':source_hash,'state_tensors':len(state),'frozen_tensors':len(frozen),'baseline_aliases':len(aliases),'trainable_tensors':len(trainable),'encoder_trainable_tensors':189,'trainable_parameters':row['initialization']['trainable_parameters'],'total_parameters_receipted':row['initialization']['total_parameters'],'source_training_steps_verified':len(source_steps),'strict_output_reload_independently_reexecuted':False})
        del p,state,frozen
    assert pair[0]==pair[1] and frow['endpoints']['control']['initialization']==frow['endpoints']['smooth_ap']['initialization']
    del original,source_state
pair=[]
for end in ('control','smooth_ap'):
    tr,pixels=replay(M0/('overfit_'+end),protocol['folds'][0],end,'overfit');pair.append(pixels)
    row=summary['overfit'][end];assert tr==row['training']
    k=len(protocol['folds'][0]['source_ids']);p=1-.1+.1/k;q=.1/k
    floor=.75*(-p*math.log(p)-(k-1)*q*math.log(q))
    first,last=tr['steps'][0]['loss'],tr['steps'][-1]['loss'];ratio=(last-floor)/(first-floor)
    assert abs(floor-row['gate']['minimum_loss'])<1e-14 and abs(ratio-row['gate']['loss_ratio'])<1e-14 and ratio<=.1
    out['overfit'].append({'endpoint':end,'classes':k,'floor':floor,'first_loss':first,'last_loss':last,'excess_ratio':ratio,'historic_candidates':0})
assert pair[0]==pair[1]
assert len(out['steps'])==248 and sum(x['matrix_elements'] for x in out['endpoints'])==4945920
assert not torch.cuda.is_initialized()
out['elapsed_seconds']=time.perf_counter()-started
out['status']='PASS_ALL_FEASIBLE_M0_SAVED_ARRAY_AND_CHECKPOINT_ARITHMETIC'
print(json.dumps(out))
