"""Independent full Q1 arithmetic. Only stdlib, NumPy and tensor deserialization/BLAS.

No project objectives, verifiers, model code, image reads, training or filesystem writes.
AP is rebuilt from positive hit ranks. Smooth-AP uses explicit positive-versus-valid
pair sums in float64. Saved gradient norms remain runtime witnesses.
"""
from pathlib import Path
from collections import OrderedDict, Counter
import datetime
import hashlib
import json
import math
import time
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
OUTS=('baseline_only','fused','cnn','transformer','mamba')
ENDS=('control','cross_scene')
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
errors={}
def near(actual,expected,key,tol):
    delta=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    errors[key]=max(errors.get(key,0.),delta)
    assert delta<=tol,(key,delta,tol)

def softap(d,ids,scenes,cross):
    result=[];counts=[]
    for i in range(64):
        own=ids==ids[i]
        if cross:
            positive=own & (scenes!=scenes[i]);valid=~(own & (scenes==scenes[i]))
        else:
            positive=own.copy();positive[i]=False
            valid=np.ones(len(ids),dtype=bool);valid[i]=False
        p=np.flatnonzero(positive);counts.append(len(p))
        if not len(p):result.append(0.);continue
        # s_j-s_p = (d_p^2-d_j^2)/2; exclude each positive's own comparison.
        logits=(d[i,p,None]**2-d[i,None,:]**2)/.02
        comparisons=1/(1+np.exp(-logits))
        comparisons[:,~valid]=0
        comparisons[np.arange(len(p)),p]=0
        precision=(1+comparisons[:,positive].sum(1))/(1+comparisons.sum(1))
        result.append(float(precision.mean()))
    ap=np.asarray(result);counts=np.asarray(counts)
    selected=ap[counts>0]
    return float(1-selected.mean()) if len(selected) else 0.,ap,counts

def hard(d,ids):
    own=ids[:64,None]==ids[None,:]
    own[np.arange(64),np.arange(64)]=False
    negative=ids[:64,None]!=ids[None,:]
    far=np.where(own,d,-np.inf).max(1)
    close=np.where(negative,d,np.inf).min(1)
    return float(np.maximum(far-close+.3,0).mean()),far,close

def metric(ap,first):
    return dict(mAP=float(np.mean(ap)*100),**{'Rank-'+str(k):float(np.mean(np.asarray(first)<=k)*100) for k in (1,5,10)})

def bootstrap(delta,labels):
    unique=np.unique(labels)
    totals=np.array([delta[labels==i].sum() for i in unique]);counts=np.array([(labels==i).sum() for i in unique])
    samples=np.random.default_rng(42).integers(len(unique),size=(10000,len(unique)))
    values=totals[samples].sum(1)/counts[samples].sum(1)
    return float(np.percentile(values,2.5,method='linear'))

start=time.perf_counter();torch.set_num_threads(56)
spec_path=ROOT/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
spec=read(spec_path);base=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
signal=read(ROOT/base['BASELINE']['CONFIG']);protocol=read(ROOT/signal['protocol'])
meta=read(ROOT/base['SOURCE_METADATA']['PATH']);baseline=read(base['BASELINE']['SUMMARY'])
summary=read(RUN/'q1/summary.json');cpu=read(RUN/'q1_cpu.json');pipeline=read(RUN/'pipeline.json')
assert pipeline['status']=='COMPLETE_VERIFIED_'+summary['status']
assert all(x['exit_code']==0 for x in pipeline['stages'])
assert sha(RUN/'q1/summary.json')==cpu['summary_sha256']==pipeline['terminal_summary_sha256']
assert sha(RUN/'q1_cpu.json')==pipeline['terminal_cpu_sha256']
assert summary['config_sha256']==sha(spec_path)==pipeline['config_sha256']
assert summary['optimizer_steps']==1560 and summary['heldout_record_forwards']==2064 and summary['official_image_reads']==0
support=read(ROOT/spec['source_support'])
training=[];step_results=[];epoch_results=[];query_results=[];fold_metrics={e:[] for e in ENDS}
aps={e:{o:[] for o in OUTS} for e in ENDS};firsts={e:{o:[] for o in OUTS} for e in ENDS}
distance_results=[];files={};zero_batches=[];held_gallery_total=0
all_unique_source=set();same_warmup_losses=[]
for f,fold,b0,md in zip(protocol['folds'],summary['folds'],baseline['folds'],meta['folds']):
    assert f['fold']==fold['fold']==b0['fold']==md['fold']
    k=f['fold'];source=set(f['source_record_indices']);paired_pixels=[];paired_start=[];paired_warmup=[]
    for end in ENDS:
        folder=RUN/'q1'/f'fold_{k}_{end}';item=fold['endpoints'][end];tr=item['training']
        assert read(folder/'receipt.json')==item and read(folder/'training.json')==tr
        assert tr['mode']=='comparison' and tr['epochs']==20 and tr['optimizer_steps']==260
        assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and not tr['missing_nonzero_gradients']
        assert tr['overflow_events']==0 and all(item['engineering_checks'].values())
        assert tr['initial_state_sha256']==item['initialization']['initial_state_sha256']
        paired_start.append(item['initialization']);paired_warmup.append(tr['steps'][:65])
        logs=[json.loads(s) for s in (folder/'memory_steps.jsonl').read_text().splitlines()]
        assert len(logs)==len(tr['steps'])==260
        for n,rec in tr['audit_files'].items():
            assert sha(folder/n)==rec['sha256'] and (folder/n).stat().st_size==rec['bytes']
        paired_pixels.append([(a['record_indices'],a['pixel_sha256']) for a in logs])
        cache=OrderedDict();fresh_count=64;vjp_count=0;distance_count=0;history_count=0;eligible_active=0;unique_used=set()
        fzero=[];expected_offset=0
        with (folder/'memory_distances.f32').open('rb') as stream:
            for n,(row,a) in enumerate(zip(tr['steps'],logs)):
                idx=a['record_indices'];unique_used.update(idx);all_unique_source.update(idx)
                assert idx==row['sampled_record_indices']==md['batches'][n]['record_indices']
                assert len(idx)==64 and set(idx)<=source
                assert row['step']==a['step']==n+1 and a['zero_based_step']==n
                assert row['epoch']==n//13+1
                ids=[protocol['records'][i]['identity'] for i in idx];sc=[protocol['records'][i]['scene'] for i in idx]
                assert ids==a['identities'] and sc==a['scenes']
                assert sorted(Counter(ids).values())==[8]*8
                for record in [record for record,age in cache.items() if n-age>8]:del cache[record]
                wanted=[dict(record_index=r,identity=protocol['records'][r]['identity'],scene=protocol['records'][r]['scene'],age=n-at,stored_step=at)
                        for r,at in cache.items() if r not in idx]
                assert wanted==a['memory'];m=len(wanted);history_count+=m
                assert a['warmup_steps']==65 and a['replacement_active']==(n>=65)
                target=('smooth_ap' if end=='control' else 'cross_scene_smooth_ap') if n>=65 else 'hard_triplet'
                assert row['active_fused_metric']==target
                assert a['history_anchor_count']==0 and a['current_anchor_count']==64 and a['coordinate_rule']=='fresh'
                assert a['distance_offset_bytes']==stream.tell()==expected_offset
                arr=np.fromfile(stream,dtype='<f4',count=4*64*(64+m))
                assert len(arr)==a['distance_float_count']==4*64*(64+m) and np.isfinite(arr).all() and (arr>=0).all()
                expected_offset+=arr.nbytes;distance_count+=len(arr)
                values=arr.astype('float64').reshape(4,64,64+m)
                poolids=np.array(ids+[r['identity'] for r in wanted]);poolsc=np.array(sc+[r['scene'] for r in wanted])
                basic,hp,hn=hard(values[0,:,:64],poolids[:64]);pooled,php,phn=hard(values[0],poolids)
                standard,ap,counts=softap(values[0],poolids,poolsc,False)
                cross,cap,ccounts=softap(values[0],poolids,poolsc,True)
                r=a['relation_objective'];eligible=int((ccounts>0).sum())
                assert r['temperature']==.01 and r['cross_scene_reduction']=='mean_eligible_anchors'
                assert r['ineligible_ap_storage_value']==0 and r['positive_counts']==counts.tolist()
                assert r['cross_scene_positive_counts']==ccounts.tolist() and r['cross_scene_eligible_anchors']==eligible
                near(r['per_anchor_smoothed_ap'],ap,'standard_ap',2e-6);near(r['cross_scene_per_anchor_ap'],cap,'cross_scene_ap',2e-6)
                for label,expected in (('hard_loss',pooled),('smooth_ap_loss',standard),('cross_scene_loss',cross)):
                    near(r[label],expected,label,2e-6)
                near(a['original_triplet'],basic,'current_hard',2e-6)
                near(a['statistics']['current_triplet'],basic,'statistics_current',2e-6)
                near(a['statistics']['expanded_triplet'],pooled,'statistics_pooled',2e-6)
                if n>=65:eligible_active+=eligible
                registered=support['folds'][k]['batches'][n]
                assert registered['counts']['anchors_with_pool_cross']==eligible
                assert registered['counts']['cross_positive_positions']==int(ccounts.sum())
                selected=(standard if end=='control' else cross) if n>=65 else basic
                comp=row['components'];weights=base['LOSS']
                expected_names={'id_fused','triplet_fused'}|{t+'_'+role for role in OUTS[2:] for t in ('id','triplet','id_residual','triplet_residual')}
                assert set(comp)==expected_names and all(math.isfinite(v) and v>=0 for v in comp.values())
                near(comp['triplet_fused'],selected,'selected_fused_objective',2e-6)
                for j,role in enumerate(OUTS[2:],1):
                    rolehard,_,_=hard(values[j,:,:64],poolids[:64]);near(comp['triplet_'+role],rolehard,'role_hard',2e-6)
                total=weights['ID_FUSED']*comp['id_fused']+weights['TRIPLET_FUSED']*comp['triplet_fused']
                for role in OUTS[2:]:
                    total+=sum(weights[w]*comp[t+'_'+role] for w,t in (('ID_BRANCH','id'),('TRIPLET_BRANCH','triplet'),('ID_RESIDUAL','id_residual'),('TRIPLET_RESIDUAL','triplet_residual')))
                near(row['loss'],total,'weighted_14_term_ledger',1e-5)
                assert row['amp_scale_after']>=row['amp_scale_before']>0
                group_count=len({r['stored_step'] for r in wanted})
                assert a['fresh_role_record_forwards']==64*group_count;fresh_count+=64*group_count
                upstream=np.asarray(a['historical_leaf_upstream_norms'])
                assert len(upstream)==m and np.isfinite(upstream).all() and (upstream>=0).all()
                active_groups=sorted({r['stored_step'] for j,r in enumerate(wanted) if upstream[j]>0})
                assert active_groups==a['history_vjp_groups'] and a['history_vjp_record_forwards']==64*len(active_groups)
                vjp_count+=a['history_vjp_record_forwards']
                assert a['gradient_weight']==1 and not a['direct_single_group_check']
                assert all(a[name] for name in ('history_candidate_vjp_applied','selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'))
                for role in OUTS[2:]:
                    pair=a['roles'][role]['total_vs_history'];both=a['roles'][role]['total_vs_both'];applied=a['applied_gradients'][role]
                    for g in (pair,both,applied):
                        x,y,d=g['first_norm'],g['second_norm'],g['difference_norm'];cos=g['cosine']
                        assert all(math.isfinite(v) and v>=0 for v in (x,y,d))
                        assert (cos is None)==(x==0 or y==0)
                        if cos is not None:
                            assert abs(cos)<=1.00001
                            near(d*d,x*x+y*y-2*x*y*cos,'gradient_norm_identity',1e-7*max(1,x*x+y*y))
                    near(both['difference_norm'],pair['second_norm'],'history_norm_addition',1e-6*max(1,pair['second_norm']))
                    assert both==applied
                if not eligible:
                    z=dict(fold=k,endpoint=end,step=n+1,epoch=row['epoch'],replacement_active=n>=65,
                           cross_loss=cross,active_loss=row['loss'],historical_candidates=m,
                           historical_upstream_nonzero=int((upstream>0).sum()),history_vjp_forwards=a['history_vjp_record_forwards'])
                    zero_batches.append(z);fzero.append(n+1)
                    if end=='cross_scene' and n>=65:
                        assert comp['triplet_fused']==0 and all(v==0 for v in upstream)
                        assert not active_groups and row['loss']>0
                        assert all(a['roles'][role]['total_vs_history']['second_norm']==0 for role in OUTS[2:])
                step_results.append(dict(fold=k,endpoint=end,step=n+1,epoch=row['epoch'],eligible=eligible,
                    historical_candidates=m,standard_loss=standard,cross_scene_loss=cross,hard_pooled_loss=pooled,
                    active_loss=row['loss'],other_thirteen_weighted_loss=total-comp['triplet_fused'],
                    fresh_forwards=64*group_count,history_vjp_forwards=a['history_vjp_record_forwards']))
                if n>=65:
                    for idx0 in idx:cache.pop(idx0,None);cache[idx0]=n
                    while len(cache)>512:cache.popitem(last=False)
            assert stream.read()==b''
        assert fresh_count==tr['extra_fresh_role_record_forwards'] and vjp_count==tr['extra_history_vjp_record_forwards']
        assert tr['extra_direct_check_record_forwards']==0
        for h in tr['history']:
            ep=h['epoch'];selected=[r for r in tr['steps'] if r['epoch']==ep]
            assert len(selected)==h['optimizer_steps']==13
            near(h['mean_loss'],np.mean([r['loss'] for r in selected]),'epoch_mean',1e-12)
            multiplier=ep/5 if ep<=5 else (1+math.cos(math.pi*(ep-6)/15))/2
            near(h['learning_rate'],.00035*multiplier,'learning_rate',1e-15)
            epoch_results.append(dict(fold=k,endpoint=end,**h))
        training.append(dict(fold=k,endpoint=end,steps=260,source_anchor_exposures=16640,unique_source_records_observed=len(unique_used),
            source_registry_records=len(source),post_warmup_anchor_exposures=12480,post_warmup_eligible_exposures=eligible_active,
            historical_candidate_exposures=history_count,distance_elements=distance_count,zero_eligible_steps=fzero,
            extra_fresh_role_record_forwards=fresh_count,extra_history_vjp_record_forwards=vjp_count,direct_checks=0,
            gradient_nonzero_coverage='cumulative 203/203; not every-step parameter-vector evidence'))
        retrieval=item['retrieval'];arrays=torch.load(folder/'retrieval_arrays.pt',map_location='cpu',weights_only=True)
        ref=torch.load(Path(b0['checkpoint']).parent/'retrieval_arrays.pt',map_location='cpu',weights_only=True)
        ranks=read(folder/'rankings.json')
        assert sha(folder/'retrieval_arrays.pt')==retrieval['retrieval_arrays_sha256']
        assert sha(folder/'rankings.json')==retrieval['rankings_sha256']
        assert retrieval['query_rows']==f['query_rows']
        gal=[protocol['records'][i] for i in f['gallery_record_indices']]
        assert retrieval['gallery_manifest']==gal
        positions=[r['gallery_position'] for r in f['query_rows']]
        assert arrays['query_gallery_positions']==positions and arrays['gallery_record_indices']==f['gallery_record_indices']
        assert torch.equal(arrays['features']['baseline_only'],ref['features'])
        assert torch.equal(arrays['distances']['baseline_only'],ref['distances'])
        assert torch.equal(arrays['features']['fused'][:,:3072],ref['features'])
        held_gallery_total+=len(gal);fm={}
        for output in OUTS:
            feat=arrays['features'][output].float();saved=arrays['distances'][output]
            assert feat.shape==(len(gal),3072 if output=='baseline_only' else 7680 if output=='fused' else 4608)
            assert torch.isfinite(feat).all() and torch.isfinite(saved).all()
            unit=feat/torch.linalg.vector_norm(feat,dim=1,keepdim=True).clamp_min(1e-12)
            q=unit[positions];rebuilt=(q*q).sum(1,keepdim=True)+(unit*unit).sum(1)[None,:]
            rebuilt.addmm_(q,unit.T,beta=1,alpha=-2)
            assert torch.equal(rebuilt,saved),(k,end,output,'distance backend equality')
            order=np.argsort(saved.numpy(),axis=1)
            assert order.tolist()==ranks[output]
            ap=[];first=[]
            for qi,order0 in enumerate(order):
                assert sorted(order0.tolist())==list(range(len(gal)))
                query=gal[positions[qi]]
                legal=[gal[j] for j in order0 if not(gal[j]['identity']==query['identity'] and gal[j]['scene']==query['scene'])]
                hit=[rank for rank,r in enumerate(legal,1) if r['identity']==query['identity']]
                assert len(hit)==f['query_rows'][qi]['valid_positives']
                value=sum(n/rank for n,rank in enumerate(hit,1))/len(hit)
                ap.append(value);first.append(hit[0])
                query_results.append(dict(fold=k,endpoint=end,output=output,record_index=query['index'],identity=query['identity'],scene=query['scene'],ap=value,first_rank=hit[0]))
            score=retrieval['outputs'][output]
            near(ap,score['average_precision'],'retrieval_ap',1e-14);assert first==score['first_match_rank']
            fm[output]=metric(ap,first)
            for name,value in fm[output].items():near(value,score['metrics'][name],'retrieval_metric',1e-10)
            aps[end][output].extend(ap);firsts[end][output].extend(first)
            distance_results.append(dict(fold=k,endpoint=end,output=output,distance_elements=saved.numel(),
                                         exact_distance_reconstruction=True,complete_rank_permutations=True,metrics=fm[output]))
        fold_metrics[end].append(fm)
        for p in sorted(folder.iterdir()):
            if p.is_file():files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    assert paired_pixels[0]==paired_pixels[1] and paired_start[0]==paired_start[1]
    diffs=[abs(a['loss']-b['loss']) for a,b in zip(*paired_warmup)]
    component_diffs={key:max(abs(a['components'][key]-b['components'][key]) for a,b in zip(*paired_warmup)) for key in paired_warmup[0][0]['components']}
    same_warmup_losses.append(dict(fold=k,all_first65_step_records_exact=paired_warmup[0]==paired_warmup[1],
        maximum_loss_difference=max(diffs),nonidentical_loss_steps=sum(v!=0 for v in diffs),
        first_nonidentical_loss_step=next((n+1 for n,v in enumerate(diffs) if v!=0),None),maximum_component_differences=component_diffs))
assert len(query_results)==6000 and sum(r['distance_elements'] for r in distance_results)==2069520
assert len(step_results)==1560 and len(epoch_results)==120 and held_gallery_total==2064
labels=np.array([r['identity'] for r in query_results if r['endpoint']=='control' and r['output']=='fused'])
assert len(labels)==600 and len(set(labels))==60
aps={e:{o:np.array(v) for o,v in outputs.items()} for e,outputs in aps.items()}
comparison=summary['comparison'];endpoints={};identity_results=[]
for end in ENDS:
    metrics={o:metric(aps[end][o],firsts[end][o]) for o in OUTS}
    gains={o:metrics[o]['mAP']-metrics['baseline_only']['mAP'] for o in OUTS}
    fg=[r['fused']['mAP']-r['baseline_only']['mAP'] for r in fold_metrics[end]]
    lower=bootstrap((aps[end]['fused']-aps[end]['baseline_only'])*100,labels)
    checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_gains_nonnegative=all(v>=0 for v in fg),
        all_full_branches_not_below_signal=all(gains[o]>=0 for o in OUTS[2:]),identity_bootstrap_lower_positive=lower>0,
        fused_strictly_best=all(metrics['fused']['mAP']>metrics[o]['mAP'] for o in OUTS if o!='fused'))
    saved=comparison['endpoints'][end]
    assert checks==saved['scientific_checks'] and all(checks.values())==saved['scientific_passed']
    near(lower,saved['identity_bootstrap']['lower_bound_pp'],'bootstrap',1e-10)
    near(fg,saved['fold_fused_gains_pp'],'fold_gains',1e-10)
    for o in OUTS:
        for key,value in metrics[o].items():near(value,saved['metrics'][o][key],'aggregate_metric',1e-10)
    endpoints[end]=dict(metrics=metrics,gains_over_signal_pp=gains,fold_gains=fg,bootstrap_lower_pp=lower,checks=checks)
for ident in sorted(set(labels)):
    mask=labels==ident;saved=next(r for r in comparison['paired_per_identity'] if r['identity']==ident)
    assert saved['query_count']==int(mask.sum())
    for o in OUTS:
        gain=float(((aps['cross_scene'][o]-aps['control'][o])*100)[mask].mean())
        near(gain,saved['gains_mAP'][o],'identity_gain',1e-10)
        identity_results.append(dict(identity=int(ident),output=o,query_count=int(mask.sum()),gain_pp=gain))
gain={o:float(((aps['cross_scene'][o]-aps['control'][o])*100).mean()) for o in OUTS}
fg=[fold_metrics['cross_scene'][i]['fused']['mAP']-fold_metrics['control'][i]['fused']['mAP'] for i in range(3)]
lower=bootstrap((aps['cross_scene']['fused']-aps['control']['fused'])*100,labels)
checks=dict(fused_gain_at_least_1pp=gain['fused']>=1,all_fold_fused_nonnegative=all(v>=0 for v in fg),
    all_role_gains_nonnegative=all(gain[o]>=0 for o in OUTS[2:]),paired_identity_bootstrap_lower_positive=lower>0,
    candidate_fused_strictly_best=endpoints['cross_scene']['checks']['fused_strictly_best'])
assert checks==comparison['paired_checks'] and all(checks.values())==comparison['paired_pass']
near(lower,comparison['paired_bootstrap_lower_pp'],'paired_bootstrap',1e-10)
near(fg,comparison['fold_fused_gains_mAP'],'paired_fold',1e-10)
for o in OUTS:near(gain[o],comparison['matched_gains_mAP'][o],'paired_gain',1e-10)
qualified=all(checks.values()) and all(endpoints['cross_scene']['checks'].values())
assert qualified==comparison['next_phase_qualified']==(summary['status']=='Q1_PASS')
assert np.array_equal(aps['control']['baseline_only'],aps['cross_scene']['baseline_only'])
totals=dict(training_steps=len(step_results),training_anchor_exposures=len(step_results)*64,
    unique_source_records_across_folds=len(all_unique_source),training_distance_elements=sum(r['distance_elements'] for r in training),
    fresh_role_record_forwards=sum(r['extra_fresh_role_record_forwards'] for r in training),
    history_vjp_record_forwards=sum(r['extra_history_vjp_record_forwards'] for r in training),
    direct_check_record_forwards=0,heldout_record_forwards=held_gallery_total,query_output_rows=len(query_results),
    unique_queries=600,unique_query_identities=60,identity_output_changes=len(identity_results))
assert totals['training_distance_elements']==cpu['checked_memory_distance_elements']
assert totals['history_vjp_record_forwards']==cpu['checked_history_vjp_record_forwards']
print(json.dumps(dict(status='PASS_COMPLETE_INDEPENDENT_Q1_ARITHMETIC',scientific_qualification='PASS' if qualified else 'FAIL',
    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
    totals=totals,errors=errors,training=training,steps=step_results,epochs=epoch_results,zero_batches=zero_batches,
    distances=distance_results,query_rows=query_results,identity_rows=identity_results,endpoints=endpoints,
    paired_gains_pp=gain,paired_fold_gains_pp=fg,paired_bootstrap_lower_pp=lower,paired_checks=checks,
    warmup_pairing=same_warmup_losses,files=files,model_forwards=0,optimizer_updates=0,official_test_reads=0,
    limits=['No original model forwards or parameter-gradient vectors regenerated; gradient norm checks verify saved scalar consistency.',
            'Ten of fourteen supervised component scalars are inputs to ledger checks, not reconstructed from logits/residual features.',
            'Stored feature provenance is bound to runtime receipts; CPU distance reconstruction does not rerun the model.'])))
