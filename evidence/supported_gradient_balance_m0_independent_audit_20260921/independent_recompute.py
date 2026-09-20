"""Fresh M0 audit. No project imports, model construction, forwards, or updates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.nice(10)
from pathlib import Path
from collections import Counter,OrderedDict
import hashlib,json,math,re,time,datetime,subprocess,gc
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
START=time.perf_counter()
REPO=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
COMMIT='1381639f778f77f124a2726ee55c07610092a438'
ROLES=('cnn','transformer','mamba')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def state_sha(d):
    h=hashlib.sha256()
    for name,v in sorted(d.items()):h.update(name.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def close(a,b,tol=1e-7):
    assert math.isfinite(a) and math.isfinite(b),(a,b)
    assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b,tol)
def absolute(a,b,tol):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert math.isfinite(error) and error<=tol,(error,tol)
    return error
pair_count=0
def pair(p):
    global pair_count
    a,b,d=p['first_norm'],p['second_norm'],p['difference_norm'];c=p['cosine']
    assert all(math.isfinite(x) and x>=0 for x in (a,b,d))
    assert (c is None)==(not a or not b)
    if c is not None:assert math.isfinite(c) and abs(c)<=1.00001
    close(d*d,a*a+b*b-2*(0 if c is None else a*b*c))
    pair_count+=1
def all_pairs(obj):
    if isinstance(obj,dict):
        if {'first_norm','second_norm','difference_norm','cosine'}<=obj.keys():pair(obj)
        else:
            for value in obj.values():all_pairs(value)
    elif isinstance(obj,list):
        for value in obj:all_pairs(value)
def dot(p):return 0 if p['cosine'] is None else p['first_norm']*p['second_norm']*p['cosine']
CFG_PATH=REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json'
SPEC=read(CFG_PATH);CFG_SHA=sha(CFG_PATH)
assert hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+str(CFG_PATH.relative_to(REPO))],cwd=REPO)).hexdigest()==CFG_SHA
BASE=read(REPO/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
SIGNAL=read(REPO/BASE['BASELINE']['CONFIG'])
PROTOCOL=read(REPO/SIGNAL['protocol']);RECORDS=PROTOCOL['records']
META=read(REPO/BASE['SOURCE_METADATA']['PATH'])
SUPPORT=read(REPO/SPEC['source_support'])
T0=read(RUN/'t0.json');SUMMARY=read(RUN/'m0/summary.json');CPU=read(RUN/'m0_cpu.json')
B0=read(BASE['BASELINE']['SUMMARY'])
assert SUMMARY['status']=='PASS_ENGINEERING_ONLY' and SUMMARY['mode']=='m0'
assert SUMMARY['project_commit']==COMMIT
assert SUMMARY['config_sha256']==T0['config_sha256']==CFG_SHA
assert SUMMARY['runner_sha256']==sha(REPO/'tools/train_msvr_supported_gradient_balance.py')
assert CPU['summary_sha256']==sha(RUN/'m0/summary.json')
assert SUMMARY['seed']==42 and SUMMARY['heldout_record_forwards']==SUMMARY['official_image_reads']==0
assert SUMMARY['gradient_balance']==SPEC['gradient_balance']
assert SPEC['implementation_revision']=='r2_direct_auxiliary'
assert SPEC['gradient_balance']==dict(ema_decay=.9,exponent=.5,ratio_min=.25,ratio_max=4.,epsilon=1e-12,weight_sum=2.)
assert SPEC['memory']==dict(capacity=512,maximum_age=8,warmup_steps=65,capacity_warmup_steps=2,overfit_warmup_steps=2)
assert META['protocol_sha256']==sha(REPO/SIGNAL['protocol'])
for i,r in enumerate(RECORDS):
    assert r['index']==i and len(r['paths'])==3
    for mod,p in zip(('vis','ni','th'),r['paths']):
        parts=Path(p).parts
        assert parts[:3]==('bounding_box_train',f"{r['identity']:04d}",mod)
        match=re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_\d+\.jpg',parts[-1]);assert match
        assert tuple(map(int,match.groups()))==(r['identity'],r['scene'],r['camera'])
ids=sorted({r['identity'] for r in RECORDS})
scene_members={i:{r['scene'] for r in RECORDS if r['identity']==i} for i in ids}
assert PROTOCOL['identity_scene_membership']=={str(i):sorted(s) for i,s in scene_members.items()}
eligible=[i for i in ids if len(scene_members[i])>1];ineligible=[i for i in ids if len(scene_members[i])==1]
assert len(RECORDS)==1032 and (len(ids),len(eligible),len(ineligible))==(155,60,95)
source_coverage=[];t0_totals=Counter()
def queue_read(q,step,current):
    for i in list(q):
        if step-q[i]>8:del q[i]
    return [dict(record_index=i,identity=RECORDS[i]['identity'],scene=RECORDS[i]['scene'],age=step-s,stored_step=s) for i,s in q.items() if i not in set(current)]
def queue_update(q,step,current):
    for i in current:
        q.pop(i,None);q[i]=step
    while len(q)>512:q.popitem(last=False)
def support_values(current,history):
    rows=[RECORDS[i] for i in current];allrows=rows+[RECORDS[r['record_index']] for r in history]
    counts=[sum(c['identity']==r['identity'] and c['scene']!=r['scene'] for c in allrows) for r in rows]
    supported={r['identity'] for r,n in zip(rows,counts) if n}
    pairs={(r['identity'],r['scene'],c['scene']) for r in rows for c in allrows if c['identity']==r['identity'] and c['scene']!=r['scene']}
    return counts,dict(eligible_anchors=sum(n>0 for n in counts),eligible_identities=len(supported),identity_directed_scene_relations=len(pairs))
for fi,(fold,md,tf,sf) in enumerate(zip(PROTOCOL['folds'],META['folds'],T0['folds'],SUPPORT['folds'],strict=True)):
    expected_hold=sorted(eligible[fi::3]+ineligible[fi::3])
    assert fold['heldout_ids']==expected_hold
    assert fold['source_ids']==sorted(set(ids)-set(expected_hold))
    assert not set(fold['source_ids'])&set(fold['heldout_ids'])
    src=[i for i,r in enumerate(RECORDS) if r['identity'] in set(fold['source_ids'])]
    gal=[i for i,r in enumerate(RECORDS) if r['identity'] in set(fold['heldout_ids'])]
    assert src==fold['source_record_indices'] and gal==fold['gallery_record_indices']
    assert fold['source_label_map']=={str(i):n for n,i in enumerate(fold['source_ids'])}
    expected_queries=[];excluded=[]
    for pos,index in enumerate(gal):
        r=RECORDS[index];same=[RECORDS[g] for g in gal if RECORDS[g]['identity']==r['identity']]
        removed=sum(x['scene']==r['scene'] for x in same);n=len(same)-removed
        if not n:excluded.append(index);continue
        expected_queries.append(dict(record_index=index,gallery_position=pos,identity=r['identity'],scene=r['scene'],valid_positives=n,removed_same_identity_same_scene=removed,retained_gallery=len(gal)-removed,negative_identity_distractors=len(gal)-len(same)))
    assert expected_queries==fold['query_rows'] and excluded==fold['excluded_query_record_indices']
    q=OrderedDict();counts=Counter();seen=set();zero=[]
    assert len(md['batches'])==len(tf['steps'])==len(sf['batches'])==260
    for step,(batch,row,registered) in enumerate(zip(md['batches'],tf['steps'],sf['batches'],strict=True)):
        current=batch['record_indices'];seen.update(current)
        assert len(current)==64 and set(current)<=set(src)
        assert sorted(Counter(RECORDS[i]['identity'] for i in current).values())==[8]*8
        history=queue_read(q,step,current);pc,sv=support_values(current,history)
        assert row==dict(step=step+1,historical_records=len(history),ages=[r['age'] for r in history],cross_scene_positive_counts=pc,eligible_anchors=sv['eligible_anchors'])
        cr=[RECORDS[i] for i in current];pool=cr+[RECORDS[r['record_index']] for r in history]
        cc=[sum(c['identity']==r['identity'] and c['scene']!=r['scene'] for c in cr) for r in cr]
        allpos=sum(sum(c['identity']==r['identity'] for c in pool)-1 for r in cr)
        crossuniq=sum(len({c['index'] for c in pool if c['identity']==r['identity'] and c['scene']!=r['scene']}) for r in cr)
        global_good=[len(scene_members[r['identity']])>1 for r in cr]
        expected_counts=dict(anchor_exposures=64,all_positive_positions=allpos,cross_positive_positions=sum(pc),same_scene_positive_positions=allpos-sum(pc),cross_unique_records=crossuniq,negative_positions=sum(sum(c['identity']!=r['identity'] for c in pool) for r in cr),anchors_with_current_cross=sum(n>0 for n in cc),anchors_with_pool_cross=sv['eligible_anchors'],anchors_rescued_by_history=sum(not a and b>0 for a,b in zip(cc,pc)),globally_eligible_anchor_exposures=sum(global_good),globally_eligible_but_pool_missing=sum(g and not n for g,n in zip(global_good,pc)),globally_ineligible_anchor_exposures=sum(not g for g in global_good),batches=1,batches_without_legal_anchor=int(not any(pc)))
        assert registered['counts']==expected_counts,(fi,step,'support counts')
        assert registered['cross_positive_count_histogram']==dict(Counter(str(n) for n in pc))
        assert registered['eligible_identities']==sv['eligible_identities']
        assert registered['step']==step+1 and registered['history_records']==len(history)
        counts.update(batches=1,anchor_exposures=64,candidates=len(history),positive_positions=sum(pc),zero_eligible=int(not any(pc)),active_zero_eligible=int(step>=65 and not any(pc)))
        if not any(pc):zero.append(dict(step=step+1,ap_active=step>=65))
        if step>=65:queue_update(q,step,current)
    assert sum(x['historical_records'] for x in tf['steps'])==tf['total_historical_candidate_records']
    assert max(x['historical_records'] for x in tf['steps'])==tf['maximum_historical_records']
    assert sum(bool(x['historical_records']) for x in tf['steps'])==tf['active_steps']
    source_coverage.append(dict(fold=fi,source_identities=len(fold['source_ids']),heldout_identities=len(expected_hold),source_records=len(src),source_queue_unique_records=len(seen),gallery_records=len(gal),query_records=len(expected_queries),counts=dict(counts),zero_support_rows=zero));t0_totals.update(counts)

def smoothed_aps(matrix,identities,scenes,cross):
    score=1.0-np.asarray(matrix,dtype=np.float64)**2/2.0
    result=[];count=[]
    for i in range(64):
        positive=np.array([v==identities[i] for v in identities]);valid=np.ones(len(identities),dtype=bool)
        if cross:
            same=np.array([s==scenes[i] for s in scenes]);positive&=~same
            valid=~(np.array([v==identities[i] for v in identities])&same)
        else:positive[i]=False;valid[i]=False
        p=np.flatnonzero(positive);count.append(len(p))
        if not len(p):result.append(0.0);continue
        fractions=[]
        for j in p:
            legal=valid.copy();legal[j]=False
            rank_sigmoid=1/(1+np.exp((score[i,j]-score[i])/0.01))
            fractions.append((1+rank_sigmoid[legal&positive].sum())/(1+rank_sigmoid[legal].sum()))
        result.append(float(np.mean(fractions)))
    return np.array(result),np.array(count)

per_end=[];all_pixels={};all_audits={};end_counts=Counter();maxima=Counter();reference_rows=[];all_files={}
for fold in PROTOCOL['folds']:
    fi=fold['fold']
    for endpoint in ('control','balanced'):
        directory=RUN/'m0'/f'fold_{fi}_{endpoint}'
        receipt=read(directory/'receipt.json');tr=read(directory/'training.json')
        assert receipt==SUMMARY['folds'][fi]['endpoints'][endpoint] and tr==receipt['training']
        per_end.append((directory,fold,endpoint,'capacity',tr,receipt))
for endpoint in ('control','balanced'):
    receipt=SUMMARY['overfit'][endpoint];directory=RUN/'m0'/('overfit_'+endpoint)
    tr=read(directory/'training.json');assert tr==receipt['training']
    per_end.append((directory,PROTOCOL['folds'][0],endpoint,'overfit',tr,receipt))
training_results=[]
for directory,fold,endpoint,mode,tr,receipt in per_end:
    rows=[json.loads(s) for s in (directory/'memory_steps.jsonl').read_text().splitlines()]
    assert len(rows)==len(tr['steps'])==tr['optimizer_steps']==(8 if mode=='capacity' else 100)
    assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and not tr['missing_nonzero_gradients']
    assert tr['overflow_events']==0 and tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
    assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
    assert tr['initial_state_sha256']==receipt['initialization']['initial_state_sha256']!=tr['final_state_sha256']
    assert tr['gradient_balancing_applied']==(endpoint=='balanced')
    assert tr['epochs']==1 and tr['classification_head_rule']=='original_current_total_gradient'
    assert tr['peak_reserved_mib']<24576 and tr['peak_allocated_mib']<=tr['peak_reserved_mib']
    assert tr['history_candidate_vjp_applied'] and tr['fresh_history_both_endpoints'] and tr['history_anchor_count']==0
    assert tr['current_rank_backward_calls']==tr['current_auxiliary_backward_calls']==len(rows)
    for name,proof in tr['audit_files'].items():
        p=directory/name;assert p.stat().st_size==proof['bytes'] and sha(p)==proof['sha256'];all_files[str(p)]=proof
    source=set(fold['source_record_indices']);q=OrderedDict();states={r:None for r in ROLES};stats=Counter();seen=set()
    prior_update={};role_values={r:dict(rank_weights=[],auxiliary_weights=[],update_norms=[],subtraction_relative=[],direct_sum_relative=[]) for r in ROLES}
    with (directory/'memory_distances.f32').open('rb') as stream:
        for step,(a,t) in enumerate(zip(rows,tr['steps'],strict=True)):
            assert a['step']==t['step']==step+1 and a['zero_based_step']==step
            current=a['record_indices'];seen.update(current)
            assert current==t['sampled_record_indices']==META['folds'][fold['fold']]['batches'][0 if mode=='overfit' else step]['record_indices']
            assert set(current)<=source and len(current)==64
            ids_here=[RECORDS[i]['identity'] for i in current];scenes_here=[RECORDS[i]['scene'] for i in current]
            assert sorted(Counter(ids_here).values())==[8]*8
            assert a['identities']==ids_here and a['scenes']==scenes_here
            history=queue_read(q,step,current);assert history==a['memory']
            pc,sv=support_values(current,history);assert sv==a['support']
            assert a['warmup_steps']==2 and a['replacement_active']==(step>=2)
            supported=step>=2 and any(pc)
            stats.update(steps=1,warmup_steps=int(step<2),supported_steps=int(supported),active_zero_support_steps=int(step>=2 and not supported),eligible_anchors=sv['eligible_anchors'],history_steps=int(bool(history)),candidates=len(history))
            assert stream.tell()==a['distance_offset_bytes']
            n=4*64*(64+len(history));assert n==a['distance_float_count']
            data=np.fromfile(stream,dtype=np.float32,count=n);assert len(data)==n and np.isfinite(data).all() and (data>=0).all()
            dist=data.reshape(4,64,-1);stats['distance_elements']+=n
            maxima['distance_max']=max(maxima['distance_max'],float(data.max()))
            maxima['current_distance_symmetry_error']=max(maxima['current_distance_symmetry_error'],float(np.max(np.abs(dist[:,:,:64]-dist[:,:,:64].transpose(0,2,1)))))
            maxima['fused_squared_vs_mean_branch_squared_error']=max(maxima['fused_squared_vs_mean_branch_squared_error'],float(np.max(np.abs(dist[0].astype(float)**2-np.mean(dist[1:].astype(float)**2,axis=0)))))
            dc,dh=dist[0,:,:64],dist[0,:,64:]
            idarr=np.array(ids_here);scene=np.array(scenes_here);mid=np.array([r['identity'] for r in history]);msc=np.array([r['scene'] for r in history])
            cp=(idarr[:,None]==idarr[None,:])&~np.eye(64,dtype=bool);cn=idarr[:,None]!=idarr[None,:]
            hp=np.where(cp,dc,-np.inf).max(1);hn=np.where(cn,dc,np.inf).min(1)
            mp=idarr[:,None]==mid;mn=~mp
            if len(history):
                mph=np.where(mp,dh,-np.inf).max(1);mnh=np.where(mn,dh,np.inf).min(1)
                fullhp=np.maximum(hp,mph);fullhn=np.minimum(hn,mnh)
                harder_positive=int((mph>hp).sum());harder_negative=int((mnh<hn).sum())
            else:fullhp,fullhn=hp,hn;harder_positive=harder_negative=0
            basic=float(np.maximum(hp-hn+np.float32(.3),0).mean());hard=float(np.maximum(fullhp-fullhn+np.float32(.3),0).mean())
            allids=ids_here+[r['identity'] for r in history];allscenes=scenes_here+[r['scene'] for r in history]
            standard,_=smoothed_aps(dist[0],allids,allscenes,False);cross,computed_pc=smoothed_aps(dist[0],allids,allscenes,True)
            crossloss=float(1-cross[computed_pc>0].mean()) if any(computed_pc) else 0.0
            rel=a['relation_objective']
            assert pc==computed_pc.tolist()==rel['cross_scene_positive_counts']
            assert rel['positive_counts']==[sum(y==x for y in allids)-1 for x in ids_here]
            assert rel['cross_scene_eligible_anchors']==sum(n>0 for n in pc)
            assert rel['temperature']==.01 and rel['cross_scene_reduction']=='mean_eligible_anchors' and rel['ineligible_ap_storage_value']==0
            for key,actual,expected in [('standard_ap',rel['per_anchor_smoothed_ap'],standard),('cross_ap',rel['cross_scene_per_anchor_ap'],cross),('current_hard',a['original_triplet'],basic),('hard_loss',rel['hard_loss'],hard),('standard_loss',rel['smooth_ap_loss'],1-standard.mean()),('cross_loss',rel['cross_scene_loss'],crossloss),('fused_component',t['components']['triplet_fused'],crossloss if step>=2 else basic)]:
                maxima[key+'_error']=max(maxima[key+'_error'],absolute(actual,expected,2e-6))
            expected_stats=dict(memory_records=len(history),memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int(mn.sum()),memory_cross_scene_positive_pairs=int((mp&(scene[:,None]!=msc)).sum()),memory_negative_violations_against_batch_hard_positive=int((mn&(dh<hp[:,None]+np.float32(.3))).sum()),harder_positive_anchors=harder_positive,harder_negative_anchors=harder_negative,current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((fullhp>=fullhn).sum()),current_triplet=basic,expanded_triplet=hard,expanded_hinge_positive_anchors=int((fullhp-fullhn+np.float32(.3)>0).sum()),maximum_memory_age=max((r['age'] for r in history),default=0))
            for k,v in expected_stats.items():
                if k.endswith('triplet'):absolute(a['statistics'][k],v,2e-6)
                else:assert a['statistics'][k]==v,(directory.name,step,k)
            assert set(a['statistics'])==set(expected_stats)
            components=t['components'];weights=BASE['LOSS'];expected_keys={'id_fused','triplet_fused'}|{p+'_'+r for r in ROLES for p in ('id','triplet','id_residual','triplet_residual')}
            assert set(components)==expected_keys and all(math.isfinite(v) and v>=0 for v in components.values())
            total=components['id_fused']*weights['ID_FUSED']+components['triplet_fused']*weights['TRIPLET_FUSED']
            for r in ROLES:
                total+=sum(components[p+'_'+r]*weights[w] for p,w in [('id','ID_BRANCH'),('triplet','TRIPLET_BRANCH'),('id_residual','ID_RESIDUAL'),('triplet_residual','TRIPLET_RESIDUAL')])
            maxima['weighted_loss_error']=max(maxima['weighted_loss_error'],absolute(t['loss'],total,1e-5))
            assert t['amp_scale_after']==t['amp_scale_before']==256.0
            assert t['active_fused_metric']==('hard_triplet' if step<2 else 'cross_scene_smooth_ap')
            assert a['saved_space_order']==['fused',*ROLES] and a['coordinate_rule']=='fresh' and a['current_anchor_count']==64 and a['history_anchor_count']==0 and a['gradient_weight']==1
            for flag in ('history_candidate_vjp_applied','classification_head_gradients_unchanged','selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'):assert a[flag]
            upstream=a['historical_leaf_upstream_norms'];assert len(upstream)==len(history) and all(math.isfinite(x) and x>=0 for x in upstream)
            groups=sorted({r['stored_step'] for r,norm in zip(history,upstream,strict=True) if norm>0})
            assert groups==a['history_vjp_groups'] and a['history_vjp_record_forwards']==64*len(groups)
            assert a['fresh_role_record_forwards']==64*len({r['stored_step'] for r in history})
            stats.update(history_vjp_groups=len(groups),history_vjp_record_forwards=64*len(groups),fresh_role_record_forwards=a['fresh_role_record_forwards'])
            assert a['current_rank_backward_calls']==a['current_auxiliary_backward_calls']==1
            all_pairs(a)
            refs=a['rank_auxiliary_reference_checks'];assert bool(refs)==bool(a['direct_single_group_check'])
            assert a['direct_component_backward_calls']==4*bool(refs)
            if refs:
                assert mode=='capacity' and step==3 and set(refs)==set(ROLES)
                assert len({r['stored_step'] for r in history})==1
                direct=a['direct_single_group_check'];close(direct['relative_l2_error'],direct['difference_norm']/direct['first_norm'],1e-12)
                assert direct['relative_l2_error']<=.005 and direct['all_four_reencoded_outputs_bitwise_equal']
                stats['direct_reference_steps']+=1
            for role in ROLES:
                b=a['gradient_balance'][role];p=b['rank_vs_auxiliary'];rnorm,anorm=p['first_norm'],p['second_norm'];ra=dot(p)
                assert b['before']==states[role] and b['supported']==bool(supported)
                ratio=None;wr=wa=1.0
                if supported:
                    prev=states[role]
                    states[role]=dict(rank=rnorm,auxiliary=anorm,supported_steps=1) if prev is None else dict(rank=.9*prev['rank']+.1*rnorm,auxiliary=.9*prev['auxiliary']+.1*anorm,supported_steps=prev['supported_steps']+1)
                    ratio=min(4.0,max(.25,math.sqrt((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))))
                    wr=2*ratio/(1+ratio);wa=2/(1+ratio)
                assert b['after']==states[role] and b['ratio']==ratio
                close(b['proposed_rank_weight'],wr);close(b['proposed_auxiliary_weight'],wa)
                if endpoint=='control' or not supported:wr=wa=1.
                close(b['applied_rank_weight'],wr);close(b['applied_auxiliary_weight'],wa)
                assert .4<=wr<=1.6 and .4<=wa<=1.6;close(wr+wa,2)
                current_history=b['current_rank_vs_history'];close(rnorm*rnorm,current_history['first_norm']**2+current_history['second_norm']**2+2*dot(current_history))
                direct_sum=b['direct_sum_vs_original'];close(direct_sum['first_norm']**2,rnorm*rnorm+anorm*anorm+2*ra)
                close(b['subtraction_auxiliary_vs_direct']['second_norm'],anorm)
                applied=b['rank_vs_applied']['second_norm']
                for x in (b['auxiliary_vs_applied'],b['original_sum_vs_applied'],a['applied_gradients'][role]):close(applied,x['second_norm'])
                close(b['rank_vs_applied']['first_norm'],rnorm);close(b['auxiliary_vs_applied']['first_norm'],anorm)
                close(direct_sum['second_norm'],b['original_sum_vs_applied']['first_norm'])
                if endpoint=='balanced' and supported:
                    close(applied*applied,wr*wr*rnorm*rnorm+wa*wa*anorm*anorm+2*wr*wa*ra)
                    close(b['rank_vs_applied']['difference_norm']**2,(wr-1)**2*rnorm*rnorm+wa*wa*anorm*anorm+2*(wr-1)*wa*ra)
                    close(b['auxiliary_vs_applied']['difference_norm']**2,wr*wr*rnorm*rnorm+(wa-1)**2*anorm*anorm+2*wr*(wa-1)*ra)
                else:
                    assert b['original_sum_vs_applied']['difference_norm']==0
                    assert a['applied_gradients'][role]==a['roles'][role]['total_vs_both']
                current_total=a['roles'][role]['total_vs_history'];both=a['roles'][role]['total_vs_both']
                close(both['second_norm']**2,current_total['first_norm']**2+current_total['second_norm']**2+2*dot(current_total))
                close(both['difference_norm'],current_total['second_norm'],1e-6)
                if step>=2 and not supported:assert rnorm==current_history['first_norm']==current_history['second_norm']==0
                update=a['actual_parameter_updates'][role]
                if role in prior_update:assert update['first_norm']==prior_update[role]
                prior_update[role]=update['second_norm'];assert update['difference_norm']>0
                rv=role_values[role];rv['rank_weights'].append(wr);rv['auxiliary_weights'].append(wa);rv['update_norms'].append(update['difference_norm'])
                rv['subtraction_relative'].append(b['subtraction_auxiliary_vs_direct']['difference_norm']/anorm if anorm else 0)
                rv['direct_sum_relative'].append(direct_sum['difference_norm']/direct_sum['first_norm'] if direct_sum['first_norm'] else 0)
                if refs:
                    assert set(refs[role])=={'current_rank','historical_rank','full_rank','auxiliary','applied'}
                    for part,v in refs[role].items():
                        assert v['passed']
                        if v['first_norm']:
                            close(v['relative_l2_error'],v['difference_norm']/v['first_norm'],1e-12);assert v['relative_l2_error']<=.005
                        else:assert v['relative_l2_error'] is None and v['difference_norm']<=1e-8
                        reference_rows.append(dict(endpoint=directory.name,step=step+1,role=role,component=part,**v))
            if step>=2:queue_update(q,step,current)
        assert stream.read()==b''
    assert states==tr['gradient_balance_state']
    assert stats['fresh_role_record_forwards']+64==tr['extra_fresh_role_record_forwards']
    assert stats['history_vjp_record_forwards']==tr['extra_history_vjp_record_forwards']
    assert stats['direct_reference_steps']==(1 if mode=='capacity' else 0)
    assert stats['direct_reference_steps']*64==tr['extra_direct_check_record_forwards']
    assert stats['direct_reference_steps']*4==tr['direct_component_backward_calls']
    hist=tr['history'];assert len(hist)==1 and hist[0]['optimizer_steps']==len(rows)
    assert hist[0]['learning_rate']==BASE['OPTIMIZATION']['NEW_MODULE_LR']==.00035
    assert hist[0]['mean_loss']==float(np.mean([x['loss'] for x in tr['steps']]))
    if mode=='capacity':
        witness=tr['historical_parameter_gradient_witness'];assert witness['roles']==rows[witness['step']-1]['roles']
        assert all(witness['roles'][r]['total_vs_history']['second_norm']>0 for r in ROLES)
        assert all(receipt['engineering_checks'].values()) and all(receipt['preflight'][k] for k in ('standalone_signal_bitwise_equal','zero_update_replay_bitwise_equal','all_model_state_unchanged')) and receipt['strict_reload_all_outputs_bitwise_equal']
    else:
        assert not stats['candidates'] and not stats['history_vjp_record_forwards']
        classes=len(fold['source_ids']);eps=.1;p=1-eps+eps/classes;other=eps/classes
        floor=(-p*math.log(p)-(classes-1)*other*math.log(other))*(BASE['LOSS']['ID_FUSED']+3*BASE['LOSS']['ID_BRANCH']+3*BASE['LOSS']['ID_RESIDUAL'])
        first,last=tr['steps'][0]['loss'],tr['steps'][-1]['loss'];ratio=(last-floor)/(first-floor)
        g=receipt['gate'];assert g==dict(passed=ratio<=.1,initial_loss=first,final_loss=last,minimum_loss=floor,initial_excess_loss=first-floor,final_excess_loss=last-floor,loss_ratio=ratio,maximum_loss_ratio=.1)
        assert all(receipt['checks'].values()) and g['passed']
    all_pixels[directory.name]=[(x['record_indices'],x['pixel_sha256']) for x in rows]
    all_audits[directory.name]=rows
    training_results.append(dict(endpoint=directory.name,counts=dict(stats),unique_source_records=len(seen),unique_source_identities=len({RECORDS[i]['identity'] for i in seen}),gate=receipt.get('gate'),initial_state_sha256=tr['initial_state_sha256'],frozen_state_sha256=tr['frozen_state_after_sha256'],peak_reserved_mib=tr['peak_reserved_mib'],elapsed_training_seconds=hist[0]['elapsed_seconds'],role_statistics={r:{k:dict(minimum=min(v),maximum=max(v)) for k,v in rv.items()} for r,rv in role_values.items()}))
    end_counts.update(stats)
for fi in range(3):
    assert all_pixels[f'fold_{fi}_control']==all_pixels[f'fold_{fi}_balanced']
    pair_end=SUMMARY['folds'][fi]['endpoints'];assert pair_end['control']['initialization']==pair_end['balanced']['initialization']
    assert SUMMARY['folds'][fi]['all_paired_source_pixels_exact']
assert all_pixels['overfit_control']==all_pixels['overfit_balanced']
for endpoint in ('control','balanced'):
    assert SUMMARY['overfit'][endpoint]['initialization']==SUMMARY['folds'][0]['endpoints'][endpoint]['initialization']

checkpoint_results=[]
for fold,b0 in zip(PROTOCOL['folds'],B0['folds'],strict=True):
    path=Path(b0['checkpoint']);assert sha(path)==b0['checkpoint_sha256']
    baseline=torch.load(path,map_location='cpu',weights_only=True)
    assert baseline['fold']==fold['fold'] and baseline['source_ids']==fold['source_ids'] and baseline['heldout_ids']==fold['heldout_ids']
    state0=baseline['model_state_dict'];assert state_sha(state0)==b0['training']['final_state_sha256']
    array_path=path.parent/'retrieval_arrays.pt';assert sha(array_path)==b0['retrieval']['retrieval_arrays_sha256']
    for endpoint in ('control','balanced'):
        receipt=SUMMARY['folds'][fold['fold']]['endpoints'][endpoint];tr=receipt['training'];p=Path(receipt['checkpoint'])
        assert sha(p)==receipt['checkpoint_sha256']
        payload=torch.load(p,map_location='cpu',weights_only=True)
        assert set(payload)=={'role_state_dict','baseline_aliases','binding','fold','source_ids','heldout_ids','config_sha256'}
        assert payload['binding']==receipt['initialization'] and payload['config_sha256']==CFG_SHA
        assert payload['fold']==fold['fold'] and payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids']
        names=payload['binding']['trainable_names'];assert len(names)==len(set(names))==203
        groups={r:[n for n in names if n.startswith('encoder.'+r+'_')] for r in ROLES}
        assert sum(map(len,groups.values()))==189 and len([n for n in names if not n.startswith('encoder.')])==14
        assert not payload['binding']['role_weights_loaded'] and payload['binding']['role_initialization_seed']==42
        assert payload['binding']['signal_checkpoint_sha256']==b0['checkpoint_sha256']
        assert payload['binding']['signal_state_sha256']==state_sha(state0)==tr['signal_state_after_sha256']
        aliases=payload['baseline_aliases'];assert all(k.startswith('baseline.') and v in state0 for k,v in aliases.items())
        role=payload['role_state_dict'];assert all(not k.startswith('baseline.') for k in role)
        state={k:state0[v] for k,v in aliases.items()};state.update(role)
        assert state_sha(state)==tr['final_state_sha256']==receipt['strict_reload_state_sha256']
        buffers={n for n in role if n.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
        frozen={n:v for n,v in state.items() if n.startswith('baseline.') or (n not in names and n not in buffers)}
        assert state_sha(frozen)==tr['frozen_state_after_sha256']
        assert all(torch.isfinite(v).all() for v in state.values())
        for r,ns in groups.items():
            finalnorm=math.sqrt(sum(float(role[n].double().square().sum()) for n in ns))
            assert finalnorm==all_audits[f"fold_{fold['fold']}_{endpoint}"][-1]['actual_parameter_updates'][r]['second_norm']
        checkpoint_results.append(dict(fold=fold['fold'],endpoint=endpoint,checkpoint=str(p),sha256=receipt['checkpoint_sha256'],baseline_checkpoint=str(path),baseline_sha256=b0['checkpoint_sha256'],state_tensors=len(state),baseline_aliases=len(aliases),role_state_tensors=len(role),role_parameter_counts={r:len(v) for r,v in groups.items()},head_tensors=14,frozen_state_sha256=state_sha(frozen),strict_reload_state_sha256=state_sha(state),final_update_parameter_norms_match=True))
        del payload,state,role,frozen
    del baseline,state0;gc.collect()
assert end_counts['steps']==SUMMARY['optimizer_steps']==CPU['checked_training_steps']==248
assert end_counts['distance_elements']==CPU['checked_memory_distance_elements']
assert end_counts['history_vjp_record_forwards']==CPU['checked_history_vjp_record_forwards']
assert CPU['checked_retrieval_distance_and_rank_elements']==0
for path,proof in CPU['files'].items():assert sha(path)==proof['sha256'] and Path(path).stat().st_size==proof['bytes']
assert not torch.cuda.is_initialized()
result=dict(status='PASS_INDEPENDENT_COMPLETE_M0_SAVED_EVIDENCE',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    run=str(RUN),execution_commit=COMMIT,config_sha256=CFG_SHA,summary_sha256=sha(RUN/'m0/summary.json'),
    source_folds=source_coverage,t0_counts=dict(t0_totals),m0_counts=dict(end_counts),
    training=training_results,checkpoint_checks=checkpoint_results,reference_checks=reference_rows,
    scalar_pair_identities_checked=pair_count,maxima=dict(maxima),
    source_exposures=end_counts['steps']*64,current_rank_backward_calls=248,current_auxiliary_backward_calls=248,
    direct_component_backward_calls=24,extra_direct_total_backward_calls=6,
    actual_m0_unsupported_ap_steps=end_counts['active_zero_support_steps'],
    final_checkpoint_count=len(checkpoint_results),baseline_checkpoint_count=3,
    all_cpu_inventory_files_rehashed=len(CPU['files']),
    runtime_limits=['No saved full per-step gradients or optimizer states; scalar identities do not independently reconstruct gradients or AdamW.',
                    'Historical feature/RNG/buffer/head/strict-output equality are saved runtime assertions; no new model forwards.',
                    'All four saved distance spaces checked for finite values and geometry; raw step embeddings are not saved.',
                    'M0 contains no real active zero-support AP update; this branch is synthetic and queue evidence only.',
                    'No retrieval scores or Q1 partial outputs read.'],
    python_version=__import__('sys').version,torch_version=torch.__version__,numpy_version=np.__version__,cpu_threads=2,interop_threads=1,
    cuda_initialized=False,model_forwards=0,optimizer_updates=0,image_reads=0,q1_result_reads=0,
    elapsed_seconds=time.perf_counter()-START)
print(json.dumps(result,indent=2))
