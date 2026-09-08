"""Auditor-authored full text replay; no imports from the experiment implementation."""
from pathlib import Path
from collections import OrderedDict, Counter
import json, hashlib, math, re
import numpy as np
OUT=Path(__file__).parent
R=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
D=Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_complete_20260909')
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,obj):
    with (OUT/n).open('x',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def eq(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b,tol)
P=load(R/'protocols/msvr310_train_oof_v1.json');S=load(D/'q1/summary.json');X=load(OUT/'remote_probe.result.json')
cfg=X['source_config'];meta=load(R/cfg['SOURCE_METADATA']['PATH'])
O=['baseline_only','fused','cnn','transformer','mamba'];E=['control','smooth_ap']
report={'status':'PASS','scope':'All 6 Q1 endpoints, 1560 updates, 600 queries x 5 outputs x 2 endpoints; no model, image, or optimizer execution','input_identity':{},'protocol':{},'folds':[],'training':[]}
inv=load(D/'remote_terminal_inventory.json')
for row in inv['files']:
    if row['path'].startswith('q1') or row['path']=='pipeline.json':
        p=D/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];report['input_identity'][row['path']]=row['sha256']
assert S['status']=='Q1_FAIL' and S['optimizer_steps']==1560 and S['seed']==42 and S['heldout_record_forwards']==2064 and S['official_image_reads']==0
assert S['config_sha256']==sha(R/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json')
assert sha(D/'q1/summary.json')==load(D/'pipeline.json')['terminal_summary_sha256']
assert sha(D/'q1_cpu.json')==load(D/'pipeline.json')['terminal_cpu_sha256']
for r in P['records']:
    assert r['index']==P['records'].index(r)
    for modal,path in zip(['vis','ni','th'],r['paths']):
        m=re.fullmatch(r'bounding_box_train/(\d+)/'+modal+r'/(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg',path);assert m,path
        assert int(m[1])==int(m[2])==r['identity'] and int(m[3])==r['scene'] and int(m[4])==r['camera']
assert len({p for r in P['records'] for p in r['paths']})==3096
global_ids=sorted({r['identity'] for r in P['records']});assert len(global_ids)==155
AP={e:{o:[] for o in O} for e in E};FIRST={e:{o:[] for o in O} for e in E};QIDS=[];FOLDMAP={e:[] for e in E}
rows_saved=[];train_by={};rankcount=0
for fi,fold in enumerate(P['folds']):
    assert fold['fold']==fi
    multiscene=[x for x in global_ids if len({r['scene'] for r in P['records'] if r['identity']==x})>=2]
    singlescene=[x for x in global_ids if x not in multiscene]
    held=set(multiscene[fi::3]+singlescene[fi::3]);source=set(global_ids)-held
    assert held==set(fold['heldout_ids']) and source==set(fold['source_ids']) and not held&source
    si=[r['index'] for r in P['records'] if r['identity'] in source];gi=[r['index'] for r in P['records'] if r['identity'] in held]
    assert si==fold['source_record_indices'] and gi==fold['gallery_record_indices']
    gallery=[P['records'][i] for i in gi];valid=[];excluded=[]
    for pos,r in enumerate(gallery):
        positives=[x for x in gallery if x['identity']==r['identity'] and x['scene']!=r['scene']]
        if positives:
            removes=sum(x['identity']==r['identity'] and x['scene']==r['scene'] for x in gallery)
            q=dict(record_index=r['index'],gallery_position=pos,identity=r['identity'],scene=r['scene'],valid_positives=len(positives),removed_same_identity_same_scene=removes,retained_gallery=len(gallery)-removes,negative_identity_distractors=sum(x['identity']!=r['identity'] for x in gallery));valid.append(q)
        else:excluded.append(r['index'])
    assert valid==fold['query_rows'] and excluded==fold['excluded_query_record_indices']
    QIDS.extend(q['identity'] for q in valid)
    fr=dict(fold=fi,source_records=len(si),source_identities=len(source),gallery_records=len(gi),heldout_identities=len(held),queries=len(valid),query_identities=len({q['identity'] for q in valid}),gallery_only_records=len(excluded),gallery_only_identities=len(held)-len({q['identity'] for q in valid}),endpoints={})
    pixelpairs=[]
    for e in E:
        folder=D/'q1'/f'fold_{fi}_{e}';receipt=load(folder/'receipt.json');tr=load(folder/'training.json');ranks=load(folder/'rankings.json');aud=[json.loads(line) for line in (folder/'memory_steps.jsonl').read_text().splitlines()]
        assert receipt==S['folds'][fi]['endpoints'][e] and tr==receipt['training'];assert set(ranks)==set(O)
        ret=receipt['retrieval'];assert ret['gallery_manifest']==gallery and ret['query_rows']==valid
        assert ret['rankings_sha256']==sha(folder/'rankings.json')
        scores={};maps={}
        for o in O:
            assert len(ranks[o])==len(valid)
            aps=[];first=[]
            for qi,(q,rank) in enumerate(zip(valid,ranks[o])):
                assert len(rank)==len(gi) and sorted(rank)==list(range(len(gi)));rankcount+=len(rank)
                legal=[gallery[k] for k in rank if not(gallery[k]['identity']==q['identity'] and gallery[k]['scene']==q['scene'])]
                pp=[i+1 for i,r in enumerate(legal) if r['identity']==q['identity']];assert len(pp)==q['valid_positives']
                ap=sum((j+1)/k for j,k in enumerate(pp))/len(pp);f=pp[0];aps.append(ap);first.append(f)
                eq(ap,ret['outputs'][o]['average_precision'][qi],1e-14);assert f==ret['outputs'][o]['first_match_rank'][qi]
                rows_saved.append(dict(fold=fi,endpoint=e,output=o,query_index=qi,record_index=q['record_index'],identity=q['identity'],average_precision=ap,first_match_rank=f,positive_count=len(pp)))
            ap=np.array(aps);first=np.array(first);metrics={'mAP':float(ap.mean()*100),**{f'Rank-{k}':float((first<=k).mean()*100) for k in [1,5,10]}}
            for k,v in metrics.items():eq(v,ret['outputs'][o]['metrics'][k])
            AP[e][o].extend(aps);FIRST[e][o].extend(first.tolist());maps[o]=metrics['mAP'];scores[o]=metrics
        FOLDMAP[e].append(maps);fr['endpoints'][e]=scores
        assert tr['mode']=='comparison' and tr['epochs']==20 and tr['optimizer_steps']==len(aud)==len(tr['steps'])==260
        assert tr['initial_state_sha256']==receipt['initialization']['initial_state_sha256']
        assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256'] and tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
        assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and tr['missing_nonzero_gradients']==[] and tr['overflow_events']==0
        assert all(receipt['engineering_checks'].values())
        assert receipt['initialization']['source_ids']==fold['source_ids'] and receipt['initialization']['heldout_ids']==fold['heldout_ids']
        assert receipt['initialization']['role_weights_loaded'] is False and receipt['initialization']['role_initialization_seed']==42
        assert receipt['initialization']['signal_checkpoint_sha256']==X['baseline_summary']['folds'][fi]['checkpoint_sha256']
        assert tr['audit_files']['memory_steps.jsonl']['sha256']==sha(folder/'memory_steps.jsonl')
        cache=OrderedDict();offset=0;active=0;history_steps=0;candidates=0;positive_exposures=0;fresh=64;vjp=0;loss_error=0.;role_zero={o:0 for o in O[2:]};max_norm_identity_error=0.;unique=set()
        for i,(row,a) in enumerate(zip(tr['steps'],aud)):
            assert row['step']==a['step']==i+1 and row['epoch']==i//13+1 and a['zero_based_step']==i
            idx=row['sampled_record_indices'];assert idx==a['record_indices']==meta['folds'][fi]['batches'][i]['record_indices'] and len(idx)==64
            assert set(idx)<=set(si);unique.update(idx)
            ids=[P['records'][j]['identity'] for j in idx];assert a['identities']==ids and sorted(Counter(ids).values())==[8]*8
            assert a['scenes']==[P['records'][j]['scene'] for j in idx]
            for key in list(cache):
                if i-cache[key]>8:del cache[key]
            wanted=[dict(record_index=j,identity=P['records'][j]['identity'],scene=P['records'][j]['scene'],age=i-s,stored_step=s) for j,s in cache.items() if j not in idx]
            assert a['memory']==wanted
            assert a['replacement_active']==(i>=65) and a['warmup_steps']==65
            assert a['coordinate_rule']=='fresh' and a['current_anchor_count']==64 and a['history_anchor_count']==0
            assert row['active_fused_metric']==('smooth_ap' if e=='smooth_ap' and i>=65 else 'hard_triplet')
            active+=i>=65;history_steps+=bool(wanted);candidates+=len(wanted)
            assert a['distance_offset_bytes']==offset and a['distance_float_count']==4*64*(64+len(wanted));offset+=a['distance_float_count']*4
            mids=[m['identity'] for m in wanted];count=[ids.count(x)+mids.count(x)-1 for x in ids];assert count==a['relation_objective']['positive_counts'];positive_exposures+=sum(count)
            assert a['relation_objective']['temperature']==.01 and len(a['relation_objective']['per_anchor_smoothed_ap'])==64
            assert all(0<=x<=1 for x in a['relation_objective']['per_anchor_smoothed_ap']);eq(1-np.mean(a['relation_objective']['per_anchor_smoothed_ap']),a['relation_objective']['smooth_ap_loss'],2e-7)
            desired=a['relation_objective']['smooth_ap_loss' if e=='smooth_ap' else 'hard_loss'] if i>=65 else a['original_triplet']
            eq(row['components']['triplet_fused'],desired,1e-7)
            c=row['components'];w=cfg['LOSS'];assert len(c)==14
            total=w['ID_FUSED']*c['id_fused']+w['TRIPLET_FUSED']*c['triplet_fused']
            total+=sum(w[k]*c[n+'_'+r] for r in O[2:] for k,n in [('ID_BRANCH','id'),('TRIPLET_BRANCH','triplet'),('ID_RESIDUAL','id_residual'),('TRIPLET_RESIDUAL','triplet_residual')])
            loss_error=max(loss_error,abs(total-row['loss']));eq(total,row['loss'],1e-5)
            assert row['amp_scale_after']>=row['amp_scale_before']>0 and all(math.isfinite(v) for v in c.values())
            groups={x['stored_step'] for x in wanted};assert a['fresh_role_record_forwards']==64*len(groups);fresh+=64*len(groups)
            norms=a['historical_leaf_upstream_norms'];assert len(norms)==len(wanted) and all(math.isfinite(n) and n>=0 for n in norms)
            live=sorted({m['stored_step'] for m,n in zip(wanted,norms) if n>0});assert a['history_vjp_groups']==live and a['history_vjp_record_forwards']==64*len(live);vjp+=64*len(live)
            assert a['direct_single_group_check']=={}
            for key in ['history_candidate_vjp_applied','selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise']:assert a[key]
            for r in O[2:]:
                pair=a['roles'][r]['total_vs_history'];both=a['roles'][r]['total_vs_both'];applied=a['applied_gradients'][r]
                role_zero[r]+=pair['second_norm']==0
                for v in [pair,both,applied]:
                    aa,bb,dd,co=v['first_norm'],v['second_norm'],v['difference_norm'],v['cosine'];assert all(math.isfinite(t) and t>=0 for t in [aa,bb,dd]);assert (co is None)==(aa==0 or bb==0)
                    if co is not None:
                        err=abs(dd*dd-(aa*aa+bb*bb-2*aa*bb*co))/max(1,aa*aa+bb*bb);max_norm_identity_error=max(err,max_norm_identity_error);assert err<=1e-7
                eq(both['difference_norm'],pair['second_norm'],1e-6*max(1,pair['second_norm']))
                for key in both:
                    if both[key] is None:assert applied[key] is None
                    else:eq(both[key],applied[key],1e-6*max(1,abs(both[key])))
            if i>=65:
                for j in idx:
                    if j in cache:del cache[j]
                    cache[j]=i
                while len(cache)>512:cache.popitem(last=False)
        assert fresh==tr['extra_fresh_role_record_forwards'] and vjp==tr['extra_history_vjp_record_forwards'] and offset==tr['audit_files']['memory_distances.f32']['bytes']
        assert tr['extra_direct_check_record_forwards']==0
        for epoch in tr['history']:
            step_rows=[r for r in tr['steps'] if r['epoch']==epoch['epoch']];assert len(step_rows)==epoch['optimizer_steps']==13;eq(np.mean([r['loss'] for r in step_rows]),epoch['mean_loss'],0)
        witness=tr['historical_parameter_gradient_witness'];assert witness['roles']==aud[witness['step']-1]['roles'] and witness['step']==67
        pixelpairs.append([(a['record_indices'],a['pixel_sha256']) for a in aud]);train_by[fi,e]=tr
        report['training'].append(dict(fold=fi,endpoint=e,updates=260,source_record_exposures=16640,unique_source_records=len(unique),objective_active_steps=active,steps_with_history=history_steps,historical_candidates=candidates,positive_position_exposures=positive_exposures,distance_elements=offset//4,extra_fresh_forwards=fresh,extra_vjp_forwards=vjp,direct_full_graph_checks=0,maximum_loss_ledger_error=loss_error,maximum_norm_identity_relative_error=max_norm_identity_error,zero_history_role_steps=role_zero,fit_elapsed_seconds=sum(h['elapsed_seconds'] for h in tr['history']),peak_reserved_mib=tr['peak_reserved_mib'],first_loss=tr['steps'][0]['loss'],last_loss=tr['steps'][-1]['loss']))
    assert pixelpairs[0]==pixelpairs[1] and S['folds'][fi]['all_paired_source_pixels_exact'];assert S['folds'][fi]['endpoints']['control']['initialization']==S['folds'][fi]['endpoints']['smooth_ap']['initialization']
    a=train_by[fi,'control']['steps'];b=train_by[fi,'smooth_ap']['steps'];fr['warmup_component_max_difference']=max(abs(x['components'][c]-y['components'][c]) for x,y in zip(a[:65],b[:65]) for c in x['components']);fr['warmup_first_differing_step']=next((i+1 for i,(x,y) in enumerate(zip(a[:65],b[:65])) if x['components']!=y['components']),None)
    report['folds'].append(fr)
ids=np.array(QIDS);assert len(ids)==600 and len(set(ids))==60 and len(set(q['record_index'] for f in P['folds'] for q in f['query_rows']))==600
report['protocol']=dict(train_records=1032,train_modality_paths=3096,train_identities=155,queries=600,query_identities=60,gallery_only_records=sum(len(f['excluded_query_record_indices']) for f in P['folds']),ranking_positions=rankcount)
def bootstrap(delta):
    identity_order=sorted(set(QIDS));groups=[delta[ids==i] for i in identity_order];draw=np.random.default_rng(42).integers(0,len(groups),size=(10000,len(groups)))
    sums=np.array([sum(g) for g in groups]);counts=np.array([len(g) for g in groups]);samples=np.sum(sums[draw],axis=1)/np.sum(counts[draw],axis=1)
    return float(np.quantile(samples,.025,method='linear'))
for e in E:
    for o in O:AP[e][o]=np.array(AP[e][o]);FIRST[e][o]=np.array(FIRST[e][o])
metrics={e:{o:{'mAP':float(AP[e][o].mean()*100),**{f'Rank-{k}':float((FIRST[e][o]<=k).mean()*100) for k in [1,5,10]}} for o in O} for e in E}
report['metrics']=metrics;report['endpoint_gates']={}
for e in E:
    rec=S['comparison']['endpoints'][e];gain={o:metrics[e][o]['mAP']-metrics[e]['baseline_only']['mAP'] for o in O};fg=[m['fused']-m['baseline_only'] for m in FOLDMAP[e]];lo=bootstrap((AP[e]['fused']-AP[e]['baseline_only'])*100)
    gates=dict(fused_gain_at_least_1pp=gain['fused']>=1,all_fold_fused_gains_nonnegative=all(x>=0 for x in fg),all_full_branches_not_below_signal=all(gain[o]>=0 for o in O[2:]),identity_bootstrap_lower_positive=lo>0,fused_strictly_best=all(metrics[e]['fused']['mAP']>metrics[e][o]['mAP'] for o in [O[0],*O[2:]]))
    assert gates==rec['scientific_checks'] and all(gates.values())==rec['scientific_passed'];eq(lo,rec['identity_bootstrap']['lower_bound_pp']);assert np.allclose(fg,rec['fold_fused_gains_pp'],atol=1e-10,rtol=0)
    for o in O:
        for k in metrics[e][o]:eq(metrics[e][o][k],rec['metrics'][o][k])
        eq(gain[o],rec['gains_over_signal_pp'][o])
        for r in rec['per_identity']:assert r['query_count']==int(sum(ids==r['identity']));eq(float(AP[e][o][ids==r['identity']].mean()*100),r['map_by_output'][o])
        if o!='baseline_only':
            d=(AP[e][o]-AP[e]['baseline_only'])*100;old=FIRST[e]['baseline_only'];new=FIRST[e][o];changes=dict(ap_improved=int(sum(d>0)),ap_declined=int(sum(d<0)),ap_unchanged=int(sum(d==0)),rank1_repaired=int(sum((old>1)&(new==1))),rank1_new_errors=int(sum((old==1)&(new>1))));assert changes==rec['query_changes'][o]
    report['endpoint_gates'][e]=dict(checks=gates,lower_bound_pp=lo,fold_fused_gain_pp=fg)
gain={o:float(np.mean((AP['smooth_ap'][o]-AP['control'][o])*100)) for o in O};fg=[b['fused']-a['fused'] for a,b in zip(FOLDMAP['control'],FOLDMAP['smooth_ap'])];lo=bootstrap((AP['smooth_ap']['fused']-AP['control']['fused'])*100)
gates=dict(fused_gain_at_least_1pp=gain['fused']>=1,all_fold_fused_nonnegative=all(x>=0 for x in fg),all_role_gains_nonnegative=all(gain[o]>=0 for o in O[2:]),paired_identity_bootstrap_lower_positive=lo>0,candidate_fused_strictly_best=all(metrics['smooth_ap']['fused']['mAP']>metrics['smooth_ap'][o]['mAP'] for o in [O[0],*O[2:]]))
assert gates==S['comparison']['paired_checks'];eq(lo,S['comparison']['paired_bootstrap_lower_pp']);assert np.allclose(fg,S['comparison']['fold_fused_gains_mAP'],atol=1e-10,rtol=0)
for o in O:eq(gain[o],S['comparison']['matched_gains_mAP'][o])
for r in S['comparison']['paired_per_identity']:
    assert r['query_count']==int(sum(ids==r['identity']))
    for o in O:eq(float(np.mean((AP['smooth_ap'][o]-AP['control'][o])[ids==r['identity']])*100),r['gains_mAP'][o])
report['paired']=dict(gains_mAP= gain,fold_fused_gains=fg,bootstrap_lower_pp=lo,checks=gates)
assert not S['comparison']['next_phase_qualified'] and not S['comparison']['paired_pass'] and not S['comparison']['vehicle_baseline_pass']
log=[json.loads(l) for l in (D/'q1.log').read_text().splitlines() if l.startswith('{')];epochlog=[r for r in log if r.get('event')=='smooth_ap_epoch'];assert len(epochlog)==120
for r in epochlog:
    want=train_by[r['fold'],r['endpoint']]['history'][r['epoch']-1]
    for k,v in want.items():assert r[k]==v
report['terminal_epoch_log_rows_checked']=120
write('independent_per_query_metrics.json',rows_saved);write('independent_text_results.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['input_identity']},indent=2))
