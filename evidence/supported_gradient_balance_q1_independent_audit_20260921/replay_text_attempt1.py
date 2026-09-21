"""Independent complete local text audit; no project evaluation imports."""
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict
import csv, hashlib, json, math, struct
import numpy as np

OUT=Path(__file__).parent
ROOT=OUT.parent/'trifusion_supported_balance_q1_complete_20260921'
REPO=OUT/'snapshots/repo'
OUTPUTS=('baseline_only','fused','cnn','transformer','mamba')
ROLES=OUTPUTS[2:]
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def csvsave(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b,tol=1e-10):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b,tol)
maxima={};original_failures=[]
def equation(kind,a,b,loc,accept=True):
    threshold=1e-7*max(1,abs(a),abs(b));ratio=abs(a-b)/threshold
    assert math.isfinite(ratio)
    record=dict(kind=kind,**loc,left=a,right=b,absolute_error=abs(a-b),threshold=threshold,ratio=ratio)
    if kind not in maxima or ratio>maxima[kind]['ratio']:maxima[kind]=record
    if ratio>1:
        if accept:raise AssertionError(record)
        original_failures.append(record)
def pair(x,loc):
    a,b,d=x['first_norm'],x['second_norm'],x['difference_norm'];c=x['cosine']
    assert all(math.isfinite(z) and z>=0 for z in (a,b,d))
    assert (c is None)==(a==0 or b==0)
    if c is not None: assert abs(c)<=1.00001
    equation('pair_identity',d*d,a*a+b*b-(0 if c is None else 2*a*b*c),loc)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def metrics(rows):
    return dict(mAP=float(np.mean([r['ap'] for r in rows])*100),**{f'Rank-{k}':float(np.mean([r['first_rank']<=k for r in rows])*100) for k in (1,5,10)})
def bootstrap(delta,labels):
    ids=np.unique(labels); totals=np.array([delta[labels==i].sum() for i in ids]); counts=np.array([(labels==i).sum() for i in ids])
    draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    return float(np.quantile(totals[draw].sum(1)/counts[draw].sum(1),.025,method='linear'))

protocol=load(REPO/'protocols/msvr310_train_oof_v1.json')
config=load(REPO/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
metadata=load(REPO/config['SOURCE_METADATA']['PATH'])
summary=load(ROOT/'q1/summary.json');cpu=load(ROOT/'q1_cpu_arithmetic_recheck/verification.json');pipeline=load(ROOT/'pipeline.json')
assert pipeline['status']=='STOPPED_AT_Q1_CPU' and pipeline['stages'][-1]['exit_code']==1
assert not (ROOT/'q1_cpu.json').exists()
assert cpu['summary_sha256']==sha(ROOT/'q1/summary.json') and cpu['status']=='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1'
assert load(ROOT/'q1_cpu_arithmetic_recheck/receipt.json')['exit_code']==0
assert summary['project_commit']==pipeline['code_commit']=='1381639f778f77f124a2726ee55c07610092a438'
assert summary['config_sha256']==pipeline['config_sha256']==sha(REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json')
assert len(protocol['records'])==1032 and summary['optimizer_steps']==1560
all_ids=sorted({r['identity'] for r in protocol['records']})
members={i:{r['scene'] for r in protocol['records'] if r['identity']==i} for i in all_ids}
multi=[i for i in all_ids if len(members[i])>=2];single=[i for i in all_ids if len(members[i])==1]
assert (len(all_ids),len(multi),len(single))==(155,60,95)
counts=Counter(); endpoints=[];gradient_rows=[];epoch_rows=[];rank_rows=[];fold_scope=[];warmup=[];sqrt_differences=[]
rank_by=defaultdict(list);fold_maps={e:[] for e in ('control','balanced')}
for fold,fr,md in zip(protocol['folds'],summary['folds'],metadata['folds'],strict=True):
    f=fold['fold'];assert f==fr['fold']==md['fold']
    expected_hold=sorted(multi[f::3]+single[f::3]);assert fold['heldout_ids']==expected_hold
    assert set(fold['source_ids'])==set(all_ids)-set(expected_hold)
    assert set(fold['source_ids']).isdisjoint(expected_hold)
    expected_gallery=[r['index'] for r in protocol['records'] if r['identity'] in expected_hold]
    assert fold['gallery_record_indices']==expected_gallery
    gallery=[protocol['records'][i] for i in expected_gallery]
    eligible=[r['index'] for r in gallery if len(members[r['identity']])>1]
    assert [q['record_index'] for q in fold['query_rows']]==eligible
    assert set(fold['source_record_indices'])=={r['index'] for r in protocol['records'] if r['identity'] in fold['source_ids']}
    fold_scope.append(dict(fold=f,source_ids=len(fold['source_ids']),heldout_ids=len(expected_hold),gallery=len(gallery),queries=len(eligible),query_ids=len({r['identity'] for r in gallery if r['index'] in eligible}),distractor_ids=sum(len(members[i])==1 for i in expected_hold),distractor_records=sum(len(members[r['identity']])==1 for r in gallery)))
    paired_logs=[]
    for end in ('control','balanced'):
        folder=ROOT/'q1'/f'fold_{f}_{end}';receipt=load(folder/'receipt.json');tr=load(folder/'training.json')
        assert receipt==fr['endpoints'][end] and tr==receipt['training'] and all(receipt['engineering_checks'].values())
        assert tr['epochs']==20 and tr['optimizer_steps']==260 and len(tr['steps'])==260
        assert tr['nonzero_gradient_tensors']==tr['trainable_tensors']==203 and not tr['missing_nonzero_gradients']
        assert tr['overflow_events']==0 and tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
        assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']==receipt['initialization']['signal_state_sha256']
        assert tr['initial_state_sha256']==receipt['initialization']['initial_state_sha256'] and tr['final_state_sha256']==receipt['strict_reload_state_sha256']
        assert tr['history_anchor_count']==0 and tr['current_rank_backward_calls']==tr['current_auxiliary_backward_calls']==260 and tr['direct_component_backward_calls']==tr['extra_direct_check_record_forwards']==0
        assert tr['classification_head_rule']=='original_current_total_gradient'
        logpath=folder/'memory_steps.jsonl';assert sha(logpath)==tr['audit_files']['memory_steps.jsonl']['sha256']
        logs=[json.loads(line) for line in logpath.read_text().splitlines()];assert len(logs)==260;paired_logs.append((tr,logs))
        states={role:None for role in ROLES};cache=OrderedDict();stats=Counter();unique_seen=set();initial_fields=None;offset=0
        for step,(t,row,batch) in enumerate(zip(tr['steps'],logs,md['batches'],strict=True)):
            loc=dict(fold=f,endpoint=end,step=step+1)
            assert row['step']==t['step']==step+1 and row['zero_based_step']==step and t['epoch']==step//13+1
            indices=t['sampled_record_indices'];assert indices==row['record_indices']==batch['record_indices']
            unique_seen.update(indices); assert set(indices)<=set(fold['source_record_indices'])
            ids=[protocol['records'][i]['identity'] for i in indices]; scenes=[protocol['records'][i]['scene'] for i in indices]
            assert row['identities']==ids and row['scenes']==scenes and sorted(Counter(ids).values())==[8]*8
            for i in list(cache):
                if step-cache[i]>8:del cache[i]
            wanted=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],age=step-s,stored_step=s) for i,s in cache.items() if i not in set(indices)]
            assert wanted==row['memory'];active=step>=65
            assert row['replacement_active']==active and row['warmup_steps']==65 and row['history_anchor_count']==0 and row['current_anchor_count']==64
            assert t['active_fused_metric']==('cross_scene_smooth_ap' if active else 'hard_triplet')
            candidates=list(zip(ids,scenes))+[(r['identity'],r['scene']) for r in wanted]
            pc=[sum(i==j and s!=u for j,u in candidates) for i,s in zip(ids,scenes)]
            rel=row['relation_objective'];assert rel['cross_scene_positive_counts']==pc
            eligible_n=sum(n>0 for n in pc);support=dict(eligible_anchors=eligible_n,eligible_identities=len({i for i,n in zip(ids,pc) if n}),identity_directed_scene_relations=len({(i,s,u) for i,s,n in zip(ids,scenes,pc) if n for j,u in candidates if i==j and s!=u}))
            assert row['support']==support and rel['cross_scene_eligible_anchors']==eligible_n
            ap=rel['cross_scene_per_anchor_ap'];assert len(ap)==64 and all(ap[i]==0 for i,n in enumerate(pc) if not n)
            objective=1-float(np.mean([a for a,n in zip(ap,pc) if n])) if eligible_n else 0
            close(rel['cross_scene_loss'],objective,2e-6)
            close(t['components']['triplet_fused'],rel['cross_scene_loss'] if active else row['original_triplet'],2e-6)
            supported=active and eligible_n>0
            stats['supported_steps']+=int(supported);stats['warmup_steps']+=int(not active);stats['unsupported_postwarmup_steps']+=int(active and not supported)
            stats['eligible_anchor_exposures']+=eligible_n;stats['postwarmup_eligible_anchor_exposures']+=eligible_n*active
            stats['current_anchor_exposures']+=64;stats['historical_candidate_exposures']+=len(wanted)
            stats['memory_distance_elements']+=row['distance_float_count'];assert row['distance_offset_bytes']==offset;offset+=4*row['distance_float_count']
            assert row['distance_float_count']==4*64*(64+len(wanted))
            fresh=64*len({m['stored_step'] for m in wanted});assert row['fresh_role_record_forwards']==fresh
            stats['fresh_role_record_forwards']+=fresh
            upstream=row['historical_leaf_upstream_norms'];assert len(upstream)==len(wanted)
            assert all(math.isfinite(u) and u>=0 for u in upstream)
            groups=sorted({m['stored_step'] for m,u in zip(wanted,upstream) if u>0});assert groups==row['history_vjp_groups']
            assert row['history_vjp_record_forwards']==64*len(groups);stats['history_vjp_record_forwards']+=64*len(groups)
            for key in ('classification_head_gradients_unchanged','history_candidate_vjp_applied','selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'): assert row[key]
            assert not row['direct_single_group_check'] and not row['rank_auxiliary_reference_checks'] and row['direct_component_backward_calls']==0
            assert row['current_rank_backward_calls']==row['current_auxiliary_backward_calls']==1
            assert t['amp_scale_before']==t['amp_scale_after']==256.0
            c=t['components'];assert len(c)==14
            total=.25*c['id_fused']+c['triplet_fused']+sum((c['id_'+e]+c['id_residual_'+e])/12+.25*(c['triplet_'+e]+c['triplet_residual_'+e]) for e in ROLES)
            close(t['loss'],total,1e-5)
            for role in ROLES:
                rl=dict(**loc,role=role);b=row['gradient_balance'][role]
                for k in ('rank_vs_auxiliary','current_rank_vs_history','rank_vs_applied','auxiliary_vs_applied','original_sum_vs_applied','subtraction_auxiliary_vs_direct','direct_sum_vs_original'):pair(b[k],rl)
                pair(row['actual_parameter_updates'][role],rl)
                assert b['supported']==supported and b['before']==states[role]
                ra=b['rank_vs_auxiliary'];r=ra['first_norm'];a=ra['second_norm'];dot=0 if ra['cosine'] is None else r*a*ra['cosine']
                ratio=None;wr=wa=1.
                if supported:
                    before=states[role];states[role]=dict(rank=r,auxiliary=a,supported_steps=1) if before is None else dict(rank=.9*before['rank']+.1*r,auxiliary=.9*before['auxiliary']+.1*a,supported_steps=before['supported_steps']+1)
                    quotient=(states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12)
                    ratio=min(4.,max(.25,math.sqrt(quotient)));oldratio=min(4.,max(.25,quotient**.5))
                    if ratio!=oldratio:sqrt_differences.append(dict(**rl,sqrt=ratio,power=oldratio,recorded=b['ratio']))
                    wr=2*ratio/(1+ratio);wa=2/(1+ratio)
                assert b['after']==states[role] and b['ratio']==ratio
                assert b['proposed_rank_weight']==wr and b['proposed_auxiliary_weight']==wa
                if end=='control' or not supported:wr=wa=1.
                assert b['applied_rank_weight']==wr and b['applied_auxiliary_weight']==wa and .4<=wr<=1.6 and .4<=wa<=1.6
                close(wr+wa,2,1e-14)
                applied=b['rank_vs_applied']['second_norm'];equation('direct_sum_norm',b['direct_sum_vs_original']['first_norm']**2,r*r+a*a+2*dot,rl)
                if end=='balanced' and supported:
                    equation('original_fp64_weighted_norm',applied**2,wr*wr*r*r+wa*wa*a*a+2*wr*wa*dot,rl,False)
                    x,y=f32(wr),f32(wa);equation('correct_fp32_weighted_norm',applied**2,x*x*r*r+y*y*a*a+2*x*y*dot,rl)
                else:assert b['original_sum_vs_applied']['difference_norm']==0
                h=b['current_rank_vs_history'];u,v=h['first_norm'],h['second_norm'];uv=0 if h['cosine'] is None else u*v*h['cosine']
                equation('rank_current_history',r*r,u*u+v*v+2*uv,rl)
                if active and not supported:assert r==u==v==0
                gradient_rows.append(dict(**rl,supported=supported,rank_norm=r,auxiliary_norm=a,cosine=ra['cosine'],rank_weight=wr,auxiliary_weight=wa,current_rank_norm=u,history_rank_norm=v,applied_norm=applied,direct_sum_relative_difference=b['direct_sum_vs_original']['difference_norm']/max(b['direct_sum_vs_original']['first_norm'],1e-300),subtraction_auxiliary_relative_difference=b['subtraction_auxiliary_vs_direct']['difference_norm']/max(a,1e-300),actual_parameter_update_norm=row['actual_parameter_updates'][role]['difference_norm']))
            if active:
                for i in indices:cache.pop(i,None);cache[i]=step
                while len(cache)>512:cache.popitem(last=False)
        assert states==tr['gradient_balance_state'] and offset==tr['audit_files']['memory_distances.f32']['bytes']
        assert stats['fresh_role_record_forwards']+64==tr['extra_fresh_role_record_forwards'] and stats['history_vjp_record_forwards']==tr['extra_history_vjp_record_forwards']
        for h in tr['history']:
            rows=[s for s in tr['steps'] if s['epoch']==h['epoch']];assert len(rows)==h['optimizer_steps']==13
            close(h['mean_loss'],float(np.mean([s['loss'] for s in rows])),1e-14)
            ep=h['epoch'];mult=ep/5 if ep<=5 else .5*(1+math.cos(math.pi*(ep-5)/15))
            close(h['learning_rate'],.00035*mult,1e-16)
            epoch_rows.append(dict(fold=f,endpoint=end,**h,**{k:float(np.mean([s['components'][k] for s in rows])) for k in rows[0]['components']}))
        counts.update(stats);endpoints.append(dict(fold=f,endpoint=end,**stats,unique_source_records_exposed=len(unique_seen),source_identities_exposed=len({protocol['records'][i]['identity'] for i in unique_seen}),seconds=sum(h['elapsed_seconds'] for h in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'],peak_reserved_mib=tr['peak_reserved_mib']))
        ranks=load(folder/'rankings.json');assert sha(folder/'rankings.json')==receipt['retrieval']['rankings_sha256']
        ret=receipt['retrieval'];assert ret['gallery_manifest']==gallery and ret['query_rows']==fold['query_rows'];localmaps={}
        for out in OUTPUTS:
            rows=[];assert len(ranks[out])==len(eligible)
            for q,order in zip(fold['query_rows'],ranks[out],strict=True):
                assert sorted(order)==list(range(len(gallery)));query=gallery[q['gallery_position']];assert query['index']==q['record_index']
                legal=[gallery[j] for j in order if not(gallery[j]['identity']==query['identity'] and gallery[j]['scene']==query['scene'])]
                positives=[j+1 for j,g in enumerate(legal) if g['identity']==query['identity']];assert positives
                expected_neg=[g for g in gallery if g['identity']!=query['identity']]
                assert len([g for g in legal if g['identity']!=query['identity']])==len(expected_neg)
                ap=sum((j+1)/r for j,r in enumerate(positives))/len(positives)
                row=dict(fold=f,endpoint=end,output=out,record_index=query['index'],identity=query['identity'],scene=query['scene'],ap=ap,first_rank=positives[0],positive_count=len(positives),negative_count=len(expected_neg))
                rows.append(row);rank_rows.append(row);counts['rank_positions']+=len(order)
            saved=ret['outputs'][out];assert np.allclose([r['ap'] for r in rows],saved['average_precision'],rtol=0,atol=1e-14)
            assert [r['first_rank'] for r in rows]==saved['first_match_rank'];scores=metrics(rows)
            for k,v in scores.items():close(v,saved['metrics'][k])
            rank_by[(end,out)].extend(rows);localmaps[out]=scores
        fold_maps[end].append(localmaps)
    ta,la=paired_logs[0];tb,lb=paired_logs[1]
    assert fr['endpoints']['control']['initialization']==fr['endpoints']['balanced']['initialization']
    assert [(r['record_indices'],r['pixel_sha256']) for r in la]==[(r['record_indices'],r['pixel_sha256']) for r in lb]
    assert ta['steps'][:65]==tb['steps'][:65]
    assert la[:65]==lb[:65]
    warmup.append(dict(fold=f,matched_current_input_steps=260,all_warmup_training_and_audit_rows_exact=65))

labels=np.array([r['identity'] for r in rank_by[('control','fused')]]);assert len(labels)==600 and len(set(labels))==60
results={};comparison=summary['comparison']
for end in ('control','balanced'):
    scores={o:metrics(rank_by[(end,o)]) for o in OUTPUTS};aps={o:np.array([r['ap'] for r in rank_by[(end,o)]]) for o in OUTPUTS}
    gains={o:scores[o]['mAP']-scores['baseline_only']['mAP'] for o in OUTPUTS};fg=[m['fused']['mAP']-m['baseline_only']['mAP'] for m in fold_maps[end]]
    lower=bootstrap((aps['fused']-aps['baseline_only'])*100,labels)
    gates=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_gains_nonnegative=all(v>=0 for v in fg),all_full_branches_not_below_signal=all(gains[o]>=0 for o in ROLES),identity_bootstrap_lower_positive=lower>0,fused_strictly_best=all(scores['fused']['mAP']>scores[o]['mAP'] for o in OUTPUTS if o!='fused'))
    saved=comparison['endpoints'][end];assert gates==saved['scientific_checks'] and all(gates.values())==saved['scientific_passed'];close(lower,saved['identity_bootstrap']['lower_bound_pp'])
    for o in OUTPUTS:
        for k,v in scores[o].items():close(v,saved['metrics'][o][k])
    results[end]=dict(metrics=scores,gains_over_signal=gains,fold_fused_gains=fg,bootstrap_lower=lower,gates=gates)
assert [r['ap'] for r in rank_by[('control','baseline_only')]]==[r['ap'] for r in rank_by[('balanced','baseline_only')]]
deltas={o:(np.array([r['ap'] for r in rank_by[('balanced',o)]])-np.array([r['ap'] for r in rank_by[('control',o)]]))*100 for o in OUTPUTS}
gains={o:float(deltas[o].mean()) for o in OUTPUTS};lower=bootstrap(deltas['fused'],labels)
fg=[b['fused']['mAP']-a['fused']['mAP'] for a,b in zip(fold_maps['control'],fold_maps['balanced'])]
gates=dict(fused_gain_at_least_1pp=gains['fused']>=1,all_fold_fused_nonnegative=all(v>=0 for v in fg),all_role_gains_nonnegative=all(gains[o]>=0 for o in ROLES),paired_identity_bootstrap_lower_positive=lower>0,candidate_fused_strictly_best=results['balanced']['gates']['fused_strictly_best'])
assert gates==comparison['paired_checks'];close(lower,comparison['paired_bootstrap_lower_pp'])
for o in OUTPUTS:close(gains[o],comparison['matched_gains_mAP'][o])
for a,b in zip(fg,comparison['fold_fused_gains_mAP']):close(a,b)
qualified=all(gates.values()) and all(results['balanced']['gates'].values())
assert qualified==comparison['next_phase_qualified']==(summary['status']=='Q1_PASS')
identity_rows=[];changes={}
for o in OUTPUTS:
    before=np.array([r['first_rank'] for r in rank_by[('control',o)]]);after=np.array([r['first_rank'] for r in rank_by[('balanced',o)]])
    changes[o]=dict(ap_improved=int((deltas[o]>0).sum()),ap_declined=int((deltas[o]<0).sum()),ap_unchanged=int((deltas[o]==0).sum()),rank1_repaired=int(((before>1)&(after==1)).sum()),rank1_new_errors=int(((before==1)&(after>1)).sum()))
    for identity in np.unique(labels):
        mask=labels==identity;saved=next(r for r in comparison['paired_per_identity'] if r['identity']==identity)
        assert saved['query_count']==int(mask.sum());gain=float(deltas[o][mask].mean());close(gain,saved['gains_mAP'][o])
        identity_rows.append(dict(identity=int(identity),output=o,queries=int(mask.sum()),gain_pp=gain))
assert counts['rank_positions']==2069520 and len(rank_rows)==6000 and len(identity_rows)==300 and len(gradient_rows)==4680 and len(epoch_rows)==120
csvsave('all6000_query_endpoint_output_rows.csv',rank_rows);csvsave('all300_identity_output_deltas.csv',identity_rows)
csvsave('all4680_gradient_role_rows.csv',gradient_rows);csvsave('all120_training_epochs.csv',epoch_rows)
save('independent_text_replay.json',dict(status='PASS_COMPLETE_INDEPENDENT_TEXT_REPLAY',scientific_status=summary['status'],next_phase_qualified=qualified,fold_scope=fold_scope,counts=counts,endpoints=endpoints,paired_warmup=warmup,ranking=results,paired_gains=gains,paired_fold_gains=fg,paired_bootstrap_lower=lower,paired_gates=gates,paired_query_changes=changes,equation_maxima=maxima,original_formula_failures=original_failures,sqrt_power_mismatches=sqrt_differences,model_forwards=0,optimizer_updates=0,gradient_reconstruction=False))
print(json.dumps(dict(status='PASS_COMPLETE_INDEPENDENT_TEXT_REPLAY',scientific_status=summary['status'],counts=counts,paired_gains=gains,paired_fold_gains=fg,paired_bootstrap_lower=lower,paired_gates=gates,original_formula_failures=original_failures,sqrt_mismatches=len(sqrt_differences)),indent=2))
