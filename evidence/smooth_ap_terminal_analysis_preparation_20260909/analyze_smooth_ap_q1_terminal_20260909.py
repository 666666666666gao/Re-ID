"""Complete terminal source-log description; no model calls or gate changes."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,math


def analyze(rows,steps):
    counts=Counter();losses=Counter();components=Counter();metrics=Counter();ages=Counter()
    gradient={e:{k:[] for k in ('current_norm','history_norm','history_current_ratio','current_history_cosine','current_both_cosine')} for e in ('cnn','transformer','mamba')}
    ap_values=[];positive_counts=[]
    for row,tr in zip(rows,steps,strict=True):
        assert row['step']==tr['step'] and row['current_anchor_count']==64 and row['history_anchor_count']==0
        assert row['coordinate_rule']=='fresh' and row['history_candidate_vjp_applied']
        rel=row['relation_objective'];aps=rel['per_anchor_smoothed_ap'];pos=rel['positive_counts']
        assert len(aps)==len(pos)==64 and all(0<=x<=1 for x in aps) and all(x>0 for x in pos)
        assert abs(rel['smooth_ap_loss']-(1-sum(aps)/64))<=2e-6
        assert len(tr['components'])==14
        ap_values.extend(aps);positive_counts.extend(pos)
        counts['steps']+=1;counts['anchor_exposures']+=64;counts['history_record_exposures']+=len(row['memory'])
        counts['steps_with_history']+=bool(row['memory'])
        counts['positive_position_exposures']+=sum(pos)
        counts['ap_exactly_one_anchor_exposures']+=sum(x==1 for x in aps)
        counts['positive_upstream_history_records']+=sum(x>0 for x in row['historical_leaf_upstream_norms'])
        counts['history_vjp_groups']+=len(row['history_vjp_groups'])
        counts['fresh_role_record_forwards']+=row['fresh_role_record_forwards']
        counts['history_vjp_record_forwards']+=row['history_vjp_record_forwards']
        ages.update(row['zero_based_step']-m['stored_step'] for m in row['memory'])
        metrics[tr['active_fused_metric']]+=1
        components.update(tr['components'])
        losses.update(actual_total=tr['loss'],batch_hard=row['statistics']['current_triplet'],expanded_hard=rel['hard_loss'],smooth_ap=rel['smooth_ap_loss'])
        for e in gradient:
            x=row['roles'][e]['total_vs_history'];b=row['roles'][e]['total_vs_both'];g=gradient[e]
            g['current_norm'].append(x['first_norm']);g['history_norm'].append(x['second_norm'])
            if x['first_norm']>0:g['history_current_ratio'].append(x['second_norm']/x['first_norm'])
            if x['cosine'] is not None:g['current_history_cosine'].append(x['cosine'])
            if b['cosine'] is not None:g['current_both_cosine'].append(b['cosine'])
    n=counts['steps'];assert n>0
    def distribution(v):
        assert all(math.isfinite(x) for x in v)
        return dict(count=len(v),mean=sum(v)/len(v),minimum=min(v),maximum=max(v)) if v else dict(count=0)
    return dict(counts=dict(counts),active_metric_step_counts=dict(metrics),history_age_histogram=dict(ages),mean_losses={k:v/n for k,v in losses.items()},mean_components={k:v/n for k,v in components.items()},per_anchor_smoothed_ap=distribution(ap_values),positive_positions_per_anchor=distribution(positive_counts),gradient_runtime_witness={e:{k:distribution(v) for k,v in g.items()} for e,g in gradient.items()})


def paired_phase(first,second):
    assert len(first)==len(second)>0
    differences=[(a['step'],abs(a['loss']-b['loss'])) for a,b in zip(first,second,strict=True)]
    changed=[s for s,d in differences if d!=0]
    return dict(steps=len(first),max_abs_total_loss_difference=max(d for _,d in differences),total_loss_exact_steps=len(first)-len(changed),first_total_loss_difference_step=changed[0] if changed else None,max_abs_component_difference={k:max(abs(a['components'][k]-b['components'][k]) for a,b in zip(first,second,strict=True)) for k in first[0]['components']})


def main(root,output):
    assert not output.exists()
    inventory=json.loads((root/'remote_terminal_inventory.json').read_bytes())
    assert inventory['pipeline']['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL')
    assert not any(inventory['original_pid_exists'].values())
    intake=json.loads((root/'intake_complete.json').read_bytes())
    assert intake['status']=='RECEIVED_ALL_REGISTERED_TERMINAL_TEXT'
    assert intake['inventory_sha256']==hashlib.sha256((root/'remote_terminal_inventory.json').read_bytes()).hexdigest()
    for item in inventory['files']:
        b=(root/item['path']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
    summary=json.loads((root/'q1/summary.json').read_bytes());cpu=json.loads((root/'q1_cpu.json').read_bytes())
    assert cpu['status']=='PASS_COMPLETE_SMOOTH_AP_Q1' and cpu['checked_training_steps']==1560
    assert cpu['summary_sha256']==hashlib.sha256((root/'q1/summary.json').read_bytes()).hexdigest()
    ends=[];paired=[]
    for fold in range(3):
        groups={}
        for end in ('control','smooth_ap'):
            p=root/'q1'/f'fold_{fold}_{end}';tr=json.loads((p/'training.json').read_bytes())
            rows=list(map(json.loads,(p/'memory_steps.jsonl').read_text().splitlines()))
            assert len(rows)==len(tr['steps'])==tr['optimizer_steps']==260
            assert all(r['warmup_steps']==65 and r['replacement_active']==(r['zero_based_step']>=65) for r in rows)
            assert all(s['active_fused_metric']==('smooth_ap' if end=='smooth_ap' and s['step']>65 else 'hard_triplet') for s in tr['steps'])
            phases={name:analyze(rows[a:b],tr['steps'][a:b]) for name,a,b in [('all_steps_1_260',0,260),('warmup_steps_1_65',0,65),('post_warmup_steps_66_260',65,260),('last_five_epochs_steps_196_260',195,260)]}
            ends.append(dict(fold=fold,endpoint=end,phases=phases,cost=dict(fresh_role_record_forwards_including_zero_check=tr['extra_fresh_role_record_forwards'],initial_zero_check_record_forwards=64,historical_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],fit_epoch_seconds=sum(x['elapsed_seconds'] for x in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'])))
            groups[end]=(rows,tr['steps'])
        a,at=groups['control'];b,bt=groups['smooth_ap']
        assert all((x['record_indices'],x['pixel_sha256'])==(y['record_indices'],y['pixel_sha256']) for x,y in zip(a,b,strict=True))
        starts=[next(i for i,r in enumerate(x) if r['memory']) for x in (a,b)];assert starts[0]==starts[1]
        paired.append(dict(fold=fold,paired_records_pixels_exact=True,first_history_step=starts[0]+1,warmup=paired_phase(at[:65],bt[:65]),before_first_history=paired_phase(at[:starts[0]],bt[:starts[0]])))
    result=dict(status='COMPLETE_DESCRIPTIVE_SOURCE_LOG_ANALYSIS',q1_summary_sha256=cpu['summary_sha256'],input_inventory_sha256=intake['inventory_sha256'],endpoints=ends,paired_warmup=paired,scientific_status=summary['status'],model_forwards=0,optimizer_updates=0,scope='All six completed source training logs. Repeated exposures are not independent instances; smoothed training AP is not retrieval AP. Gradient statistics are saved runtime witnesses, not independent model backward. Actual totals use different active objectives, while hard and AP columns preserve common definitions. Does not change gates or substitute for complete retrieval verification and independent audit.')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],endpoints=len(ends),steps=1560,output=str(output))))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
