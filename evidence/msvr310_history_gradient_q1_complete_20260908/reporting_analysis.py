from pathlib import Path
import csv
import hashlib
import json

tmp = Path('C:/Users/gb/.codex_tmp')
root = tmp / 'history_gradient_q1_terminal_processing_20260908'
intake = tmp / 'history_gradient_q1_complete_20260908'
a = json.loads((root / 'training_text_analysis.json').read_bytes())
def combined(rows, key):
    stats = [r[key] for r in rows if r[key]['count']]
    n = sum(s['count'] for s in stats)
    return dict(count=n, mean=sum(s['count'] * s['mean'] for s in stats) / n)
result = dict(scope='Reporting aggregation of complete saved text, no model recomputation', phases={}, identities={}, costs=[], prehistory=[])
for endpoint in ('control', 'history_gradient'):
    result['phases'][endpoint] = {}
    for label, lo, hi in [('all',1,20),('warmup',1,5),('epochs6_10',6,10),('epochs11_15',11,15),('epochs16_20',16,20)]:
        rows = [x for x in a['all_epochs'] if x['endpoint'] == endpoint and lo <= x['epoch'] <= hi]
        out = {k: combined(rows,k) for k in ('current_triplet','expanded_triplet','total_loss')}
        out['components'] = {k: combined([x['all14_components'] for x in rows],k) for k in rows[0]['all14_components']}
        out['mining_exposures'] = {k: sum(x['mining_exposures'][k] for x in rows) for k in rows[0]['mining_exposures']}
        result['phases'][endpoint][label] = out
    ends = [x for x in a['endpoints'] if x['endpoint'] == endpoint]
    result['phases'][endpoint]['roles_all_history'] = {}
    for role in ('cnn','transformer','mamba'):
        rows = [x['roles'][role] for x in ends]
        result['phases'][endpoint]['roles_all_history'][role] = {
            'history_to_total_norm_ratio': combined(rows, 'history_to_current_total_norm_ratio'),
            'cos_total_history': combined([x['total_vs_history'] for x in rows], 'cosine'),
            'cos_total_both': combined([x['total_vs_both'] for x in rows], 'cosine'),
            'applied_updates': sum(x['updates_with_applied_difference'] for x in rows)}
for x in a['endpoints']:
    t = json.loads((intake / f"q1/fold_{x['fold']}_{x['endpoint']}/training.json").read_bytes())
    result['costs'].append({**{k:x[k] for k in ('fold','endpoint','training_seconds','peak_allocated_mib','history_vjp_record_forwards','selected_vjp_group_exposures','zero_upstream_group_skip_exposures')}, 'receipt_fresh_record_forwards':t['extra_fresh_role_record_forwards'], 'overflow_events':t['overflow_events'], 'missing_nonzero_gradients':t['missing_nonzero_gradients']})
for fold in range(3):
    ts = [json.loads((intake/f'q1/fold_{fold}_{e}/training.json').read_bytes()) for e in ('control','history_gradient')]
    pairs = list(zip(ts[0]['steps'][:66],ts[1]['steps'][:66]))
    result['prehistory'].append(dict(fold=fold, steps=66, first_nonidentical_loss_step=next((x['step'] for x,y in pairs if x['loss']!=y['loss']),None), max_abs_total_loss_difference=max(abs(x['loss']-y['loss']) for x,y in pairs), max_abs_component_difference=max(abs(x['components'][k]-y['components'][k]) for x,y in pairs for k in x['components'])))
ids = list(csv.DictReader((root/'rankings/all300_identity_output_changes.csv').open(encoding='utf-8')))
queries = list(csv.DictReader((root/'rankings/all3000_query_output_changes.csv').open(encoding='utf-8')))
for output in ('baseline_only','fused','cnn','transformer','mamba'):
    rows = [x for x in ids if x['output']==output]
    assert len(rows) == 60
    for x in rows:
        x['query_count']=int(x['query_count'])
        x['delta_mAP_pp']=float(x['delta_mAP_pp'])
        x['query_weighted_contribution_pp']=x['query_count']*x['delta_mAP_pp']/600
    rows.sort(key=lambda x:x['query_weighted_contribution_pp'])
    q = [x for x in queries if x['output']==output]
    new = [x for x in q if x['rank1_new_error'].lower() in ('true','1')]
    groups = {}
    for old_right,new_right in ((True,True),(False,False),(False,True),(True,False)):
        members=[x for x in q if (int(x['control_first_match_rank'])==1)==old_right and (int(x['candidate_first_match_rank'])==1)==new_right]
        groups[f'{old_right}_to_{new_right}']=dict(queries=len(members),query_weighted_map_change_pp=sum(float(x['delta_ap_pp']) for x in members)/600)
    assert sum(x['queries'] for x in groups.values())==600
    result['identities'][output]=dict(improved=sum(x['delta_mAP_pp']>0 for x in rows), declined=sum(x['delta_mAP_pp']<0 for x in rows), unchanged=sum(x['delta_mAP_pp']==0 for x in rows), net_pp=sum(x['query_weighted_contribution_pp'] for x in rows), most_negative=rows[:5], most_positive=rows[-5:][::-1], new_rank1_errors=len(new), new_error_same_scene_nearest_negative=sum(x['scene']==x['candidate_nearest_negative_scene'] for x in new),rank1_transition_groups=groups)
result['inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'training_text_analysis.json',root/'rankings/all300_identity_output_changes.csv',root/'rankings/all3000_query_output_changes.csv')}
out=tmp/'history_gradient_q1_reporting_analysis_20260908.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(costs=result['costs'],prehistory=result['prehistory'],fused_identities=result['identities']['fused'],phases={e:{k:{s:v[s] for s in ('current_triplet','expanded_triplet','total_loss')} for k,v in result['phases'][e].items() if k!='roles_all_history'} for e in result['phases']},roles={e:result['phases'][e]['roles_all_history'] for e in result['phases']}),ensure_ascii=False,indent=2))
