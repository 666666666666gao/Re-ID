"""Check every descriptive source statistic, report table and CSV against raw text.
Uses fresh independent remote arithmetic for AP/metrics; no executor imports.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import numpy as np

OUT=Path(__file__).resolve().parent
TMP=OUT.parent
RAW=TMP/'trifusion_cross_scene_q1_complete_20260921'
REPORT=TMP/'trifusion_cross_scene_q1_analysis_20260921_v2'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arithmetic=read(OUT/'07_remote_arithmetic.stdout')
source=read(TMP/'trifusion_cross_scene_q1_source_log_analysis_20260921.json')
executor=read(REPORT/'terminal_analysis.json')
checks=0;max_error=0.
def same(a,b,path='root'):
    global checks,max_error
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(v,w) in enumerate(zip(a,b)):same(v,w,path+'.'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        delta=abs(a-b);max_error=max(max_error,delta)
        assert delta<=1e-10*max(1,abs(a),abs(b)),(path,a,b)
        checks+=1
    else:assert a==b,(path,a,b);checks+=1
def dist(v):
    return dict(count=len(v),mean=math.fsum(v)/len(v),minimum=min(v),maximum=max(v)) if v else dict(count=0)
def phase(a,t):
    rel=[x['relation_objective'] for x in a]
    counter={
        'cross_scene_eligible_anchor_exposures':sum(r['cross_scene_eligible_anchors'] for r in rel),
        'cross_scene_positive_position_exposures':sum(sum(r['cross_scene_positive_counts']) for r in rel),
        'cross_scene_zero_eligible_batches':sum(r['cross_scene_eligible_anchors']==0 for r in rel),
        'steps':len(a),'anchor_exposures':64*len(a),'history_record_exposures':sum(len(x['memory']) for x in a),
        'steps_with_history':sum(bool(x['memory']) for x in a),
        'positive_position_exposures':sum(sum(r['positive_counts']) for r in rel),
        'ap_exactly_one_anchor_exposures':sum(v==1 for r in rel for v in r['per_anchor_smoothed_ap']),
        'positive_upstream_history_records':sum(v>0 for x in a for v in x['historical_leaf_upstream_norms']),
        'history_vjp_groups':sum(len(x['history_vjp_groups']) for x in a),
        'fresh_role_record_forwards':sum(x['fresh_role_record_forwards'] for x in a),
        'history_vjp_record_forwards':sum(x['history_vjp_record_forwards'] for x in a)}
    gradients={}
    for role in ('cnn','transformer','mamba'):
        gs=[x['roles'][role]['total_vs_history'] for x in a]
        vals=dict(current_norm=[g['first_norm'] for g in gs],history_norm=[g['second_norm'] for g in gs],
            history_current_ratio=[g['second_norm']/g['first_norm'] for g in gs if g['first_norm']>0],
            current_history_cosine=[g['cosine'] for g in gs if g['cosine'] is not None],
            current_both_cosine=[x['roles'][role]['total_vs_both']['cosine'] for x in a if x['roles'][role]['total_vs_both']['cosine'] is not None])
        gradients[role]={k:dist(v) for k,v in vals.items()}
    losses=dict(actual_total=[r['loss'] for r in t],batch_hard=[r['original_triplet'] for r in a],
                expanded_hard=[r['hard_loss'] for r in rel],smooth_ap=[r['smooth_ap_loss'] for r in rel],cross_scene_ap=[r['cross_scene_loss'] for r in rel])
    return dict(counts=counter,active_metric_step_counts=dict(Counter(x['active_fused_metric'] for x in t)),
        history_age_histogram={str(k):v for k,v in Counter(m['age'] for x in a for m in x['memory']).items()},
        mean_losses={k:math.fsum(v)/len(v) for k,v in losses.items()},
        mean_components={k:math.fsum(r['components'][k] for r in t)/len(t) for k in t[0]['components']},
        per_anchor_smoothed_ap=dist([v for r in rel for v in r['per_anchor_smoothed_ap']]),
        positive_positions_per_anchor=dist([v for r in rel for v in r['positive_counts']]),gradient_runtime_witness=gradients)

raw_tr={};raw_aud={};all_steps=[]
for saved in source['endpoints']:
    f,e=saved['fold'],saved['endpoint'];p=RAW/'q1'/f'fold_{f}_{e}'
    tr=read(p/'training.json');a=[json.loads(s) for s in (p/'memory_steps.jsonl').read_text().splitlines()]
    raw_tr[f,e]=tr;raw_aud[f,e]=a
    for name,begin,end in [('all_steps_1_260',0,260),('warmup_steps_1_65',0,65),('post_warmup_steps_66_260',65,260),('last_five_epochs_steps_196_260',195,260)]:
        same(phase(a[begin:end],tr['steps'][begin:end]),saved['phases'][name],f'{f}.{e}.{name}')
    cost=dict(fresh_role_record_forwards_including_zero_check=tr['extra_fresh_role_record_forwards'],initial_zero_check_record_forwards=64,
              historical_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],
              fit_epoch_seconds=math.fsum(x['elapsed_seconds'] for x in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'])
    same(cost,saved['cost'])
    for row in tr['steps']:all_steps.append(dict(fold=f,endpoint=e,step=row['step'],active_fused_metric=row['active_fused_metric'],total_loss=row['loss'],**row['components']))
for saved,ind in zip(source['paired_warmup'],arithmetic['warmup_pairing']):
    f=saved['fold'];assert f==ind['fold'] and saved['paired_records_pixels_exact']
    a=raw_aud[f,'control'];b=raw_aud[f,'cross_scene']
    assert [(x['record_indices'],x['pixel_sha256']) for x in a]==[(x['record_indices'],x['pixel_sha256']) for x in b]
    first=next(n for n,x in enumerate(a,1) if x['memory']);assert first==saved['first_history_step']==67
    for key,last in (('warmup',65),('before_first_history',first-1)):
        ta=raw_tr[f,'control']['steps'][:last];tb=raw_tr[f,'cross_scene']['steps'][:last]
        d=[abs(x['loss']-y['loss']) for x,y in zip(ta,tb)]
        expected=dict(steps=last,max_abs_total_loss_difference=max(d),total_loss_exact_steps=sum(x==0 for x in d),
            first_total_loss_difference_step=next((n for n,x in enumerate(d,1) if x),None),
            max_abs_component_difference={k:max(abs(x['components'][k]-y['components'][k]) for x,y in zip(ta,tb)) for k in ta[0]['components']})
        same(expected,saved[key])
    same(saved['warmup']['max_abs_total_loss_difference'],ind['maximum_loss_difference'])

qindex={(r['fold'],r['endpoint'],r['output'],r['record_index']):r for r in arithmetic['query_rows']}
query_changes=[];changes={};identity_index={(r['identity'],r['output']):r for r in arithmetic['identity_rows']}
for f in range(3):
    item=read(RAW/'q1'/f'fold_{f}_cross_scene'/'receipt.json')['retrieval']
    rankings=read(RAW/'q1'/f'fold_{f}_cross_scene'/'rankings.json')
    gallery=item['gallery_manifest']
    for output in ('baseline_only','fused','cnn','transformer','mamba'):
        for qi,q in enumerate(item['query_rows']):
            a=qindex[f,'control',output,q['record_index']];b=qindex[f,'cross_scene',output,q['record_index']]
            row=dict(fold=f,output=output,query_record_index=q['record_index'],identity=q['identity'],scene=q['scene'],
                control_ap=a['ap'],candidate_ap=b['ap'],gain_ap_pp=(b['ap']-a['ap'])*100,
                control_first_match_rank=a['first_rank'],candidate_first_match_rank=b['first_rank'],
                rank1_repaired=a['first_rank']!=1 and b['first_rank']==1,
                rank1_new_error=a['first_rank']==1 and b['first_rank']!=1)
            if row['rank1_new_error']:
                top=next(gallery[j] for j in rankings[output][qi] if not(gallery[j]['identity']==q['identity'] and gallery[j]['scene']==q['scene']))
                assert top['identity']!=q['identity']
                row.update(new_error_top_identity=top['identity'],new_error_top_scene=top['scene'],new_error_top_same_scene=top['scene']==q['scene'])
            query_changes.append(row)
for output in ('baseline_only','fused','cnn','transformer','mamba'):
    rows=[r for r in query_changes if r['output']==output]
    ids=[r['gain_pp'] for r in arithmetic['identity_rows'] if r['output']==output]
    changes[output]=dict(query_count=600,ap_improved=sum(r['gain_ap_pp']>0 for r in rows),ap_declined=sum(r['gain_ap_pp']<0 for r in rows),
        ap_unchanged=sum(r['gain_ap_pp']==0 for r in rows),rank1_repaired=sum(r['rank1_repaired'] for r in rows),
        rank1_new_errors=sum(r['rank1_new_error'] for r in rows),new_errors_same_scene=sum(r.get('new_error_top_same_scene',False) for r in rows),
        identities_improved=sum(v>0 for v in ids),identities_declined=sum(v<0 for v in ids),identities_unchanged=sum(v==0 for v in ids))
same(changes,executor['paired_query_changes'])
same(executor['paired_checks'],arithmetic['paired_checks']);same(executor['signal_checks'],arithmetic['endpoints']['cross_scene']['checks'])
same(executor['paired_gains'],arithmetic['paired_gains_pp']);same(executor['fold_gains'],arithmetic['paired_fold_gains_pp'])
same(executor['paired_bootstrap_lower_pp'],arithmetic['paired_bootstrap_lower_pp'])
assert len(executor['actual_postwarmup_zero_support'])==4
for z in executor['actual_postwarmup_zero_support']:
    match=next(r for r in arithmetic['zero_batches'] if r['fold']==z['fold'] and r['endpoint']=='cross_scene' and r['step']==z['step'])
    assert match['replacement_active'] and match['cross_loss']==z['rank_loss']==0 and match['historical_candidates']==z['history_records']
    assert match['historical_upstream_nonzero']==match['history_vjp_forwards']==z['vjp_groups']==0 and z['history_upstream_all_zero']
pooled={}
for end in ('control','cross_scene'):
    selected=[r for r in source['endpoints'] if r['endpoint']==end]
    pooled[end]=dict(last65_common_mean_losses={k:math.fsum(r['phases']['last_five_epochs_steps_196_260']['mean_losses'][k] for r in selected)/3 for k in selected[0]['phases']['last_five_epochs_steps_196_260']['mean_losses']},
        fit_epoch_seconds=math.fsum(r['cost']['fit_epoch_seconds'] for r in selected),
        fresh_role_record_forwards_including_zero_check=sum(r['cost']['fresh_role_record_forwards_including_zero_check'] for r in selected),
        history_vjp_record_forwards=sum(r['cost']['historical_vjp_record_forwards'] for r in selected),
        peak_allocated_mib=max(r['cost']['peak_allocated_mib'] for r in selected))
same(pooled,executor['pooled_source']);same(source['paired_warmup'],executor['warmup'])

csv_counts={}
def csvcheck(filename,expected):
    actual=list(csv.DictReader((REPORT/filename).open(encoding='utf-8',newline='')))
    assert len(actual)==len(expected),(filename,len(actual),len(expected))
    for row,wanted in zip(actual,expected):
        for key,value in wanted.items():
            if isinstance(value,bool):assert row[key]==str(value)
            elif isinstance(value,(int,float)):same(float(row[key]),value,filename+'.'+key)
            else:assert row[key]==value
        assert all(v=='' for key,v in row.items() if key not in wanted)
    csv_counts[filename]=len(actual)
csvcheck('all_paired_queries.csv',query_changes)
csvcheck('all_training_step_components.csv',all_steps)
csvcheck('all_training_epochs.csv',arithmetic['epochs'])
foldrows=[]
for f in range(3):
    for output in ('baseline_only','fused','cnn','transformer','mamba'):
        for end in ('control','cross_scene'):
            r=next(r for r in arithmetic['distances'] if (r['fold'],r['endpoint'],r['output'])==(f,end,output))
            foldrows.append(dict(fold=f,endpoint=end,output=output,queries=[210,207,183][f],gallery=[360,349,323][f],**r['metrics']))
csvcheck('all_fold_output_metrics.csv',foldrows)
idrows=[dict(identity=i,query_count=identity_index[i,'fused']['query_count'],**{o:identity_index[i,o]['gain_pp'] for o in ('baseline_only','fused','cnn','transformer','mamba')}) for i in sorted({k[0] for k in identity_index})]
csvcheck('all_paired_identities.csv',idrows)
for name,record in read(REPORT/'manifest.json').items():
    assert sha(REPORT/name)==record['sha256'] and (REPORT/name).stat().st_size==record['bytes']
inventory=read(OUT/'03_remote_intake.stdout')['inventory'];storage={}
for item in inventory:
    relative=item['path'].split('msvr310_cross_scene_smooth_ap_v1_seed42_d35864d/',1)[1]
    category=relative.split('/',1)[0] if '/' in relative else 'root_text'
    storage[category]=storage.get(category,0)+item['bytes']
jsonout=dict(status='PASS_ALL_DESCRIPTIVE_STATISTICS_AND_CSV_CLAIMS',scalar_checks=checks,maximum_numeric_difference=max_error,
    source_log_phases_checked=24,csv_rows=csv_counts,paired_query_changes=changes,pooled_source=pooled,storage_bytes=storage,
    immutable_primary_hashes={str(p):sha(p) for p in (TMP/'trifusion_cross_scene_q1_source_log_analysis_20260921.json',REPORT/'REPORT.md',REPORT/'terminal_analysis.json')},
    report_scope_findings=['Numerical and descriptive claims match complete raw evidence.',
        'Registered Q1 scientific gates remain failed; one fixed seed and reused development folds do not establish causal robustness.',
        'Runtime gradient witness terminology is appropriate; there is no Q1 direct-gradient reconstruction.',
        'Tracker and repository AGENTS still describe a running phase at the inspected snapshot.'],
    model_forwards=0,optimizer_updates=0)
(OUT/'descriptive_claim_checks.json').write_text(json.dumps(jsonout,indent=2)+'\n',encoding='utf-8')
for name,rows in [('all6000_independent_query_output_rows.csv',arithmetic['query_rows']),('all300_independent_identity_output_changes.csv',arithmetic['identity_rows']),('all1560_independent_step_arithmetic.csv',arithmetic['steps']),('all120_independent_epoch_rows.csv',arithmetic['epochs']),('all_zero_eligible_rows.csv',arithmetic['zero_batches'])]:
    with (OUT/name).open('x',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
extra=[TMP/'analyze_cross_scene_terminal_results_20260921.py',TMP/'analyze_cross_scene_smooth_ap_q1_20260921.py',TMP/'trifusion_cross_scene_q1_analysis_20260921/FAILURE.md',TMP/'trifusion_cross_scene_q1_analysis_20260921/failed_v1.py']
extras=[]
for i,p in enumerate(extra):
    target=OUT/'snapshots'/('extra_'+str(i)+'_'+p.name);shutil.copyfile(p,target)
    extras.append(dict(path=str(p),snapshot=str(target.relative_to(OUT)),sha256=sha(p)))
(OUT/'additional_input_manifest.json').write_text(json.dumps(extras,indent=2)+'\n',encoding='utf-8')
print(json.dumps(jsonout))
