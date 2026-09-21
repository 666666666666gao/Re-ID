from pathlib import Path
from collections import Counter
import csv,difflib,hashlib,json,math,shutil,statistics
OUT=Path(__file__).parent;TMP=OUT.parent;REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');ROOT=TMP/'trifusion_supported_balance_q1_complete_20260921'
def js(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<tol,(a,b)
remote=js(OUT/'remote_bindings.json');text=js(OUT/'independent_text_replay.json');arrays=js(OUT/'remote_arrays_attempt2.json')
for name,content in remote['texts'].items():
    p=OUT/'snapshots/remote_text'/name.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
extra=['tools/build_msvr310_train_oof_protocol.py','tools/msvr310_exact_signal_inference.py','tools/build_v12_complete_path_oof_targets.py','results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md','refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md','AGENTS.md']
for name in extra:
    p=OUT/'latest_claims'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,p)
reader_names=['analyze_supported_gradient_balance_q1_20260921_repaired.py','analyze_trifusion_supported_balance_q1_gradients_20260921_repaired.py','export_supported_balance_training_tables_20260921_repaired.py','audit_msvr_paired_ranking_text_repaired_20260921.py']
for n in reader_names:
    p=OUT/'snapshots/readers'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(TMP/n,p)
diff=''.join(difflib.unified_diff((REPO/'tools/audit_msvr_paired_ranking_text.py').read_text().splitlines(True),(TMP/reader_names[-1]).read_text().splitlines(True),fromfile='original_ranking_reader',tofile='repaired_ranking_reader'))
(OUT/'ranking_reader.diff').write_text(diff,encoding='utf-8')
eqroot=TMP/'trifusion_supported_balance_cpu_equations_audit_20260921'
for n in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','primary_scalar_audit.json','minimum_repair_scalar_replay.json','collect_primary_scalars.py','verify_minimum_repair_readonly.py'):
    p=OUT/'snapshots/arithmetic_audit'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(eqroot/n,p)
assert js(eqroot/'minimum_repair_scalar_replay.json')['status'].startswith('PASS')
source=js(TMP/'trifusion_supported_balance_q1_source_log_analysis_20260921.json')
grad=js(TMP/'trifusion_supported_balance_q1_gradient_analysis_20260921/summary.json')
rank=js(TMP/'trifusion_supported_balance_q1_ranking_analysis_20260921/ranking_replay.json')
for output,v in text['paired_gains'].items():close(v,rank['paired_gains'][output])
assert text['paired_gates']==rank['paired_gates'] and text['paired_query_changes']==rank['paired_changes']
source_endpoints=source['endpoints']
steps_export=csvread(TMP/'trifusion_supported_balance_q1_training_tables_20260921/all_training_steps.csv')
epochs_export=csvread(TMP/'trifusion_supported_balance_q1_training_tables_20260921/all_training_epochs.csv')
gradient_export=csvread(TMP/'trifusion_supported_balance_q1_gradient_analysis_20260921/role_steps.csv')
assert (len(steps_export),len(epochs_export),len(gradient_export))==(1560,120,4680)
steps_export={(r['endpoint'],int(r['step'])):r for r in steps_export};epochs_export={(r['endpoint'],int(r['epoch'])):r for r in epochs_export};gradient_export={(r['endpoint'],int(r['step']),r['role']):r for r in gradient_export}
cost={};late=[];zeros=[];role_summaries=[];source_checks=0
protocol=js(REPO/'protocols/msvr310_train_oof_v1.json');support=js(REPO/'evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json')
for r in protocol['records']:
    names={Path(p).name for p in r['paths']};assert len(names)==1;n=next(iter(names));assert (int(n[:4]),int(n[11]),int(n[6:9]))==(r['identity'],r['camera'],r['scene'])
    assert [Path(p).parts[-2] for p in r['paths']]==['vis','ni','th']
for f in range(3):
    for end in ('control','balanced'):
        name=f'fold_{f}_{end}';p=ROOT/'q1'/name;tr=js(p/'training.json');logs=[json.loads(s) for s in (p/'memory_steps.jsonl').read_text().splitlines()]
        for s,row in zip(tr['steps'],logs):
            ex=steps_export[(name,s['step'])]
            for k,v in dict(loss=s['loss'],amp_scale_before=s['amp_scale_before'],amp_scale_after=s['amp_scale_after'],**s['components']).items():close(ex[k],v)
            assert ex['active_fused_metric']==s['active_fused_metric']
            registered=support['folds'][f]['batches'][s['step']-1]
            assert registered['history_records']==len(row['memory'])
            assert registered['counts']['cross_positive_positions']==sum(row['relation_objective']['cross_scene_positive_counts'])
            assert registered['counts']['anchors_with_pool_cross']==row['support']['eligible_anchors'];source_checks+=1
            if row['support']['eligible_anchors']==0:zeros.append(dict(fold=f,endpoint=end,step=s['step'],active=row['replacement_active']))
            for role,b in row['gradient_balance'].items():
                exg=gradient_export[(name,s['step'],role)]
                for k,v in dict(rank_norm=b['rank_vs_auxiliary']['first_norm'],auxiliary_norm=b['rank_vs_auxiliary']['second_norm'],applied_rank_weight=b['applied_rank_weight'],applied_auxiliary_weight=b['applied_auxiliary_weight'],direct_sum_to_original_difference=b['direct_sum_vs_original']['difference_norm'],subtraction_to_direct_auxiliary_difference=b['subtraction_auxiliary_vs_direct']['difference_norm'],actual_adamw_parameter_delta_norm=row['actual_parameter_updates'][role]['difference_norm']).items():close(exg[k],v)
        for h in tr['history']:
            ex=epochs_export[(name,h['epoch'])];rows=[s for s in tr['steps'] if s['epoch']==h['epoch']]
            for k in ('optimizer_steps','learning_rate','mean_loss','elapsed_seconds'):close(ex[k],h[k])
            for k in rows[0]['components']:close(ex[k],sum(s['components'][k] for s in rows)/len(rows))
        sr=next(s for s in source_endpoints if s['fold']==f and s['endpoint']==end)
        losskeys=dict(batch_hard=lambda r:r['statistics']['current_triplet'],expanded_hard=lambda r:r['relation_objective']['hard_loss'],smooth_ap=lambda r:r['relation_objective']['smooth_ap_loss'],cross_scene_ap=lambda r:r['relation_objective']['cross_scene_loss'])
        phases=[('all_steps_1_260',0,260),('warmup_steps_1_65',0,65),('post_warmup_steps_66_260',65,260),('last_five_epochs_steps_196_260',195,260)]
        for phase,a,b in phases:
            for k,fn in losskeys.items():close(sr['phases'][phase]['mean_losses'][k],sum(fn(r) for r in logs[a:b])/(b-a))
        late.append(dict(fold=f,endpoint=end,**{k:sum(fn(r) for r in logs[195:])/65 for k,fn in losskeys.items()}))
        for role in ('cnn','transformer','mamba'):
            rows=[r['gradient_balance'][role] for r in logs if r['gradient_balance'][role]['supported']]
            row=dict(fold=f,endpoint=end,role=role,n=len(rows),median_rank_weight=statistics.median([r['applied_rank_weight'] for r in rows]),median_rank_to_auxiliary=statistics.median([r['rank_vs_auxiliary']['first_norm']/r['rank_vs_auxiliary']['second_norm'] for r in rows]),direct_sum_relative_median=statistics.median([r['direct_sum_vs_original']['difference_norm']/r['direct_sum_vs_original']['first_norm'] for r in rows]),direct_sum_relative_max=max(r['direct_sum_vs_original']['difference_norm']/r['direct_sum_vs_original']['first_norm'] for r in rows),negative_cosines=sum(r['rank_vs_auxiliary']['cosine'] is not None and r['rank_vs_auxiliary']['cosine']<0 for r in rows))
            gs=next(r for r in grad['role_summaries'] if r['endpoint']==name and r['role']==role);close(row['median_rank_weight'],gs['applied_rank_weight']['median']);role_summaries.append(row)
        ecost=dict(epoch_seconds=sum(h['elapsed_seconds'] for h in tr['history']),peak_reserved_mib=tr['peak_reserved_mib'],peak_allocated_mib=tr['peak_allocated_mib'],current_record_forwards=64*260,fresh_role_forwards=tr['extra_fresh_role_record_forwards'],history_vjp_forwards=tr['extra_history_vjp_record_forwards'],rank_backward=tr['current_rank_backward_calls'],auxiliary_backward=tr['current_auxiliary_backward_calls'],history_vjp_groups=tr['extra_history_vjp_record_forwards']//64)
        cost[name]=ecost
claims=(OUT/'latest_claims/results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md').read_text()
for line in claims.splitlines():
    if line.startswith('| ') and line.split('|')[1].strip() in ('baseline_only','fused','cnn','transformer','mamba'):
        _,out,a,b,d,r1a,r1b,_=line.split('|');out=out.strip()
        expected=(text['ranking']['control']['metrics'][out]['mAP'],text['ranking']['balanced']['metrics'][out]['mAP'],text['paired_gains'][out],text['ranking']['control']['metrics'][out]['Rank-1'],text['ranking']['balanced']['metrics'][out]['Rank-1'])
        for x,y in zip((a,b,d,r1a,r1b),expected):close(x,y,5.1e-7)
for end in ('control','balanced'):
    mean={k:sum(r[k] for r in late if r['endpoint']==end)/3 for k in ('batch_hard','expanded_hard','smooth_ap','cross_scene_ap')}
    parts=next(l for l in claims.splitlines() if l.startswith('| '+end+' |')).split('|')[2:6]
    for p,k in zip(parts,mean):close(p,mean[k],5.1e-9)
    cost[end+'_total']=dict(epoch_seconds=sum(v['epoch_seconds'] for k,v in list(cost.items()) if k.endswith('_'+end)),fresh_role_forwards=sum(v['fresh_role_forwards'] for k,v in list(cost.items()) if k.endswith('_'+end)),history_vjp_forwards=sum(v['history_vjp_forwards'] for k,v in list(cost.items()) if k.endswith('_'+end)))
late_direction=[]
for f in range(3):
    a=next(r for r in late if r['fold']==f and r['endpoint']=='control');b=next(r for r in late if r['fold']==f and r['endpoint']=='balanced')
    delta={k:b[k]-a[k] for k in ('batch_hard','expanded_hard','smooth_ap','cross_scene_ap')};assert delta['batch_hard']>0 and delta['expanded_hard']>0 and delta['cross_scene_ap']<0;late_direction.append(dict(fold=f,**delta))
summary=dict(status='PASS_ALL_LATEST_RESULT_NUMBERS_AND_TABLES',counts=dict(training_steps=1560,epoch_rows=120,role_rows=4680,query_endpoint_output_rows=6000,identity_output_rows=300,source_support_schedule_matches=source_checks),cost=cost,late_losses=late,late_direction=late_direction,role_summaries=role_summaries,zero_eligible_batches=zeros,normalization_check='Dataset count denominators and unit-feature normalization; no self-maximum performance scaling',source_isolation='All source metadata and original B0 1950 source steps checked, no new census or M0 rerun',latest_claim_path='latest_claims/results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md',latest_claim_sha256=sha(OUT/'latest_claims/results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md'))
save(OUT/'claims_cost_replay.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('role_summaries','zero_eligible_batches')},indent=2))
