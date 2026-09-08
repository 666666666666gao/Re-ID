"""Independent stdlib-only exhaustive text audit. No author code imports or model calls."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, hashlib, json, math, re, statistics, struct, subprocess, sys

R=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
T=Path('C:/Users/gb/.codex_tmp')
O=Path(__file__).parent
S=T/'history_gradient_complete_source_20260908'
P=T/'history_gradient_complete_processing_20260908'
A=T/'history_gradient_complete_analysis_20260908'
F=T/'history_gradient_complete_figures_20260908'
Q=R/'evidence/msvr310_fresh_coordinate_complete_q1_20260908'
hashes={}
counts=Counter()
maxima=defaultdict(float)

def raw(p):
    b=p.read_bytes(); hashes[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return b
def js(p):return json.loads(raw(p))
def sha(p):raw(p);return hashes[str(p)]['sha256']
def eq(a,b,kind):
    assert a==b,(kind,a,b); counts[kind]+=1
def close(a,b,kind,rtol=1e-12,atol=1e-12):
    assert math.isclose(a,b,rel_tol=rtol,abs_tol=atol),(kind,a,b)
    counts[kind]+=1;maxima[kind+'_max_absolute_difference']=max(maxima[kind+'_max_absolute_difference'],abs(a-b))
def quantiles(v):
    s=sorted(v);n=len(s)
    return [s[math.floor((n-1)*p)]+((n-1)*p%1)*(s[math.ceil((n-1)*p)]-s[math.floor((n-1)*p)]) for p in [0,.25,.5,.75,1]]
def aggregate(v):
    valid=[x for x in v if x is not None]
    return dict(defined=len(valid),undefined=len(v)-len(valid),mean=statistics.fmean(valid) if valid else None,
                minimum=min(valid) if valid else None,maximum=max(valid) if valid else None,
                negative=sum(x<0 for x in valid),quantiles=quantiles(valid) if valid else None)
def recursive_match(a,b,kind):
    if isinstance(a,dict):
        eq(set(a),set(b),kind+'_keys')
        for k in a:recursive_match(a[k],b[k],kind)
    elif isinstance(a,list):
        eq(len(a),len(b),kind+'_length')
        for x,y in zip(a,b):recursive_match(x,y,kind)
    elif isinstance(a,float):close(a,b,kind+'_numeric')
    else:eq(a,b,kind+'_value')

manifest=js(S/'intake_manifest.json')
eq(len(manifest['files']),25,'intake_file_count')
for item in manifest['files']:
    p=S/item['path'];b=raw(p)
    eq(len(b),item['bytes'],'intake_size');eq(sha(p),item['sha256'],'intake_sha256')
pipe=js(S/'pipeline.json');summary=js(S/'source/summary.json');cpu=js(S/'source/cpu_verification.json')
post=js(P/'all_statistics_postcheck.json');analysis=js(A/'complete_text_reaggregation.json');plot=js(F/'plot_receipt.json')
spec=js(R/'configs/MSVR310/TriFusion-history-candidate-gradient-v1.json')
protocol=js(R/'protocols/msvr310_train_oof_v1.json')
eq(pipe['status'],'COMPLETE_VERIFIED_SOURCE_ONLY','terminal_status')
eq(summary['status'],'PASS_COMPLETE_FIXED_STATE_PROBE','terminal_status')
eq(summary['mode'],'source','source_mode')
eq(cpu['status'],'PASS_COMPLETE_FIXED_STATE_PROBE_CPU','terminal_status')
eq([x['stage'] for x in pipe['stages']],['t0','preflight','preflight_cpu','source','source_cpu'],'stage_order')
for stage in pipe['stages']:eq(stage['exit_code'],0,'stage_exit')
for obj in [pipe,summary,cpu,post,analysis]:eq(obj['optimizer_updates'],0,'zero_optimizer_witness')
eq(sha(S/'source/summary.json'),pipe['summary_sha256'],'terminal_sha')
eq(sha(S/'source/cpu_verification.json'),pipe['cpu_sha256'],'terminal_sha')
for obj in [cpu,post,analysis,plot]:eq(obj['summary_sha256'],sha(S/'source/summary.json'),'summary_sha')
for obj in [pipe,summary,post]:eq(obj['config_sha256'],sha(R/'configs/MSVR310/TriFusion-history-candidate-gradient-v1.json'),'config_sha')
eq(post['cpu_sha256'],sha(S/'source/cpu_verification.json'),'post_cpu_sha')
eq(post['protocol_sha256'],sha(R/'protocols/msvr310_train_oof_v1.json'),'post_protocol_sha')
eq(post['verifier_sha256'],sha(R/'tools/verify_msvr_history_gradient_all_statistics.py'),'post_verifier_sha')
remote_binding=js(P/'all_statistics_remote_binding.json')
eq(remote_binding['sha256'],sha(P/'all_statistics_postcheck.json'),'post_receipt_sha')
eq(remote_binding['bytes'],len(raw(P/'all_statistics_postcheck.json')),'post_receipt_size')
execution=js(P/'execution_binding.json')
for p,h in execution['scripts'].items():eq(sha(R/p),h,'postprocessing_script_sha')
for field,name in [('script_sha256','finish_history_gradient_source_20260908.py'),('observer_sha256','observe_history_gradient_source_20260908.py'),('intake_sha256','receive_history_gradient_complete_source_20260908.py')]:
    eq(sha(T/name),execution[field],'processing_entry_sha')
eq(plot['aggregate_sha256'],sha(A/'complete_text_reaggregation.json'),'plot_input_sha')
eq(plot['csv_sha256'],sha(A/'all_role_history_steps.csv'),'plot_input_sha')
eq(plot['cpu_sha256'],sha(S/'source/cpu_verification.json'),'plot_input_sha')
eq(plot['plot_script_sha256'],sha(R/'tools/plot_msvr_history_candidate_gradients.py'),'plot_input_sha')
for name,h in plot['artifacts'].items():eq(sha(F/name),h,'figure_sha')
png=raw(F/'msvr310_history_candidate_gradients_source.png');eq(png[:8],b'\x89PNG\r\n\x1a\n','png_signature')
png_dimensions=struct.unpack('>II',png[16:24])

# Five config layers actually reached by the source context, including the inherited loader/model code.
config_names=['TriFusion-history-candidate-gradient-v1.json','TriFusion-fresh-coordinate-paired-v1.json','TriFusion-instance-memory-paired-v1.json','TriFusion-source-style-paired-v1-r2.json','Signal-source-oof-v1.json']
bindings=[]
for name in config_names:
    d=js(R/'configs/MSVR310'/name)
    for key in ['project_file_sha256','project_source_file_sha256']:
        for path,h in d.get(key,{}).items():
            b=raw(R/path);exact=hashlib.sha256(b).hexdigest()==h;lf=hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==h
            assert exact or lf,(name,path)
            bindings.append(dict(config=name,path=path,expected_sha256=h,local_sha256=hashlib.sha256(b).hexdigest(),exact=exact,lf_equivalent=lf))
base=js(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
metadata=js(R/base['SOURCE_METADATA']['PATH'])
eq(sha(R/base['SOURCE_METADATA']['PATH']),base['SOURCE_METADATA']['SHA256'],'registered_order_sha')
eq(sha(R/base['BASELINE']['CONFIG']),base['BASELINE']['CONFIG_SHA256'],'base_config_sha')
qsummary=js(Q/'q1/summary.json');qcpu=js(Q/'q1_cpu.json')
eq(sha(Q/'q1/summary.json'),spec['fixed_file_sha256'][spec['q1_summary']],'inherited_q1_summary_sha')
eq(sha(Q/'q1_cpu.json'),spec['fixed_file_sha256'][spec['q1_cpu']],'inherited_q1_cpu_sha')

# Derive all protocol records and fold identity assignments from the original label inventory.
labels_path=R/'evidence/vehicle_query_protocol_labels_20260905.json'
labels=js(labels_path);eq(sha(labels_path),protocol['label_evidence_sha256'],'label_inventory_sha')
train=next(x for x in labels['datasets'] if x['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
eq(len(train),1032,'label_record_count');eq(len(protocol['records']),1032,'protocol_record_count')
for i,(r,original) in enumerate(zip(protocol['records'],train)):
    basename=PurePosixPath(original['path']).name
    identity,camera,scene=int(basename[:4]),int(basename[11]),int(basename[6:9])
    expected=dict(index=i,identity=identity,camera=camera,scene=scene,
                  paths=[f'bounding_box_train/{identity:04d}/{m}/{basename}' for m in ['vis','ni','th']])
    eq(r,expected,'all_protocol_record_mapping')
    eq([identity,camera,scene],[original['identity'],original['camera'],original['scene']],'filename_label_mapping')
identities=sorted({r['identity'] for r in protocol['records']})
scene_map={i:sorted({r['scene'] for r in protocol['records'] if r['identity']==i}) for i in identities}
eq({str(i):v for i,v in scene_map.items()},protocol['identity_scene_membership'],'identity_scene_membership')
held=[set() for _ in range(3)]
for group in [[i for i in identities if len(scene_map[i])>1],[i for i in identities if len(scene_map[i])==1]]:
    for j,i in enumerate(group):held[j%3].add(i)
fold_scope=[]
for f in protocol['folds']:
    k=f['fold'];source_ids=sorted(set(identities)-held[k])
    eq(f['heldout_ids'],sorted(held[k]),'fold_partition');eq(f['source_ids'],source_ids,'fold_partition')
    eq(f['source_label_map'],{str(i):j for j,i in enumerate(source_ids)},'legal_class_zero_map')
    eq(f['source_record_indices'],[i for i,r in enumerate(protocol['records']) if r['identity'] in source_ids],'fold_source_indices')
    eq(f['gallery_record_indices'],[i for i,r in enumerate(protocol['records']) if r['identity'] in held[k]],'fold_heldout_indices')
    fold_scope.append(dict(fold=k,source_identities=len(source_ids),source_records=len(f['source_record_indices']),source_scenes=len(f['source_scene_values']),class_zero_identity=source_ids[0]))

all_rows=[];states=[];absent=[];paired={};proofs=[];all_stat_sums={};memory_exposures=0;nonzero_exposures=0;norm_points=[]
fields=['history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine']
for fold in summary['folds']:
    k=fold['fold'];f=protocol['folds'][k]
    eq(set(fold['states']),{'initial','control','fresh_memory'},'state_set')
    for name,rec in fold['states'].items():
        d=S/'source'/f'fold_{k}_{name}';eq(js(d/'receipt.json'),rec,'state_receipt_equality')
        for fn,h in rec['files'].items():
            p=d/fn
            if p.exists():eq(sha(p),h['sha256'],'state_file_sha');eq(len(raw(p)),h['bytes'],'state_file_size')
            else:absent.append(dict(path=str(p),**h))
        eq(rec['initial_state_sha256'],rec['final_state_sha256'],'fixed_state_witness')
        old=qsummary['folds'][k]['endpoints']['control' if name=='initial' else name]
        expected_state=old['initialization']['initial_state_sha256'] if name=='initial' else old['training']['final_state_sha256']
        eq(rec['initial_state_sha256'],expected_state,'inherited_state_digest')
        trainable=old['initialization']['trainable_names']
        encoder=[x for x in trainable if x.startswith('encoder.')]
        role_counts={role:sum(n.startswith('encoder.'+role+'_') for n in encoder) for role in ['cnn','transformer','mamba']}
        eq(sum(role_counts.values()),len(encoder),'complete_encoder_role_partition')
        assert all(role_counts.values())
        if name!='initial':eq(old['checkpoint_sha256'],spec['fixed_file_sha256'][old['checkpoint']],'inherited_checkpoint_reference')
        old_rows=[json.loads(x) for x in raw(Q/'q1'/f'fold_{k}_{"control" if name=="initial" else name}'/'memory_steps.jsonl').splitlines()]
        rows=[json.loads(x) for x in raw(d/'steps.jsonl').splitlines()]
        eq(len(rows),260,'complete_rows_per_state');eq(rec['batches'],260,'state_batches')
        queue={};seen=set();own=[];offset=0;stats_totals=Counter();history_rows=0;state_nonzero=0;class_zero_exposures=0
        qc=dict(age_expired_record_exposures=0,capacity_evicted_record_exposures=0,current_record_duplicate_exposures=0,excluded_current_history_exposures=0,available_history_group_exposures=0,selected_vjp_group_exposures=0,zero_upstream_group_skip_exposures=0,batches_with_age_expiry=0,batches_with_capacity_eviction=0,batches_with_zero_upstream_group_skip=0,maximum_selected_history_records=0,maximum_selected_history_age=0,maximum_queue_after_update=0)
        proof=rec['candidate_gradient_chain_rule'];eq(proof['step'],67,'direct_proof_first_history_step');eq(proof['history_groups'],1,'direct_proof_groups')
        a=proof['agreement'];relative=a['difference_norm']/max(a['first_norm'],a['second_norm'])
        close(relative,proof['relative_l2_error'],'direct_proof_relative_rederive');assert relative<=.005
        proofs.append(dict(fold=k,state=name,relative_l2_error=relative))
        for step,row in enumerate(rows):
            eq(row['step'],step+1,'step_sequence');eq(row['epoch'],step//13+1,'epoch_sequence');eq(row['optimizer_updates'],0,'row_zero_update')
            eq(row['historical_coordinate_mode'],'fixed_current_parameters','row_coordinate_scope')
            eq(row['age_semantics'],'batch_recency_not_parameter_updates','row_age_scope')
            indices=row['record_indices'];current=set(indices);seen.update(indices)
            eq(len(indices),64,'batch_size');assert current<=set(f['source_record_indices'])
            eq(indices,metadata['folds'][k]['batches'][step]['record_indices'],'registered_batch_order')
            eq(indices,old_rows[step]['record_indices'],'inherited_batch_order')
            eq(row['pixel_sha256'],old_rows[step]['pixel_sha256'],'inherited_pixel_sha')
            eq(set(row['pixel_sha256']),{'RGB','NI','TI'},'three_modal_sha_keys')
            assert all(re.fullmatch('[0-9a-f]{64}',x) for x in row['pixel_sha256'].values())
            ids=[protocol['records'][i]['identity'] for i in indices];scenes=[protocol['records'][i]['scene'] for i in indices]
            eq(row['identities'],ids,'row_identity_mapping');eq(row['scenes'],scenes,'row_scene_mapping')
            eq(sorted(Counter(ids).values()),[8]*8,'batch_p8_k8');class_zero_exposures+=ids.count(f['source_ids'][0])
            expired=[i for i,stored in queue.items() if step-stored>8]
            qc['age_expired_record_exposures']+=len(expired);qc['batches_with_age_expiry']+=bool(expired)
            for i in expired:del queue[i]
            qc['current_record_duplicate_exposures']+=64-len(current);qc['excluded_current_history_exposures']+=len(current&set(queue))
            selected=[i for i in queue if i not in current]
            expected=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],age=step-queue[i],stored_step=queue[i]) for i in selected]
            eq(row['memory'],expected,'full_queue_metadata_replay')
            members=row['memory'];m=len(members);history_rows+=bool(m);memory_exposures+=m
            stats=row['statistics'];eq(len(stats),13,'thirteen_statistics_present');assert all(math.isfinite(v) for v in stats.values()) and math.isfinite(row['loss'])
            pos=sum(i==r['identity'] for i in ids for r in members)
            cross=sum(i==r['identity'] and sc!=r['scene'] for i,sc in zip(ids,scenes) for r in members)
            for key,value in dict(memory_records=m,memory_positive_pairs=pos,memory_negative_pairs=64*m-pos,memory_cross_scene_positive_pairs=cross,maximum_memory_age=max([r['age'] for r in members],default=0)).items():eq(stats[key],value,'metadata_statistics_independent')
            assert 0<=stats['harder_positive_anchors']<=64 and 0<=stats['harder_negative_anchors']<=64
            assert 0<=stats['current_wrong_order_anchors']<=stats['expanded_wrong_order_anchors']<=stats['expanded_hinge_positive_anchors']<=64
            assert 0<=stats['memory_negative_violations_against_batch_hard_positive']<=stats['memory_negative_pairs']
            assert 0<=stats['current_triplet']<=stats['expanded_triplet']+2e-6
            if not m:eq(stats['current_triplet'],stats['expanded_triplet'],'empty_history_loss_identity')
            stats_totals.update(stats)
            values=4096+64*m;eq(row['distance_float_count'],values,'matrix_count');eq(row['distance_offset_bytes'],offset*4,'matrix_offset');offset+=values
            available={r['stored_step'] for r in members};groups=row['candidate_vjp_groups'];assert len(set(groups))==len(groups) and set(groups)<=available
            eq(groups,sorted(groups),'vjp_group_order');skipped=len(available)-len(groups)
            qc['available_history_group_exposures']+=len(available);qc['selected_vjp_group_exposures']+=len(groups);qc['zero_upstream_group_skip_exposures']+=skipped;qc['batches_with_zero_upstream_group_skip']+=skipped>0
            qc['maximum_selected_history_records']=max(qc['maximum_selected_history_records'],m);qc['maximum_selected_history_age']=max(qc['maximum_selected_history_age'],max([r['age'] for r in members],default=0))
            eq(row['extra_role_record_forwards'],64*len(groups)+(64 if step+1==proof['step'] else 0),'extra_role_forward_accounting')
            nz=row['history_nonzero_gradient_records'];assert 0<=nz<=m;state_nonzero+=nz;nonzero_exposures+=nz
            eq(set(row['roles']),{'cnn','transformer','mamba'} if m else set(),'role_coverage')
            for role,v in row['roles'].items():
                eq(set(v),{'current_vs_history','current_vs_both','total_vs_both','current_repeat_noise','history_repeat_noise'},'role_comparison_set')
                for cmp in v.values():
                    x,y,delta,cos=cmp['first_norm'],cmp['second_norm'],cmp['difference_norm'],cmp['cosine']
                    assert all(math.isfinite(x) and x>=0 for x in [x,y,delta]);eq(cos is None,x==0 or y==0,'undefined_cosine_rule')
                    if cos is not None:
                        assert math.isfinite(cos) and abs(cos)<=1+1e-12
                        closure=abs(delta*delta-(x*x+y*y-2*x*y*cos))/max(x*x+y*y,1e-30)
                        assert closure<1e-10,('compare_norm_closure',k,name,step,role,closure)
                        maxima['all_comparison_norm_closure']=max(maxima['all_comparison_norm_closure'],closure);counts['all_comparison_norm_closure']+=1
                uv=v['current_vs_history'];both=v['current_vs_both'];total=v['total_vs_both'];u,h=uv['first_norm'],uv['second_norm']
                eq(both['first_norm'],u,'current_norm_alignment');eq(v['current_repeat_noise']['first_norm'],u,'current_noise_alignment');eq(v['history_repeat_noise']['first_norm'],h,'history_noise_alignment')
                close(both['difference_norm'],h,'triplet_delta_is_history',rtol=1e-5,atol=1e-8)
                close(total['difference_norm'],base['LOSS']['TRIPLET_FUSED']*h,'task_delta_is_weighted_history',rtol=1e-5,atol=1e-8)
                dot=u*h*uv['cosine'] if uv['cosine'] is not None else 0
                closure=abs(both['second_norm']**2-(u*u+h*h+2*dot))/max(u*u+h*h,1e-30)
                assert closure<1e-5;maxima['gradient_sum_norm_closure']=max(maxima['gradient_sum_norm_closure'],closure);counts['gradient_sum_norm_closure']+=1
                if u>0 and both['second_norm']>0:close(both['cosine'],(u*u+dot)/(u*both['second_norm']),'gradient_sum_cosine_rederive',rtol=1e-7,atol=1e-8)
                maxima['current_repeat_noise']=max(maxima['current_repeat_noise'],v['current_repeat_noise']['difference_norm']);maxima['history_repeat_noise']=max(maxima['history_repeat_noise'],v['history_repeat_noise']['difference_norm'])
                point=dict(fold=k,state=name,step=step+1,epoch=row['epoch'],role=role,current_norm=u,history_norm=h,history_to_current_ratio=h/u if u>0 else None,current_history_cosine=uv['cosine'],current_both_cosine=both['cosine'],task_total_both_cosine=total['cosine'],history_noise=v['history_repeat_noise']['difference_norm'],history_above_repeat_noise=h>v['history_repeat_noise']['difference_norm'],history_nonzero_record_exposures=nz)
                own.append(point);all_rows.append(point)
                norm_points.append(dict(fold=k,state=name,role=role,current_norm=u,history_norm=h,both_triplet_norm=both['second_norm'],task_current_norm=total['first_norm'],task_both_norm=total['second_norm'],triplet_both_to_current=both['second_norm']/u if u>0 else None,task_both_to_current=total['second_norm']/total['first_norm'] if total['first_norm']>0 else None,history_noise_to_norm=v['history_repeat_noise']['difference_norm']/h if h>0 else None,current_noise_to_norm=v['current_repeat_noise']['difference_norm']/u if u>0 else None))
            evicted=0
            if step>=65:
                for i in indices:queue.pop(i,None);queue[i]=step
                while len(queue)>512:del queue[next(iter(queue))];evicted+=1
            qc['capacity_evicted_record_exposures']+=evicted;qc['batches_with_capacity_eviction']+=bool(evicted);qc['maximum_queue_after_update']=max(qc['maximum_queue_after_update'],len(queue))
        eq(offset*4,rec['files']['distances.f32']['bytes'],'distance_receipt_bytes');counts['distance_elements_from_metadata']+=offset
        eq(sorted(seen),rec['observed_source_records'],'receipt_seen_records');eq(sorted(seen),f['source_record_indices'],'complete_source_coverage')
        eq(history_rows,194,'history_rows_per_state');eq(rec['history_batches'],history_rows,'history_receipt_count')
        eq(rec['extra_role_record_forwards'],sum(r['extra_role_record_forwards'] for r in rows)+64,'state_extra_forward_total')
        counts['batches']+=len(rows);counts['history_batches']+=history_rows
        agg_state=next(x for x in analysis['states'] if (x['fold'],x['state'])==(k,name))
        for key,value in qc.items():eq(value,agg_state['queue_and_vjp_coverage'][key],'queue_coverage_aggregate')
        cp=next(x for x in cpu['states'] if (x['fold'],x['state'])==(k,name));eq(cp['history_nonzero_record_exposures'],state_nonzero,'cpu_nonzero_exposures')
        expected_post=next(x for x in post['states'] if (x['fold'],x['state'])==(k,name))['sum_of_per_batch_statistics']
        for key,value in stats_totals.items():
            if key in ['current_triplet','expanded_triplet']:close(value,expected_post[key],'post_statistics_text_sum',rtol=0,atol=260*2e-6)
            else:eq(value,expected_post[key],'post_statistics_count_sum')
        role_aggregates={}
        for role in ['cnn','transformer','mamba']:
            points=[x for x in own if x['role']==role];cr=cp['roles'][role]
            expected=dict(rows=len(points),history_nonzero=sum(x['history_norm']>0 for x in points),above_repeated_noise=sum(x['history_above_repeat_noise'] for x in points),current_history_cosines=[x['current_history_cosine'] for x in points],total_change_cosines=[x['task_total_both_cosine'] for x in points],history_parameter_norm=[x['history_norm'] for x in points],current_parameter_norm=[x['current_norm'] for x in points])
            recursive_match(expected,cr,'cpu_role_aggregate')
            a={field:aggregate([x[field] for x in points]) for field in fields};a['history_above_repeat_noise']=sum(x['history_above_repeat_noise'] for x in points)
            recursive_match(a,agg_state['roles'][role],'analysis_role_aggregate');role_aggregates[role]=a
        paired[(k,name)]=[(r['record_indices'],r['pixel_sha256'],r['memory']) for r in rows]
        states.append(dict(fold=k,state=name,batches=len(rows),history_batches=history_rows,source_records=len(seen),class_zero_exposures=class_zero_exposures,roles=role_aggregates,role_parameter_tensor_counts=role_counts,queue_and_vjp_coverage=qc,statistics_sums=dict(stats_totals),history_nonzero_record_exposures=state_nonzero,extra_role_record_forwards=rec['extra_role_record_forwards'],chain_rule_relative_error=relative,peak_allocated_mib=rec['peak_allocated_mib'],elapsed_seconds=rec['elapsed_seconds']))
for k in range(3):eq(paired[(k,'initial')],paired[(k,'control')],'fixed_inputs_across_states');eq(paired[(k,'control')],paired[(k,'fresh_memory')],'fixed_inputs_across_states')
csv_rows=list(csv.DictReader(raw(A/'all_role_history_steps.csv').decode('utf-8').splitlines()))
eq(len(csv_rows),len(all_rows),'csv_row_count')
for expected,actual in zip(all_rows,csv_rows):
    eq(set(expected),set(actual),'csv_columns')
    for key,value in expected.items():
        if isinstance(value,float):close(value,float(actual[key]),'csv_float')
        else:eq('' if value is None else str(value),actual[key],'csv_value')
points=[]
for state in states:
    for role in ['cnn','transformer','mamba']:
        for field in fields:
            a=state['roles'][role][field];points.append(dict(fold=state['fold'],state=state['state'],role=role,metric=field,mean=a['mean'],defined=a['defined'],undefined=a['undefined']))
recursive_match(points,plot['points'],'plot_points')
eq(len(points),108,'plotted_cells');eq(sum(p['defined'] for p in points),plot['contributing_metric_values'],'plot_contributing_values');eq(sum(p['undefined'] for p in points),plot['undefined_metric_values'],'plot_undefined_values')
eq(counts['batches'],2340,'complete_batch_count');eq(len(all_rows),5238,'role_history_count');eq(counts['distance_elements_from_metadata'],cpu['distance_elements'],'remote_distance_count')
eq(counts['distance_elements_from_metadata'],post['distance_elements'],'post_distance_count');eq(counts['batches']*13,post['statistic_checks'],'post_claimed_stat_checks')
close(maxima['gradient_sum_norm_closure'],analysis['max_gradient_norm_identity_relative_error'],'published_max_norm_closure')
log_lines=raw(S/'source.log').splitlines();events=[json.loads(x) for x in log_lines if x.startswith(b'{')]
eq(sum(x.get('event')=='candidate_gradient_state' for x in events),9,'runtime_state_log_count');eq(sum(x.get('event')=='candidate_gradient_epoch' for x in events),180,'runtime_epoch_log_count')
source_state_events=[(x['fold'],x['state']) for x in events if x['event']=='candidate_gradient_state']
eq(source_state_events,[(x['fold'],x['state']) for x in states],'runtime_state_order')
for j in range(9):
    epochs=events[j*21+1:j*21+21]
    eq([x['epoch'] for x in epochs],list(range(1,21)),'all_runtime_epochs')
    eq([x['batches'] for x in epochs],[13*i for i in range(1,21)],'all_runtime_epoch_batches')
changes=subprocess.check_output(['git','diff','--name-only',pipe['code_commit'],summary['project_commit']],cwd=R,text=True).splitlines()
eq(changes,['AGENTS.md','MANIFEST.md','docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md','evidence/msvr310_history_gradient_r1_launch_20260908/launch.json','evidence/msvr310_history_gradient_r1_launch_20260908/observation.json','refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md'],'launch_source_commit_delta')
for d in [P,A,F]:
    for p in d.iterdir():
        if p.is_file():raw(p)
for path in ['docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md','tools/build_msvr310_train_oof_protocol.py','data/datasets/msvr310.py','evidence/msvr310_style_readiness_20260907/msvr310_style_prelaunch_source_bytes_20260907.json']:
    raw(R/path)
raw(Path(__file__))
totals={key:sum(x['queue_and_vjp_coverage'][key] for x in states) for key in states[0]['queue_and_vjp_coverage'] if not key.startswith('maximum')}
totals.update(batches=counts['batches'],current_record_exposures=counts['batches']*64,history_batches=counts['history_batches'],history_anchor_exposures=counts['history_batches']*64,role_history_rows=len(all_rows),history_record_exposures=memory_exposures,history_nonzero_record_exposures=nonzero_exposures,extra_role_record_forwards=sum(x['extra_role_record_forwards'] for x in states),history_above_repeat_noise=sum(x['history_above_repeat_noise'] for x in all_rows),negative_current_history_cosines=sum(x['current_history_cosine'] is not None and x['current_history_cosine']<0 for x in all_rows),undefined_plot_metrics=sum(x['undefined'] for x in points),distance_elements=counts['distance_elements_from_metadata'],remote_reported_statistic_checks=post['statistic_checks'])
norm_distributions=[]
for state in states:
    for role in ['cnn','transformer','mamba']:
        own=[p for p in norm_points if (p['fold'],p['state'],p['role'])==(state['fold'],state['state'],role)]
        norm_distributions.append(dict(fold=state['fold'],state=state['state'],role=role,statistics={field:aggregate([p[field] for p in own]) for field in own[0] if field not in ['fold','state','role']}))
result=dict(status='PASS_EXHAUSTIVE_INDEPENDENT_TEXT_CHECKS',generated_at=datetime.now(timezone.utc).isoformat(),python=sys.version,scope='All complete source text rows and all states; independent stdlib arithmetic, labels, queue, source bindings and figure values. No model/optimizer/inference and no distance binary or gradient vector re-execution.',counts=dict(counts),maxima=dict(maxima),totals=totals,fold_scope=fold_scope,states=states,direct_proofs=proofs,norm_distributions=norm_distributions,project_bindings=bindings,absent_distance_binaries=absent,launch_commit=pipe['code_commit'],source_start_commit=summary['project_commit'],commit_changed_paths=changes,png_dimensions=list(png_dimensions),input_hashes=hashes)
(O/'independent_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(O/'audited_input_hashes.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],totals=totals,maxima=dict(maxima),input_files=len(hashes),bindings=len(bindings),exact_bindings=sum(x['exact'] for x in bindings),missing_distance_bytes=sum(x['bytes'] for x in absent),png_dimensions=png_dimensions),indent=2))
