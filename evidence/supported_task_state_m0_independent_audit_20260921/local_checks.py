import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import ast

OUT=Path(__file__).resolve().parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_m0_complete_20260921')
ROLES=('cnn','transformer','mamba')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def pair(p):
    a,b,d=p['first_norm'],p['second_norm'],p['difference_norm'];c=p['cosine']
    assert all(math.isfinite(v) and v>=0 for v in (a,b,d))
    assert (c is None)==(a==0 or b==0)
    assert c is None or math.isfinite(c) and abs(c)<=1.00001
    expected=a*a+b*b if c is None else a*a+b*b-2*a*b*c
    assert abs(d*d-expected)<1e-7*max(1,a*a+b*b)
summary=js(INTAKE/'m0/summary.json');cpu=js(INTAKE/'m0_cpu.json');pipeline=js(INTAKE/'pipeline_at_intake.json')
assert all(next(s for s in pipeline['stages'] if s['stage']==stage)['exit_code']==0 for stage in ('t0','m0','m0_cpu'))
assert 'exit_code' not in next(s for s in pipeline['stages'] if s['stage']=='q1')
protocol=js(REPO/'protocols/msvr310_train_oof_v1.json')
labels=REPO/'evidence/vehicle_query_protocol_labels_20260905.json'
assert sha(labels)==protocol['label_evidence_sha256']
dataset=next(x for x in js(labels)['datasets'] if x['dataset']=='MSVR310')
train=dataset['record_manifest']['bounding_box_train']
assert len(train)==len(protocol['records'])==1032
for raw,record in zip(train,protocol['records'],strict=True):
    assert all(raw[k]==record[k] for k in ('identity','camera','scene'))
    assert record['paths'][0]=='bounding_box_train/'+raw['path']
labels_by_id=collections.defaultdict(set)
for row in train:labels_by_id[row['identity']].add(row['scene'])
heldout=[set() for _ in range(3)]
for group in ([i for i in sorted(labels_by_id) if len(labels_by_id[i])>1],[i for i in sorted(labels_by_id) if len(labels_by_id[i])==1]):
    for index,identity in enumerate(group):heldout[index%3].add(identity)
for i,fold in enumerate(protocol['folds']):
    assert sorted(heldout[i])==fold['heldout_ids']
    gallery=[protocol['records'][j] for j in fold['gallery_record_indices']]
    queries=[]
    for position,row in enumerate(gallery):
        positives=sum(r['identity']==row['identity'] and r['scene']!=row['scene'] for r in gallery)
        if positives:queries.append((row['index'],position,positives))
    assert queries==[(q['record_index'],q['gallery_position'],q['valid_positives']) for q in fold['query_rows']]
allrows={};endpoint_metrics=[];reference_rows=[];gate_checks=[]
input_hashes=js(OUT/'local_input_hashes.json')
additional=['protocols/msvr310_train_oof_v1.json','tools/build_msvr310_train_oof_protocol.py','evidence/vehicle_query_protocol_labels_20260905.json','tools/audit_vehicle_query_protocol_labels.py','evidence/msvr310_style_t0_runtime_binding_20260907/msvr_style_metadata_remote_numpy_20260907.json','evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json','tools/check_msvr_cross_scene_smooth_ap_math.py','tools/check_msvr_smooth_ap_math.py']
for rel in additional:
    p=REPO/rel;target=OUT/'snapshots'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);input_hashes[str(p)]=sha(p)
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
for name in [f'fold_{f}_{a}' for f in range(3) for a in ('control','split')]+['overfit_control','overfit_split']:
    path=INTAKE/'m0'/name;tr=js(path/'training.json');rows=[json.loads(s) for s in (path/'memory_steps.jsonl').read_text().splitlines()];allrows[name]=rows
    overfit=name.startswith('overfit');arm=name.split('_')[-1];f=0 if overfit else int(name.split('_')[1])
    item=summary['overfit'][arm] if overfit else summary['folds'][f]['endpoints'][arm]
    assert item['training']==tr
    if not overfit:assert js(path/'receipt.json')==item
    assert tr['current_rank_backward_calls']==tr['current_auxiliary_backward_calls']==len(rows)
    assert tr['direct_component_backward_calls']==(0 if overfit else 4)
    assert tr['initial_state_sha256']==item['initialization']['initial_state_sha256']
    assert item['initialization']['role_initialization_seed']==42 and not item['initialization']['role_weights_loaded']
    assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']==item['initialization']['signal_state_sha256']
    computed={'all_trainable_gradients_live':tr['missing_nonzero_gradients']==[], 'overflow_zero':tr['overflow_events']==0,'frozen_state_unchanged':tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256'],'signal_state_unchanged':tr['signal_state_before_sha256']==tr['signal_state_after_sha256'],'role_state_updated':tr['initial_state_sha256']!=tr['final_state_sha256'],'capacity_below_24gib':tr['peak_reserved_mib']<24*1024}
    if overfit:computed.update(fixed_100_steps=tr['optimizer_steps']==100,original_overfit_gate=item['gate']['loss_ratio']<=.1)
    else:computed['fixed_training_length']=tr['optimizer_steps']==8
    assert computed==(item['checks'] if overfit else item['engineering_checks']) and all(computed.values())
    gate_checks.append(dict(endpoint=name,checks=computed))
    assert len(tr['history'])==1 and tr['history'][0]['learning_rate']==.00035
    assert abs(sum(r['loss'] for r in tr['steps'])/len(rows)-tr['history'][0]['mean_loss'])<1e-12
    for r in rows:
        for role in ROLES:
            for p in r['roles'][role].values():pair(p)
            pair(r['assembled_gradients'][role]);pair(r['actual_parameter_updates'][role])
        direct=r['direct_single_group_check']
        if direct:
            pair(direct);assert direct['all_four_reencoded_outputs_bitwise_equal'];assert direct['relative_l2_error']<=.005
            assert abs(direct['relative_l2_error']-direct['difference_norm']/max(direct['first_norm'],1e-12))<1e-12
            for role,checks in r['rank_auxiliary_reference_checks'].items():
                for component,p in checks.items():
                    pair(p);assert p['passed'];assert p['relative_l2_error']<=.005 if p['first_norm'] else p['difference_norm']<=1e-8
                    reference_rows.append(dict(endpoint=name,step=r['step'],role=role,component=component,relative_l2_error=p['relative_l2_error'],difference_norm=p['difference_norm']))
    endpoint_metrics.append(dict(endpoint=name,steps=len(rows),unique_sampled_records=len({i for r in rows for i in r['record_indices']}),unique_sampled_ids=len({i for r in rows for i in r['identities']}),peak_reserved_mib=tr['peak_reserved_mib'],warmup_steps=sum(not r['replacement_active'] for r in rows),ap_steps=sum(r['replacement_active'] for r in rows),actual_unsupported_steps=sum(not r['rank_observed'] for r in rows),history_steps=sum(bool(r['memory']) for r in rows),historical_candidates=sum(len(r['memory']) for r in rows),history_vjp_forwards=sum(r['history_vjp_record_forwards'] for r in rows),direct_full_check_steps=[r['step'] for r in rows if r['direct_single_group_check']],full_direct_error=max([r['direct_single_group_check']['relative_l2_error'] for r in rows if r['direct_single_group_check']]+[0]),last_task_states=rows[-1]['task_states'],overfit_gate=item['gate'] if overfit else None))
paired=[]
for prefix in ['fold_0','fold_1','fold_2','overfit']:
    a=allrows[prefix+'_control'];b=allrows[prefix+'_split']
    assert [(r['record_indices'],r['pixel_sha256']) for r in a]==[(r['record_indices'],r['pixel_sha256']) for r in b]
    exact=a[0]['assembled_gradients']==b[0]['assembled_gradients']
    gradient_deltas={role:{key:b[0]['assembled_gradients'][role][key]-a[0]['assembled_gradients'][role][key] for key in ('first_norm','second_norm','difference_norm','cosine')} for role in ROLES}
    paired.append(dict(pair=prefix,first_step_update_norm_ratios_split_to_control={role:b[0]['actual_parameter_updates'][role]['difference_norm']/a[0]['actual_parameter_updates'][role]['difference_norm'] for role in ROLES},initial_current_gradient_witnesses_bitwise_equal=exact,first_step_gradient_summary_scalar_deltas=gradient_deltas,scope='Matched initialization and pixels verified; scalar summaries do not prove cross-arm vector equality. Bitwise gradient equality is not a registered gate.'))
assert sum(x['steps'] for x in endpoint_metrics)==248 and len(reference_rows)==90
local_raw_mismatches=[r for r in js(OUT/'config_binding_checks.json') if not r['passed']]
assert all(r['lf_matches'] for r in local_raw_mismatches)
result=dict(status='PASS_COMPLETE_LOCAL_TEXT_AND_PROVENANCE_CHECKS',generated_at=datetime.now(timezone.utc).isoformat(),pipeline_m0_and_cpu_original_exit_zero=True,intake_file_count=28,matched_pairs=paired,endpoint_metrics=endpoint_metrics,total_steps=248,total_role_update_records=744,direct_reference_count=90,direct_reference_max_relative_l2=max(r['relative_l2_error'] or 0 for r in reference_rows),total_history_vjp_forwards=sum(x['history_vjp_forwards'] for x in endpoint_metrics),actual_unsupported_steps=0,label_manifest_sha256=sha(labels),protocol_sha256=sha(REPO/'protocols/msvr310_train_oof_v1.json'),label_manifest_records_checked=1032,ground_truth_source='bound dataset filename-derived identity/camera/scene metadata; not model outputs',local_crlf_only_source_files=sorted({r['path'] for r in local_raw_mismatches}),model_forwards=0,optimizer_updates=0)
write('local_checks.json',result);write('original_m0_gate_checks.json',gate_checks);write('reference_checks.json',reference_rows);write('local_input_hashes.json',input_hashes)
print(json.dumps({k:v for k,v in result.items() if k!='endpoint_metrics'},indent=2))
print(json.dumps([ {k:v for k,v in r.items() if k not in ('last_task_states','overfit_gate')} for r in endpoint_metrics],indent=2))
