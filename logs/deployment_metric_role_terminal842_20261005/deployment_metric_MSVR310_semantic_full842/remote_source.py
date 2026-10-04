
from datetime import datetime
from pathlib import Path
import hashlib,json,math,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
dataset,variant='MSVR310','semantic'
campaign=root/'logs/deployment_metric_role_v1_20261005_837'
scope=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text())
assert len(scope['source_sha256'])==339
def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
assert all(sha(root/name)==digest for name,digest in scope['source_sha256'].items())
state=json.loads((campaign/'campaign.json').read_text())
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==(dataset,variant,'full'))
assert job['status']=='COMPLETE' and job['exit_code']==0
accept_path=campaign/'acceptance'/f'{dataset}_{variant}.json'
accepted=json.loads(accept_path.read_text())
assert accepted['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
assert all(sha(Path(name))==digest for name,digest in accepted['artifact_sha256'].items())
retirements=[json.loads(line) for line in (campaign/'probe_retirement.jsonl').read_text().splitlines()]
retired=[r for r in retirements if (r['dataset'],r['variant'])==(dataset,variant)]
assert len(retired)==1 and retired[0]['acceptance_sha256']==sha(accept_path)
assert all(retired[0][k]==v for k,v in accepted['probe'].items())
assert not Path(accepted['probe']['path']).exists()
row=accepted['row'];assert row==job['result']
run,m0_dir=Path(row['run_dir']),Path(row['m0_dir'])
assert run.resolve()==root/f'trained-model/{campaign.name}_full_{variant}_{dataset}'
assert m0_dir.resolve()==root/f'trained-model/{campaign.name}_m0_{variant}_{dataset}'
assert {p.name for p in run.glob('*.pth')}=={'best_map.pth'}
assert not list(m0_dir.glob('*.pth'))
training=json.loads((run/'training.json').read_text());official=json.loads((run/'official_metrics.json').read_text())
m0=json.loads((m0_dir/'training.json').read_text())
binding=json.loads((campaign/'initialization'/f'{dataset}_{variant}.json').read_text())['binding']
assert row['initializer']==training['initializer']==m0['initializer']==binding
assert binding['role_metric_policy']=='role_triplet_joint_1536_l2_h_global_triplet_original_raw_parts'
assert binding['objective_gradient_policy']=='author_global_loss_to_shared_and_heads_fused_loss_to_roles_only'
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
assert training['schema']==official['schema']==m0['schema']=='trifusion-deployment-metric-role-v1'
assert training['condition']==official['condition']
assert [r['epoch'] for r in training['history']]==list(range(1,51))
best=max(training['history'],key=lambda r:(r['official_fused']['mAP'],r['epoch']))
assert row['best_epoch']==training['best_epoch']==official['selected_epoch']==best['epoch']
assert row['metrics']==official['metrics']
assert all(abs(official['metrics'][name]-best['official_fused'][name])<1e-5 for name in row['metrics'])
assert official['seed']==42 and official['training_epochs']==50
assert official['protocol_sha256']==binding['protocol_sha256']
assert official['checkpoint_sha256']==row['checkpoint_sha256']==sha(run/'best_map.pth')
assert official['distance_sha256']==row['distance_sha256']==sha(run/'official_distances.pt')
assert official['training_best_distance_sha256']==sha(run/'best_epoch_distances.pt')
assert official['independent_upstream_metrics_equal'] and not official['reranking']
assert m0['status']=='M0_PASS' and m0['production_m0_diagnostics']['effective_optimizer_updates']==8
assert m0['m0']['reload_probe_sha256']==accepted['probe']['sha256']
assert m0['m0']['trainable_parameters']==m0['m0']['nonzero_gradient_parameters']==binding['trainable_parameter_tensors']
assert all(count==8 for count in m0['production_m0_diagnostics']['author_bn_batches_tracked'].values())
if variant=='native':
 detail=m0['production_m0_diagnostics']
 assert len(detail['detail_parameters'])==14 and len(detail['detail_updates'])==8
 assert all(detail['detail_updates'][-1][name]['parameter_delta_from_initial_max_abs']>0 for name in detail['detail_parameters'])
seal_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)==scope['control_seal_sha256']
controls=json.loads(seal_path.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==187
assert all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
matched=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==(dataset,variant))
excluded=('architecture','entry_sha256','scope','role_metric_policy')
assert {k:v for k,v in binding.items() if k not in excluded}=={k:v for k,v in matched['initializer'].items() if k not in excluded}
assert (run/'training_batch_order.jsonl').read_bytes()==(Path(matched['run_dir'])/'training_batch_order.jsonl').read_bytes()
steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
batches=[json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
assert len(steps)==len(batches)==sum(r['steps'] for r in training['history'])
assert [(r['epoch'],r['batch']) for r in steps]==[(r['epoch'],r['batch']) for r in batches]
heads=1 if dataset=='RGBNT201' else 3
assert all(len(r['head_losses'])==len(r['global_head_losses'])==heads for r in steps)
assert all(math.isfinite(r['global_loss']) and math.isfinite(r['fused_role_loss']) for r in steps)
assert all(abs(r['loss']-r['global_loss']-r['fused_role_loss'])<=1e-5 for r in steps)
assert all(r['role_metric_width']==1536 and abs(r['role_metric_norm_mean']-1)<1e-6 for r in steps)
trajectory=[]
fields=('global_loss','fused_role_loss','shared_global_norm_mean','correction_norm_mean',
        'actual_scaled_correction_norm_mean','actual_scaled_correction_global_ratio_mean','role_metric_norm_mean')
for epoch in range(1,51):
 epoch_steps=[r for r in steps if r['epoch']==epoch]
 assert len(epoch_steps)==training['history'][epoch-1]['steps']
 trajectory.append(dict(epoch=epoch,steps=len(epoch_steps),**{n:sum(r[n] for r in epoch_steps)/len(epoch_steps) for n in fields}))
deltas={}
for name in (variant,'global_only'):
 control=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==(dataset,name))
 delta={key:row['metrics'][key]-control['metrics'][key] for key in row['metrics']}
 deltas[name]=dict(control_metrics=control['metrics'],delta_metrics=delta,phase_progress=delta['mAP']>=.5 and delta['Rank-1']>=0)
names=[run/'training.json',run/'official_metrics.json',m0_dir/'training.json',accept_path,
 campaign/'initialization'/f'{dataset}_{variant}.json',campaign/'manifest.json',
 campaign/f'prepare_{dataset}_{variant}.log',campaign/f'{dataset}_{variant}_m0.log',
 campaign/f'{dataset}_{variant}_train.log',campaign/f'{dataset}_{variant}_evaluate.log']
files={}
for path in names:
 data=path.read_bytes();files[path.relative_to(root).as_posix()]=dict(text=data.decode('utf-8'),sha256=hashlib.sha256(data).hexdigest())
snapshot=dict(campaign=state,probe_retirement_records=retirements)
summary=dict(status='ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED',at=datetime.now().astimezone().isoformat(),row=row,
 formal_epochs=50,formal_steps=len(steps),source_files_verified_unchanged=339,control_artifacts_verified_unchanged=187,
 actual_training_batch_order_equal=True,metric_width=1536,head_loss_sum_count=heads,global_and_role_losses_verified=True,
 deltas=deltas,training_task_and_norm_trajectory=trajectory,
 best_to_last_map_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
 training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
 full_acceptance_sha256=sha(accept_path),retired_probe=retired[0],disk_free_bytes=shutil.disk_usage(root).free,
 boundary='One closed endpoint only. Original full50/first strict verified from immutable acceptance inputs,never old retired M0 verifier or model replay. Scalar paired deltas only; all-query15pairCPU report remains original once-only final stage. No reselection,new NN,power/temperature action or seed/SOTA claim.')
print(json.dumps(dict(summary=summary,files=files,snapshot=snapshot)))
