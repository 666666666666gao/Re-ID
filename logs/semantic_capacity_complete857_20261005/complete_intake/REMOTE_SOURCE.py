from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_control_v1_20261005_856'
launch=root/'logs/semantic_capacity_launch_20261005_856'
completion=root/'logs/semantic_capacity_first_eval_completion_20261005_857'
completion_launch=root/'logs/semantic_capacity_first_eval_launch_20261005_857'
report_root=root/'results/semantic_capacity_control_v1_complete_20261005_856'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
original_exit=json.loads((launch/'EXIT.json').read_text())
exit_record=json.loads((completion_launch/'EXIT.json').read_text())
state=json.loads((completion/'campaign.json').read_text())
plan=json.loads((completion/'PLAN.json').read_text())
original_state=json.loads((campaign/'campaign.json').read_text())
assert original_exit['exit_code']==1 and original_state['report_invocations']==0
assert state['new_training_invocations']==0 and state['inherited_training_endpoints']==3
assert all(sha(Path(n))==d for n,d in plan['original_failure_sha256'].items())
manifest=json.loads((campaign/'manifest.json').read_text())
matrix=json.loads((completion/'accepted_matrix.json').read_text())
report=json.loads((report_root/'SUMMARY.json').read_text())
assert exit_record['exit_code']==0
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==6 and all(r['status']=='COMPLETE' and r['exit_code']==0 for r in state['jobs'])
assert len(manifest['source_sha256'])==345 and all(sha(root/n)==d for n,d in manifest['source_sha256'].items())
assert matrix['schema']==report['schema']=='trifusion-semantic-capacity-control-v1'
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==report['accepted']==3
assert report['status']=='COMPLETE' and report['formal_epochs']==150 and report['formal_steps']==6484
assert report['reader_parameters']==159096 and report['native_reader_parameters']==159296 and report['parameter_gap']==-200
seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
controls=json.loads(seal.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==187
assert all(sha(Path(n))==d for n,d in controls['artifact_sha256'].items())
datasets=('RGBNT201','MSVR310','RGBNT100')
accepted={r['dataset']:r for r in matrix['rows']}
assert set(accepted)==set(datasets)
assert {r['dataset'] for r in report['rows']}==set(datasets)
assert len(report['pairs'])==9
assert {(p['dataset'],p['control']) for p in report['pairs']}=={(d,c) for d in datasets for c in ('raw_native','raw_semantic','raw_global_only')}
paths={campaign/'manifest.json',campaign/'campaign.json',completion/'accepted_matrix.json',
 completion/'manifest.json',completion/'campaign.json',completion/'report.log',completion/'PLAN.json',
 completion/'ADMIN_COMPLETE857.py',completion_launch/'LAUNCH.json',completion_launch/'EXIT.json',
 completion_launch/'console.log',completion_launch/'SUPERVISOR.py',completion_launch/'supervisor.log',
 launch/'LAUNCH.json',launch/'EXIT.json',launch/'console.log',
 report_root/'SUMMARY.json',report_root/'REPORT.md',seal}
paths.update(p for p in campaign.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
paths.update(p for p in completion.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
paths.update(Path(r['run_dir'])/'training.json' for r in controls['rows'])
retirement=root/'logs/obsolete_scale_weight_retirement_20261005_856'
paths.update((retirement/'PREPARE.json',retirement/'RETIREMENT.json'))
metric_retirement=root/'logs/capacity_closed_metric_best_retirement_20261005_857'
paths.update((metric_retirement/'PREPARE.json',metric_retirement/'RETIREMENT.json'))
artifacts,orders,rows={}, {}, []
for row in report['rows']:
 dataset=row['dataset']; original=accepted[dataset]
 assert all(row[k]==v for k,v in original.items())
 receipt_path=campaign/'acceptance'/f'{dataset}_native.json'
 acceptance=json.loads(receipt_path.read_text())
 assert acceptance['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT' and acceptance['row']==original
 assert all(sha(Path(n))==d for n,d in acceptance['artifact_sha256'].items())
 assert not Path(acceptance['probe']['path']).exists()
 run=Path(row['run_dir']);m0=Path(row['m0_dir'])
 training=json.loads((run/'training.json').read_text())
 receipt=json.loads((run/'official_metrics.json').read_text())
 smoke=json.loads((m0/'training.json').read_text())
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history'])==50
 assert [e['epoch'] for e in training['history']]==list(range(1,51))
 assert training['history'][row['best_epoch']-1]['official_fused']==row['metrics']
 assert receipt['status']=='COMPLETE' and receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
 assert receipt['metrics']==row['metrics'] and receipt['selected_epoch']==row['best_epoch']
 assert smoke['status']=='M0_PASS' and smoke['m0']['reload_max_abs_difference']==0
 assert smoke['m0']['nonzero_gradient_parameters']==smoke['m0']['trainable_parameters']
 assert smoke['production_m0_diagnostics']['effective_optimizer_updates']==8
 assert smoke['production_m0_diagnostics']['active_reader_parameters']==159096
 assert [p.name for p in run.glob('*.pth')]==['best_map.pth']
 bound={}
 for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('best_epoch_distances.pt','training_best_distance_sha256')):
  p=run/name;actual=sha(p);assert actual==receipt[key]
  bound[name]=dict(sha256=actual,bytes=p.stat().st_size)
 assert sha(run/'official_metrics.json')==row['receipt_sha256']
 artifacts[dataset]=bound
 data=(run/'training_batch_order.jsonl').read_bytes()
 steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
 batches=[json.loads(line) for line in data.splitlines()]
 assert len(steps)==len(batches)==row['formal_steps']==sum(e['steps'] for e in training['history'])
 assert [(s['epoch'],s['batch']) for s in steps]==[(b['epoch'],b['batch']) for b in batches]
 old_rows=[r for r in controls['rows'] if r['dataset']==dataset]
 assert len(old_rows)==3 and all(data==(Path(r['run_dir'])/'training_batch_order.jsonl').read_bytes() for r in old_rows)
 orders[dataset]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),lines=len(batches),controls_equal=3)
 paths.update(p for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log','.csv'))
 paths.update(p for p in m0.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log','.csv'))
 rows.append(dict(dataset=dataset,source='semantic_capacity',best_epoch=row['best_epoch'],metrics=row['metrics'],
  formal_steps=row['formal_steps'],last_epoch_metrics=training['history'][-1]['official_fused'],
  best_to_last_map_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
  complete_training_and_epoch_eval_seconds=row['complete_training_and_epoch_eval_seconds']))
pairs=[]
for pair in report['pairs']:
 diag=pair['paired_diagnosis'];delta=diag['delta_metrics']
 assert pair['actual_training_batch_order_equal'] and diag['status']=='CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE'
 assert len(diag['query_changes'])==sum(r['queries'] for r in diag['identity_changes'])
 assert [r['query_index'] for r in diag['query_changes']]==list(range(len(diag['query_changes'])))
 assert pair['phase_progress']==(delta['mAP']>=.5 and delta['Rank-1']>=0)
 pairs.append(dict(dataset=pair['dataset'],control=pair['control'],delta=delta,phase_progress=pair['phase_progress'],
  rank1_repairs=diag['rank1_repairs'],rank1_new_errors=diag['rank1_new_errors'],
  identity_macro_mean_delta_ap_points=diag['identity_macro_mean_delta_ap_points'],query_count=len(diag['query_changes'])))
files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
print(json.dumps(dict(status='CAPACITY_COMPLETE_WITH_FIRST_EVAL_CONTINUATION_VERIFIED',at=datetime.now().astimezone().isoformat(),
 source_files_verified_unchanged=345,control_artifacts_verified=187,report_invocations=1,formal_completed=3,
 formal_epochs=150,formal_steps=6484,batch_orders=orders,rows=rows,pairs=pairs,artifacts=artifacts,files=files,
 m0_probes_retired_after_original_strict=True,completion_launch_exit=exit_record,original_launch_exit=original_exit,
 original_failure_sha256=plan['original_failure_sha256'],new_training_invocations=0,
 disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Text-only intake and immutable SHA checks, no torch import/model/report replay. Three original trainings, first strict100 and once-only report through explicit continuation; original disk-stop parent EXIT1/state/counters preserved. Seed42 near-capacity control, no causal uniqueness, equivalence, multiple-seed robustness or SOTA claim.')))
