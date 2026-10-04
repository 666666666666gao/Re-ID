
from datetime import datetime
from pathlib import Path
import hashlib,json,math,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(root))
from tools import queue_role_input_detach as panel
panel.configure()
campaign=root/'logs/role_input_detach_v1_20261004_813'
state=json.loads((campaign/'campaign.json').read_text())
selected=[r for r in state['jobs'] if (r['phase'],r['dataset'],r['variant'])==('full','MSVR310','native')]
assert len(selected)==1 and selected[0]['status']=='COMPLETE' and selected[0]['exit_code']==0
manifest=panel.base.require_sources(campaign)
controls=panel.require_controls()
row=panel.base.verify(campaign,'MSVR310','native')
assert row==selected[0]['result']
semantic_job=[r for r in state['jobs'] if (r['phase'],r['dataset'],r['variant'])==('full','MSVR310','semantic')]
assert len(semantic_job)==1 and semantic_job[0]['status']=='COMPLETE' and semantic_job[0]['exit_code']==0
new_semantic=panel.base.verify(campaign,'MSVR310','semantic')
assert new_semantic==semantic_job[0]['result']
run=Path(row['run_dir'])
training=json.loads((run/'training.json').read_text())
steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
order=(run/'training_batch_order.jsonl').read_bytes()
batches=[json.loads(line) for line in order.splitlines()]
assert len(steps)==len(batches)==sum(item['steps'] for item in training['history'])
assert [(r['epoch'],r['batch']) for r in steps]==[(r['epoch'],r['batch']) for r in batches]
assert all(math.isfinite(r['loss']) for r in steps)
original=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==('MSVR310','native'))
global_only=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==('MSVR310','global_only'))
assert order==(Path(original['run_dir'])/'training_batch_order.jsonl').read_bytes()
assert order==(Path(new_semantic['run_dir'])/'training_batch_order.jsonl').read_bytes()
assert {k:v for k,v in row['initializer'].items() if k not in ('architecture','entry_sha256','role_input_gradient_policy')}=={k:v for k,v in original['initializer'].items() if k not in ('architecture','entry_sha256')}
assert [p.name for p in run.glob('*.pth')]==['best_map.pth']
metrics=json.loads((run/'official_metrics.json').read_text())
artifacts={}
for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('best_epoch_distances.pt','training_best_distance_sha256')):
 path=run/name
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 assert actual==metrics[key]
 artifacts[str(path)]={'bytes':path.stat().st_size,'sha256':actual}
m0=Path(row['m0_dir'])
m0_receipt=json.loads((m0/'training.json').read_text())
path=m0/'m0_reload_probe.pth'
actual=hashlib.sha256(path.read_bytes()).hexdigest()
assert actual==m0_receipt['m0']['reload_probe_sha256']
artifacts[str(path)]={'bytes':path.stat().st_size,'sha256':actual}
files={}
paths=[run/'training.json',run/'official_metrics.json',m0/'training.json',campaign/'initialization/MSVR310_native.json',campaign/'MSVR310_native_train.log',campaign/'MSVR310_native_evaluate.log']
for path in paths:
 files[path.relative_to(root).as_posix()]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
norms={}
for key in ('shared_global_norm_mean','correction_norm_mean','actual_scaled_correction_norm_mean','actual_scaled_correction_global_ratio_mean'):
 values=[r[key] for r in steps]
 assert all(math.isfinite(v) for v in values)
 norms[key]={'step_mean':sum(values)/len(values),'minimum':min(values),'maximum':max(values),'first':values[0],'last':values[-1]}
print(json.dumps({'status':'NEW_MSVR_PAIR_FULL50_AND_STRICT_EVALUATIONS_VERIFIED','at':datetime.now().astimezone().isoformat(),
 'campaign_status':state['status'],'formal_completed':sum(r['phase']=='full' and r['status']=='COMPLETE' for r in state['jobs']),
 'active_command':state.get('active_command'),'completed_job':selected[0],'original_control':original,'original_global_only':global_only,
 'new_semantic':new_semantic,'delta_new_semantic':{k:row['metrics'][k]-new_semantic['metrics'][k] for k in row['metrics']},
 'source_count':len(manifest['source_sha256']),'source_sha_unchanged':True,'original61control_artifacts_unchanged':True,
 'actual_batch_orders_equal':True,'actual_batch_order':{'sha256':hashlib.sha256(order).hexdigest(),'bytes':len(order),'lines':len(batches)},
 'formal_steps':len(steps),'all_losses_finite':True,'one_best_weight':True,'own_m0_probe_sha_verified':True,
 'artifacts':artifacts,'files':files,'training_norm_observations':norms,
 'delta_original_native':{k:row['metrics'][k]-original['metrics'][k] for k in row['metrics']},
 'delta_original_global_only':{k:row['metrics'][k]-global_only['metrics'][k] for k in row['metrics']},
 'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'First new MSVR310 pair completed full50 and own strict reload; corresponding original controls historical, not new seed repeats. New native minus new semantic is a paired detail comparison, separate from the registered gradient intervention gate. Scalar training norms are descriptive batch-step observations, not query retrieval utility. No new model execution, report replay, old retired M0 verifier, tuning, power/temp or engineering repair. Scientific Goal active/unmet.'}))
