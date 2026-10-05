launch={'status': 'FINAL_PREVIOUSLY_UNSTARTED_REGION_RECONSTRUCTION_ENDPOINT_LAUNCHED', 'at': '2026-10-05T20:47:07.303930+08:00', 'pid': 2072932, 'start_ticks': 45976806, 'campaign': '/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_final_training_completion_20261005_858', 'launch': '/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_final_training_launch_20261005_858', 'original_campaign': '/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_v1_20261005_858', 'report_dir': '/data/gaob/Re-ID/Trifusion/results/region_reconstruction_v1_complete_20261005_858', 'coordinator_sha256': '91159f218b2b49a26d63373781ee05deeac4ce200ec0ae95f9d120c633c31142', 'plan_sha256': 'c9d3bf72934c2ee1b7343c05647fb1942872d54c237fe0468560369a27747c80', 'original_failure_sha256': {'/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_v1_20261005_858/campaign.json': '22de0447f290430350b50e2ddc174d53cfbb240862fddf10c0217f16d819b686', '/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_launch_20261005_858/EXIT.json': '9cc857422087866d16b4d6d9a4d320b70a4303df8795f8906756e29866573cab', '/data/gaob/Re-ID/Trifusion/logs/region_reconstruction_launch_20261005_858/console.log': '15c12304b64298a3b680e53b387732a0ec294144c9eead75a57328f8d6477498'}, 'new_training_invocations': 1, 'free_bytes': 3750723584}
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_region_reconstruction as panel
panel.configure()
campaign=Path(launch['campaign']);folder=Path(launch['launch']);original=Path(launch['original_campaign']);report=Path(launch['report_dir'])
state=json.loads((campaign/'campaign.json').read_text());exit_record=json.loads((folder/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert state['inherited_formal_endpoints']==5 and state['new_training_invocations']==1
assert exit_record['exit_code']==0 and state['original_parent_exit_code']==1
stat=Path('/proc/'+str(launch['pid'])+'/stat')
assert not stat.exists() or int(stat.read_text().split(') ')[1].split()[19])!=launch['start_ticks']
assert all(panel.base.sha(Path(n))==d for n,d in launch['original_failure_sha256'].items())
assert panel.base.sha(campaign/'ADMIN_COMPLETE858.py')==launch['coordinator_sha256']
assert panel.base.sha(campaign/'PLAN.json')==launch['plan_sha256']
manifest=panel.base.require_sources(campaign)
assert len(manifest['source_sha256'])==354 and manifest['query_modes']=={'semantic':'patch','native':'mean'}
controls=panel.previous.require_controls()
matrix=json.loads((campaign/'accepted_matrix.json').read_text());summary=json.loads((report/'SUMMARY.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==6
assert len(state['jobs'])==12 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert summary['accepted']==6 and summary['formal_epochs']==300 and len(summary['pairs'])==15
assert summary['active_reconstruction_parameters']==105232 and summary['active_reconstruction_tensors']==15
files=set();suffixes={'.json','.jsonl','.log','.md','.csv','.py','.txt'}
for directory in (campaign,folder,original,root/'logs/region_reconstruction_launch_20261005_858',report,
                  root/'logs/closed_capacity_best_retirement_20261005_858',
                  root/'logs/closed_normalized_control_retirement_20261005_858'):
 files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in suffixes)
steps=0
for row in matrix['rows']:
 dataset,variant=row['dataset'],row['variant']
 assert panel.previous.accepted_row(campaign,dataset,variant)==row
 run=Path(row['run_dir']);m0=Path(row['m0_dir'])
 assert not (m0/'m0_reload_probe.pth').exists() and {p.name for p in run.glob('*.pth')}=={'best_map.pth'}
 training=json.loads((run/'training.json').read_text())
 assert [r['epoch'] for r in training['history']]==list(range(1,51))
 steps+=sum(r['steps'] for r in training['history'])
 for old in (r for r in controls['rows'] if r['dataset']==dataset):
  assert (run/'training_batch_order.jsonl').read_bytes()==(Path(old['run_dir'])/'training_batch_order.jsonl').read_bytes()
 files.update(p for directory in (run,m0) for p in directory.iterdir() if p.is_file() and p.suffix in suffixes)
assert steps==summary['formal_steps']==12968
for row in controls['rows']:
 run=Path(row['run_dir']);files.update(run/n for n in ('training.json','official_metrics.json','training_steps.jsonl','training_batch_order.jsonl'))
files.add(root/'refine-logs/region_evidence_reconstruction_v1/SOURCE_SCOPE.json');files.add(panel.CONTROLS)
hashes={str(p.relative_to(root)):panel.base.sha(p) for p in sorted(files)}
print(json.dumps(dict(status='SIX_REGION_RECONSTRUCTION_ENDPOINTS_ADMINISTRATIVE_COMPLETION_VERIFIED',
 verified_at=datetime.now().astimezone().isoformat(),accepted=6,formal_epochs=300,formal_steps=steps,pairs=15,
 report_invocations=1,new_training_invocations=1,inherited_formal_endpoints=5,original_parent_exit_code=1,
 completion_launch_exit=exit_record,original_failure_sha256=launch['original_failure_sha256'],manifest=manifest,
 summary=summary,text_sha256=hashes,control_artifact_count=187,free_bytes=shutil.disk_usage(root).free,
 boundary='Six registered endpoints completed without redoing five accepted trainings. Original parent prelaunch storage failure preserved, last fresh50/firststrict and original15-pair CPU report completed once through separate administrative lineage. Only texts copied; no NN/report re-execution or retired binary replay. Single seed consumed benchmarks; broad Goal remains ACTIVE/UNMET.')))
