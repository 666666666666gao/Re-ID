"""Collect complete six-endpoint study while preserving original parent EXIT1."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
launch=json.loads((base/'region_reconstruction_final_training_launch858/stdout.json').read_bytes())
packet=base/'region_reconstruction_completed_admin_intake858'
code='launch='+repr(launch)+'\n'+r'''from pathlib import Path
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
'''
compile(code,'collect_completed_admin858','exec')
assert not packet.exists();packet.mkdir();(packet/'COLLECTOR_SOURCE.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('env CUDA_VISIBLE_DEVICES= /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -')
stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n')
assert status==0,error.decode()
value=json.loads(data);sftp=client.open_sftp()
for name,digest in value['text_sha256'].items():
 destination=packet/'received'/name;destination.parent.mkdir(parents=True,exist_ok=True)
 sftp.get('/data/gaob/Re-ID/Trifusion/'+name,str(destination))
 assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest,name
sftp.close();client.close()
print(json.dumps(dict(status=value['status'],accepted=6,formal_steps=12968,pairs=15,
 texts_received=len(value['text_sha256']),free_bytes=value['free_bytes']),indent=2))
