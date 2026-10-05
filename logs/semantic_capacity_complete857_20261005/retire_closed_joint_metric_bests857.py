"""Retire four closed, non-selected metric candidates; retain the best 201 candidate."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'capacity_first_eval_closed_metric_retirement857'
seal = json.loads((base / 'deployment_metric_five_fixed_best_seal850/INPUT_SEAL.json').read_bytes())
rows = [{k: r[k] for k in ('dataset', 'variant', 'run_dir', 'best_epoch', 'metrics')}
        for r in seal['rows'] if r['variant'] != 'global_only'
        and (r['dataset'], r['variant']) != ('RGBNT201', 'native')]
assert len(rows) == 4
code = rf'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_control_v1_20261005_856'
launch=root/'logs/semantic_capacity_launch_20261005_856'
journal=root/'logs/capacity_closed_metric_best_retirement_20261005_857'
assert not journal.exists()
assert not Path('/proc/642951').exists() and not Path('/proc/642952').exists()
state=json.loads((campaign/'campaign.json').read_text())
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1 and state['report_invocations']==0
full=next(r for r in state['jobs'] if (r['dataset'],r['phase'])==('RGBNT100','full'))
assert len(full['steps'])==1 and full['steps'][0]['mode']=='train' and full['steps'][0]['exit_code']==0
assert not (campaign/'RGBNT100_evaluate.log').exists()
assert not (root/'trained-model/semantic_capacity_control_v1_20261005_856_full_native_RGBNT100/official_metrics.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for body in iter(lambda:stream.read(1024*1024),b''):h.update(body)
 return h.hexdigest()
manifest=json.loads((campaign/'manifest.json').read_text())
assert len(manifest['source_sha256'])==345
assert all(sha(root/n)==d for n,d in manifest['source_sha256'].items())
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(protected)==187 and all(sha(Path(n))==d for n,d in protected.items())
old={seal['artifact_sha256']!r}
rows={rows!r}
verified=[]
for row in rows:
 folder=Path(row['run_dir']);target=folder/'best_map.pth'
 assert target.resolve().is_relative_to((root/'trained-model').resolve()) and target.name=='best_map.pth'
 assert folder.name.startswith(('deployment_metric_role_v1_20261005_837_full_',
  'deployment_metric_role_pending100_20261005_842_full_',
  'deployment_metric_storage_continuation_20261005_848_full_'))
 assert str(target) not in protected and target.relative_to(root).as_posix() not in manifest['source_sha256']
 training=json.loads((folder/'training.json').read_text());receipt=json.loads((folder/'official_metrics.json').read_text())
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history'])==50
 assert receipt['status']=='COMPLETE' and receipt['selected_epoch']==training['best_epoch']==row['best_epoch']
 assert receipt['metrics']==row['metrics']
 assert sha(target)==old[str(target)]==receipt['checkpoint_sha256']
 assert sha(folder/'training.json')==old[str(folder/'training.json')]
 assert sha(folder/'official_metrics.json')==old[str(folder/'official_metrics.json')]
 assert sha(folder/'official_distances.pt')==receipt['distance_sha256']
 assert sha(folder/'best_epoch_distances.pt')==receipt['training_best_distance_sha256']
 verified.append(dict(dataset=row['dataset'],variant=row['variant'],path=str(target),
  bytes=target.stat().st_size,sha256=sha(target),metrics=row['metrics'],
  training_sha256=sha(folder/'training.json'),receipt_sha256=sha(folder/'official_metrics.json'),
  distance_sha256=receipt['distance_sha256'],best_epoch_distance_sha256=receipt['training_best_distance_sha256']))
preserved=root/'trained-model/deployment_metric_role_v1_20261005_837_full_native_RGBNT201/best_map.pth'
assert sha(preserved)==old[str(preserved)]
failure_files=(campaign/'campaign.json',launch/'EXIT.json',launch/'console.log')
failure_hashes={{str(p):sha(p) for p in failure_files}}
journal.mkdir()
(journal/'PREPARE.json').write_bytes((json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=verified,
 failure_sha256=failure_hashes),indent=2)+'\n').encode())
before=shutil.disk_usage(root).free
for row in verified:
 target=Path(row['path']);target.unlink();assert not target.exists()
assert all(sha(Path(n))==d for n,d in failure_hashes.items())
assert sha(preserved)==old[str(preserved)]
value=dict(status='FOUR_CLOSED_NON_SELECTED_METRIC_BESTS_RETIRED',at=datetime.now().astimezone().isoformat(),
 retired_files=4,retired_bytes=sum(r['bytes'] for r in verified),rows=verified,
 disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,
 retained_highest_metric_201_native_sha256=sha(preserved),original_failure_sha256=failure_hashes,
 boundary='Closed joint-metric stage rejected; retain its highest201 native best and all current345/187 dependencies, capacity bests/probe100, raw controls, authors and public initialization. Original metrics/distances/history remain. These four old weight-reload paths retired, including their historical five-model seal binary replay. No training/re-evaluation/source change or power/temperature action.')
(journal/'RETIREMENT.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
print(json.dumps(value))
'''
compile(code, 'retire_closed_joint_metric857.py', 'exec')
assert not packet.exists()
packet.mkdir()
(packet / 'REMOTE_SOURCE.py').write_bytes(code.encode())
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(300)
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
client.close()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_bytes((json.dumps(dict(exit_code=status, at=datetime.now().astimezone().isoformat())) + '\n').encode())
assert status == 0, error.decode()
value = json.loads(data)
print(json.dumps({k: value[k] for k in ('status', 'at', 'retired_files', 'retired_bytes', 'disk_free_before', 'disk_free_after')}))
