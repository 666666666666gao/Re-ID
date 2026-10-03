"""Receive the original complete F3 report and all six terminal texts; no replay."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import paramiko

PRIVATE = Path('C:/Users/gb/.codex_tmp')
LAUNCH = json.loads((PRIVATE / 'training_feature_scale_split755/deploy/LAUNCH.json').read_bytes())
ROOT = LAUNCH['root']
SOURCE = PRIVATE / 'training_feature_scale_source_intake749/source'
OUT = PRIVATE / 'metric_feature_scale_complete_v1'
assert LAUNCH['port'] == 2026 and LAUNCH['training_ports'] == [2026]
assert not OUT.exists()
assert LAUNCH['gpu_scope'] == [0,1,2,3] and LAUNCH['max_parallel'] == 4
metric_source_names = (
 'tools/run_metric_feature_scale.py','tools/queue_metric_feature_scale.py',
 'tools/check_metric_feature_scale_pair.py','tools/report_metric_feature_scale.py',
 'refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md',
 'refine-logs/metric_feature_scale_v1/EXPERIMENT_CODE_REVIEW.md',
)
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
original = json.loads((PRIVATE/'training_feature_scale_complete754/INTAKE.json').read_bytes())['source_sha256']
assert len(original) == 259
source_input = {name:SOURCE/name for name in original}
source_input.update({name:repo/name for name in metric_source_names})
assert len(source_input) == 265 and set(source_input) == set(LAUNCH['source_sha256'])
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == LAUNCH['source_sha256'][name]
           for name,path in source_input.items())
code = f'''
from datetime import datetime
import hashlib,json,shutil
from pathlib import Path
root=Path({ROOT!r})
campaign=Path({LAUNCH['campaign']!r})
report=root/'results/metric_feature_scale_complete_20261003'
finish=root/'logs/metric_feature_scale_storage_finish766_20261003'
retirement=root/'logs/closed_m0_retirement766_20261003'
finish_source=root/'logs/metric_storage_finish_source766_20261003'
finish_source_v2=root/'logs/metric_storage_finish_source766_v2_20261003'
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
finish_record=json.loads((finish/'STATUS.json').read_text())
assert finish_record['status']=='ALL6_STRICT_AND_FIRST_REPORT_COMPLETE'
assert finish_record['evaluation_invocations']==1 and finish_record['evaluation_exit_code']==0
assert finish_record['report_invocations']==1 and finish_record['report_exit_code']==0
assert json.loads((finish/'original_campaign.json').read_text())['status']=='FAILED'
assert state['original_failed_campaign_sha256']==hashlib.sha256((finish/'original_campaign.json').read_bytes()).hexdigest()
assert not (Path('/proc')/str(finish_record['controller_pid'])).exists()
assert len(state['jobs'])==12 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert all(j['gpu'] in (0,1,2,3) for j in state['jobs'])
manifest=json.loads((campaign/'manifest.json').read_text())
source=manifest['source_sha256']
assert len(source)==265 and source=={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source}}
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
summary=json.loads((report/'SUMMARY.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==6
assert summary['status']=='COMPLETE' and summary['accepted']==6 and summary['formal_epochs']==300
assert summary['formal_steps']==20416 and len(summary['rows'])==6 and len(summary['pairs'])==3
text=list(campaign.rglob('*.json'))+list(campaign.rglob('*.log'))+list(report.glob('*.json'))+list(report.glob('*.md'))
text.append(Path({LAUNCH['log']!r}))
text.extend(finish.glob('*.json'));text.extend(finish.glob('*.log'))
text.extend(retirement.glob('*.json'))
text.extend(finish_source.glob('*.json'));text.extend(finish_source.glob('*.md'))
text.extend(finish_source.glob('*.py'));text.extend(finish_source.glob('*.log'))
text.extend(finish_source_v2.glob('*.json'));text.extend(finish_source_v2.glob('*.md'))
text.extend(finish_source_v2.glob('*.py'));text.extend(finish_source_v2.glob('*.log'))
binary=[]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
 for variant in ('normalized','metric_raw'):
  for phase in ('m0','full'):
   run=root/'trained-model'/f'{{campaign.name}}_{{phase}}_{{variant}}_{{dataset}}'
   text.extend(run.glob('*.json'));text.extend(run.glob('*.jsonl'))
   names=('m0_reload_probe.pth',) if phase=='m0' else ('best_map.pth','best_epoch_distances.pt','official_distances.pt')
   for name in names:
    path=run/name
    digest=hashlib.sha256()
    with path.open('rb') as stream:
     for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    binary.append({{'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}})
   if phase=='full':
    training=json.loads((run/'training.json').read_text())
    metrics=json.loads((run/'official_metrics.json').read_text())
    assert [r['epoch'] for r in training['history']]==list(range(1,51))
    assert metrics['status']=='COMPLETE' and metrics['training_epochs']==50 and metrics['seed']==42
    assert metrics['independent_upstream_metrics_equal'] and not metrics['reranking']
    actual={{Path(r['path']).name:r['sha256'] for r in binary if Path(r['path']).parent==run}}
    for name,key in (('best_map.pth','checkpoint_sha256'),('best_epoch_distances.pt','training_best_distance_sha256'),('official_distances.pt','distance_sha256')):
     assert actual[name]==metrics[key]
files={{str(path.relative_to(root)):{{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}} for path in text}}
assert len(files)==len(text) and len(binary)==24
print(json.dumps({{'status':'ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE','observed_at':datetime.now().astimezone().isoformat(),
 'port':2026,'source_sha256':source,'text':files,'binary':binary,'controller_state':state,
 'controller_proc_exists':(Path('/proc')/str(state['controller_pid'])).exists(),
 'accepted_matrix':matrix,'disk_free_bytes':shutil.disk_usage(root).free,
 'terminal_summary_sha256':files[str((report/'SUMMARY.json').relative_to(root))]['sha256']}}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
receipt = json.loads(raw)
assert all(hashlib.sha256(source_input[name].read_bytes()).hexdigest() == digest
           for name, digest in receipt['source_sha256'].items())
OUT.mkdir()
for name,path in source_input.items():
    destination = OUT/'source'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == receipt['source_sha256'][name]
sftp = client.open_sftp()
for name, entry in receipt['text'].items():
    with sftp.open(ROOT + '/' + name, 'rb') as stream:
        data = stream.read()
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], name
    path = OUT / 'raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
sftp.close()
client.close()
receipt['received_at'] = datetime.now().astimezone().isoformat()
receipt['local_source_directory'] = str(OUT/'source')
receipt['boundary'] = 'Complete executor text/hash intake of six original full50 trainings, their strict evaluation and the first unmodified CPU report. The last original worker stopped before evaluation at a disk guard; separately recorded storage-only finish invoked that evaluation for the first time, preserving the original FAILED receipt and immutable checkpoint. No training/scoring replay; current binaries remain remote. Retired historical engineering M0 binary limits are explicit. Fresh integrity and claim reviews not yet performed.'
(OUT / 'INTAKE.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps({'status': receipt['status'], 'text_files': len(receipt['text']),
                  'remote_binary_records': len(receipt['binary']), 'output': str(OUT)}, indent=2), flush=True)
