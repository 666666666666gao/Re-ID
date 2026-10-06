"""Receive the original three full runs and once-only report after queue exit0."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/independent_role_heads_complete873')
code = '''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/independent_role_heads_v1_20261006_873'
launch=root/'logs/independent_role_heads_launch_20261006_873'
report_root=root/'results/independent_role_heads_v1_20261006_873'
datasets=('RGBNT201','MSVR310','RGBNT100')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
exit_record=json.loads((launch/'EXIT.json').read_text())
state=json.loads((campaign/'campaign.json').read_text())
assert exit_record['exit_code']==0
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==6 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
manifest=json.loads((campaign/'manifest.json').read_text())
assert manifest['physical_gpus']==[0,1] and manifest['max_parallel_jobs']==1
assert len(manifest['source_sha256'])==385
assert all(sha(root/n)==digest for n,digest in manifest['source_sha256'].items())
assert all(sha(Path(n))==digest for n,digest in manifest['initialization_sha256'].items())
seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal)==manifest['control_seal_sha256']=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
controls=json.loads(seal.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==187
assert all(sha(Path(n))==digest for n,digest in controls['artifact_sha256'].items())
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
report=json.loads((report_root/'SUMMARY.json').read_text())
assert matrix['schema']==report['schema']=='trifusion-independent-role-heads-v1'
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==report['accepted']==3
assert report['status']=='COMPLETE' and report['formal_epochs']==150 and report['formal_steps']==6484
assert len(report['rows'])==3 and {r['dataset'] for r in report['rows']}==set(datasets)
assert len(report['pairs'])==6
assert {(p['dataset'],p['control']) for p in report['pairs']}=={(d,c) for d in datasets for c in ('raw_semantic','raw_global_only')}
accepted={r['dataset']:r for r in matrix['rows']}
retired=[json.loads(line) for line in (campaign/'probe_retirement.jsonl').read_text().splitlines()]
assert len(retired)==3
paths={seal,report_root/'SUMMARY.json',report_root/'REPORT.md'}
paths.update(p for folder in (campaign,launch) for p in folder.rglob('*')
             if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.py'))
rows,orders,artifacts=[],{},{}
for row in report['rows']:
 dataset=row['dataset'];original=accepted[dataset]
 assert all(row[k]==v for k,v in original.items())
 acceptance_path=campaign/'acceptance'/f'{dataset}_semantic.json'
 acceptance=json.loads(acceptance_path.read_text())
 assert acceptance['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
 assert acceptance['row']==original
 assert all(sha(Path(n))==digest for n,digest in acceptance['artifact_sha256'].items())
 match=[r for r in retired if (r['dataset'],r['variant'])==(dataset,'semantic')]
 assert len(match)==1 and match[0]['acceptance_sha256']==sha(acceptance_path)
 assert all(match[0][key]==value for key,value in acceptance['probe'].items())
 assert not Path(acceptance['probe']['path']).exists()
 run,m0=Path(row['run_dir']),Path(row['m0_dir'])
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
 diag=smoke['production_m0_diagnostics']
 assert diag['effective_optimizer_updates']==8 and len(diag['role_head_updates'])==8
 assert len(diag['role_head_parameters'])==row['initializer']['role_head_tensors']
 assert all(v==8 for v in diag['role_bn_batches_tracked'])
 assert all(diag['role_head_updates'][-1][name]['parameter_delta_from_initial_max_abs']>0 for name in diag['role_head_parameters'])
 assert {p.name for p in run.glob('*.pth')}=={'best_map.pth'}
 assert not list(m0.glob('*.pth'))
 bound={}
 for name,field in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('best_epoch_distances.pt','training_best_distance_sha256')):
  p=run/name;actual=sha(p);assert actual==receipt[field]
  bound[name]=dict(sha256=actual,bytes=p.stat().st_size)
 assert sha(run/'official_metrics.json')==row['receipt_sha256']
 assert bound['best_map.pth']['sha256']==row['checkpoint_sha256']
 assert bound['official_distances.pt']['sha256']==row['distance_sha256']
 artifacts[dataset]=bound
 order=(run/'training_batch_order.jsonl').read_bytes()
 batches=[json.loads(line) for line in order.splitlines()]
 steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
 assert len(steps)==len(batches)==row['formal_steps']==sum(e['steps'] for e in training['history'])
 assert [(s['epoch'],s['batch']) for s in steps]==[(b['epoch'],b['batch']) for b in batches]
 for variant in ('semantic','global_only'):
  old=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==(dataset,variant))
  assert order==(Path(old['run_dir'])/'training_batch_order.jsonl').read_bytes()
  paths.add(Path(old['run_dir'])/'training.json')
 orders[dataset]=dict(sha256=hashlib.sha256(order).hexdigest(),bytes=len(order),formal_steps=len(batches),matched_controls=2)
 for folder in (run,m0):
  paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.csv'))
 rows.append(dict(dataset=dataset,best_epoch=row['best_epoch'],metrics=row['metrics'],formal_steps=row['formal_steps'],
     last_epoch_metrics=training['history'][-1]['official_fused'],
     best_to_last_map_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
     complete_training_and_epoch_eval_seconds=row['complete_training_and_epoch_eval_seconds'],
     additional_head_parameters=row['initializer']['role_head_parameters']))
pairs=[]
for pair in report['pairs']:
 diag=pair['paired_diagnosis'];delta=diag['delta_metrics']
 assert pair['actual_training_batch_order_equal'] and diag['status']=='CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE'
 assert len(diag['query_changes'])==sum(r['queries'] for r in diag['identity_changes'])
 assert [r['query_index'] for r in diag['query_changes']]==list(range(len(diag['query_changes'])))
 assert pair['phase_progress']==(delta['mAP']>=.5 and delta['Rank-1']>=0)
 pairs.append(dict(dataset=pair['dataset'],control=pair['control'],delta_metrics=delta,phase_progress=pair['phase_progress'],
     rank1_repairs=diag['rank1_repairs'],rank1_new_errors=diag['rank1_new_errors'],
     identity_macro_mean_delta_ap_points=diag['identity_macro_mean_delta_ap_points'],query_count=len(diag['query_changes'])))
files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
print(json.dumps(dict(status='THREE_FULL50_FIRST_STRICT_AND_ORIGINAL_ONCE_REPORT_VERIFIED',at=datetime.now().astimezone().isoformat(),
     formal_completed=3,formal_epochs=150,formal_steps=6484,report_invocations=1,report_exit_code=0,
     launch_exit=exit_record,source_files_verified_unchanged=385,protected_raw_artifacts_verified=187,
     rows=rows,pairs=pairs,batch_orders=orders,artifacts=artifacts,files=files,disk_free_bytes=shutil.disk_usage(root).free,
     controller_present=(Path('/proc')/str(state['controller_pid'])).exists(),
     boundary='Read-only text intake and SHA verification only: zero model construction, inference, updates, report replay or deletion. Three fresh seed42 full runs and original six-pair report; added trainable head capacity/BN buffers disclosed. Official post-selection exploratory evidence, not training-seed stability, exclusive cause, SOTA, or full goal completion.'))) 
'''


def main():
    compile(code, 'remote_independent_head_complete_intake873', 'exec')
    assert not packet.exists()
    packet.mkdir()
    (packet / 'REMOTE_SOURCE.py').write_bytes(code.encode('utf-8'))
    client = paramiko.SSHClient()
    client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138', port=2026, username='gaob',
                   key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
    stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code)
    stdin.channel.shutdown_write()
    stdout.channel.settimeout(300)
    data, error = stdout.read(), stderr.read()
    status = stdout.channel.recv_exit_status()
    (packet / 'stdout.json').write_bytes(data)
    (packet / 'stderr.txt').write_bytes(error)
    (packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=status, at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
    assert status == 0, error.decode()
    record = json.loads(data)
    sftp = client.open_sftp()
    for name, info in record['files'].items():
        with sftp.open('/data/gaob/Re-ID/Trifusion/' + name, 'rb') as stream:
            body = stream.read()
        assert len(body) == info['bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']
        destination = packet / 'received' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    sftp.close()
    client.close()
    (packet / 'SUMMARY.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in ('files','artifacts')}, indent=2))


if __name__ == '__main__':
    main()
