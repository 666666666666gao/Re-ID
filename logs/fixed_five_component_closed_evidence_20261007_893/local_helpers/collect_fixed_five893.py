from pathlib import Path
import json
import paramiko

private = Path('C:/Users/gb/.codex_tmp')
packet = private / 'independent_evidence_draft/fixed_five_complete893'
assert not packet.exists()
packet.mkdir()
code = r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/fixed_five_incremental_best_launch_20261007_892'
campaign=root/'logs/fixed_five_incremental_best_diagnosis_20261007_892'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exit=json.loads((journal/'EXIT.json').read_text())
state=json.loads((campaign/'campaign.json').read_text())
assert state['status'] in ('FAILED','COMPLETE')
assert (exit['exit_code']==0)==(state['status']=='COMPLETE')
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
seal_path=root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==424 and len(seal['artifact_sha256'])==75
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
rows=[]
for job in state['jobs']:
    if job['status']=='COMPLETE':
        p=campaign/(job['dataset']+'_'+job['objective'])/'DIAGNOSIS.json'
        report=json.loads(p.read_text())
        assert report['status']=='COMPLETE' and report['model_state_before_sha256']==report['model_state_after_sha256']
        for name,item in report['artifacts'].items():
            assert (p.parent/name).stat().st_size==item['bytes'] and sha(p.parent/name)==item['sha256']
        row=next(r for r in seal['rows'] if r['dataset']==job['dataset'] and r['variant']==job['objective'])
        assert report['input_checkpoint_sha256']==row['checkpoint_sha256']
        assert all(abs(report['scores']['fused']['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
        rows.append(report)
if exit['exit_code']==0:
    assert len(rows)==5
    assert sum(row['diagnostic'][s]['global_norm']['count'] for row in rows for s in ('query','gallery'))==16926
files={}
for directory in (journal,campaign):
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.py','.log','.txt'):
            files[p.relative_to(root).as_posix()]=dict(text=p.read_text(),sha256=sha(p),bytes=p.stat().st_size)
files[seal_path.relative_to(root).as_posix()]=dict(text=seal_path.read_text(),sha256=sha(seal_path),bytes=seal_path.stat().st_size)
print(json.dumps(dict(status='FIXED_FIVE_COMPONENT_DIAGNOSIS_TERMINAL_RECEIVED',at=datetime.now().astimezone().isoformat(),
    original_exit=exit,campaign=state,rows=rows,source_count=424,artifact_count=75,
    current_inputs_physically_unchanged=True,free_bytes=shutil.disk_usage(root).free,files=files,
    boundary='Original selected best weights only, no optimizer/NN rerun/new checkpoint/gain/threshold. Tensor artifacts remain26. Exact component retrieval is post-selection descriptive and not an independent training ablation. Failed100repair remains missing.')))
'''
compile(code, 'collect_fixed_five_893_remote', 'exec')
(packet / 'SOURCE.py').write_bytes(code.encode('utf-8'))
c = paramiko.SSHClient()
c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
i, o, e = c.exec_command('/usr/bin/python3 -B -')
i.write(code)
i.channel.shutdown_write()
o.channel.settimeout(300)
data, error = o.read(), e.read()
status = o.channel.recv_exit_status()
c.close()
(packet / 'REMOTE.json').write_bytes(data)
(packet / 'STDERR.txt').write_bytes(error)
(packet / 'EXIT.json').write_bytes((json.dumps(dict(exit_code=status)) + '\n').encode())
assert status == 0, error.decode()
result = json.loads(data)
text = packet / 'primary_text'
text.mkdir()
mapping = {}
import hashlib
for original, row in result['files'].items():
    dest = text / (f'{len(mapping):04d}_' + Path(original).name)
    dest.write_bytes(row['text'].encode('utf-8'))
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == row['sha256']
    mapping[original] = dict(local_file=str(dest), sha256=row['sha256'], bytes=row['bytes'])
(packet / 'FILE_MAP.json').write_bytes((json.dumps(mapping, indent=2) + '\n').encode())
compact = dict(status=result['status'], original_exit=result['original_exit'], source_count=424, artifact_count=75,
    rows=[dict(dataset=r['dataset'],objective=r['objective'],selected_epoch=r['selected_epoch'],
        metrics={n:v['metrics'] for n,v in r['scores'].items()},comparisons={n:{k:v for k,v in d.items() if k not in ('query_changes','identity_changes')} for n,d in r['comparisons'].items()},
        diagnostic=r['diagnostic'],seconds=r['elapsed_seconds']) for r in result['rows']],
    free_bytes=result['free_bytes'],boundary=result['boundary'])
(packet / 'COMPONENT_SUMMARY.json').write_bytes((json.dumps(compact, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps(dict(status=result['status'],original_exit=result['original_exit'],complete=len(result['rows']),texts=len(mapping),free_bytes=result['free_bytes']),ensure_ascii=False))
