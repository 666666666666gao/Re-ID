from pathlib import Path
import hashlib
import json
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891')
assert not packet.exists()
packet.mkdir()
code = r'''from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/qualified_five_incremental_full_launch_20261007_888'
campaign=root/'logs/qualified_five_incremental_full_v1_20261007_888'
assert json.loads((journal/'EXIT.json').read_text())['exit_code']==0
assert json.loads((campaign/'campaign.json').read_text())['status']=='COMPLETE'
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        words=(p/'cmdline').read_bytes().split(bytes([0]))
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in words),words
scope_path=root/'refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_SOURCE_SCOPE.json'
scope=json.loads(scope_path.read_text())['source_sha256']
assert len(scope)==420
files={}
for name,digest in scope.items():
    p=root/name
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest,name
    files[name]={'text':data.decode('utf-8'),'sha256':digest,'bytes':len(data)}
extras=['refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_SOURCE_SCOPE.json',
        'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json',
        'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_INPUT_SEAL.json']
extras += ['logs/training_feature_scale_protocols_20261002/'+d+'.json' for d in ('RGBNT201','RGBNT100','MSVR310')]
for name in extras:
    data=(root/name).read_bytes()
    files[name]={'text':data.decode('utf-8'),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
print(json.dumps({'status':'EXACT_FROZEN_420_SOURCES_AND_SIX_PROTOCOL_SEAL_TEXTS','source_count':420,'files':files}))
'''
compile(code, 'qualified_five_audit_sources891_remote', 'exec')
(packet / 'SOURCE.py').write_text(code, encoding='utf-8')
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
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': status}) + '\n', encoding='utf-8')
assert status == 0, error.decode()
r = json.loads(data)
mapping = {}
for index, (name, item) in enumerate(r['files'].items()):
    p = packet / f'{index:04d}_{Path(name).name}'
    p.write_bytes(item['text'].encode('utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
    mapping[name] = {'local_file': p.name, 'sha256': item['sha256'], 'bytes': p.stat().st_size}
(packet / 'FILE_MAP.json').write_text(json.dumps(mapping, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': r['status'], 'text_files': len(mapping), 'bytes': sum(v['bytes'] for v in mapping.values())}))
