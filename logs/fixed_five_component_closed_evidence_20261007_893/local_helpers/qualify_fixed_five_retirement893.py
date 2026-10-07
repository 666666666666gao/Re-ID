from pathlib import Path
import json
import paramiko

private = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
received = json.loads((private / 'fixed_five_complete893/REMOTE.json').read_bytes())
assert received['original_exit']['exit_code'] == 0 and len(received['rows']) == 5
packet = private / 'fixed_five_retirement_qualification893'
assert not packet.exists()
packet.mkdir()
code = r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion'); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==424 and all(sha(root/n)==d for n,d in scope.items())
seal=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==75 and all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
exit=json.loads((root/'logs/fixed_five_incremental_best_launch_20261007_892/EXIT.json').read_text())
assert exit==RECEIVED['original_exit'] and exit['exit_code']==0
rows=[]; retained={}
for original, component in zip(seal['rows'],RECEIVED['rows']):
    assert (original['dataset'],original['variant'])==(component['dataset'],component['objective'])
    run=Path(original['run_dir']); diag=root/'logs/fixed_five_incremental_best_diagnosis_20261007_892'/(original['dataset']+'_'+original['variant'])
    assert json.loads((diag/'DIAGNOSIS.json').read_text())==component
    p=run/'best_map.pth'
    assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.stat().st_nlink==1
    assert sha(p)==original['checkpoint_sha256']
    rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='closed_nonadvancing_five_formal_best'))
    for name in ('official_metrics.json','official_distances.pt','training.json','training_steps.jsonl','training_batch_order.jsonl'):
        retained[str(run/name)]=sha(run/name)
    retained[str(diag/'DIAGNOSIS.json')]=sha(diag/'DIAGNOSIS.json')
    retained[str(diag/'diagnostic_distances.pt')]=component['artifacts']['diagnostic_distances.pt']['sha256']
    for name in ('query_features.pt','gallery_features.pt'):
        p=diag/name
        assert p.resolve().is_relative_to((root/'logs/fixed_five_incremental_best_diagnosis_20261007_892').resolve()) and p.stat().st_nlink==1
        assert sha(p)==component['artifacts'][name]['sha256']
        rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='closed_component_feature_cache'))
protected={}
for name in ('INPUT_SEAL.json','INITIAL_BOUNDARY_INPUT_SEAL.json'):
    protected.update(json.loads((root/'refine-logs/incremental_role_objective_v1'/name).read_text())['artifact_sha256'])
assert not set(protected)&{r['path'] for r in rows}
assert all(sha(Path(n))==d for n,d in protected.items())
assert all(sha(Path(n))==d for n,d in retained.items())
assert len(rows)==15 and len({r['path'] for r in rows})==15
print(json.dumps(dict(status='FIVE_CLOSED_NONADVANCING_WEIGHTS_AND_TEN_FEATURE_CACHES_QUALIFIED_NOT_RETIRED',
    at=datetime.now().astimezone().isoformat(),rows=rows,count=15,bytes=sum(r['bytes'] for r in rows),
    free_bytes_before=shutil.disk_usage(root).free,retained_artifact_sha256=retained,protected_artifact_sha256=protected,
    source_count=424,boundary='Read-only qualification. Five current runs all below registered advancement and existing stronger preserved routes; final fixed-best consumer EXIT0 and complete query evidence retained. Existing45/48 controls, authors/public/strongwinners untouched. Actual retirement only after text claim review closes; direct five-PTH/feature-cache replay will require regeneration after retirement, no historical pass/fail changed.')))
'''
code = 'RECEIVED = ' + repr({k: v for k, v in received.items() if k not in ('files', 'campaign')}) + '\n' + code
compile(code, 'qualify_fixed_five_retirement_893_remote', 'exec')
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
value = json.loads(data)
print(json.dumps({k:v for k,v in value.items() if k not in ('rows','retained_artifact_sha256','protected_artifact_sha256')},ensure_ascii=False))
