from pathlib import Path
import hashlib
import json
import paramiko

private=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
acceptance=json.loads((private/'fixed_five_review_acceptance893.json').read_bytes())
assert acceptance['status']=='TEXT_REVIEW_CLOSED'
review=private/'fixed_five_result_to_claim893/CLAIMS_FROM_RESULTS.json'
assert hashlib.sha256(review.read_bytes()).hexdigest()==acceptance['review_sha256']
qualified=json.loads((private/'fixed_five_retirement_qualification893/REMOTE.json').read_bytes())
assert qualified['count']==15 and qualified['status'].endswith('QUALIFIED_NOT_RETIRED')
packet=private/'fixed_five_closed_retirement894'
assert not packet.exists()
packet.mkdir()
code=r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==424 and all(sha(root/n)==d for n,d in scope.items())
assert json.loads((root/'logs/fixed_five_incremental_best_launch_20261007_892/EXIT.json').read_text())['exit_code']==0
rows=QUALIFIED['rows'];assert len(rows)==15 and len({r['path'] for r in rows})==15
protected=QUALIFIED['protected_artifact_sha256'];retained=QUALIFIED['retained_artifact_sha256']
assert not set(protected)&{r['path'] for r in rows}
assert all(sha(Path(n))==d for n,d in protected.items())
assert all(sha(Path(n))==d for n,d in retained.items())
for row in rows:
    p=Path(row['path'])
    target=root/('trained-model' if row['kind']=='closed_nonadvancing_five_formal_best' else 'logs/fixed_five_incremental_best_diagnosis_20261007_892')
    assert p.resolve().is_relative_to(target.resolve()) and p.stat().st_nlink==1
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
journal=root/'logs/fixed_five_closed_binary_retirement_20261007_894'
assert not journal.exists();journal.mkdir()
free=shutil.disk_usage(root).free
(journal/'QUALIFIED.json').write_text(json.dumps(dict(QUALIFIED,actual_free_before=free,
    text_review_closed=ACCEPTANCE,pre_retirement_verified_at=datetime.now().astimezone().isoformat()),indent=2)+'\n')
for row in rows:Path(row['path']).unlink()
assert all(not Path(r['path']).exists() for r in rows)
assert all(sha(Path(n))==d for n,d in protected.items())
assert all(sha(Path(n))==d for n,d in retained.items())
assert all(sha(root/n)==d for n,d in scope.items())
result=dict(status='FIVE_NONADVANCING_OWN_BEST_AND_TEN_CACHES_RETIRED',at=datetime.now().astimezone().isoformat(),
    rows=rows,count=15,bytes=sum(r['bytes'] for r in rows),free_bytes_before=free,free_bytes_after=shutil.disk_usage(root).free,
    source_count=424,current45_and48_unchanged=True,retained_artifact_sha256=retained,
    required_original_five_full_bytes=5192548352,
    boundary='All model consumers/NN and fresh text review closed before deletion. Authorized useless OWN binaries only. Original official/diagnostic distances, full query tables, histories/receipts kept; authors/public/current controls/strongwinners/source424 unchanged. Direct five-PTH/feature-cache replay requires regeneration. OldFAIL/missing100/result gates unchanged. No new queue/storage qualification claimed.')
(journal/'RETIRED.json').write_text(json.dumps(result,indent=2)+'\n')
result['files']={p.relative_to(root).as_posix():dict(text=p.read_text(),sha256=sha(p),bytes=p.stat().st_size) for p in journal.iterdir() if p.is_file()}
print(json.dumps(result))
'''
code='QUALIFIED = '+repr(qualified)+'\nACCEPTANCE = '+repr(acceptance)+'\n'+code
compile(code,'retire_fixed_five_894_remote','exec')
(packet/'SOURCE.py').write_bytes(code.encode('utf-8'))
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status))+'\n').encode())
assert status==0,error.decode()
value=json.loads(data)
print(json.dumps({k:v for k,v in value.items() if k not in ('rows','files','retained_artifact_sha256')},ensure_ascii=False))
