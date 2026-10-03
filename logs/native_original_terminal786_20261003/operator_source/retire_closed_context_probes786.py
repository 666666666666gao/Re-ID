from pathlib import Path
from datetime import datetime
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
spec=json.loads((base/'closed_context_probe_inventory786/INVENTORY.json').read_bytes())
assert spec['status']=='EXACT_FIFTEEN_CLOSED_CONTEXT_PROBES_VERIFIED' and len(spec['candidates'])==15
target=base/'closed_context_probe_retirement786'
assert not target.exists()
target.mkdir()
(target/'SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion').resolve(strict=True);spec={spec!r}
receipt=root/'logs/closed_context_probes_retirement786_20261003/RETIREMENT.json'
assert not receipt.parent.exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert all(not (Path('/proc')/str(pid)).exists() for pid in spec['absent_pids'])
assert sha(Path(spec['campaign_path']))==spec['campaign_sha256']
assert sha(Path(spec['accepted_path']))==spec['accepted_sha256']
assert all(sha(Path(n))==h for n,h in spec['protected_sha256'].items())
assert all(sha(root/n)==h for n,h in spec['source_sha256'].items())
rows=[]
for row in spec['candidates']:
 p=Path(row['path'])
 assert p.resolve(strict=True)==p and not p.is_symlink() and p.is_relative_to(root/'trained-model')
 assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('correspondence_context_identity_20260929_') and p.parent.name.endswith('_seed42_m0')
 assert p.stat().st_size==row['bytes'] and sha(p)==row['probe_sha256']
 for field in ('training','formal_best','formal_distance','formal_receipt'):
  assert sha(Path(row[field+'_path']))==row[field+'_sha256']
 rows.append(dict(row,deleted=False))
record={{'status':'VERIFIED_NOT_YET_RETIRED','started_at':datetime.now().astimezone().isoformat(),
 'free_bytes_before':shutil.disk_usage(root).free,'candidates':rows,
 'protected_sha256':spec['protected_sha256'],'source_sha256':spec['source_sha256'],
 'boundary':'User-authorized useless-weight cleanup: exact15 closed context15 M0 probes after full50/best/reload/report closure. All15 formal bests and distances, training/reload/source/failed records retained. Direct binary M0 replay retired; no current/public/author/other-project weights removed.'}}
receipt.parent.mkdir();receipt.write_text(json.dumps(record,indent=2)+'\\n')
for row in rows:
 Path(row['path']).unlink();row.update(deleted=True,deleted_at=datetime.now().astimezone().isoformat())
 receipt.write_text(json.dumps(record,indent=2)+'\\n')
assert all(not Path(r['path']).exists() for r in rows)
assert all(sha(Path(n))==h for n,h in spec['protected_sha256'].items())
for row in rows:
 for field in ('training','formal_best','formal_distance','formal_receipt'):
  assert sha(Path(row[field+'_path']))==row[field+'_sha256']
record.update(status='RETIRED_EXACT_FIFTEEN_CLOSED_CONTEXT_PROBES',retired_bytes=sum(r['bytes'] for r in rows),
 completed_at=datetime.now().astimezone().isoformat(),free_bytes_after=shutil.disk_usage(root).free)
receipt.write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();rc=stdout.channel.recv_exit_status();client.close()
(target/'stdout.json').write_bytes(data);(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc,'at':datetime.now().astimezone().isoformat()})+'\n')
assert rc==0,error.decode()
(target/'RETIREMENT.json').write_bytes(data)
record=json.loads(data)
print(json.dumps({k:record[k] for k in ('status','started_at','completed_at','retired_bytes','free_bytes_before','free_bytes_after','boundary')}))
