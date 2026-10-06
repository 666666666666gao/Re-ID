from pathlib import Path
import json
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_retirement_r2')
assert not packet.exists();packet.mkdir()
qualified=json.loads(Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_v3/QUALIFICATION.json').read_bytes())
code='payload='+repr(qualified)+'\n'+r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/selection_storage_retirement_20261006_865'
assert not journal.exists()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for directory in Path('/proc').iterdir():
 if directory.name.isdigit() and (directory/'cmdline').is_file():
  command=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  assert '/data/gaob/Re-ID/Trifusion/tools/' not in command
sources=json.loads((root/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_text())['source_sha256']
protected=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(payload['qualified'])==6 and len(sources)==366 and len(protected)==187
assert all(sha(root/n)==d for n,d in sources.items()) and all(sha(n)==d for n,d in protected.items())
for pair in payload['qualified']:
 for row in (pair['target'],pair['winner']):
  folder=Path(row['directory']);assert folder.resolve().is_relative_to((root/'trained-model').resolve())
  assert all(sha(n)==d for n,d in row['files_sha256'].items())
  t=json.loads((folder/'training.json').read_text());r=json.loads((folder/'official_metrics.json').read_text())
  assert [x['epoch'] for x in t['history']]==list(range(1,51)) and r['status']=='COMPLETE'
  assert r['metrics']==row['metrics'] and r['checkpoint_sha256']==row['checkpoint_sha256']
 p=Path(pair['target']['directory'])/'best_map.pth'
 assert str(p) not in protected and str(p.relative_to(root)) not in sources
 assert all(pair['winner']['metrics'][k]>=v for k,v in pair['target']['metrics'].items())
before=shutil.disk_usage(root).free
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(payload,indent=2)+'\n')
for pair in payload['qualified']:
 target=pair['target'];p=Path(target['directory'])/'best_map.pth'
 assert sha(p)==target['checkpoint_sha256'] and p.stat().st_size==target['bytes']
 p.unlink();assert not p.exists()
 with (journal/'deleted.jsonl').open('a') as stream:stream.write(json.dumps(dict(path=str(p),bytes=target['bytes'],sha256=target['checkpoint_sha256']))+'\n')
 assert all(sha(n)==d for n,d in target['files_sha256'].items() if n!=str(p))
 assert all(sha(n)==d for n,d in pair['winner']['files_sha256'].items())
assert all(sha(root/n)==d for n,d in sources.items()) and all(sha(n)==d for n,d in protected.items())
result=dict(status='SIX_CLOSED_DOMINATED_OWN_WEIGHTS_RETIRED',at=datetime.now().astimezone().isoformat(),removed_count=6,removed_bytes=sum(r['target']['bytes'] for r in payload['qualified']),free_before=before,free_after=shutil.disk_usage(root).free,rows=payload['qualified'],preserved_sources=366,preserved_raw_artifacts=187,boundary='Only explicitly qualified clean_joint and EV1 semantic best binaries. Current producer set empty; no dependency on targets in current study; original full50/strict/arrays/text retained. Three dominating winners unchanged. No25/otherprojects/power/temp action.')
(journal/'RETIREMENT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
compile(code,'retirement865','exec');(packet/'SOURCE.py').write_text(code)
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'STDOUT.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode()
s=c.open_sftp()
for name in ('PREPARE.json','deleted.jsonl','RETIREMENT.json'):s.get('/data/gaob/Re-ID/Trifusion/logs/selection_storage_retirement_20261006_865/'+name,str(packet/name))
s.close();c.close()
r=json.loads(data);print(json.dumps({k:r[k] for k in ('status','removed_count','removed_bytes','free_before','free_after')}))
