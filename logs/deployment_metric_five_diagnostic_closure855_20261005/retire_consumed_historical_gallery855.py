"""Retire six consumed historical gallery feature caches, preserving every best."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'deployment_metric_consumed_gallery_retirement855'
old=json.loads((base/'global_task_role_fixed_best_complete/stdout.json').read_bytes())
candidates=[]
for name,info in old['files'].items():
    if name.endswith('/DIAGNOSIS.json'):
        r=json.loads((base/'global_task_role_fixed_best_complete/texts'/(Path(name).parent.name+'_DIAGNOSIS.json')).read_bytes())
        assert r['status']=='COMPLETE'
        candidates.append(dict(diagnosis=name,diagnosis_sha256=info['sha256'],
            target=str(Path(name).with_name('gallery_features.pt')).replace('\\','/'),
            **r['artifacts']['gallery_features.pt']))
assert len(candidates)==6
code=f'''from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/deployment_metric_consumed_gallery_retirement_20261005_855'
assert not journal.exists()
assert json.loads((root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854/EXIT.json').read_text())['exit_code']==0
seal=json.loads((root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==271
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for r in {candidates!r}:
 target=root/r['target'];diagnosis=root/r['diagnosis']
 assert target.resolve().is_relative_to((root/'results/global_task_role_fixed_best_20261004_v1').resolve())
 assert target.name=='gallery_features.pt' and str(target) not in seal['artifact_sha256']
 assert sha(diagnosis)==r['diagnosis_sha256'] and json.loads(diagnosis.read_text())['status']=='COMPLETE'
 exists=target.is_file()
 if exists:assert target.stat().st_size==r['bytes'] and sha(target)==r['sha256']
 rows.append(dict(r,present_before=exists))
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows),indent=2)+'\\n')
before=shutil.disk_usage(root).free
for r in rows:
 if r['present_before']:(root/r['target']).unlink()
 assert not (root/r['target']).exists()
after=shutil.disk_usage(root).free
record=dict(status='CONSUMED_HISTORICAL_GALLERY_ONLY_RETIREMENT_COMPLETE',at=datetime.now().astimezone().isoformat(),rows=rows,
 retired_files=sum(r['present_before'] for r in rows),retired_bytes=sum(r['bytes'] for r in rows if r['present_before']),
 disk_free_before=before,disk_free_after=after,
 boundary='Only already diagnosed historical gallery_features caches, not in current271 dependencies. Every formal best, initializer, original evaluation distances, query features, DIAGNOSIS and source retained. No new model, current cache, failure weight, recursive deletion or scientific result change.')
(journal/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
assert not packet.exists();packet.mkdir();(packet/'REMOTE_SOURCE.py').write_bytes(code.encode())
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n').encode())
assert status==0,error.decode();record=json.loads(data)
print(json.dumps({k:record[k] for k in ('status','at','retired_files','retired_bytes','disk_free_before','disk_free_after')}))
