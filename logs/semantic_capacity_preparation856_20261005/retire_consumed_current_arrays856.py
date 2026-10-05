"""Only closed accepted diagnostic caches, preserving raw controls and failures."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/semantic_capacity_space856')
assert not packet.exists();packet.mkdir()
code='''from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/consumed_current_diagnostic_array_retirement_20261005_856'
assert not journal.exists()
assert json.loads((root/'logs/deployment_metric_pending_fixed_best_launch_20261005_854/EXIT.json').read_text())['exit_code']==0
dependencies=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for folder,models in (('results/deployment_metric_fixed_best_five_20261005_850',('RGBNT201_semantic','RGBNT201_native')),
                      ('results/deployment_metric_fixed_best_pending100_20261005_854',('RGBNT100_semantic','RGBNT100_native'))):
 for model in models:
  directory=root/folder/model
  diagnosis=directory/'DIAGNOSIS.json'
  record=json.loads(diagnosis.read_text());assert record['status']=='COMPLETE'
  for name in ('query_features.pt','gallery_features.pt','distances.pt'):
   target=directory/name
   assert target.resolve().is_relative_to(directory.resolve()) and str(target) not in dependencies
   info=record['artifacts'][name]
   assert target.is_file() and target.stat().st_size==info['bytes'] and sha(target)==info['sha256']
   rows.append(dict(path=str(target),**info,diagnosis_sha256=sha(diagnosis)))
assert len(rows)==12
journal.mkdir()
(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows),indent=2)+'\\n')
before=shutil.disk_usage(root).free
for r in rows:
 Path(r['path']).unlink();assert not Path(r['path']).exists()
after=shutil.disk_usage(root).free
value=dict(status='CONSUMED_ACCEPTED_DIAGNOSTIC_CACHES_RETIRED',at=datetime.now().astimezone().isoformat(),
 rows=rows,retired_files=len(rows),retired_bytes=sum(r['bytes'] for r in rows),disk_free_before=before,disk_free_after=after,
 boundary='Only 12 consumed query/gallery/diagnostic-distance files for four accepted normalized-metric diagnoses. DIAGNOSIS and original formal distances/bests retained, raw187 controls and MSVR failed saved arrays preserved. Old diagnostic cache revalidation is retired, no result rewrite or recursive deletion.')
(journal/'RETIREMENT.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
'''
compile(code,'remote_retirement.py','exec')
(packet/'REMOTE_SOURCE.py').write_bytes(code.encode())
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n').encode())
assert status==0,error.decode()
value=json.loads(data)
print(json.dumps({key:value[key] for key in ('status','at','retired_files','retired_bytes','disk_free_before','disk_free_after')}))
