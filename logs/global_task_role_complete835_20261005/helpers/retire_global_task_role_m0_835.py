"""Retire six completed M0 probes after full intake and diagnostic input sealing."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet=base/'global_task_role_m0_retirement835'
assert not packet.exists()
assert json.loads((base/'global_task_role_complete_report/EXIT.json').read_bytes())['exit_code']==0
assert json.loads((base/'global_task_role_fixed_best_seal/EXIT.json').read_bytes())['exit_code']==0
record=json.loads((base/'global_task_role_complete_report/stdout.json').read_bytes())
sealpath=repo/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal=json.loads(sealpath.read_bytes())
sealsha=hashlib.sha256(sealpath.read_bytes()).hexdigest()
assert len(seal['rows'])==9 and len(seal['source_sha256'])==332
targets=[dict(item['m0_reload_probe.pth']) for item in record['artifacts'].values()]
assert len(targets)==6 and not set(t['path'] for t in targets).intersection(seal['artifact_sha256'])
code=f'''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
bound={seal!r}
targets={targets!r}
assert all(sha(root/n)==d for n,d in bound['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in bound['artifact_sha256'].items())
state=json.loads((root/'logs/global_task_role_v1_20261004_824/campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs']) and len(state['jobs'])==12
for t in targets:
 p=Path(t['path'])
 assert p.resolve()==p and p.resolve().is_relative_to((root/'trained-model').resolve())
 assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('global_task_role_v1_20261004_824_m0_')
 assert p.stat().st_size==t['bytes'] and sha(p)==t['sha256']
 receipt=json.loads((p.parent/'training.json').read_text())
 assert receipt['status']=='M0_PASS' and receipt['m0']['reload_probe_sha256']==t['sha256']
 assert receipt['production_m0_diagnostics']['effective_optimizer_updates']==8
journal=root/'logs/global_task_role_m0_retirement835_20261005'
assert not journal.exists()
journal.mkdir()
before=shutil.disk_usage(root).free
(journal/'PLAN.json').write_text(json.dumps(dict(status='SIX_ORIGINAL_M0_PROBES_DEPENDENCIES_CLOSED',at=datetime.now().astimezone().isoformat(),targets=targets,input_seal_sha256={sealsha!r},formal_best_and_all_sealed_artifacts_retained=True),indent=2)+'\\n')
for t in targets:
 p=Path(t['path']);assert sha(p)==t['sha256'];p.unlink();assert not p.exists()
 with (journal/'RETIREMENT.jsonl').open('a') as f:f.write(json.dumps(dict(t,retired_at=datetime.now().astimezone().isoformat()))+'\\n')
assert all(sha(root/n)==d for n,d in bound['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in bound['artifact_sha256'].items())
result=dict(status='SIX_COMPLETED_CURRENT_M0_PROBES_RETIRED',completed_at=datetime.now().astimezone().isoformat(),retired_files=6,retired_bytes=sum(t['bytes'] for t in targets),disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,all332_source_and_sealed_artifacts_unchanged=True,input_seal_sha256={sealsha!r},boundary='Only exact successful M0 probes retired after all6 full50/first-strict and original once CPU report intake. Formal best, original distances, M0 text, controls and diagnosis inputs retained. No model or report replay; no power/temperature action. Probe direct replay no longer available.')
(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(dict(result=result,journal={{p.name:p.read_text() for p in journal.iterdir()}})))
'''
compile(code,'remote_retire_current_m0.py','exec')
packet.mkdir()
(packet/'remote_retire_current_m0.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(180)
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code,at=datetime.now().astimezone().isoformat()))+'\n',encoding='utf-8')
client.close();assert exit_code==0,error.decode()
result=json.loads(data)
for name,content in result['journal'].items():(packet/name).write_text(content,encoding='utf-8')
print(json.dumps(result['result']))
