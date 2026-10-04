"""Retire exact old accepted M0 binaries; retain receipts and every formal best."""
from pathlib import Path
from datetime import datetime
import hashlib,json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
read=base/'historical_probe_receipts841'
assert json.loads((read/'EXIT.json').read_bytes())['exit_code']==0
record=json.loads((read/'stdout.json').read_bytes());rows=record['rows']
assert len(rows)==70 and sum(r['bytes'] for r in rows)==807017706
assert all(json.loads(r['training_text'])['status']=='M0_PASS' and json.loads(r['training_text'])['m0']['reload_probe_sha256']==r['probe_sha256'] for r in rows)
packet=base/'historical_probe_retirement841b';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=ROWS
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
scope=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text())
assert len(scope['source_sha256'])==339
assert all(sha(root/name)==digest for name,digest in scope['source_sha256'].items())
seal=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['artifact_sha256'])==187 and all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
protected=set(seal['artifact_sha256'])
active=json.loads((root/'logs/deployment_metric_role_v1_20261005_837/campaign.json').read_text())['active_command']
assert active['status']=='RUNNING'
active_command=' '.join(active['command'])
for row in rows:
 p=Path(row['path']);assert p.resolve().is_relative_to(root/'trained-model') and p.name=='m0_reload_probe.pth'
 assert row['path'] not in protected and '20261005_837' not in str(p)
 assert p.stat().st_size==row['bytes'] and sha(p)==row['probe_sha256']
 receipt=p.parent/'training.json';assert sha(receipt)==row['training_sha256']
 value=json.loads(receipt.read_text());assert value['status']=='M0_PASS' and value['checkpoint'] is None and value['best_epoch'] is None
 assert datetime.fromisoformat(value['completed_at']).date()<datetime.now().astimezone().date()
 assert value['m0']['reload_probe_sha256']==row['probe_sha256'] and value['m0']['reload_max_abs_difference']==0
 assert str(p.parent) not in active_command
archive=root/'logs/historical_m0_probe_retirement_20261005_841';assert not archive.exists();archive.mkdir()
certificate=dict(status='EXACT70_HISTORICAL_M0_PROBES_VERIFIED_BEFORE_RETIREMENT',at=datetime.now().astimezone().isoformat(),
 rows=rows,expected_bytes=sum(r['bytes'] for r in rows),source_files_verified=339,current_control_artifacts_verified=187,
 boundary='User-authorized useless weight retirement. Each old M0_PASS/no formal checkpoint/probe SHA verified;not current inputs or current project active command directory. All formal best and receipts retained. Historical direct M0 binary replay now retired;no model/power-temperature action.')
accept=archive/'ACCEPTANCE.json';accept.write_text(json.dumps(certificate,indent=2)+'\\n')
before=shutil.disk_usage(root).free
with (archive/'RETIREMENT.jsonl').open('x') as journal:
 for row in rows:
  Path(row['path']).unlink()
  journal.write(json.dumps(dict(path=row['path'],bytes=row['bytes'],probe_sha256=row['probe_sha256'],
                  receipt_sha256=row['training_sha256'],acceptance_sha256=sha(accept),at=datetime.now().astimezone().isoformat()))+'\\n');journal.flush()
assert all(not Path(r['path']).exists() and sha(Path(r['path']).parent/'training.json')==r['training_sha256'] for r in rows)
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
summary=dict(status='EXACT70_HISTORICAL_M0_PROBES_RETIRED',at=datetime.now().astimezone().isoformat(),count=len(rows),
 retired_bytes=certificate['expected_bytes'],disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,
 acceptance_sha256=sha(accept),journal_sha256=sha(archive/'RETIREMENT.jsonl'))
(archive/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\\n')
print(json.dumps(dict(summary=summary,files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in archive.iterdir()})))
'''.replace('ROWS',repr(rows))
compile(code,'remote_historical_probe_retirement841.py','exec')
packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode()
value=json.loads(data)
for name,row in value['files'].items():
 target=packet/'received'/Path(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row['text'].encode('utf-8'))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
(packet/'SUMMARY.json').write_text(json.dumps(value['summary'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(value['summary']))
