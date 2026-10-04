from pathlib import Path
import hashlib
import json
import paramiko

private=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
plan_path=private/'closed_m3_probe_retirement_preflight_v2_20261004/stdout.json'
plan=json.loads(plan_path.read_bytes())
assert plan['status']=='READY' and len(plan['rows'])==12 and plan['total_bytes']==162103728
packet=private/'closed_m3_probe_retirement_v2_20261004'
assert not packet.exists()
packet.mkdir()
code=r'''
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
plan=PLAN
root=Path(plan['root']).resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert root==Path('/data/gaob/Re-ID/Trifusion')
assert len(plan['rows'])==12
assert all(sha(root/name)==digest for name,digest in plan['current_source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in plan['control_artifacts_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in plan['protected_sha256'].items())
for row in plan['rows']:
    path=Path(row['path'])
    assert not path.is_symlink() and path.resolve()==path
    assert path.parent.parent==root/'trained-model'
    assert path.name=='m0_reload_probe.pth' and path.parent.name.startswith('correspondence_m3_prediction_20260929_m3_')
    assert path.parent.name.endswith('_seed42_m0')
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
journal=root/'logs/closed_m3_probe_retirement_20261004'
assert not journal.exists()
journal.mkdir()
(journal/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
free_before=shutil.disk_usage(root).free
deleted=[]
with (journal/'DELETION_EVENTS.jsonl').open('x') as events:
    for row in plan['rows']:
        path=Path(row['path'])
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        path.unlink()
        event={**row,'deleted':not path.exists(),'deleted_at':datetime.now().astimezone().isoformat()}
        assert event['deleted']
        events.write(json.dumps(event)+'\n')
        events.flush()
        deleted.append(event)
assert all(sha(Path(name))==digest for name,digest in plan['protected_sha256'].items())
assert all(sha(root/name)==digest for name,digest in plan['current_source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in plan['control_artifacts_sha256'].items())
record={'schema':plan['schema'],'status':'RETIRED_EXACT_12_CLOSED_M3_M0_PROBES','completed_at':datetime.now().astimezone().isoformat(),
    'rows':deleted,'retired_bytes':sum(row['bytes'] for row in deleted),'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(root).free,
    'protected_sha256':plan['protected_sha256'],'control_artifact_count':61,'source_file_count':322,'current_required_probes_preserved':5,
    'plan_sha256':sha(journal/'PLAN.json'),'boundary':plan['boundary']}
(journal/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('PLAN',repr(plan),1)
compile(code, '<remote exact M3 retirement>', 'exec')
(packet/'remote.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(90)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
assert record['status']=='RETIRED_EXACT_12_CLOSED_M3_M0_PROBES' and record['retired_bytes']==162103728
sftp=client.open_sftp()
for name in ('PLAN.json','DELETION_EVENTS.jsonl','RETIREMENT.json'):
    sftp.get('/data/gaob/Re-ID/Trifusion/logs/closed_m3_probe_retirement_20261004/'+name,str(packet/name))
sftp.close()
client.close()
assert json.loads((packet/'RETIREMENT.json').read_bytes())==record
print(json.dumps({key:record[key] for key in ('status','completed_at','retired_bytes','free_bytes_before','free_bytes_after','current_required_probes_preserved','boundary')},indent=2))
