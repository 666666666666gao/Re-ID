from datetime import datetime
from pathlib import Path
import hashlib,json,paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
launch=json.loads((base/'cpu_boundary_deploy779/stdout.json').read_bytes())
sources=launch['source_sha256']
assert len(sources)==304 and launch['port']==2026
target=base/'cpu_boundary_terminal780'
assert not target.exists()
target.mkdir()
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=Path({launch['campaign']!r})
job=json.loads((campaign/'JOB.json').read_text())
assert job['status'] in ('FAILED','COMPLETE')
assert not Path('/proc/'+str({launch['controller_pid']})).exists()
assert not Path('/proc/'+str(job['child_pid'])).exists()
expected={sources!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in expected.items())
assert not list(campaign.rglob('*.pth')) and not list(campaign.rglob('*.npy'))
paths=[p for p in campaign.rglob('*') if p.is_file()]
assert all(p.suffix in ('.json','.py','.log','.txt') for p in paths)
measured=json.loads((campaign/'observations/MEASURED.json').read_text())
assert measured['optimizer_updates']==0 and measured['forward_backwards']==2
fields={{}}
for row in measured['metadata_differences']:
 for name in row['changed']:fields[name]=fields.get(name,0)+1
print(json.dumps({{'at':datetime.now().astimezone().isoformat(),'status':'DIAGNOSTIC_TERMINAL_NOT_TRAINING',
 'job':job,'primary':{{str(p.relative_to(root)):{{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}} for p in paths}},
 'source_sha256':expected,'summary':{{'outputs':measured['outputs'],'boundaries':measured['boundaries'],
 'gradient_parameters':len(measured['gradients']),
 'failed_gradients':sum(row.get('fixed_gate_pass') is False for row in measured['gradients'].values()),
 'packs':len(measured['packs']),'unpacks':len(measured['unpacks']),'metadata_changed_fields':fields,
 'post_checks':measured['post_checks'],'memory':measured['memory']}},
 'boundary':'Read-only terminal intake. Two diagnostic backwards/zero updates/no weights; originalv4FAIL retained; no M0/formal/official/scorer/restart.'}},indent=2))
'''
i,o,e=client.exec_command('/usr/bin/python3 -B -')
i.write(code)
i.channel.shutdown_write()
data,error=o.read(),e.read()
rc=o.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
assert rc==0,error.decode()
record=json.loads(data)
sftp=client.open_sftp()
for name,row in record['primary'].items():
    path=target/'primary'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    sftp.get('/data/gaob/Re-ID/Trifusion/'+name,str(path))
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for name,digest in sources.items():
    path=target/'_source'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    sftp.get('/data/gaob/Re-ID/Trifusion/'+name,str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
sftp.close()
client.close()
(target/'INTAKE.json').write_bytes(data)
(target/'EXIT.json').write_text(json.dumps({'exit_code':0,'at':datetime.now().astimezone().isoformat(),'boundary':'Collector only; diagnostic child exit remains in JOB'})+'\n')
print(json.dumps({'status':record['status'],'job_status':record['job']['status'],'diagnostic_exit_code':record['job']['exit_code'],
                  'summary':record['summary'],'primary_files':len(record['primary']),'source_files':len(sources)},indent=2))
