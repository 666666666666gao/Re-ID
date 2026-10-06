from pathlib import Path
import hashlib,json,paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'signal_selection_failed_campaign865';assert not packet.exists();packet.mkdir()
code=r'''from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/signal_selection_reference_v1_20261006_865'
launch=root/'logs/signal_selection_reference_launch_20261006_865'
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
state=json.loads((campaign/'campaign.json').read_text())
assert len(state['preparation'])==3 and all(s['exit_code']==0 for s in state['preparation'])
assert state['failed_job']['phase']=='m0' and state['report_invocations']==0
for folder in (root/'trained-model').glob('signal_selection_reference_v1_20261006_865_*'):assert not folder.exists()
files=[p for folder in (campaign,launch) for p in folder.rglob('*') if p.is_file()]
assert all(p.suffix in ('.json','.log','.txt','.py') for p in files)
print(json.dumps(dict(status='FAILED_BEFORE_MODEL_BUILD_OR_OPTIMIZER',state=state,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})))
'''
compile(code,'failed_campaign865_intake','exec');(packet/'SOURCE.py').write_text(code)
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error);assert status==0,error.decode()
d=json.loads(data);s=c.open_sftp()
for name,row in d['files'].items():
 p=packet/'received'/name;p.parent.mkdir(parents=True,exist_ok=True);s.get('/data/gaob/Re-ID/Trifusion/'+name,str(p));assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
s.close();c.close();print(json.dumps(dict(status=d['status'],files=len(d['files']),prepare_pass=3,m0_updates=0,formal=0)))
