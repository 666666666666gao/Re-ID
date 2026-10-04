"""Read the original terminal queue and failed native M0; do not rerun it."""
from pathlib import Path,PurePosixPath
from datetime import datetime
import hashlib,json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'deployment_metric_original_terminal842';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/deployment_metric_role_v1_20261005_837';parent=root/'logs/deployment_metric_role_launch_20261005_837'
state=json.loads((campaign/'campaign.json').read_text());terminal=json.loads((parent/'EXIT.json').read_text())
assert state['status']=='FAILED' and state['active_command'] is None and terminal['exit_code']==1
failed=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','native','m0'))
assert failed['status']=='FAILED' and failed['exit_code']==1
names=[campaign/'campaign.json',campaign/'manifest.json',campaign/'MSVR310_native_m0.log',campaign/'prepare_MSVR310_native.log',
       campaign/'initialization/MSVR310_native.json',parent/'EXIT.json',parent/'console.log',parent/'supervisor.py',parent/'LAUNCH.json']
m0=root/'trained-model/deployment_metric_role_v1_20261005_837_m0_native_MSVR310'
names.extend(p for p in m0.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.txt'))
files={str(p.relative_to(root)):dict(text=p.read_bytes().decode('utf-8'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in names}
print(json.dumps(dict(status='ORIGINAL_QUEUE_FAILED_NATIVE_M0_INTAKE_ONLY',at=datetime.now().astimezone().isoformat(),
 campaign=state,parent_exit=terminal,failed=failed,failed_m0_files=sorted(p.name for p in m0.iterdir()),files=files,
 disk_free_bytes=shutil.disk_usage(root).free,boundary='Original terminal failure read only. No new model,retry,threshold change or power/temperature query. Completed3formal preserved;pending3 not claimed.')))
'''
compile(code,'remote_terminal842.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode();value=json.loads(data)
for name,row in value['files'].items():
 target=packet/'received'/str(PurePosixPath(name));target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row['text'].encode('utf-8'))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
print(value['files']['logs/deployment_metric_role_v1_20261005_837/MSVR310_native_m0.log']['text'][-7000:])
print(json.dumps(dict(at=value['at'],status=value['status'],failed_m0_files=value['failed_m0_files'],disk_free_bytes=value['disk_free_bytes'])))
