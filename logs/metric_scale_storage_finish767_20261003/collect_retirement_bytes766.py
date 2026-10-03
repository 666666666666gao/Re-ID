"""Receive exact remote retirement bytes after the transport digest failure."""
from pathlib import Path
from datetime import datetime
import hashlib,json,paramiko

base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
spec=json.loads((base/'FINISH_SPEC.json').read_bytes())
path=base/'retirement/REMOTE_RETIREMENT_BYTES.json';assert not path.exists()
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
sftp=client.open_sftp()
with sftp.open(spec['retirement_receipt'],'rb') as stream:data=stream.read()
local=(base/'retirement/RETIREMENT.json').read_bytes()
assert data==local.replace(b'\r\n',b'\n') and json.loads(data)==json.loads(local)
path.write_bytes(data)
_,stdout,stderr=client.exec_command("/usr/bin/python3 -B -c \"from pathlib import Path; import json; print(json.dumps({'finish_exists':Path('/data/gaob/Re-ID/Trifusion/logs/metric_feature_scale_storage_finish766_20261003').exists(),'launcher_log_exists':Path('/data/gaob/Re-ID/Trifusion/logs/metric_storage_finish_source766_20261003/launcher.log').exists(),'launch_receipt_exists':Path('/data/gaob/Re-ID/Trifusion/logs/metric_storage_finish_source766_20261003/LAUNCH.json').exists()}))\"")
out,error=stdout.read(),stderr.read();assert stdout.channel.recv_exit_status()==0,error.decode()
state=json.loads(out);assert not any(state.values())
record={'recorded_at':datetime.now().astimezone().isoformat(),'status':'REMOTE_LF_LOCAL_CRLF_SAME_JSON_CONFIRMED_NO_LAUNCH',
 'remote_bytes':len(data),'remote_sha256':hashlib.sha256(data).hexdigest(),
 'local_previous_sha256':hashlib.sha256(local).hexdigest(),'remote_path':spec['retirement_receipt'],
 'launch_state':state,'original_failed_transport':'deploy_metric_storage_finish766.py remote line7 before Popen'}
(base/'retirement/TRANSPORT_DIAGNOSIS.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
sftp.close();client.close();print(json.dumps(record,indent=2))
