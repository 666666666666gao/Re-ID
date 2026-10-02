from pathlib import Path
import json
import paramiko

output=Path('C:/Users/gb/.codex_tmp/training_feature_scale_hardware754')
assert not output.exists()
output.mkdir()
code="""
from datetime import datetime
import json,subprocess
command=['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader,nounits']
print(json.dumps({'observed_at':datetime.now().astimezone().isoformat(),'port':2026,
 'command':command,'hardware':subprocess.check_output(command,text=True),
 'boundary':'Read-only hardware inventory; no model, benchmark, restart or configuration change.'}))
"""
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
client.close()
(output/'stdout.txt').write_bytes(data)
(output/'stderr.txt').write_bytes(error)
assert exit_code==0,error.decode()
record=json.loads(data)
(output/'HARDWARE.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record,indent=2))
