from datetime import datetime
from pathlib import Path
import json
import sys
import time
import paramiko

delay = int(sys.argv[1])
assert 0 <= delay <= 300
time.sleep(delay)
packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach_fixed_best_observations') / datetime.now().strftime('%H%M%S_%f')
packet.mkdir(parents=True)
code = '''
from datetime import datetime
from pathlib import Path
import json,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/role_input_detach_fixed_best_20261004_v1'
launch=root/'logs/role_input_detach_fixed_best_launch_20261004_v1'
record=json.loads((launch/'LAUNCH.json').read_text())
state=json.loads((output/'campaign.json').read_text()) if (output/'campaign.json').exists() else None
process=Path('/proc/'+str(record['pid'])+'/stat')
proc=None
if process.exists():
 text=process.read_text();fields=text[text.rfind(')')+2:].split();assert int(fields[19])==record['start_ticks'];proc={'pid':record['pid'],'state':fields[0],'start_ticks':int(fields[19])}
exit=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
jobs=[] if state is None else [{key:job.get(key) for key in ('dataset','variant','status','exit_code')} for job in state['jobs']]
console=(launch/'console.log').read_text().splitlines()[-10:] if (launch/'console.log').exists() else []
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'status':None if state is None else state['status'],'jobs':jobs,'process':proc,'exit':exit,'console_tail':console,'gpu_observation':gpu}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, out, err = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(60)
data, error = out.read(), err.read()
status = out.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'remote_observer.py').write_text(code, encoding='utf-8')
(packet / 'EXIT.json').write_text(json.dumps({'exit_code': status}) + '\n', encoding='utf-8')
client.close()
assert status == 0, error.decode()
print(json.dumps({'packet':str(packet), **json.loads(data)}))
