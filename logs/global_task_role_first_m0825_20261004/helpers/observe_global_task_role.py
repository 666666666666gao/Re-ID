"""Read an existing campaign at one estimated milestone; never launch/restart."""
from datetime import datetime
from pathlib import Path
import json
import sys
import time
import paramiko

delay=int(sys.argv[1])
assert 0<=delay<=300
time.sleep(delay)
packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_observations')/datetime.now().strftime('%H%M%S_%f')
packet.mkdir(parents=True)
code='''
from datetime import datetime
from pathlib import Path
import json,subprocess,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/global_task_role_v1_20261004_824'
launch=root/'logs/global_task_role_launch_20261004_824'
record=json.loads((launch/'LAUNCH.json').read_text())
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else None
process=Path('/proc/'+str(record['pid'])+'/stat')
proc=None
if process.exists():
 text=process.read_text();fields=text[text.rfind(')')+2:].split();assert int(fields[19])==record['start_ticks'];proc=dict(pid=record['pid'],state=fields[0],start_ticks=int(fields[19]))
exit_record=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
jobs=[] if state is None else [{k:row.get(k) for k in ('dataset','variant','phase','status','exit_code')} for row in state['jobs']]
logs={p.name:p.read_text().splitlines()[-12:] for p in campaign.glob('*.log')}
console=(launch/'console.log').read_text().splitlines()[-16:] if (launch/'console.log').exists() else []
active=[]
if state:
 rows=state['preparation']+[step for job in state['jobs'] for step in job.get('steps',[])]
 for row in rows:
  if row.get('status')=='RUNNING':
   p=Path('/proc/'+str(row['pid'])+'/stat')
   stat=p.read_text() if p.exists() else None
   active.append(dict(pid=row['pid'],mode=row['mode'],process_stat=stat,command=row['command']))
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status=None if state is None else state['status'],
 controller_pid=None if state is None else state['controller_pid'],jobs=jobs,supervisor=proc,exit=exit_record,
 active_processes=active,logs=logs,console_tail=console,gpu_observation=gpu,disk_free_bytes=shutil.disk_usage(root).free)))
'''
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(60)
data,error=out.read(),err.read()
status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data)
(packet/'stderr.txt').write_bytes(error)
(packet/'remote_observer.py').write_text(code,encoding='utf-8')
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n',encoding='utf-8')
client.close()
assert status==0,error.decode()
print(json.dumps(dict(packet=str(packet),**json.loads(data))))
