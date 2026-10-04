from pathlib import Path
from datetime import datetime
import json,paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_fixed_best_seal')
target=base/'TIMEOUT_OBSERVATION.json'
assert not target.exists()
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=r'''
from pathlib import Path
from datetime import datetime
import json,os
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or not (p/'cmdline').exists():continue
 if (p/'cmdline').stat().st_uid!=os.getuid():continue
 b=(p/'cmdline').read_bytes()
 if b'/usr/bin/python3\x00-B\x00-\x00' in b or b'diagnose_global_task_role_best.py' in b:
  stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
  rows.append(dict(pid=int(p.name),state=fields[0],ppid=int(fields[1]),start_ticks=int(fields[19]),command=b.decode().replace('\x00',' ')))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),processes=rows,boundary='Static seal observer after local SSH read timeout; no model or original validation rerun.')))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();data,error=stdout.read(),stderr.read()
assert stdout.channel.recv_exit_status()==0,error.decode()
client.close();result=json.loads(data)
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
failure=dict(status='LOCAL_STATIC_SEAL_READ_TIMEOUT',at=datetime.now().astimezone().isoformat(),failed_line=95,exception='Paramiko stdout.read TimeoutError after 60s; original remote validation does not persist result.',local_seal_created=False,model_inference_started=False,observation=target.name)
(base/'LOCAL_TIMEOUT.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
