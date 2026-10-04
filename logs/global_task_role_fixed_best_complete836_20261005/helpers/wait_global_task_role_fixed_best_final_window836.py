"""One scheduled observation near the registered diagnosis's expected completion."""
from datetime import datetime,timedelta
from pathlib import Path
import json,time,paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
launch=json.loads((base/'global_task_role_fixed_best_launch/stdout.json').read_bytes())
assert launch['status']=='LAUNCHED_ONCE' and launch['pid']==3503728 and launch['start_ticks']==39019368
packet=base/'global_task_role_fixed_best_final_window836'
assert not packet.exists()
started=datetime.now().astimezone()
target=datetime.fromisoformat(launch['at'])+timedelta(minutes=14)
delay=(target-started).total_seconds()
assert delay>0
packet.mkdir()
(packet/'START.json').write_text(json.dumps(dict(started_at=started.isoformat(),target_at=target.isoformat(),delay_seconds=delay,launch_pid=launch['pid'],launch_start_ticks=launch['start_ticks'],boundary='Single scheduled observer near revised final window after observed4/6complete; no remote polling before target or power/temp action.'),indent=2)+'\n',encoding='utf-8')
time.sleep(delay)
code=r'''
from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/global_task_role_fixed_best_20261004_v1'
launch=root/'logs/global_task_role_fixed_best_launch_20261004_v1'
original=json.loads((launch/'LAUNCH.json').read_text())
assert original['pid']==3503728 and original['start_ticks']==39019368
state=json.loads((output/'campaign.json').read_text())
proc=Path('/proc/3503728/stat')
process_stat=proc.read_text() if proc.exists() else None
if process_stat is not None:assert int(process_stat[process_stat.rfind(')')+2:].split()[19])==39019368
exit_record=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
logs={p.name:p.read_text().splitlines()[-4:] for p in output.glob('*.log')}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status=state['status'],jobs=state['jobs'],process_stat=process_stat,exit=exit_record,logs=logs,disk_free_bytes=shutil.disk_usage(root).free,boundary='Read existing diagnosis status only; no model, collection replay or power/temperature query.')))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=exit_code,completed_at=datetime.now().astimezone().isoformat()))+'\n',encoding='utf-8')
client.close();assert exit_code==0,error.decode()
record=json.loads(data)
print(json.dumps({k:record[k] for k in ('at','status','jobs','exit','disk_free_bytes')}))
