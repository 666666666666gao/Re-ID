from pathlib import Path
from datetime import datetime,timedelta
import json,time,paramiko

base=Path('C:/Users/gb/.codex_tmp')
packet=base/'independent_evidence_draft/signal_selection_launch865'
launch=json.loads((packet/'LAUNCH.json').read_bytes())
deadline=datetime.fromisoformat(launch['started_at'])+timedelta(seconds=180)
assert not (packet/'OBSERVATION.json').exists() and not (packet/'OBSERVER_STARTED.json').exists()
(packet/'OBSERVER_STARTED.json').write_text(json.dumps(dict(started_at=datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),boundary='One observation at first180-second production milestone, no restart or source changes.'),indent=2)+'\n')
time.sleep(max(0,(deadline-datetime.now().astimezone()).total_seconds()))
code=r'''from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
launch=root/'logs/signal_selection_reference_launch_20261006_865'
campaign=root/'logs/signal_selection_reference_v1_20261006_865'
supervisor=json.loads((launch/'LAUNCH.json').read_text())
child=json.loads((launch/'CHILD.json').read_text())
processes=[]
for row in (supervisor,child):
 directory=Path('/proc')/str(row['pid'])
 processes.append(dict(pid=row['pid'],present=directory.exists(),start_ticks=int((directory/'stat').read_text().split()[21]) if directory.exists() else None,command=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace') if directory.exists() else None))
state=json.loads((campaign/'campaign.json').read_text())
steps=[s for job in state['jobs'] for s in job.get('steps',[])]+state['preparation']
running=[s for s in steps if s['status']=='RUNNING']
actual=[]
for row in running:
 directory=Path('/proc')/str(row['pid'])
 actual.append(dict(**row,present=directory.exists(),actual_start_ticks=int((directory/'stat').read_text().split()[21]) if directory.exists() else None,actual_command=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace') if directory.exists() else None))
training=[]
for folder in (root/'trained-model').glob('signal_selection_reference_v1_20261006_865_*'):
 path=folder/'training.json'
 if path.is_file():
  r=json.loads(path.read_text());training.append(dict(folder=str(folder),status=r['status'],completed_epochs=len(r['history']),last_epoch=r['history'][-1] if r['history'] else None,m0=r.get('m0'),selection_reference=r.get('selection_reference')))
result=dict(at=datetime.now().astimezone().isoformat(),supervisor_exit=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').is_file() else None,processes=processes,campaign=state,actual_running_steps=actual,training=training,free_bytes=shutil.disk_usage(root).free,controller_stderr=(launch/'stderr.txt').read_text()[-6000:],last_logs={s['mode']+'_'+s.get('dataset','')+'_'+s.get('selection',''): (campaign/(f'prepare_{s["dataset"]}_{s["selection"]}.log' if s['mode']=='prepare' else f'{s["command"][s["command"].index("--dataset")+1]}_{s["command"][s["command"].index("--selection")+1]}_{s["mode"]}.log')).read_text()[-4000:] for s in running},boundary='Passive original handles only; no NN/temperature/power/25 query or retry.')
print(json.dumps(result))
'''
compile(code,'observer865','exec');(packet/'OBSERVATION_SOURCE.py').write_text(code)
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'OBSERVATION.json').write_bytes(data);(packet/'OBSERVER_STDERR.txt').write_bytes(error)
assert status==0,error.decode()
r=json.loads(data);print(json.dumps(dict(at=r['at'],supervisor_exit=r['supervisor_exit'],processes=r['processes'],running=r['actual_running_steps'],completed=[dict(dataset=j['dataset'],selection=j['selection'],phase=j['phase']) for j in r['campaign']['jobs'] if j['status']=='COMPLETE'],training=r['training'],free_bytes=r['free_bytes'],controller_stderr=r['controller_stderr']),indent=2))
