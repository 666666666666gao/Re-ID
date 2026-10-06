from pathlib import Path
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
