from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');launch=root/'logs/signal_selection_reference_launch_20261006_869';campaign=root/'logs/signal_selection_reference_v1_20261006_869'
def alive(r):
 p=Path('/proc')/str(r['pid'])
 return p.exists() and int((p/'stat').read_text().split()[21])==r['start_ticks']
parents={n:dict(json.loads((launch/n).read_text()),actual_alive=alive(json.loads((launch/n).read_text()))) for n in ('LAUNCH.json','CHILD.json') if (launch/n).is_file()}
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').is_file() else None
running=[]
if state:
 for job in state['jobs']:
  if job.get('reuse_no_execution'):continue
  for step in job['steps']:
   if step.get('status')!='RUNNING':continue
   row=dict(dataset=job['dataset'],selection=job['selection'],phase=job['phase'],mode=step['mode'],pid=step['pid'],start_ticks=step['start_ticks'],actual_alive=alive(step),started_at=step['started_at'])
   folder=root/'trained-model'/f'{campaign.name}_{job["phase"]}_{job["selection"]}_{job["dataset"]}'
   training=folder/'training.json'
   if training.is_file():
    data=json.loads(training.read_text());row.update(recorded_status=data['status'],epochs_completed=len(data['history']),steps_completed=sum(h['steps'] for h in data['history']),epoch_seconds=[h['seconds'] for h in data['history'][-5:]])
   running.append(row)
result=dict(at=datetime.now().astimezone().isoformat(),parents=parents,exit=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').is_file() else None,state_status=state['status'] if state else None,accepted=sum(j['phase']=='full' and j['status']=='COMPLETE' for j in state['jobs']) if state else 0,m0_accepted=sum(j['phase']=='m0' and j['status']=='COMPLETE' for j in state['jobs']) if state else 0,report_invocations=state['report_invocations'] if state else 0,running=running,free_bytes=shutil.disk_usage(root).free,stderr=(launch/'stderr.txt').read_text())
print(json.dumps(result))
