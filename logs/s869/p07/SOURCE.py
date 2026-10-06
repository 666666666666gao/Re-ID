from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');launch=root/'logs/signal_selection_reference_launch_20261006_868';campaign=root/'logs/signal_selection_reference_v1_20261006_868'
records=[json.loads((launch/n).read_text()) for n in ('LAUNCH.json','CHILD.json')]
processes=[]
for r in records:
 p=Path('/proc')/str(r['pid']);processes.append(dict(pid=r['pid'],present=p.exists(),start_ticks=int((p/'stat').read_text().split()[21]) if p.exists() else None))
state=json.loads((campaign/'campaign.json').read_text());jobs=[j for j in state['jobs'] if j['dataset']=='RGBNT100']
started=[]
for job in jobs:
 for step in job.get('steps',[]):
  if 'pid' in step:
   p=Path('/proc')/str(step['pid']);started.append(dict(**step,present=p.exists(),actual_start_ticks=int((p/'stat').read_text().split()[21]) if p.exists() else None))
training=[]
for folder in (root/'trained-model').glob('signal_selection_reference_v1_20261006_868_*RGBNT100'):
 if (folder/'training.json').is_file():
  r=json.loads((folder/'training.json').read_text());training.append(dict(folder=str(folder),status=r['status'],completed_epochs=len(r['history']),last_epoch=r['history'][-1] if r['history'] else None,m0=r.get('m0'),selection_reference=r.get('selection_reference')))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),exit=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').is_file() else None,processes=processes,status=state['status'],report_invocations=state['report_invocations'],accepted_full=len([j for j in state['jobs'] if j['phase']=='full' and j['status']=='COMPLETE']),jobs=jobs,started_steps=started,training=training,stderr=(launch/'stderr.txt').read_text()[-5000:],free_bytes=shutil.disk_usage(root).free,boundary='One passiveoriginal868transition observation afterlastactual+180s. Newstepmayexistwithoutstatus/pid becausequeuepreflightasserts beforestartedfields; onlyactuallyrecordedPID handles checked. NoNN/eval/report/retry/otherproject/GPU/25/power/temp action.')))
