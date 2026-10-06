from pathlib import Path
from datetime import datetime
import json,hashlib
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/signal_selection_reference_v1_20261006_870';launch=root/'logs/signal_selection_reference_launch_20261006_870'
state=json.loads((campaign/'campaign.json').read_text());matrix=json.loads((campaign/'accepted_matrix.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==1 and matrix['accepted']==9
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
for n in ('LAUNCH.json','CHILD.json'):
 r=json.loads((launch/n).read_text());p=Path('/proc')/str(r['pid']);assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
files=[campaign/n for n in ('campaign.json','accepted_matrix.json','manifest.json','endpoint_origins.json','batch_metadata_paths.json','report.log','retired_m0.jsonl')];files.extend(launch/n for n in ('LAUNCH.json','CHILD.json','EXIT.json','stderr.txt','stdout.txt'))
result=dict(at=datetime.now().astimezone().isoformat(),report_log=(campaign/'report.log').read_text(),rows=matrix['rows'],report_step=state['report_step'],files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
print(json.dumps(result))
