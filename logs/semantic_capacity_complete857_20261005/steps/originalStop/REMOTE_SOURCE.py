from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_control_v1_20261005_856'
launch=root/'logs/semantic_capacity_launch_20261005_856'
record=json.loads((launch/'LAUNCH.json').read_text())
assert record['pid']==642951 and record['start_ticks']==42374341
state=json.loads((campaign/'campaign.json').read_text())
process=Path('/proc/'+str(record['pid'])+'/stat')
if process.exists():
 stat=process.read_text();assert int(stat[stat.rfind(')')+2:].split()[19])==record['start_ticks']
exit=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').is_file() else None
logs={p.name:p.read_text()[-4000:] for p in [launch/'console.log',*campaign.glob('*.log')] if p.is_file()}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),campaign=state,exit=exit,logs=logs,
 acceptance={p.name:json.loads(p.read_text()) for p in (campaign/'acceptance').glob('*.json')},
 disk_free_bytes=shutil.disk_usage(root).free)))
