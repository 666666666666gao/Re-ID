from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=Path('/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_five_20261005_850');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_fixed_best_five_launch_20261005_850')
record=json.loads((launch/'LAUNCH.json').read_text())
assert record['pid']==444043 and record['start_ticks']==41860134
state=json.loads((output/'campaign.json').read_text())
proc=Path('/proc/'+str(record['pid'])+'/stat');process_stat=proc.read_text() if proc.exists() else None
if process_stat is not None:assert int(process_stat[process_stat.rfind(')')+2:].split()[19])==record['start_ticks']
terminal=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
logs={p.name:p.read_text().splitlines()[-4:] for p in output.glob('*.log')}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status=state['status'],jobs=state['jobs'],
 process_stat=process_stat,exit=terminal,logs=logs,disk_free_bytes=shutil.disk_usage(root).free)))
