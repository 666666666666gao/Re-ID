from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
output=Path('/data/gaob/Re-ID/Trifusion/results/deployment_metric_fixed_best_pending100_20261005_854');launch=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_pending_fixed_best_launch_20261005_854')
record=json.loads((launch/'LAUNCH.json').read_text())
assert record['pid']==504070 and record['start_ticks']==42015613
state=json.loads((output/'campaign.json').read_text())
proc=Path('/proc/'+str(record['pid'])+'/stat');stat=proc.read_text() if proc.exists() else None
if stat is not None:assert int(stat[stat.rfind(')')+2:].split()[19])==record['start_ticks']
terminal=json.loads((launch/'EXIT.json').read_text()) if (launch/'EXIT.json').exists() else None
logs={p.name:p.read_text().splitlines()[-4:] for p in output.glob('*.log')}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),status=state['status'],jobs=state['jobs'],
 process_stat=stat,exit=terminal,logs=logs,disk_free_bytes=shutil.disk_usage(root).free)))
