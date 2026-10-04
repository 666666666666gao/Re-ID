from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_continuation_20261005_848');parent=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_launch_20261005_848')
state=json.loads((campaign/'campaign.json').read_text())
full=next(j for j in state['jobs'] if (j['variant'],j['phase'])==('native','full'))
terminal=json.loads((parent/'EXIT.json').read_text()) if (parent/'EXIT.json').exists() else None
proc=Path('/proc/'+str(141141));alive=False
if proc.exists():
 stat=(proc/'stat').read_text();alive=int(stat[stat.rfind(')')+2:].split()[19])==41010354
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),supervisor_alive=alive,parent_exit=terminal,
 campaign=state,native_full=full,disk_free_bytes=shutil.disk_usage(root).free,
 console_tail=(parent/'console.log').read_text()[-2500:])))
