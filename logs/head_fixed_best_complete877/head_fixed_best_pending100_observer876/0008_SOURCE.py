from pathlib import Path
from datetime import datetime
import json,shutil
j=Path('/data/gaob/Re-ID/Trifusion/logs/independent_role_heads_fixed_best_pending100_launch_20261007_876')
out=Path('/data/gaob/Re-ID/Trifusion/results/independent_role_heads_fixed_best_pending100_20261007_876')
def read(p):return json.loads(p.read_text()) if p.exists() else None
launch=read(j/'LAUNCH.json');child=read(j/'CHILD.json');end=read(j/'EXIT.json')
campaign=dict(status='COMPLETE' if end and end['exit_code']==0 else 'FAILED' if end else 'RUNNING',jobs=[dict(dataset='RGBNT100',status='COMPLETE' if (out/'RGBNT100_semantic/DIAGNOSIS.json').exists() else 'PENDING')])
pid=launch['pid'];alive=Path('/proc')/str(pid)
owner_alive=alive.exists() and int((alive/'stat').read_text().split()[21])==launch['start_ticks']
terminal=end is not None and not owner_alive
assert owner_alive or terminal,'Supervisor absent without exit receipt'
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),terminal=terminal,
    owner_alive=owner_alive,child=child,exit=end,campaign=campaign,
    free_bytes=shutil.disk_usage(out.parent).free,
    controller_log_tail=(j/'controller.log').read_text()[-6000:] if (j/'controller.log').exists() else '',
    endpoint_log_tails={p.name:p.read_text()[-3500:] for p in out.glob('*_semantic.log')}
        if out.exists() else {})))
