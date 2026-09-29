from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
root=Path('/data/gaob/Re-ID/Trifusion')
source=(root/'.git/observe_cross_depth_660.py').read_text()
source=source.replace('2026-09-29T21:22:40+08:00','2026-09-29T21:42:00+08:00')
source=source.replace('cross_depth_progress_660_20260929.json','cross_depth_progress_661_20260929.json')
target=root/'.git/observe_cross_depth_661.py'
assert not target.exists()
target.write_text(source)
path=root/'.git/cross_depth_observer_661_20260929.json'
assert not path.exists()
with (root/'.git/cross_depth_observer_661_20260929.log').open('x') as log:
    process=subprocess.Popen([sys.executable,'-B',str(target)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'status':'DURABLE_ONE_SHOT_OBSERVER_STARTED','at':datetime.now().astimezone().isoformat(),
        'pid':process.pid,'not_before':'2026-09-29T21:42:00+08:00',
        'estimate':'First current MSVR full50 approximately21:44 and201 approximately21:49 from actual21:22 epoch progress; '
                   '100 slower. Check a few minutes before earliest estimated endpoint, not constant epoch polling.',
        'boundary':'Read-only observer; no retrain/retry/checkpoint selection/changes.'}
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
