from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
root = Path('/data/gaob/Re-ID/Trifusion')
path = root/'.git/cross_depth_observer_660_20260929.json'
assert not path.exists()
with (root/'.git/cross_depth_observer_660_20260929.log').open('x') as log:
    process = subprocess.Popen([sys.executable,'-B',str(root/'.git/observe_cross_depth_660.py')],
                               cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record = {'status':'DURABLE_ONE_SHOT_OBSERVER_STARTED','at':datetime.now().astimezone().isoformat(),
          'pid':process.pid,'not_before':'2026-09-29T21:22:40+08:00',
          'boundary':'Observer only; no training command/retry/update/checkpoint selection.'}
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
