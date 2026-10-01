"""Record one durable progress milestone without reading interim retrieval scores."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
DUE = datetime(2026, 10, 1, 11, 45, tzinfo=timezone(timedelta(hours=8)))
STATUS = ROOT / 'logs/visual_start_roles_observer_1145_20261001.json'
SNAPSHOT = CAMPAIGN / 'milestone_1145_snapshot.json'
assert not STATUS.exists() and not SNAPSHOT.exists()

while datetime.now().astimezone() < DUE:
    remaining = (DUE - datetime.now().astimezone()).total_seconds()
    state = {'status': 'WAITING', 'pid': os.getpid(), 'due_at': DUE.isoformat(),
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'scope': 'Wall-clock waiting only; no model execution or interim score reads.'}
    STATUS.write_text(json.dumps(state, indent=2) + '\n')
    time.sleep(min(240, remaining))

manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
           for path, digest in manifest['source_sha256'].items())
campaign = json.loads((CAMPAIGN / 'campaign.json').read_text())
rows = []
for row in campaign['jobs']:
    item = dict(row)
    child = CAMPAIGN / (CAMPAIGN.name + '_visual_start_' + row['variant'] + '_' + row['dataset'])
    if (child / 'campaign.json').exists():
        child_state = json.loads((child / 'campaign.json').read_text())
        item['child'] = child_state
        for stage in child_state['jobs']:
            if stage['status'] == 'RUNNING':
                process = Path('/proc') / str(stage['pid'])
                item['running_process_exists'] = process.exists()
                if process.exists():
                    item['running_command'] = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if stage['mode'] == 'm0' and stage['status'] == 'COMPLETE':
                item['m0'] = json.loads((Path(stage['output_dir']) / 'training.json').read_text())
    rows.append(item)
record = {'observed_at': datetime.now().astimezone().isoformat(),
          'campaign': campaign, 'rows': rows, 'source_count_verified': len(manifest['source_sha256']),
          'disk_free_bytes': shutil.disk_usage(ROOT).free,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits']).decode(),
          'scope': 'Progress, actual M0 receipts and process existence; no interim retrieval selection.'}
SNAPSHOT.write_text(json.dumps(record, indent=2) + '\n')
state.update(status='COMPLETE', exit_code=0, completed_at=datetime.now().astimezone().isoformat(),
             snapshot=str(SNAPSHOT), snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest())
STATUS.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state))
