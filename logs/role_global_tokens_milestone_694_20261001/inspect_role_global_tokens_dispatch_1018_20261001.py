"""Read actual process handles after the RGBNT100 static and MSVR310 direct endpoint handoff."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'

def live(pid):
    path = Path('/proc') / str(pid)
    return path.exists() and (path / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'

manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 215
assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest['source_sha256'].items())
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
observer = json.loads((ROOT / 'logs/role_global_tokens_observer_1030_20261001.json').read_text())
waiter = json.loads((ROOT / 'logs/role_global_tokens_analysis_waiter_20261001.json').read_text())
record = {
    'observed_at': datetime.now().astimezone().isoformat(),
    'source_count_verified': 215,
    'campaign_status': state['status'],
    'jobs': [{**{k: j[k] for k in ('dataset', 'variant', 'status')},
              **({k: j[k] for k in ('pid', 'gpu')} if 'pid' in j else {}),
              'worker_live': live(j['pid']) if 'pid' in j else None} for j in state['jobs']],
    'controller_live': live(1739616),
    'observer_receipt': observer,
    'observer_wrapper_live': live(2004872),
    'observer_child_live': live(observer['pid']),
    'analysis_waiter_receipt': waiter,
    'analysis_waiter_wrapper_live': live(1888314),
    'analysis_waiter_child_live': live(waiter['pid']),
    'analysis_waiter_status': json.loads((CAMPAIGN / 'analysis_waiter_status.json').read_text())
}
target = ROOT / 'logs/role_global_tokens_dispatch_1018_20261001.json'
assert not target.exists()
target.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
