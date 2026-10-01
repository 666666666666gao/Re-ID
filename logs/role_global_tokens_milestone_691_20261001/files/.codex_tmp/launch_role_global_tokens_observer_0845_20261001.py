"""Schedule one ETA-based observation with a real child-process wait receipt."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
SOURCE = ROOT / '.codex_tmp/observe_role_global_tokens_milestone_0845_20261001.py'
RECEIPT = ROOT / 'logs/role_global_tokens_observer_0845_20261001.json'
LOG = ROOT / 'logs/role_global_tokens_observer_0845_20261001.log'
WRAPPER_LOG = ROOT / 'logs/role_global_tokens_observer_0845_20261001_wrapper.log'
parser = argparse.ArgumentParser()
parser.add_argument('--worker', action='store_true')
args = parser.parse_args()
if args.worker:
    command = [sys.executable, '-B', str(SOURCE)]
    with LOG.open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'started_at': datetime.now().astimezone().isoformat(), 'pid': child.pid,
                  'command': command, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'observe_at': '2026-10-01T08:45:00+08:00', 'status': 'RUNNING'}
        assert not RECEIPT.exists()
        RECEIPT.write_text(json.dumps(record, indent=2) + '\n')
        code = child.wait()
    record.update(completed_at=datetime.now().astimezone().isoformat(), exit_code=code,
                  status='COMPLETE' if code == 0 else 'FAILED')
    RECEIPT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    raise SystemExit(code)
assert not RECEIPT.exists() and not LOG.exists() and not WRAPPER_LOG.exists()
launch = json.loads((ROOT / 'logs/role_global_tokens_launch_20261001_v1.json').read_text())
path = Path('/proc') / str(launch['pid'])
assert path.exists() and (path / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
with WRAPPER_LOG.open('x') as log:
    wrapper = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'wrapper_pid': wrapper.pid, 'observe_at': '2026-10-01T08:45:00+08:00',
                  'receipt': str(RECEIPT), 'status': 'OBSERVER_DISPATCHED_NOT_COMPLETE'}))