"""One scheduled read-only observation of the existing F2 controller."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

private = Path('C:/Users/gb/.codex_tmp')
observer = private / 'observe_scale746.py'
assert hashlib.sha256(observer.read_bytes()).hexdigest() == 'fca73f67c0bff564bda62f3f56ec93d19c15f6d809646505fff6a8e45c03acd0'
output = private / 'training_feature_scale_observe_wait750'
assert not output.exists()
output.mkdir()
target = datetime.fromisoformat('2026-10-03T03:12:00+08:00')
started = datetime.now().astimezone()
assert started < target
receipt = {'status': 'WAITING_ONE_READ_ONLY_OBSERVATION', 'pid': os.getpid(),
           'started_at': started.isoformat(), 'scheduled_at': target.isoformat(),
           'remote_controller_pid': 2185141, 'remote_port': 2026,
           'observer_label': 'next_full750',
           'boundary': 'Local timer then existing read-only observer once; no training, evaluation or restart.'}
(output / 'START.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt), flush=True)
time.sleep((target - started).total_seconds())
process = subprocess.run(['uv', 'run', '--with', 'paramiko', 'python', '-X', 'utf8', str(observer),
                          '--label', 'next_full750'], cwd='C:/Users/gb', capture_output=True)
(output / 'observer_stdout.txt').write_bytes(process.stdout)
(output / 'observer_stderr.txt').write_bytes(process.stderr)
result = {**receipt, 'status': 'OBSERVATION_RETURNED', 'exit_code': process.returncode,
          'completed_at': datetime.now().astimezone().isoformat()}
(output / 'END.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result), flush=True)
raise SystemExit(process.returncode)
