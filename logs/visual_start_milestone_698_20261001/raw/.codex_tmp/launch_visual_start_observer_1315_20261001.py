"""Detach one ETA milestone observer, preserving its actual wait exit."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
HELPER = ROOT / '.codex_tmp/observe_visual_start_roles_1315_20261001.py'
STATUS = ROOT / 'logs/visual_start_roles_observer_1315_wrapper_20261001.json'

def stamp():
    return datetime.now().astimezone().isoformat()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

if '--worker' not in sys.argv:
    launch = ROOT / 'logs/visual_start_roles_observer_1315_launch_20261001.json'
    assert not launch.exists() and not STATUS.exists()
    assert HELPER.exists()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker']
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    record = {'launched_at': stamp(), 'wrapper_pid': child.pid, 'command': command,
              'source_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
              'launcher_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(launch, record)
    print(json.dumps(record))
else:
    assert not STATUS.exists()
    command = [sys.executable, '-B', str(HELPER)]
    with (ROOT / 'logs/visual_start_roles_observer_1315_wrapper_20261001.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        record = {'status': 'RUNNING', 'wrapper_pid': os.getpid(), 'pid': child.pid,
                  'started_at': stamp(), 'command': command,
                  'source_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest()}
        write(STATUS, record)
        code = child.wait()
    record.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code, completed_at=stamp())
    write(STATUS, record)
    raise SystemExit(code)
