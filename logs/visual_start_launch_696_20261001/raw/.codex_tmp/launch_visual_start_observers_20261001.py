"""Detach one fixed milestone observer and one completion-only CPU waiter."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
HELPERS = {
    'observer_1145': ('observe_visual_start_roles_1145_20261001.py', '65259552b3644d930986d057779e9715e2cf65dd0a50d6f31abbe7c4f0d87a02'),
    'analysis_waiter': ('wait_visual_start_complete_analysis_20261001.py', '3370be369dad86a094c8a62e6b304eb484fe878cf3a27adb05228d6a415dc40e'),
}


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


if '--worker' not in sys.argv:
    launch = ROOT / 'logs/visual_start_roles_helpers_launch_20261001.json'
    assert not launch.exists()
    rows = []
    for name, (filename, digest) in HELPERS.items():
        helper = ROOT / '.codex_tmp' / filename
        assert hashlib.sha256(helper.read_bytes()).hexdigest() == digest
        assert not (ROOT / f'logs/visual_start_roles_{name}_wrapper_20261001.json').exists()
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', name]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        rows.append({'name': name, 'wrapper_pid': child.pid, 'source_sha256': digest, 'command': command})
    record = {'launched_at': stamp(), 'helpers': rows,
              'launcher_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(launch, record)
    print(json.dumps(record))
else:
    name = sys.argv[-1]
    filename, digest = HELPERS[name]
    helper = ROOT / '.codex_tmp' / filename
    status = ROOT / f'logs/visual_start_roles_{name}_wrapper_20261001.json'
    assert not status.exists() and hashlib.sha256(helper.read_bytes()).hexdigest() == digest
    command = [sys.executable, '-B', str(helper)]
    with (ROOT / f'logs/visual_start_roles_{name}_wrapper_20261001.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        record = {'status': 'RUNNING', 'name': name, 'wrapper_pid': os.getpid(), 'pid': child.pid,
                  'started_at': stamp(), 'command': command, 'source_sha256': digest}
        write(status, record)
        code = child.wait()
    record.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code, completed_at=stamp())
    write(status, record)
    raise SystemExit(code)
