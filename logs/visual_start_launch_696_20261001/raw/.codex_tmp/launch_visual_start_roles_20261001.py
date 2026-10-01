"""Launch the registered six-end comparison once and retain its actual exit."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
LAUNCH = ROOT / 'logs/visual_start_roles_launch_20261001_v1.json'
STATUS = ROOT / 'logs/visual_start_roles_wrapper_20261001_v1.json'
LOG = ROOT / 'logs/visual_start_roles_controller_20261001_v1.log'
ENTRY = ROOT / 'tools/queue_visual_start_roles.py'
EXPECTED_SOURCE = '492c272702e667e4ce7f2b2b62c65105ffb6d2f899447cd7665de4c823c80852'


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


assert hashlib.sha256(ENTRY.read_bytes()).hexdigest() == EXPECTED_SOURCE
if '--worker' not in sys.argv:
    assert not CAMPAIGN.exists() and not LAUNCH.exists() and not STATUS.exists() and not LOG.exists()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker']
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    record = {'status': 'WRAPPER_LAUNCHED', 'launched_at': stamp(), 'wrapper_pid': process.pid,
              'command': command, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'queue_source_sha256': EXPECTED_SOURCE, 'campaign': str(CAMPAIGN)}
    write(LAUNCH, record)
    print(json.dumps(record))
else:
    assert not CAMPAIGN.exists() and not STATUS.exists() and not LOG.exists()
    command = [sys.executable, '-B', str(ENTRY), '--campaign', str(CAMPAIGN),
               '--inputs', str(ROOT / 'pertrained-model/visual_start_reset_20261001/INPUTS.json'),
               '--initialization-witness', str(ROOT / 'refine-logs/visual_start_roles_v1/INITIALIZATION_WITNESS.json'),
               '--after-campaign', str(ROOT / 'logs/role_global_tokens_20261001_v1'),
               '--after-matrix-sha256', '6eed7c1ba6fccb80350acc04ad4c42d907e3d31381867ba4942aaec4e4d0abf5']
    with LOG.open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        record = {'status': 'RUNNING', 'started_at': stamp(), 'wrapper_pid': os.getpid(),
                  'controller_pid': process.pid, 'command': command, 'campaign': str(CAMPAIGN),
                  'queue_source_sha256': EXPECTED_SOURCE}
        write(STATUS, record)
        code = process.wait()
    record.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code, completed_at=stamp())
    write(STATUS, record)
    raise SystemExit(code)
