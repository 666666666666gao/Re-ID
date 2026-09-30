from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/slot_competition_fp32_roles_20261001_r2'
source = root / 'refine-logs/slot_competition_roles_v1/OBSERVE_PANEL_20261001.py'
receipt = root / 'logs/slot_competition_fp32_observer_launch_20261001.json'
stdout = root / 'logs/slot_competition_fp32_observer_20261001.log'
target = root / 'logs/slot_competition_fp32_observer_20261001.json'
assert not receipt.exists() and not stdout.exists() and not target.exists()
state = json.loads((campaign / 'campaign.json').read_text())
pid = state['controller_pid']
controller = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert 'tools/queue_slot_competition_fp32_roles.py' in controller
due = '2026-10-01T06:35:00+08:00'
command = [sys.executable, '-B', str(source), '--campaign', str(campaign),
           '--output', str(target), '--due', due]
with stdout.open('x') as log:
    process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
report = {'started_at': datetime.now().astimezone().isoformat(), 'observer_pid': process.pid,
          'due': due, 'poll_seconds': 240, 'controller_pid': pid, 'command': command,
          'observer_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'scope': 'Scheduled existing panel observation only; estimate not completion, no restart'}
receipt.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
