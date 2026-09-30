import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/slot_competition_fp32_roles_20261001_r2'
receipt = root / 'logs/slot_competition_fp32_launch_20261001_r2.json'
log_path = root / 'logs/slot_competition_fp32_controller_20261001_r2.log'
assert not campaign.exists() and not receipt.exists() and not log_path.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == 'a885302abd9b5be9eb4d0bf012ecaa52b5eb6558'
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root).strip()
for name in ('modeling/trifusion/slot_competition_fp32_roles.py', 'tools/run_slot_competition_fp32_roles.py',
             'tools/queue_slot_competition_fp32_roles.py', 'tools/collect_slot_competition_fp32_roles.py'):
    ast.parse((root / name).read_text())
    expected = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=root)
    assert (root / name).read_bytes() == expected
gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
devices = {int(row.split(',')[0]): int(row.split(',')[1]) for row in gpu_text.splitlines()}
assert set(devices) == set(range(4)) and any(value < 500 for value in devices.values())
command = [sys.executable, '-B', str(root / 'tools/queue_slot_competition_fp32_roles.py'),
           '--campaign', str(campaign), '--after-campaign',
           str(root / 'logs/patch_memory_roles_recovery_20261001'),
           '--after-matrix-sha256', '35067bce6690e86e43a08f963ddc259ade704549d899f8b2ce58cfb1dcb3aa56']
with log_path.open('x') as output:
    process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
result = {'started_at': datetime.now().astimezone().isoformat(), 'controller_pid': process.pid,
          'command': command, 'campaign': str(campaign), 'controller_log': str(log_path),
          'code_commit': 'a885302abd9b5be9eb4d0bf012ecaa52b5eb6558', 'gpu_memory_mib_before_launch': devices,
          'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Actual launch only; M0 and formal completion not yet claimed.'}
receipt.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
