"""Wait on the real registered controller, then execute the prepared CPU report once."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'
OUTPUT = ROOT / 'results/role_global_tokens_complete_20261001'
STATUS = CAMPAIGN / 'analysis_waiter_status.json'
REPORT = ROOT / 'tools/report_role_global_tokens_complete.py'
REPORT_SHA = '8c5856c102e8503907b70e58dc6e083bfde11ae1e7f4037030ccfe5d48656bb2'
assert not OUTPUT.exists() and not STATUS.exists()
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA
launch = json.loads((ROOT / 'logs/role_global_tokens_launch_20261001_v1.json').read_text())
while True:
    state = json.loads((CAMPAIGN / 'campaign.json').read_text())
    assert len(state['jobs']) == 9
    assert all(row['status'] in ('PENDING', 'RUNNING', 'COMPLETE') for row in state['jobs'])
    record = {'observed_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
              'controller_pid': launch['pid'], 'campaign_status': state['status'],
              'complete_parent_jobs': sum(row['status'] == 'COMPLETE' for row in state['jobs']),
              'expected_jobs': 9, 'report_source_sha256': REPORT_SHA,
              'status': 'WAITING_FOR_NINE_VERIFIED_ENDPOINTS',
              'scope': 'Status/proc reads only while waiting; no interim scores, model forwards or retries.'}
    if state['status'] == 'COMPLETE':
        assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
        break
    assert state['status'] == 'RUNNING'
    controller = Path('/proc') / str(launch['pid'])
    assert controller.exists() and (controller / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    record['controller_live'] = True
    STATUS.write_text(json.dumps(record, indent=2) + '\n')
    time.sleep(240)
assert not OUTPUT.exists()
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA
command = [sys.executable, '-B', str(REPORT), '--campaign', str(CAMPAIGN), '--output-dir', str(OUTPUT)]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
report_log = ROOT / 'logs/role_global_tokens_complete_analysis_20261001.log'
with report_log.open('x') as log:
    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(status='CPU_REPORT_RUNNING', report_pid=child.pid,
                  report_started_at=datetime.now().astimezone().isoformat(), command=command,
                  controller_live_required=False)
    STATUS.write_text(json.dumps(record, indent=2) + '\n')
    code = child.wait()
record.update(status='CPU_REPORT_COMPLETE' if code == 0 else 'CPU_REPORT_FAILED',
              exit_code=code, completed_at=datetime.now().astimezone().isoformat())
STATUS.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
raise SystemExit(code)
