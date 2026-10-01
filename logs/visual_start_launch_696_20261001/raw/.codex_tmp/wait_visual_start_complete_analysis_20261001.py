"""Execute the prepared CPU report once after all six verified endpoints finish."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
OUTPUT = ROOT / 'results/visual_start_roles_complete_20261001'
STATUS = CAMPAIGN / 'analysis_waiter_status.json'
REPORT = ROOT / 'tools/report_visual_start_roles_complete.py'
REPORT_SHA = 'faa00082ea3a4735181c18a0324e314ec4de5e3a177f84b487ef9e936edd9c13'
assert not OUTPUT.exists() and not STATUS.exists()
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA
launch = json.loads((ROOT / 'logs/visual_start_roles_wrapper_20261001_v1.json').read_text())
while True:
    state = json.loads((CAMPAIGN / 'campaign.json').read_text())
    assert len(state['jobs']) == 6
    assert all(row['status'] in ('PENDING', 'RUNNING', 'COMPLETE') for row in state['jobs'])
    record = {'observed_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
              'controller_pid': launch['controller_pid'], 'campaign_status': state['status'],
              'complete_parent_jobs': sum(row['status'] == 'COMPLETE' for row in state['jobs']),
              'expected_jobs': 6, 'report_source_sha256': REPORT_SHA,
              'status': 'WAITING_FOR_SIX_VERIFIED_ENDPOINTS',
              'scope': 'Parent status/process reads only while waiting; no interim scores, retries or model forwards.'}
    if state['status'] == 'COMPLETE':
        assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
        break
    assert state['status'] == 'RUNNING'
    controller = Path('/proc') / str(launch['controller_pid'])
    assert controller.exists() and (controller / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    STATUS.write_text(json.dumps(record, indent=2) + '\n')
    time.sleep(240)
assert not OUTPUT.exists()
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA
command = [sys.executable, '-B', str(REPORT), '--campaign', str(CAMPAIGN), '--output-dir', str(OUTPUT)]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
with (ROOT / 'logs/visual_start_roles_complete_analysis_20261001.log').open('x') as log:
    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(status='CPU_REPORT_RUNNING', report_pid=child.pid, command=command,
                  report_started_at=datetime.now().astimezone().isoformat(), controller_live_required=False)
    STATUS.write_text(json.dumps(record, indent=2) + '\n')
    code = child.wait()
record.update(status='CPU_REPORT_COMPLETE' if code == 0 else 'CPU_REPORT_FAILED', exit_code=code,
              completed_at=datetime.now().astimezone().isoformat())
STATUS.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
raise SystemExit(code)
