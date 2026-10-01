"""One durable sequential preflight, initialization witness and nine-end queue."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
CAMPAIGN = ROOT / 'logs/shared_private_evidence_20261001_v1'
PREFLIGHT = ROOT / 'logs/shared_private_preflight_20261001_v1/preflight.json'
WITNESS = ROOT / 'logs/shared_private_preflight_20261001_v1/initialization_witness.json'
AFTER = ROOT / 'logs/visual_update_control_20261001_v1'
SUMMARY = ROOT / 'results/visual_update_control_complete_20261001/SUMMARY.json'
SUMMARY_SHA = '43634cf0c6acd7b998d9c9c2e39174bd60141598c4f4b9f1ba14d81fea6b18ad'
RECEIPT = ROOT / 'logs/shared_private_evidence_launch_20261001_v1.json'


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    assert not RECEIPT.exists() and not CAMPAIGN.exists() and not PREFLIGHT.exists() and not WITNESS.exists()
    after = ['--after-campaign', str(AFTER), '--after-summary', str(SUMMARY), '--after-summary-sha256', SUMMARY_SHA]
    stages = [
        ('preflight', [PYTHON, '-B', str(ROOT / 'tools/preflight_shared_private_evidence.py'),
                       '--output', str(PREFLIGHT), '--gpus', '0', '1', '2', *after]),
        ('witness', [PYTHON, '-B', str(ROOT / 'tools/check_shared_private_initialization.py'),
                     '--output', str(WITNESS), '--preflight', str(PREFLIGHT)]),
        ('queue', [PYTHON, '-B', str(ROOT / 'tools/queue_shared_private_evidence.py'),
                   '--campaign', str(CAMPAIGN), '--preflight', str(PREFLIGHT),
                   '--initialization-witness', str(WITNESS), *after]),
    ]
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_at': stamp(), 'jobs': [],
             'campaign': str(CAMPAIGN), 'preflight': str(PREFLIGHT), 'witness': str(WITNESS),
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'after_summary_sha256': SUMMARY_SHA, 'poll_seconds': 240,
             'scope': 'One preflight and witness, then fresh full50 nine-end queue. No retries.'}
    for stage, command in stages:
        row = {'stage': stage, 'status': 'RUNNING', 'command': command, 'started_at': stamp()}
        with (ROOT / f'logs/shared_private_evidence_{stage}_20261001_v1.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            RECEIPT.write_text(json.dumps(state, indent=2) + '\n')
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=stamp())
        if code:
            state.update(status='FAILED', exit_code=code, completed_at=stamp())
            RECEIPT.write_text(json.dumps(state, indent=2) + '\n')
            return code
        RECEIPT.write_text(json.dumps(state, indent=2) + '\n')
    state.update(status='COMPLETE', exit_code=0, completed_at=stamp())
    RECEIPT.write_text(json.dumps(state, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
