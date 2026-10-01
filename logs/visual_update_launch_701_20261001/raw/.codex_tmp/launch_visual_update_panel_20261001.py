"""Once-only durable preflight, initialization witness, then twelve full50 controls."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
RECEIPT = ROOT / 'logs/visual_update_control_launch_20261001_v1.json'
WORKFLOW = ROOT / '.codex_tmp/visual_update_launch_20261001'
PREFLIGHT = ROOT / 'logs/visual_update_preflight_20261001_v1/PRECHECK.json'
WITNESS = PREFLIGHT.parent / 'INITIALIZATION_WITNESS.json'
CAMPAIGN = ROOT / 'logs/visual_update_control_20261001_v1'
AFTER = ROOT / 'logs/visual_start_roles_20261001_v1'
SUMMARY = ROOT / 'results/visual_start_roles_complete_20261001/SUMMARY.json'
SUMMARY_SHA = '214ea13646e218bb3a1f771ba5bebd9255c6065f05b99cda37d37680f2563567'


def stamp():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def worker():
    state = json.loads(RECEIPT.read_text())
    assert state['status'] == 'LAUNCHED' and state['source_sha256'] == sha(Path(__file__))
    state.update(status='RUNNING', pid=os.getpid(), actual_started_at=stamp(), jobs=[])
    write(RECEIPT, state)
    predecessor = ['--after-campaign', str(AFTER), '--after-summary', str(SUMMARY),
                   '--after-summary-sha256', SUMMARY_SHA]
    stages = [
        ('preflight', ['--output', str(PREFLIGHT), *predecessor, '--gpus', '0', '1'],
         'tools/preflight_visual_update_control.py', None),
        ('initialization', ['--preflight', str(PREFLIGHT), '--output', str(WITNESS)],
         'tools/check_visual_update_initialization.py', '0'),
        ('full50_panel', ['--campaign', str(CAMPAIGN), '--preflight', str(PREFLIGHT),
                          '--initialization-witness', str(WITNESS), *predecessor],
         'tools/queue_visual_update_control.py', None),
    ]
    for name, arguments, source, gpu in stages:
        assert sha(ROOT / source) == state['entry_source_sha256'][source]
        command = [sys.executable, '-B', str(ROOT / source), *arguments]
        row = {'stage': name, 'status': 'RUNNING', 'started_at': stamp(), 'command': command,
               'source_sha256': sha(ROOT / source)}
        env = dict(os.environ)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        with (WORKFLOW / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = child.pid
        state['jobs'].append(row)
        write(RECEIPT, state)
        code = child.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=stamp())
        write(RECEIPT, state)
        if code:
            state.update(status='FAILED', completed_at=stamp(), exit_code=code)
            write(RECEIPT, state)
            return code
    state.update(status='COMPLETE', completed_at=stamp(), exit_code=0)
    write(RECEIPT, state)
    return 0


def launch():
    assert not RECEIPT.exists() and not WORKFLOW.exists()
    assert not PREFLIGHT.exists() and not WITNESS.exists() and not CAMPAIGN.exists()
    assert sha(SUMMARY) == SUMMARY_SHA
    reviewed = json.loads((ROOT / 'logs/visual_start_complete_700_20261001/CODE_REVIEW_CALL.json').read_text())
    assert reviewed['result'] == 'NO_REMAINING_BLOCKING_ISSUE_IDENTIFIED'
    assert all(sha(ROOT / path) == digest for path, digest in reviewed['source_sha256'].items())
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT).strip()
    WORKFLOW.mkdir()
    state = {'status': 'LAUNCHED', 'launched_at': stamp(), 'launch_pid': os.getpid(),
             'source_sha256': sha(Path(__file__)), 'entry_source_sha256': reviewed['source_sha256'],
             'campaign': str(CAMPAIGN), 'preflight': str(PREFLIGHT), 'witness': str(WITNESS),
             'after_summary_sha256': SUMMARY_SHA, 'poll_seconds': 240,
             'scope': 'Fixed two actual M0s, actual twelve-build witness, then fresh twelve full50 controls. No retries.'}
    write(RECEIPT, state)
    with (WORKFLOW / 'workflow.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(WORKFLOW / 'launch.json', dict(state, workflow_pid=child.pid))
    print(json.dumps({'status': 'LAUNCHED', 'workflow_pid': child.pid, 'receipt': str(RECEIPT),
                      'source_sha256': state['source_sha256'], 'launched_at': state['launched_at']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    launch()
