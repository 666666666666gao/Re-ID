"""Observe registered training at its ETA, then execute the reviewed CPU report once."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_update_control_20261001_v1'
RECEIPT = ROOT / 'logs/visual_update_analysis_waiter_20261001.json'
STATUS = CAMPAIGN / 'analysis_waiter_status.json'
REPORT = ROOT / 'tools/report_visual_update_control_complete.py'
REPORT_SHA = '4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765'
OUTPUT = ROOT / 'results/visual_update_control_complete_20261001'
FIRST = datetime.fromisoformat('2026-10-01T14:28:30+08:00')
POLL = 240


def stamp():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def observe():
    assert sha(REPORT) == REPORT_SHA and not OUTPUT.exists()
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    assert manifest['schema'] == 'trifusion-visual-update-panel-v1' and len(manifest['jobs']) == 12
    state = {'status': 'WAITING_FOR_TWELVE_VERIFIED_ENDPOINTS', 'pid': os.getpid(),
             'started_at': stamp(), 'first_observe_at': FIRST.isoformat(), 'poll_seconds': POLL,
             'report_source_sha256': REPORT_SHA, 'report_invocations': 0}
    write(STATUS, state)
    time.sleep(max(0, (FIRST - datetime.now().astimezone()).total_seconds()))
    while True:
        parent = json.loads((CAMPAIGN / 'campaign.json').read_text())
        snapshot = {'observed_at': stamp(), 'parent_status': parent['status'],
                    'controller_pid': parent['controller_pid'], 'jobs': parent['jobs'],
                    'complete': sum(row['status'] == 'COMPLETE' for row in parent['jobs']),
                    'source_count': len(manifest['source_sha256']), 'endpoints': []}
        assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
        for job in parent['jobs']:
            child = CAMPAIGN / f"{CAMPAIGN.name}_visual_update_{job['variant']}_{job['dataset']}"
            if not (child / 'campaign.json').exists():
                continue
            row = json.loads((child / 'campaign.json').read_text())
            item = {'dataset': job['dataset'], 'variant': job['variant'], 'gpu': job['gpu'],
                    'parent_status': job['status'], 'child_status': row['status'], 'stages': row['jobs']}
            latest = row['jobs'][-1]
            item['latest_process_present'] = (Path('/proc') / str(latest['pid'])).exists()
            training_path = Path(latest['output_dir']) / 'training.json'
            if training_path.exists():
                training = json.loads(training_path.read_text())
                item['training_status'] = training['status']
                item['recorded_complete_epochs'] = len(training['history'])
            snapshot['endpoints'].append(item)
        snapshot['gpu_state'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                        '--format=csv,noheader,nounits'], text=True)
        name = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        snapshots = CAMPAIGN / 'milestone_snapshots'
        snapshots.mkdir(exist_ok=True)
        write(snapshots / (name + '.json'), snapshot)
        write(CAMPAIGN / 'milestone_current_snapshot.json', snapshot)
        state.update(updated_at=stamp(), complete=snapshot['complete'], latest_snapshot=str(snapshots / (name + '.json')))
        write(STATUS, state)
        print(json.dumps({'observed_at': snapshot['observed_at'], 'complete': snapshot['complete'],
                          'parent_status': parent['status']}), flush=True)
        if parent['status'] == 'FAILED':
            state.update(status='CAMPAIGN_FAILED_NO_REPORT', completed_at=stamp())
            write(STATUS, state)
            return 1
        if parent['status'] == 'COMPLETE':
            assert len(parent['jobs']) == 12
            assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in parent['jobs'])
            matrix = json.loads((CAMPAIGN / 'accepted_matrix.json').read_text())
            assert matrix['verified_complete'] == matrix['expected_endpoints'] == 12
            assert sha(REPORT) == REPORT_SHA and not OUTPUT.exists()
            command = [sys.executable, '-B', str(REPORT), '--campaign', str(CAMPAIGN), '--output-dir', str(OUTPUT)]
            with (CAMPAIGN / 'complete_cpu_report.log').open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                           stdout=log, stderr=subprocess.STDOUT)
            state.update(status='CPU_REPORT_RUNNING', report_pid=process.pid, report_started_at=stamp(),
                         report_command=command, report_invocations=1)
            write(STATUS, state)
            code = process.wait()
            state.update(status='CPU_REPORT_COMPLETE' if code == 0 else 'CPU_REPORT_FAILED',
                         report_exit_code=code, report_completed_at=stamp())
            write(STATUS, state)
            return code
        assert parent['status'] == 'RUNNING'
        assert (Path('/proc') / str(parent['controller_pid'])).exists()
        time.sleep(POLL)


def worker():
    record = json.loads(RECEIPT.read_text())
    record.update(status='RUNNING', wrapper_pid=os.getpid(), actual_started_at=stamp())
    with (CAMPAIGN / 'analysis_waiter_observer.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--observe'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record['observer_pid'] = process.pid
    write(RECEIPT, record)
    code = process.wait()
    record.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code, completed_at=stamp())
    write(RECEIPT, record)
    return code


def launch():
    assert not RECEIPT.exists() and not STATUS.exists() and not OUTPUT.exists()
    assert sha(REPORT) == REPORT_SHA
    record = {'status': 'LAUNCHED', 'launched_at': stamp(), 'launcher_pid': os.getpid(),
              'source_sha256': sha(Path(__file__)), 'report_source_sha256': REPORT_SHA,
              'first_observe_at': FIRST.isoformat(), 'poll_seconds': POLL,
              'boundary': 'Observe existing12 only; no model/optimizer/retry/selection. Report runs once only after all12 verified, actual wait exits retained.'}
    write(RECEIPT, record)
    with (CAMPAIGN / 'analysis_waiter_wrapper.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(CAMPAIGN / 'analysis_waiter_launch.json', dict(record, wrapper_pid=process.pid))
    print(json.dumps({'status': 'LAUNCHED', 'wrapper_pid': process.pid, 'receipt': str(RECEIPT),
                      'first_observe_at': FIRST.isoformat(), 'poll_seconds': POLL}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    if args.observe:
        raise SystemExit(observe())
    if args.worker:
        raise SystemExit(worker())
    launch()
