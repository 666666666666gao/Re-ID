"""Wait at the registered cadence and invoke the nine-end CPU report once."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'logs/shared_private_evidence_20261001_v2'
LAUNCH = ROOT / 'logs/shared_private_evidence_launch_20261001_v2.json'
REPORT = ROOT / 'tools/report_shared_private_evidence_complete.py'
OUTPUT = ROOT / 'results/shared_private_evidence_complete_20261001'
STATUS = CAMPAIGN / 'analysis_waiter_status.json'
DISPATCH = ROOT / 'logs/shared_private_analysis_waiter_20261001.json'
FIRST = datetime.fromisoformat('2026-10-01T19:40:00+08:00')
MANIFEST_SHA = '89227da9bea2b89d399fa19f814ff921c8a92b666d31c072dbf4fdaea7694109'


def stamp():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def live(pid):
    folder = Path('/proc') / str(pid)
    return folder.exists() and (folder / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'


def worker(report_sha):
    assert not STATUS.exists() and not OUTPUT.exists()
    self_sha = sha(Path(__file__))
    status = {'status': 'WAITING', 'pid': os.getpid(), 'started_at': stamp(),
              'first_observe_at': FIRST.isoformat(), 'poll_seconds': 240,
              'report_invocations': 0, 'source_sha256': self_sha,
              'report_source_sha256': report_sha, 'campaign': str(CAMPAIGN)}
    write(STATUS, status)
    time.sleep(max(0, (FIRST - datetime.now().astimezone()).total_seconds()))
    while True:
        assert sha(Path(__file__)) == self_sha and sha(REPORT) == report_sha
        assert sha(CAMPAIGN / 'manifest.json') == MANIFEST_SHA
        manifest, state, launch = (read(path) for path in (CAMPAIGN / 'manifest.json', CAMPAIGN / 'campaign.json', LAUNCH))
        assert len(manifest['source_sha256']) == 229
        assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
        observed = stamp()
        stages = []
        for job in state['jobs']:
            child = CAMPAIGN / f"{CAMPAIGN.name}_shared_private_{job['variant']}_{job['dataset']}"
            if not (child / 'campaign.json').exists():
                continue
            for item in read(child / 'campaign.json')['jobs']:
                receipt_path = Path(item['output_dir']) / 'training.json'
                receipt = read(receipt_path) if receipt_path.exists() else None
                stages.append({'dataset': job['dataset'], 'variant': job['variant'], 'mode': item['mode'],
                               'recorded_status': item['status'], 'pid': item.get('pid'),
                               'pid_live': live(item['pid']) if item.get('pid') else False,
                               'complete_epochs': len(receipt['history']) if receipt else 0,
                               'training_status': receipt['status'] if receipt else None})
        snapshot = {'observed_at': observed, 'parent_status': state['status'],
                    'parent_jobs': state['jobs'], 'wrapper_status': launch['status'], 'stages': stages,
                    'formal_accepted': sum(job['status'] == 'COMPLETE' for job in state['jobs']),
                    'controller_live': live(state['controller_pid']), 'wrapper_live': live(launch['pid']),
                    'scope': 'Recorded stages and live process checks only; no interim metric selection or neural call.'}
        folder = CAMPAIGN / 'milestone_snapshots'
        folder.mkdir(exist_ok=True)
        path = folder / (datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        write(path, snapshot)
        status.update(last_snapshot=str(path), observed_at=observed, formal_accepted=snapshot['formal_accepted'])
        if state['status'] == 'FAILED' or launch['status'] == 'FAILED':
            status.update(status='TRAINING_CAMPAIGN_FAILED', completed_at=stamp())
            write(STATUS, status)
            return 1
        if state['status'] == launch['status'] == 'COMPLETE':
            assert launch['exit_code'] == 0 and snapshot['formal_accepted'] == len(state['jobs']) == 9
            assert all(job['exit_code'] == 0 for job in state['jobs'])
            break
        if launch['status'] == 'RUNNING' and not snapshot['wrapper_live']:
            status.update(status='RECORDED_RUNNING_PROCESS_ABSENT', completed_at=stamp())
            write(STATUS, status)
            return 1
        write(STATUS, status)
        time.sleep(240)
    command = [sys.executable, '-B', str(REPORT), '--campaign', str(CAMPAIGN), '--output-dir', str(OUTPUT)]
    with (CAMPAIGN / 'complete_analysis.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status.update(status='CPU_REPORT_RUNNING', report_invocations=1, report_pid=process.pid,
                      report_started_at=stamp(), report_command=command)
        write(STATUS, status)
        code = process.wait()
    status.update(status='CPU_REPORT_COMPLETE' if code == 0 else 'CPU_REPORT_FAILED',
                  report_actual_exit_code=code, completed_at=stamp())
    write(STATUS, status)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--report-sha', required=True)
    args = parser.parse_args()
    assert sha(REPORT) == args.report_sha
    if args.worker:
        return worker(args.report_sha)
    assert not DISPATCH.exists() and not STATUS.exists() and not OUTPUT.exists()
    launch = read(LAUNCH)
    assert launch['status'] == 'RUNNING' and live(launch['pid'])
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--report-sha', args.report_sha]
    with (ROOT / 'logs/shared_private_analysis_waiter_20261001.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'launched_at': stamp(), 'pid': process.pid, 'command': command,
              'source_sha256': sha(Path(__file__)), 'report_source_sha256': args.report_sha,
              'status': 'WAITER_DISPATCHED_NOT_REPORT_COMPLETE', 'first_observe_at': FIRST.isoformat(),
              'poll_seconds': 240, 'receipt': str(STATUS)}
    write(DISPATCH, record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
