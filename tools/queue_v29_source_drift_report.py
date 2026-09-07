#!/usr/bin/env python3
"""Wait for the original source diagnostic, then export its complete verified scalars."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    run = args.run_dir.resolve()
    repo = Path.cwd().resolve()
    assert repo == Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert run.parent == Path('/root/autodl-tmp/trifusion-v2/artifacts')
    prefix = str(run) + '_report'
    launcher = json.loads(Path(str(run) + '_launcher.json').read_bytes())
    assert launcher['wrapper_pid'] == args.upstream_pid
    assert launcher['repository_commit'] == args.execution_commit
    reporter = repo / 'tools/report_v29_source_role_drift.py'
    math_module = repo / 'tools/v29_source_drift_math.py'
    assert sha(reporter) == args.reporter_sha256 and sha(math_module) == args.math_sha256
    output = run / 'complete_scalar_report'
    assert not output.exists()

    def write(suffix, value):
        with Path(prefix + suffix).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')

    started = time.time()
    write('_queue_launch.json', {
        'queue_pid': os.getpid(), 'upstream_wrapper_pid': args.upstream_pid,
        'execution_commit': args.execution_commit,
        'reporter_sha256': args.reporter_sha256, 'math_sha256': args.math_sha256,
        'queue_sha256': sha(__file__), 'poll_seconds': 240,
        'started_at': datetime.now().astimezone().isoformat(),
    })
    while Path('/proc', str(args.upstream_pid)).exists():
        time.sleep(240)
    terminal_path = Path(str(run) + '_pipeline_exit.json')
    assert terminal_path.exists(), 'Original source pipeline ended without its terminal receipt'
    terminal = json.loads(terminal_path.read_bytes())
    if terminal['exit_code'] != 0:
        write('_queue_exit.json', {
            'status': 'UPSTREAM_FAILED_NO_REPORT', 'exit_code': terminal['exit_code'],
            'upstream_terminal': terminal, 'queue_pid': os.getpid(),
            'ended_at': datetime.now().astimezone().isoformat(),
        })
        return terminal['exit_code']
    assert terminal['stage'] == 'COMPLETE_VERIFIED_SOURCE_ROLE_DRIFT'
    assert sha(reporter) == args.reporter_sha256 and sha(math_module) == args.math_sha256
    command = ['/root/miniconda3/envs/tri_reid/bin/python', '-u', '-m',
               'tools.report_v29_source_role_drift', '--run-dir', str(run),
               '--output-dir', str(output)]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(repo), OMP_NUM_THREADS='4')
    with Path(prefix + '.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL)
        write('_launch.json', {'original_pid': child.pid, 'queue_pid': os.getpid(),
                              'command': command, 'started_at': datetime.now().astimezone().isoformat()})
        code = child.wait()
    write('_exit.json', {'original_pid': child.pid, 'exit_code': code,
                         'ended_at': datetime.now().astimezone().isoformat()})
    write('_queue_exit.json', {
        'status': 'COMPLETE_SCALAR_REPORT' if code == 0 else 'SCALAR_REPORT_FAILED',
        'exit_code': code, 'queue_pid': os.getpid(),
        'ended_at': datetime.now().astimezone().isoformat(),
        'elapsed_seconds': time.time() - started,
    })
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--upstream-pid', type=int, required=True)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--reporter-sha256', required=True)
    parser.add_argument('--math-sha256', required=True)
    raise SystemExit(main(parser.parse_args()))
