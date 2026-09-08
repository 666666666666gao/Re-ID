"""Persistent fixed T0, M0, complete CPU checks, Q1, complete CPU checks."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.train_msvr310_signal_oof import sha256, write_json


def run(args):
    repo = Path.cwd().resolve();root = args.output_dir.resolve();config = args.config.resolve()
    assert repo == Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert root.parent == Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.code_commit
    assert sha256(config) == args.config_sha256
    spec = json.loads(config.read_bytes())
    assert sha256(__file__) == spec['project_file_sha256']['tools/run_msvr_history_gradient.py']
    assert shutil.disk_usage(root.parent).free >= spec['minimum_free_bytes']
    root.mkdir()
    common = ['--config', str(config)]
    m0 = root/'m0';q1 = root/'q1'
    stages = [
        ('t0', 'tools.check_msvr_history_gradient', common+['--output', str(root/'t0.json')], False),
        ('m0', 'tools.train_msvr_history_gradient', common+['--mode', 'm0', '--output-dir', str(m0)], True),
        ('m0_cpu', 'tools.verify_msvr_history_gradient', common+['--run-dir', str(m0), '--output', str(root/'m0_cpu.json')], False),
        ('q1', 'tools.train_msvr_history_gradient', common+['--mode', 'q1', '--output-dir', str(q1),
             '--m0-receipt', str(m0/'summary.json'), '--m0-verification', str(root/'m0_cpu.json')], True),
        ('q1_cpu', 'tools.verify_msvr_history_gradient', common+['--run-dir', str(q1), '--output', str(root/'q1_cpu.json')], False),
    ]
    receipt = dict(status='RUNNING', wrapper_pid=os.getpid(), code_commit=args.code_commit,
                   config_sha256=args.config_sha256, started_at=datetime.now().astimezone().isoformat(), stages=[])
    def save(): write_json(root/'pipeline.json', receipt)
    save()
    for name, module, arguments, gpu in stages:
        command = ['/root/miniconda3/envs/tri_reid/bin/python', '-u', '-m', module, *arguments]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0' if gpu else '', OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                           PYTHONPATH=str(repo/'modeling')+':'+str(repo))
        # Preserve the independently measured B0 distance execution path.
        if name == 'q1_cpu':
            environment.update(OMP_NUM_THREADS='56', MKL_NUM_THREADS='56')
        started = time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            row = dict(stage=name, original_pid=child.pid, command=command,
                       started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save()
            code = child.wait()
        row.update(exit_code=code, ended_at=datetime.now().astimezone().isoformat(),
                   elapsed_seconds=time.monotonic()-started)
        save()
        if code:
            receipt.update(status='STOPPED_AT_'+name.upper(), ended_at=row['ended_at']);save()
            return code
    result = json.loads((q1/'summary.json').read_bytes())
    proof = json.loads((root/'q1_cpu.json').read_bytes())
    assert proof['status'] == 'PASS_COMPLETE_HISTORY_GRADIENT_Q1' and proof['summary_sha256'] == sha256(q1/'summary.json')
    receipt.update(status='COMPLETE_VERIFIED_'+result['status'], ended_at=datetime.now().astimezone().isoformat(),
                   terminal_summary_sha256=sha256(q1/'summary.json'), terminal_cpu_sha256=sha256(root/'q1_cpu.json'))
    save()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
