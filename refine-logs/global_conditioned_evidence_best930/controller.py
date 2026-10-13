"""Exactly two closed-best inference diagnostics; no training or retry."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path('/data/gaob/Re-ID/Trifusion')
ADMIN = Path(__file__).resolve().parent
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
stamp = lambda: datetime.now().astimezone().isoformat()


def verify(seal):
    for name, h in seal['source_sha256'].items():
        assert sha(ROOT / name) == h, name
    for name, h in seal['artifact_sha256'].items():
        assert sha(Path(name)) == h, name
    assert not Path('/proc/3190961').exists() and not Path('/proc/3190963').exists()
    assert json.loads(Path(seal['controller_exit']).read_text())['exit_code'] == 0


def worker():
    seal_path = ADMIN / 'INPUT_SEAL.json'
    seal = json.loads(seal_path.read_text())
    verify(seal)
    state = {'status': 'RUNNING', 'started_at': stamp(), 'worker_pid': os.getpid(),
             'optimizer_updates': 0, 'physical_gpus': [0, 1], 'jobs': [], 'goal_status': 'ACTIVE_UNMET'}
    write = lambda: (ADMIN / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n')
    write()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for exposure in ('gradient', 'random'):
        verify(seal)
        used = [tuple(int(x.strip()) for x in line.split(',')) for line in subprocess.check_output(
            ['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
        assert {v[0] for v in used} == {0, 1} and all(v[1] < 500 for v in used), used
        assert shutil.disk_usage(ROOT).free >= 256 * 1024**2
        output = ADMIN / exposure
        assert not output.exists()
        command = [PYTHON, '-B', str(ROOT / 'tools/diagnose_global_conditioned_evidence_best.py'),
                   '--seal', str(seal_path), '--exposure', exposure, '--output-dir', str(output)]
        began = time.perf_counter()
        with (ADMIN / f'{exposure}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {'exposure': exposure, 'command': command, 'started_at': stamp(), 'child_pid': child.pid,
                   'devices': used, 'status': 'RUNNING'}
            state['jobs'].append(row)
            write()
            code = child.wait()
        row.update(exit_code=code, completed_at=stamp(), elapsed_seconds=time.perf_counter() - began,
                   status='FAILED' if code else 'COMPLETE')
        (ADMIN / f'{exposure}_EXIT.json').write_text(json.dumps(row, indent=2) + '\n')
        write()
        if code:
            state.update(status='FAILED', completed_at=stamp())
            write()
            return code
        result = json.loads((output / 'DIAGNOSIS.json').read_text())
        assert result['status'] == 'COMPLETE' and result['exposure'] == exposure and result['optimizer_updates'] == 0
    verify(seal)
    state.update(status='COMPLETE_TWO_FIXED_BEST_DIAGNOSES', completed_at=stamp())
    write()
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        raise SystemExit(worker())
    if sys.argv[1:] == ['--supervisor']:
        record = {'supervisor_pid': os.getpid(), 'started_at': stamp()}
        child = subprocess.Popen(['/usr/bin/python3', str(Path(__file__)), '--worker'])
        record['worker_pid'] = child.pid
        (ADMIN / 'SUPERVISOR.json').write_text(json.dumps(record, indent=2) + '\n')
        record.update(exit_code=child.wait(), completed_at=stamp())
        (ADMIN / 'CONTROLLER_EXIT.json').write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit(record['exit_code'])
    assert len(sys.argv) == 1 and not (ADMIN / 'SUPERVISOR.log').exists()
    with (ADMIN / 'SUPERVISOR.log').open('x') as log:
        child = subprocess.Popen(['/usr/bin/python3', str(Path(__file__)), '--supervisor'],
                                 cwd=ROOT, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    value = {'supervisor_pid': child.pid, 'launched_at': stamp(), 'new_training_started': False}
    (ADMIN / 'LAUNCH.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)
