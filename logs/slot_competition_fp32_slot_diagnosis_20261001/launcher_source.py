"""Run three read-only slot diagnostics and retain actual child exit codes."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
PYTHON = Path('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')
DATASETS = ('RGBNT201', 'RGBNT100', 'MSVR310')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=3, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    assert matrix['schema'] == 'trifusion-slot-competition-fp32-roles-verification-v2'
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == 6
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in matrix['rows'])
    assert len(set(args.gpus)) == 3
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in inventory.splitlines():
        fields = [int(value) for value in line.split(',')]
        cards[fields[0]] = fields[1:]
    assert all(cards[gpu][0] <= 100 and cards[gpu][1] == 0 for gpu in args.gpus)
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    jobs = []
    for dataset, gpu in zip(DATASETS, args.gpus):
        command = [str(PYTHON), '-B', '-u', str(ROOT / 'tools/diagnose_slot_competition_fp32_slots.py'),
                   '--matrix', str(args.matrix), '--dataset', dataset,
                   '--output-dir', str(args.output_dir / dataset)]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        log = (args.output_dir / (dataset + '.log')).open('w')
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        log.close()
        jobs.append({'dataset': dataset, 'gpu': gpu, 'pid': child.pid,
                     'started_at': datetime.now().astimezone().isoformat(),
                     'command': command, 'child': child})
    print(json.dumps({'status': 'READ_ONLY_DIAGNOSTICS_STARTED',
                      'jobs': [{key: value for key, value in job.items() if key != 'child'} for job in jobs]}), flush=True)
    for job in jobs:
        job['exit_code'] = job.pop('child').wait()
        job['waited_at'] = datetime.now().astimezone().isoformat()
    passed = all(job['exit_code'] == 0 for job in jobs)
    report = {'status': 'COMPLETE' if passed else 'FAILED', 'jobs': jobs,
              'completed_at': datetime.now().astimezone().isoformat(),
              'matrix_sha256': sha(args.matrix), 'source_sha256': sha(Path(__file__)),
              'diagnostic_source_sha256': sha(ROOT / 'tools/diagnose_slot_competition_fp32_slots.py'),
              'initial_gpu_inventory': inventory,
              'scope': 'Read-only full-gallery diagnostics after six accepted endpoints; no training or retry.'}
    (args.output_dir / 'EXECUTION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    assert passed, 'A diagnostic child failed; preserve its actual exit code and log.'


if __name__ == '__main__':
    main()
