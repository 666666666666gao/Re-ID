"""Administrative continuation of two never-started fixed-best diagnoses."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
ORIGINAL = ROOT / 'results/deployment_metric_fixed_best_five_20261005_850'
SEAL = ROOT / 'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
ENTRY = ROOT / 'refine-logs/deployment_metric_fixed_best_diagnosis_v1/DIAGNOSE_FIVE.py'
OUTPUT = ROOT / 'results/deployment_metric_fixed_best_pending100_20261005_854'
EXPECTED = [('RGBNT100', 'semantic'), ('RGBNT100', 'native')]
CAMPAIGNS = {'semantic': ROOT / 'logs/deployment_metric_role_pending100_20261005_842',
             'native': ROOT / 'logs/deployment_metric_storage_continuation_20261005_848'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return datetime.now().astimezone().isoformat()


def write(value):
    (OUTPUT / 'campaign.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require_inputs(seal):
    assert len(seal['source_sha256']) == 341 and len(seal['artifact_sha256']) == 271
    assert all(sha(ROOT / name) == digest for name, digest in seal['source_sha256'].items())
    assert all(sha(Path(name)) == digest for name, digest in seal['artifact_sha256'].items())


def main():
    failed = json.loads((ORIGINAL / 'campaign.json').read_text())
    terminal = json.loads((ROOT / 'logs/deployment_metric_fixed_best_five_launch_20261005_850/EXIT.json').read_text())
    assert failed['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert [(j['dataset'], j['variant'], j['status']) for j in failed['jobs']] == [
        ('RGBNT201', 'semantic', 'COMPLETE'), ('RGBNT201', 'native', 'COMPLETE'), ('MSVR310', 'semantic', 'FAILED')]
    assert not any(j['dataset'] == 'RGBNT100' for j in failed['jobs'])
    assert not OUTPUT.exists() and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    seal = json.loads(SEAL.read_text())
    assert sha(SEAL) == failed['seal_sha256']
    require_inputs(seal)
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    devices = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    assert all(int(line.split(',')[1]) < 500 for line in devices.splitlines())
    frozen = {str(p): sha(p) for p in (ORIGINAL / 'campaign.json',
        ROOT / 'logs/deployment_metric_fixed_best_five_launch_20261005_850/EXIT.json',
        ORIGINAL / 'RGBNT201_semantic/DIAGNOSIS.json', ORIGINAL / 'RGBNT201_native/DIAGNOSIS.json',
        ORIGINAL / 'MSVR310_semantic.log')}
    OUTPUT.mkdir()
    state = dict(schema='trifusion-deployment-metric-pending100-fixed-best-v1', status='RUNNING',
        started_at=stamp(), pid=os.getpid(), planned_models=2, jobs=[], optimizer_updates=0,
        original_failure_immutable_sha256=frozen, seal_sha256=sha(SEAL), physical_gpus=[0, 1],
        boundary='Only two never-started RGBNT100 diagnoses. Original MSVRsemantic failure and two completed201 diagnoses unchanged. No retry, training, new selection or power-temperature action.')
    write(state)
    for dataset, variant in EXPECTED:
        row = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
        assert row['artifact_campaign'] == str(CAMPAIGNS[variant])
        command = [sys.executable, '-B', str(ENTRY), '--campaign', str(CAMPAIGNS[variant]),
                   '--seal', str(SEAL), '--output-dir', str(OUTPUT), '--dataset', dataset, '--variant', variant]
        job = dict(dataset=dataset, variant=variant, status='RUNNING', started_at=stamp(), command=command)
        state['jobs'].append(job)
        with (OUTPUT / f'{dataset}_{variant}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            write(state)
            code = process.wait()
        job.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=stamp())
        write(state)
        if code:
            state.update(status='FAILED', completed_at=stamp())
            write(state)
            return code
    require_inputs(seal)
    assert all(sha(Path(name)) == digest for name, digest in frozen.items())
    state.update(status='COMPLETE', completed_at=stamp())
    write(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
