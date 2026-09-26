#!/usr/bin/env python3
"""Run the registered nine source-only derivative panels after a GPU job completes."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--after-campaign', type=Path, required=True)
    args = parser.parse_args()
    assert ROOT == Path('/data/gaob/Re-ID/Trifusion') and args.gpu in (0, 1, 2, 3)
    directory = ROOT / 'logs/joint_fusion_scale_source_20260926'
    assert not directory.exists()
    directory.mkdir()
    status = {'status': 'WAITING', 'created_at': stamp(), 'gpu': args.gpu,
              'after_campaign': str(args.after_campaign), 'jobs': []}

    def save():
        (directory / 'campaign.json').write_text(json.dumps(status, indent=2) + '\n')

    save()
    while True:
        prior = json.loads(args.after_campaign.read_text())
        if prior['status'] == 'COMPLETE':
            break
        assert prior['status'] == 'RUNNING'
        time.sleep(240)
    status.update(status='RUNNING', started_at=stamp())
    save()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for method, date, state in (('SIGNAL_SIM_JOINT', '20260925', 'initial'),
                                    ('SIGNAL_SIM_JOINT', '20260925', 'final'),
                                    ('SIGNAL_SIM_JOINT_LOWLR', '20260926', 'final')):
            name = f'{dataset}_{method}_{state}'
            receipt = ROOT / 'trained-model' / f'official_extra_seed42_{dataset}_{method}_{date}' / f'{dataset}_{method}_seed42/training.json'
            job = {'dataset': dataset, 'method': method, 'state': state,
                   'status': 'RUNNING', 'started_at': stamp()}
            status['jobs'].append(job)
            save()
            command = [sys.executable, '-B', '-u', str(ROOT / 'tools/diagnose_joint_fusion_scale.py'),
                       '--training-receipt', str(receipt), '--state', state,
                       '--output-dir', str(directory / name)]
            with (directory / f'{name}.log').open('x') as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            result = json.loads((directory / name / 'summary.json').read_text())
            assert result['status'] == 'COMPLETE' and result['model_state_unchanged']
            job.update(status='COMPLETE', completed_at=stamp(), summary=result)
            save()
    status.update(status='COMPLETE', completed_at=stamp())
    save()


if __name__ == '__main__':
    main()
