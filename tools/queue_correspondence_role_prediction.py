#!/usr/bin/env python3
"""Registered four-condition M3 panel, using the existing four-GPU scheduler."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS

CONDITIONS = {'own_direct': ('own', 'direct'), 'matched_direct': ('matched', 'direct'),
              'own_predictor': ('own', 'predictor'), 'matched_predictor': ('matched', 'predictor')}
SOURCE_PATHS = ('modeling/trifusion/correspondence_roles.py', 'tools/run_correspondence_roles.py',
                'modeling/trifusion/correspondence_role_prediction.py',
                'tools/run_correspondence_role_prediction.py',
                'tools/check_correspondence_role_prediction.py',
                'tools/queue_correspondence_role_prediction.py', 'tools/queue_correspondence_refinement.py',
                'tools/collect_correspondence_role_prediction.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(dataset, variant, mode, output):
    address, prediction = CONDITIONS[variant]
    weight, digest = BASELINES[dataset]
    return [sys.executable, '-B', str(ROOT / 'tools/run_correspondence_role_prediction.py'),
            '--dataset', dataset, '--mode', mode, '--protocol', str(PROTOCOLS / f'{dataset}.json'),
            '--signal-source', str(SOURCE), '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'),
            '--baseline-checkpoint', str(WEIGHTS / weight), '--baseline-sha256', digest,
            '--output-dir', str(output), '--seed', '42', '--epochs', '50', '--width', '128',
            '--pred-weight', '0.1', '--m1', '--m2', '--m3',
            '--address-mode', address, '--prediction-mode', prediction]


def require_sources(campaign):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())


def worker(args):
    require_sources(args.campaign)
    campaign = queue.child_campaign(args.campaign, 'm3', args.dataset, args.variant)
    assert not campaign.exists()
    campaign.mkdir()
    state = {'status': 'RUNNING', 'dataset': args.dataset, 'variant': args.variant,
             'gpu': args.gpu, 'controller_pid': os.getpid(), 'started_at': queue.stamp(), 'jobs': []}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for mode in ('m0', 'train', 'evaluate'):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        suffix = 'm0' if mode == 'm0' else 'full'
        output = ROOT / f'trained-model/{campaign.name}_seed42_{suffix}'
        row = {'mode': mode, 'status': 'RUNNING', 'started_at': queue.stamp(),
               'output_dir': str(output), 'command': command(args.dataset, args.variant, mode, output)}
        with (campaign / f'{mode}.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(row['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            queue.write(campaign / 'campaign.json', state)
            code = process.wait()
        if code == 0:
            receipt = json.loads((output / ('official_metrics.json' if mode == 'evaluate' else 'training.json')).read_text())
            expected = {'m0': 'M0_PASS', 'train': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluate': 'COMPLETE'}[mode]
            assert receipt['status'] == expected
        row.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code, completed_at=queue.stamp())
        state['status'] = 'FAILED' if code else 'RUNNING'
        queue.write(campaign / 'campaign.json', state)
        if code:
            return code
    state.update(status='COMPLETE', completed_at=queue.stamp())
    queue.write(campaign / 'campaign.json', state)
    queue.require_complete(campaign, args.dataset)
    training = json.loads((output / 'training.json').read_text())
    address, prediction = CONDITIONS[args.variant]
    assert training['initializer']['prediction'] == {'address_mode': address, 'prediction_mode': prediction}
    return 0


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
            '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def coordinate(args):
    assert sha(args.prior_matrix) == args.prior_matrix_sha256
    prior = json.loads(args.prior_matrix.read_text())
    assert prior['expected_endpoints'] == prior['verified_complete'] == len(prior['rows']) == 18
    assert all(r['status'] == 'VERIFIED_COMPLETE' for r in prior['rows'])
    assert {(r['dataset'], r['variant']) for r in prior['rows'] if r['phase'] == 'm2'} == {
        (d, v) for d in queue.DATASETS for v in queue.VARIANTS}
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase': 'm3', 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
            for variant in CONDITIONS for dataset in queue.DATASETS]
    queue.write(args.campaign / 'manifest.json', {
        'schema': 'trifusion-correspondence-m3-panel-v1', 'seed': 42, 'epochs': 50,
        'poll_seconds': queue.POLL_SECONDS, 'prior_matrix': str(args.prior_matrix),
        'prior_matrix_sha256': args.prior_matrix_sha256, 'jobs': jobs,
        'source_sha256': {name: sha(ROOT / name) for name in SOURCE_PATHS},
        'fixed_contract': {'m1': True, 'm2': True, 'm3': True, 'width': 128, 'fused_width': 1536,
                           'prediction_weight': 0.1, 'selection': 'single', 'readout': 'pooled',
                           'checkpoint_policy': 'best_official_map', 'teacher_momentum': 0.996,
                           'predictor_parameters': 100608, 'target_coordinate_guidance': 'detached_teacher'},
        'boundary': 'Registered address/predictor factors; no configuration selected from M2 scores. '
                    'Teacher coordinates guide only training prediction. Predictors add trainable capacity '
                    'and are unused at inference; direct cells retain frozen copies to match common weights/RNG. '
                    'No automatic retry, preemption or tuning. Each real M0 precedes its full 50 epochs and reload.',
    })
    state = {'status': 'RUNNING', 'phase': 'm3', 'controller_pid': os.getpid(),
             'started_at': datetime.now().astimezone().isoformat(), 'jobs': jobs}
    queue.start_command = start_command
    code = queue.run_phase(args.campaign, state, 'm3')
    state.update(status='FAILED' if code else 'COMPLETE', completed_at=queue.stamp())
    queue.write(args.campaign / 'campaign.json', state)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prior-matrix', type=Path)
    parser.add_argument('--prior-matrix-sha256')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--dataset', choices=queue.DATASETS)
    parser.add_argument('--variant', choices=tuple(CONDITIONS))
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return worker(args)
    assert args.prior_matrix is not None and args.prior_matrix_sha256 is not None
    args.prior_matrix = args.prior_matrix.resolve()
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
