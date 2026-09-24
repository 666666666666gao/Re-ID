#!/usr/bin/env python3
"""Run Signal's published module-free baseline on three datasets, one GPU."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparators/Signal-cd1b0a6'
DATASET_ROOT = ROOT / 'artifacts/signal_backbone_dataset_root'
CLIP = ROOT / 'pertrained-model/ViT-B-16.pt'
OUTPUT_ROOT = ROOT / 'trained-model/signal_plain_baseline_20260924_r2'
LOG_ROOT = ROOT / 'logs/signal_plain_baseline_20260924_r2'
CAMPAIGN = ROOT / 'logs/official_extra_seed46_20260924/campaign.json'
EPOCHS = {'MSVR310': 50, 'RGBNT201': 50, 'RGBNT100': 30}


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert args.gpu == 3
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = LOG_ROOT / 'campaign.json'
    assert not status_path.exists()
    status = {'schema': 'signal-plain-baseline-campaign-v1', 'status': 'WAITING',
              'started_at': stamp(), 'gpu': args.gpu, 'source': str(SOURCE),
              'clip_weight': str(CLIP), 'dataset_root': str(DATASET_ROOT),
              'dataset_order': list(EPOCHS), 'jobs': []}

    def save():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    save()
    while True:
        prior = json.loads(CAMPAIGN.read_text(encoding='utf-8'))
        rows = [row for row in prior['jobs'] if row.get('gpu') == args.gpu]
        if rows and all(row['status'] == 'COMPLETE' for row in rows):
            break
        time.sleep(240)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    status['status'] = 'RUNNING'
    save()
    for dataset, epochs in EPOCHS.items():
        job_log_dir = LOG_ROOT / dataset
        job_log_dir.mkdir(parents=True, exist_ok=True)
        job = {'dataset': dataset, 'status': 'TRAINING', 'started_at': stamp(),
               'epochs': epochs, 'seed': 1234, 'checkpoint_policy': 'fixed_final_epoch'}
        status['jobs'].append(job)
        save()
        command = [
            sys.executable, '-B', '-u', str(SOURCE / 'train.py'),
            '--config_file', str(SOURCE / f'configs/{dataset}/Signal.yml'),
            'MODEL.USE_A', 'False', 'MODEL.USE_B', 'False',
            'MODEL.PRETRAIN_PATH_T', str(CLIP),
            'DATASETS.ROOT_DIR', str(DATASET_ROOT),
            'SOLVER.EVAL_PERIOD', str(epochs),
            'SOLVER.CHECKPOINT_PERIOD', '1000',
            'OUTPUT_DIR', str(OUTPUT_ROOT), 'ckpt_save_path', dataset,
        ]
        with (job_log_dir / 'train.log').open('x', encoding='utf-8') as handle:
            subprocess.run(command, cwd=job_log_dir, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        job['status'] = 'EVALUATING'
        job['trained_at'] = stamp()
        save()
        checkpoint = OUTPUT_ROOT / dataset / 'Signalbest.pth'
        result_path = job_log_dir / 'metrics.json'
        evaluation = [
            sys.executable, '-B', '-u', str(ROOT / 'tools/evaluate_signal_plain_baseline.py'),
            '--dataset', dataset, '--signal-source', str(SOURCE),
            '--dataset-root', str(DATASET_ROOT), '--clip-weight', str(CLIP),
            '--checkpoint', str(checkpoint), '--output', str(result_path),
        ]
        with (job_log_dir / 'evaluate.log').open('x', encoding='utf-8') as handle:
            subprocess.run(evaluation, cwd=job_log_dir, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['dataset'] == dataset
        job['status'] = 'COMPLETE'
        job['completed_at'] = stamp()
        job['checkpoint'] = str(checkpoint)
        job['checkpoint_sha256'] = result['checkpoint_sha256']
        job['metrics_path'] = str(result_path)
        job['metrics'] = result['metrics']
        save()
    status['status'] = 'COMPLETE'
    status['completed_at'] = stamp()
    save()


if __name__ == '__main__':
    main()
