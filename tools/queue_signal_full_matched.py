#!/usr/bin/env python3
"""Train complete Signal from the same public CLIP start as the plain baseline."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparators/Signal-cd1b0a6'
DATASET_ROOT = ROOT / 'artifacts/signal_backbone_dataset_root'
CLIP = ROOT / 'pertrained-model/ViT-B-16.pt'
OUTPUT_ROOT = ROOT / 'trained-model/signal_full_matched_20260925'
LOG_ROOT = ROOT / 'logs/signal_full_matched_20260925'
EPOCHS = {'MSVR310': 50, 'RGBNT201': 50, 'RGBNT100': 30}


def stamp():
    return datetime.now().astimezone().isoformat()


def finite_training_log(path):
    losses = [float(value) for value in re.findall(r'Loss: ([^,]+), Acc:',
                                                   path.read_text(encoding='utf-8'))]
    assert losses and all(math.isfinite(value) for value in losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dataset', choices=tuple(EPOCHS), required=True)
    parser.add_argument('--after-campaign', type=Path, required=True)
    parser.add_argument('--selection', choices=('fixed', 'best_map'), default='fixed')
    args = parser.parse_args()
    assert ROOT == Path('/data/gaob/Re-ID/Trifusion')
    assert args.gpu in (0, 1, 2, 3)
    assert CLIP.is_file() and SOURCE.is_dir() and DATASET_ROOT.is_dir()
    log_root = LOG_ROOT if args.selection == 'fixed' else ROOT / 'logs/signal_full_best_map_20260926'
    output_root = OUTPUT_ROOT if args.selection == 'fixed' else ROOT / 'trained-model/signal_full_best_map_20260926'
    log_dir = log_root / args.dataset
    assert not log_dir.exists() and not (output_root / args.dataset).exists()
    log_dir.mkdir(parents=True)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = (args.dataset,)
    status_path = log_dir / 'campaign.json'
    status = {'schema': 'signal-full-matched-campaign-v1', 'status': 'WAITING',
              'created_at': stamp(), 'gpu': args.gpu, 'seed': 1234,
              'dataset_order': datasets, 'public_clip': str(CLIP),
              'source': str(SOURCE), 'after_campaign': str(args.after_campaign),
              'selection': args.selection,
              'jobs': []}

    def save():
        status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')

    def run_train(dataset, epochs, directory, log_path, *, m0):
        command = [sys.executable, '-B', '-u', str(ROOT / 'tools/train_signal_full_author.py'), dataset,
                   '--config_file', str(SOURCE / f'configs/{dataset}/Signal.yml'),
                   'MODEL.USE_A', 'True', 'MODEL.USE_B', 'True',
                   'MODEL.PRETRAIN_PATH_T', str(CLIP),
                   'DATASETS.ROOT_DIR', str(DATASET_ROOT),
                   'SOLVER.SEED', '1234',
                   'SOLVER.MAX_EPOCHS', str(1 if m0 else epochs),
                   'SOLVER.EVAL_PERIOD', str(1 if args.selection == 'best_map' and not m0 else epochs),
                   'SOLVER.CHECKPOINT_PERIOD', '1000',
                   'OUTPUT_DIR', str(directory), 'ckpt_save_path', dataset]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        with log_path.open('x', encoding='utf-8') as handle:
            subprocess.run(command, cwd=log_path.parent, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        finite_training_log(log_path)

    save()
    while True:
        prior = json.loads(args.after_campaign.read_text(encoding='utf-8'))
        if prior['status'] == 'COMPLETE':
            break
        assert prior['status'] == 'RUNNING'
        time.sleep(240)
    status['status'] = 'RUNNING'
    status['started_at'] = stamp()
    save()
    assert shutil.disk_usage('/data').free > 3 * 1024**3
    for dataset in datasets:
        epochs = EPOCHS[dataset]
        job = {'dataset': dataset, 'epochs': epochs, 'seed': 1234,
               'status': 'M0', 'started_at': stamp(),
               'checkpoint_policy': 'official_mAP_each_epoch' if args.selection == 'best_map' else 'fixed_final_epoch'}
        status['jobs'].append(job)
        save()
        run_train(dataset, epochs, log_dir / 'm0', log_dir / 'm0.log', m0=True)
        job['status'] = 'TRAINING'
        job['m0_at'] = stamp()
        save()
        run_train(dataset, epochs, output_root, log_dir / 'train.log', m0=False)
        checkpoint = output_root / dataset / 'Signalbest.pth'
        assert checkpoint.is_file()
        job['status'] = 'EVALUATING'
        job['trained_at'] = stamp()
        save()
        result_path = log_dir / 'metrics.json'
        evaluation = [sys.executable, '-B', '-u',
                      str(ROOT / 'tools/evaluate_signal_plain_baseline.py'),
                      '--dataset', dataset, '--signal-source', str(SOURCE),
                      '--dataset-root', str(DATASET_ROOT), '--clip-weight', str(CLIP),
                      '--checkpoint', str(checkpoint), '--output', str(result_path),
                      '--full-signal']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        with (log_dir / 'evaluate.log').open('x', encoding='utf-8') as handle:
            subprocess.run(evaluation, cwd=log_dir, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['dataset'] == dataset
        assert result['signal_sim_gam_lam'] is True and result['feature_dim'] == 3072
        job.update(status='COMPLETE', completed_at=stamp(), checkpoint=str(checkpoint),
                   checkpoint_sha256=result['checkpoint_sha256'],
                   metrics_path=str(result_path), metrics=result['metrics'],
                   free_disk_bytes=shutil.disk_usage('/data').free)
        save()
    status['status'] = 'COMPLETE'
    status['completed_at'] = stamp()
    save()


if __name__ == '__main__':
    main()
