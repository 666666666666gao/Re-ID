#!/usr/bin/env python3
"""Evaluate a separately trained plain or complete Signal checkpoint."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('RGBNT201', 'RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--signal-source', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--clip-weight', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-signal', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.signal_source))
    from config import cfg
    from data import make_dataloader
    from modeling import make_frame
    from utils.metrics import R1_mAP, R1_mAP_eval

    cfg.merge_from_file(str(args.signal_source / f'configs/{args.dataset}/Signal.yml'))
    overrides = ['MODEL.PRETRAIN_PATH_T', str(args.clip_weight),
                 'DATASETS.ROOT_DIR', str(args.dataset_root)]
    if not args.full_signal:
        overrides.extend(['MODEL.USE_A', 'False', 'MODEL.USE_B', 'False'])
    cfg.merge_from_list(overrides)
    cfg.freeze()
    if args.full_signal and args.dataset == 'RGBNT100':
        from modeling.AddModule import useB
        from tools.signal_gram_stable import signal_gram_volume_stable
        useB.volume_computation3 = signal_gram_volume_stable
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if args.full_signal:
        assert hasattr(model, 'SIM') and hasattr(model, 'AlignM')
        assert model.use_A is True and model.use_B is True
    else:
        assert not hasattr(model, 'SIM') and not hasattr(model, 'AlignM')
        assert model.use_A is False and model.use_B is False
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.cuda().eval()

    if args.dataset == 'MSVR310':
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    feature_dim = None
    with torch.no_grad():
        for img, pid, camid, camids, target_view, paths in val_loader:
            inputs = {key: value.cuda(non_blocking=True) for key, value in img.items()}
            features = model(inputs, cam_label=camids.cuda(non_blocking=True),
                             view_label=target_view.cuda(non_blocking=True), training=False,
                             sge=cfg.MODEL.stageName)
            feature_dim = features.shape[1]
            if args.dataset == 'MSVR310':
                evaluator.update((features, pid, camid, target_view, paths))
            else:
                evaluator.update((features, pid, camid, paths))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    assert feature_dim == (3072 if args.full_signal else 1536)
    metrics = {'mAP': 100 * float(mAP), 'Rank-1': 100 * float(cmc[0]),
               'Rank-5': 100 * float(cmc[4]), 'Rank-10': 100 * float(cmc[9])}
    result = {
        'schema': ('signal-full-matched-official-v1' if args.full_signal
                   else 'signal-plain-baseline-official-v1'),
        'status': 'COMPLETE',
        'evaluated_at': datetime.now().astimezone().isoformat(),
        'dataset': args.dataset,
        'model': ('Signal_SIM_GAM_LAM' if args.full_signal
                  else 'CLIP_ViT_B_16_RGB_NIR_TIR_CLS_concat'),
        'signal_sim_gam_lam': args.full_signal,
        'trifusion_modules': False,
        'feature_dim': feature_dim,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'source': str(args.signal_source),
        'configuration': str(args.signal_source / f'configs/{args.dataset}/Signal.yml'),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256(args.checkpoint),
        'query_count': num_query,
        'gallery_count': len(evaluator.pids) - num_query,
        'filter': 'same-identity same-scene' if args.dataset == 'MSVR310' else 'same-identity same-camera',
        'feature_normalization': cfg.TEST.FEAT_NORM,
        'reranking': False,
        'metrics': metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
