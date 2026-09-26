#!/usr/bin/env python3
"""Run the published Signal trainer with the verified RGBNT100 Gram fix."""

import argparse
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparators/Signal-cd1b0a6'


def with_precise_epoch_metrics(evaluate):
    def evaluate_and_log(cfg, model, val_loader, device, evaluator, epoch, logger,
                         return_pattern=1, sge='CLS'):
        result = evaluate(cfg, model, val_loader, device, evaluator, epoch, logger,
                          return_pattern=return_pattern, sge=sge)
        mAP, cmc = result
        logger.info('SIGNAL_EPOCH_METRICS %s', json.dumps({
            'epoch': epoch,
            'metrics': {'mAP': float(mAP) * 100,
                        **{f'Rank-{rank}': float(cmc[rank - 1]) * 100
                           for rank in (1, 5, 10)}},
        }))
        return result

    return evaluate_and_log


if __name__ == '__main__':
    dataset = sys.argv[1]
    assert dataset in ('RGBNT201', 'RGBNT100', 'MSVR310')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--amp-audit-output', type=Path)
    audit_args, original_args = parser.parse_known_args(sys.argv[2:])
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SOURCE))
    if audit_args.amp_audit_output is not None:
        from torch.cuda import amp
        from tools.signal_amp_audit import audited_scaler_class
        amp.GradScaler = audited_scaler_class(amp.GradScaler, audit_args.amp_audit_output)
    if dataset == 'RGBNT100':
        from modeling.AddModule import useB
        from tools.signal_gram_stable import signal_gram_volume_stable
        useB.volume_computation3 = signal_gram_volume_stable
    from engine import processor
    processor.training_neat_eval = with_precise_epoch_metrics(processor.training_neat_eval)
    sys.argv = [str(SOURCE / 'train.py'), *original_args]
    runpy.run_path(str(SOURCE / 'train.py'), run_name='__main__')
