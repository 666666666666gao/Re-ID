#!/usr/bin/env python3
"""Run the published Signal trainer with the verified RGBNT100 Gram fix."""

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparators/Signal-cd1b0a6'


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
    sys.argv = [str(SOURCE / 'train.py'), *original_args]
    runpy.run_path(str(SOURCE / 'train.py'), run_name='__main__')
