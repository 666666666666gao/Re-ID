#!/usr/bin/env python3
"""Run the published Signal trainer with the verified RGBNT100 Gram fix."""

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparators/Signal-cd1b0a6'


if __name__ == '__main__':
    dataset = sys.argv[1]
    assert dataset in ('RGBNT201', 'RGBNT100', 'MSVR310')
    original_args = sys.argv[2:]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SOURCE))
    if dataset == 'RGBNT100':
        from modeling.AddModule import useB
        from tools.signal_gram_stable import signal_gram_volume_stable
        useB.volume_computation3 = signal_gram_volume_stable
    sys.argv = [str(SOURCE / 'train.py'), *original_args]
    runpy.run_path(str(SOURCE / 'train.py'), run_name='__main__')
