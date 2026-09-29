#!/usr/bin/env python3
"""Four M3 address/predictor conditions using the unchanged full-50-epoch runner."""

import argparse
from functools import partial
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'modeling'))

from tools import run_correspondence_roles as runner
from tools.official_three_dataset_model import sha256
from trifusion.correspondence_role_prediction import RolePredictionTriFusion

BASE_BUILD = runner.build
PREDICTION = {}


def build(args, protocol):
    assert args.m1 and args.m2 and args.m3 and args.width == 128 and args.pred_weight == 0.1
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    binding.update(
        architecture='correspondence_role_prediction_v1', prediction=PREDICTION.copy(),
        entry_sha256=sha256(Path(__file__)),
        reused_entry_sha256=sha256(ROOT / 'tools/run_correspondence_roles.py'),
        prediction_source_sha256=sha256(ROOT / 'modeling/trifusion/correspondence_role_prediction.py'),
        prediction_head_parameters=sum(p.numel() for p in model.predictors.parameters()),
        prediction_address_gradient='student' if PREDICTION['address_mode'] == 'own' else 'detached_teacher',
    )
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({'schema': 'trifusion-correspondence-role-prediction-v1',
                'dataset': args.dataset, 'seed': args.seed, 'epoch': epoch,
                'protocol_sha256': sha256(args.protocol), 'baseline_sha256': args.baseline_sha256,
                'variants': {'m1': args.m1, 'm2': args.m2, 'm3': args.m3},
                'prediction': PREDICTION.copy(), 'metrics': metrics,
                'state': runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-correspondence-role-prediction-v1'
    assert payload['dataset'] == args.dataset and payload['seed'] == args.seed
    assert payload['protocol_sha256'] == sha256(args.protocol)
    assert payload['baseline_sha256'] == args.baseline_sha256
    assert payload['variants'] == {'m1': args.m1, 'm2': args.m2, 'm3': args.m3}
    assert payload['prediction'] == PREDICTION
    assert set(payload['state']) == set(runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload['state'])
    model.load_state_dict(state, strict=True)
    return payload


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--address-mode', choices=('own', 'matched'), required=True)
    parser.add_argument('--prediction-mode', choices=('direct', 'predictor'), required=True)
    options, remaining = parser.parse_known_args()
    PREDICTION.update(address_mode=options.address_mode, prediction_mode=options.prediction_mode)
    runner.CorrespondenceTriFusion = partial(RolePredictionTriFusion, **PREDICTION)
    runner.build = build
    runner.save_checkpoint = save_checkpoint
    runner.load_checkpoint = load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == '__main__':
    main()
