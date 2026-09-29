#!/usr/bin/env python3
"""Three evidence-depth controls with the existing full50 training and scorer."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'modeling'))
from tools import run_correspondence_context_identity as entry
from trifusion.cross_depth_role_state import CrossDepthRoleStateTriFusion

BASE_BUILD, BASE_EVALUATE = entry.build, entry.evaluate
DEPTH_MODES = ('mixed_once', 'depth_mean', 'depth_recurrent')
DEPTH_MODE = None
ARCHITECTURE = 'cross_depth_role_state_v1'


def build(args, protocol):
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    assert model.depth_mode == DEPTH_MODE
    binding.update(architecture=ARCHITECTURE, depth_mode=DEPTH_MODE,
                   depth_logits_trainable=False,
                   depth_source_sha256=entry.sha256(ROOT / 'modeling/trifusion/cross_depth_role_state.py'),
                   entry_sha256=entry.sha256(Path(__file__)),
                   reused_context_entry_sha256=entry.sha256(ROOT / 'tools/run_correspondence_context_identity.py'))
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({'schema': 'trifusion-cross-depth-role-state-v1',
                'dataset': args.dataset, 'seed': args.seed, 'epoch': epoch,
                'protocol_sha256': entry.sha256(args.protocol), 'baseline_sha256': args.baseline_sha256,
                'variants': {'m1': args.m1, 'm2': args.m2, 'm3': args.m3},
                'condition': entry.CONDITION.copy(), 'depth_mode': DEPTH_MODE,
                'metrics': metrics, 'state': entry.runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-cross-depth-role-state-v1'
    assert payload['dataset'] == args.dataset and payload['seed'] == args.seed
    assert payload['protocol_sha256'] == entry.sha256(args.protocol)
    assert payload['baseline_sha256'] == args.baseline_sha256
    assert payload['variants'] == {'m1': args.m1, 'm2': args.m2, 'm3': args.m3}
    assert payload['condition'] == entry.CONDITION and payload['depth_mode'] == DEPTH_MODE == model.depth_mode
    assert set(payload['state']) == set(entry.runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload['state'])
    model.load_state_dict(state, strict=True)
    return payload


def evaluate(args, protocol):
    BASE_EVALUATE(args, protocol)
    path = args.output_dir / 'official_metrics.json'
    result = json.loads(path.read_text())
    result.update(architecture=ARCHITECTURE, depth_mode=DEPTH_MODE)
    path.write_text(json.dumps(result, indent=2) + '\n')


def main():
    global DEPTH_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--depth-mode', choices=DEPTH_MODES, required=True)
    options, remaining = parser.parse_known_args()
    DEPTH_MODE = options.depth_mode
    entry.ContextIdentityTriFusion = partial(CrossDepthRoleStateTriFusion, depth_mode=DEPTH_MODE)
    entry.build, entry.evaluate = build, evaluate
    entry.save_checkpoint, entry.load_checkpoint = save_checkpoint, load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    entry.main()


if __name__ == '__main__':
    main()
