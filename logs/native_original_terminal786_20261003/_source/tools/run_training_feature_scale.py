"""F2: matched current-package training with normalized or raw features."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_foundation_recipe as foundation

SCHEMA = 'trifusion-training-feature-scale-v1'
VARIANTS = ('normalized', 'raw')
original_build_core = foundation.build_core
original_train_loader = foundation.train_loader
original_loss_values = foundation.loss_values


class FeatureScaleFoundation(nn.Module):
    def __init__(self, baseline, variant):
        super().__init__()
        self.signal = baseline.signal
        self.neck = baseline.neck
        self.classifier = baseline.classifier
        self.variant = variant

    def forward(self, batch, *, return_aux=False):
        raw = self.signal(batch['images'], cam_label=batch['camera_ids'], training=False).float()
        normalized = F.normalize(raw, dim=1)
        if not return_aux:
            return normalized
        feature = normalized if self.variant == 'normalized' else raw
        return {'fused': feature, 'logits': self.classifier(self.neck(feature))}


def build_core(args, protocol):
    baseline_args = argparse.Namespace(**dict(vars(args), recipe='current'))
    baseline, cfg, binding = original_build_core(baseline_args, protocol)
    model = FeatureScaleFoundation(baseline, args.recipe).cuda()
    assert foundation.runner._module_state_sha256(model) == binding['initial_model_state_sha256']
    binding.update(architecture=SCHEMA, recipe=args.recipe, base_recipe='current',
                   training_feature_scaling=args.recipe, deployment_feature_scaling='L2',
                   entry_sha256=foundation.runner.sha256(Path(__file__)))
    return model, cfg, binding


class BatchOrderLoader:
    def __init__(self, loader, path):
        self.loader = loader
        self.path = path
        self.epoch = 0
        self.path.touch(exist_ok=False)

    def __iter__(self):
        self.epoch += 1
        with self.path.open('a') as log:
            for index, raw in enumerate(self.loader):
                yield raw
                # M0 stops before processing its ninth fetched batch.
                log.write(json.dumps({'epoch':self.epoch, 'batch':index,
                    'labels':raw[1].tolist(), 'cameras':raw[2].tolist(),
                    'paths':list(raw[4])})+'\n')
                log.flush()


def train_loader(args, protocol, cfg):
    baseline_args = argparse.Namespace(**dict(vars(args), recipe='current'))
    return BatchOrderLoader(original_train_loader(baseline_args, protocol, cfg),
                            args.output_dir/'training_batch_order.jsonl')


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = original_loss_values(args, output, labels, cameras, loss_fn)
    norms = output['fused'].detach().norm(dim=1)
    values.update(training_feature_norm_mean=float(norms.mean()),
                  training_feature_norm_min=float(norms.min()),
                  training_feature_norm_max=float(norms.max()))
    return loss, values


def configure():
    foundation.SCHEMA = SCHEMA
    foundation.RECIPES = VARIANTS
    foundation.build_core = build_core
    foundation.train_loader = train_loader
    foundation.loss_values = loss_values


if __name__ == '__main__':
    configure()
    foundation.main()
