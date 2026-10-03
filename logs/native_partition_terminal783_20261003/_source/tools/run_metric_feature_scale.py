"""F3: keep BN/CE input normalized; isolate the Triplet input scale."""
import argparse
from pathlib import Path
import sys

from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_training_feature_scale as f2

foundation = f2.foundation
SCHEMA = 'trifusion-metric-feature-scale-v1'
VARIANTS = ('normalized', 'metric_raw')


class MetricFeatureScaleFoundation(nn.Module):
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
        metric_feature = normalized if self.variant == 'normalized' else raw
        return {'fused': metric_feature, 'ce_feature': normalized,
                'logits': self.classifier(self.neck(normalized))}


def build_core(args, protocol):
    baseline_args = argparse.Namespace(**dict(vars(args), recipe='current'))
    baseline, cfg, binding = f2.original_build_core(baseline_args, protocol)
    model = MetricFeatureScaleFoundation(baseline, args.recipe).cuda()
    assert foundation.runner._module_state_sha256(model) == binding['initial_model_state_sha256']
    binding.update(architecture=SCHEMA, recipe=args.recipe, base_recipe='current',
                   ce_training_feature_scaling='normalized',
                   metric_training_feature_scaling='normalized' if args.recipe=='normalized' else 'raw',
                   deployment_feature_scaling='L2',
                   entry_sha256=foundation.runner.sha256(Path(__file__)))
    return model, cfg, binding


def train_loader(args, protocol, cfg):
    baseline_args = argparse.Namespace(**dict(vars(args), recipe='current'))
    return f2.BatchOrderLoader(f2.original_train_loader(baseline_args, protocol, cfg),
                              args.output_dir/'training_batch_order.jsonl')


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = f2.loss_values(args, output, labels, cameras, loss_fn)
    norms = output['ce_feature'].detach().norm(dim=1)
    values.update(ce_feature_norm_mean=float(norms.mean()),
                  ce_feature_norm_min=float(norms.min()),
                  ce_feature_norm_max=float(norms.max()))
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
