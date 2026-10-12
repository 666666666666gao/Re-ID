"""Draft actual-value identity relation supervision; unchanged1536 inference."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.region_use_heads import RegionUseHeads
from trifusion.region_use_objective import region_use_loss

SCHEMA = 'trifusion-region-use-supervision-v1'
CONDITIONS = ('uniform', 'ambiguity')
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition
original_loss_values = previous.loss_values
original_training_batch = inner.runner._training_batch
ENVIRONMENTS = {}
CURRENT_MODEL = None


def training_batch(raw):
    batch, labels = original_training_batch(raw)
    batch['retrieval_environments'] = torch.tensor(
        [ENVIRONMENTS[Path(path).name] for path in raw[4]], device=labels.device)
    return batch, labels


def build_core(args, protocol):
    global CURRENT_MODEL
    assert args.variant == 'semantic' and args.comparison in CONDITIONS
    model, cfg, binding = original_build_core(args, protocol)
    assert isinstance(model, RegionUseHeads)
    CURRENT_MODEL = model
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        region_use_comparison=args.comparison, auxiliary_weight=1.0,
        relation_scale=128**0.5, relation_score='half_global_plus_symmetric_mean_region',
        added_model_parameters=0,
        scope='Training-only actual C/T/M region relationship supervision. Original raw author tasks and1536 additive deployment unchanged. No matcher/novel-loss/SOTA claim.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args), 'region_use_comparison': args.comparison,
            'auxiliary_tasks': 'actual_region_identity_relation',
            'auxiliary_weight': 1.0, 'relation_scale': 128**0.5,
            'relation_environment': 'protocol_scene_MSVR310_camera_others',
            'inference_region_comparison': False, 'added_model_parameters': 0}


def loss_values(args, output, labels, cameras, loss_fn):
    original, values = original_loss_values(args, output, labels, cameras, loss_fn)
    auxiliary, stats = region_use_loss(output['actual_regions'], output['shared_global'],
        labels, output['retrieval_environments'], ambiguity_weighted=args.comparison == 'ambiguity')
    values.update(region_use_loss=float(auxiliary.detach()), region_use_support=stats)
    if args.mode == 'm0':
        parameters = [(name, parameter) for name, parameter in CURRENT_MODEL.named_parameters()
            if name.startswith('evidence_model.roles.') and any(part in name
                for part in ('.query_projections.', '.key_projections.', '.value_projections.'))]
        assert len(parameters) == 9
        targets = [output['shared_global'], output['actual_regions']]+[p for _, p in parameters]
        with torch.autocast('cuda', enabled=False):
            gradients = torch.autograd.grad(auxiliary*256.0, targets, retain_graph=True, allow_unused=True)
        assert gradients[0] is None
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        values['region_use_isolated_activity'] = {
            'diagnostic_vjp_scale': 256.0,
            'shared_global_gradient_absent': True,
            'actual_region_gradient_norm': float(gradients[1].float().norm()/256.0) if gradients[1] is not None else 0.0,
            'query_key_value_gradient_norms': {name: float(g.float().norm()/256.0) if g is not None else 0.0
                for (name, _), g in zip(parameters, gradients[2:])}}
    return original+auxiliary, values


def configure():
    previous.configure()
    inner.AuthorHeadEvidence = RegionUseHeads
    inner.runner._training_batch = training_batch
    inner.loss_values = loss_values
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201', 'MSVR310', 'RGBNT100'), required=True)
    parser.add_argument('--comparison', choices=CONDITIONS, required=True)
    parser.add_argument('--mode', choices=('prepare', 'm0', 'train', 'evaluate'), required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.variant = args.recipe = 'semantic'
    args.seed = 42
    args.epochs = 50
    for name in ('protocol', 'signal_source', 'clip_weight', 'initialization', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.baseline_sha256 = inner.runner.sha256(args.clip_weight)
    protocol = inner.runner.read_protocol(args.protocol, args.dataset)
    for records in protocol['records'].values():
        for row in records:
            key = Path(row['paths'][0]).name
            environment = row['scene'] if args.dataset == 'MSVR310' else row['camera']
            assert key not in ENVIRONMENTS or ENVIRONMENTS[key] == environment
            ENVIRONMENTS[key] = environment
    configure()
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        _model, _cfg, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps(dict(schema=SCHEMA, status='INITIALIZATION_VERIFIED',
            binding=binding, prepared_at=datetime.now().astimezone().isoformat()), indent=2)+'\n')
    else:
        (inner.foundation.evaluate if args.mode == 'evaluate' else inner.train)(args, protocol)


if __name__ == '__main__':
    main()
