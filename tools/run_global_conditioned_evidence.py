"""Single-seed RGBNT201 formation contrast; unchanged raw author objectives."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.global_conditioned_evidence import GlobalConditionedEvidence

SCHEMA = 'trifusion-global-conditioned-evidence-v1'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition
original_loss_values = previous.loss_values
original_training_batch = inner.runner._training_batch


def training_batch(raw):
    batch, labels = original_training_batch(raw)
    batch['identity_labels'] = labels
    return batch, labels


def build_core(args, protocol):
    assert args.dataset == 'RGBNT201' and args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    assert isinstance(model.evidence_model, GlobalConditionedEvidence)
    model.evidence_model.exposure = args.exposure
    model.evidence_model.mask_generator = torch.Generator(device='cuda:0').manual_seed(42)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        evidence_exposure=args.exposure, excluded_count_per_active_modality=32,
        sensitivity_layer=3, sensitivity_vjp_scale=256.0, added_model_parameters=0,
        inference_exclusion=False,
        scope='Training-only role exposure contrast using actual global relationship gradients versus equal-count random exclusion. No semantic-disjointness or retrieval-benefit claim; unchanged raw author objectives and1536 deployment.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args), 'evidence_exposure': args.exposure,
        'sensitivity_layer': 3, 'excluded_count_per_active_modality': 32,
        'sensitivity_vjp_scale': 256.0, 'inference_exclusion': False,
        'added_model_parameters': 0}


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = original_loss_values(args, output, labels, cameras, loss_fn)
    values['evidence_exposure'] = output['evidence_exposure']
    return loss, values


def configure():
    previous.configure()
    inner.RawFeatureSemanticTriFusion = GlobalConditionedEvidence
    inner.clean.control.GlobalTokenTriFusion = GlobalConditionedEvidence
    inner.runner._training_batch = training_batch
    inner.loss_values = loss_values
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201',), required=True)
    parser.add_argument('--exposure', choices=('gradient', 'random'), required=True)
    parser.add_argument('--mode', choices=('prepare', 'm0', 'train', 'evaluate'), required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.variant = args.recipe = 'semantic'
    args.seed, args.epochs = 42, 50
    for name in ('protocol', 'signal_source', 'clip_weight', 'initialization', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.baseline_sha256 = inner.runner.sha256(args.clip_weight)
    protocol = inner.runner.read_protocol(args.protocol, args.dataset)
    configure()
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        _model, _cfg, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps({'schema': SCHEMA,
            'status': 'INITIALIZATION_VERIFIED', 'binding': binding,
            'prepared_at': datetime.now().astimezone().isoformat()}, indent=2)+'\n')
    else:
        (inner.foundation.evaluate if args.mode == 'evaluate' else inner.train)(args, protocol)
