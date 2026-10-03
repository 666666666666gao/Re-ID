"""Matched full50 author-package semantic/native evidence controls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_foundation_recipe as foundation
from tools.run_training_feature_scale import BatchOrderLoader
from trifusion.independent_native_roles import (
    RawFeatureSemanticTriFusion, IndependentNativeTriFusion,
)
from trifusion.evidence_author_heads import (
    SharedGlobalRawFeatures, AuthorHeadEvidence, build_author_optimizer_and_loss,
)

clean = foundation.clean
runner = foundation.runner
SCHEMA = 'trifusion-independent-native-evidence-v1'
VARIANTS = ('global_only', 'semantic', 'native')
original_train_loader = foundation.train_loader
original_loss_values = foundation.loss_values


def condition(args):
    return {'variant': args.variant, 'foundation_option': 'author', 'seed': 42, 'epochs': 50,
            'public_clip_sha256': args.baseline_sha256,
            'initialization_sha256': runner.sha256(args.initialization),
            'deployment': 'L2_1536', 'auxiliary_tasks': 'none'}


def build_core(args, protocol):
    values = dict(vars(args), readout='roles')
    initialized, cfg, _config, original = clean.build_core(argparse.Namespace(**values), protocol)
    assert isinstance(initialized, RawFeatureSemanticTriFusion)
    shared = {name: runner._module_state_sha256(getattr(initialized, name))
              for name in ('backbone', 'neck', 'classifier')}
    if args.variant == 'native':
        with torch.random.fork_rng(devices=[]):
            evidence = IndependentNativeTriFusion(
                initialized.backbone.signal, num_classes=len(protocol['train_label_map']),
                grid=initialized.roles.grid, width=128, m1=True, m2=True, m3=False,
                mamba_factory=runner.production_mamba_factory, token_mode='static',
                query_mode='context', auxiliary_target='none',
            )
        state = evidence.state_dict()
        state.update(initialized.state_dict())
        evidence.load_state_dict(state, strict=True)
    elif args.variant == 'global_only':
        evidence = SharedGlobalRawFeatures(initialized)
    else:
        evidence = initialized
    assert shared == {name: runner._module_state_sha256(getattr(evidence, name)) for name in shared}
    model = AuthorHeadEvidence(evidence).cuda()
    del initialized
    visual = model.signal.clip_vision_encoder.base
    camera = model.signal.clip_vision_encoder.cv_embed
    # Constructing a new role backbone freezes Signal again; restore intended
    # visual/camera ownership after all variant constructors, for every arm.
    visual.requires_grad_(True)
    camera.requires_grad_(True)
    cfg = cfg.clone()
    cfg.defrost()
    cfg.SOLVER.MAX_EPOCHS = 50
    cfg.SOLVER.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.freeze()
    assert not cfg.MODEL.FROZEN and cfg.MODEL.NO_MARGIN
    assert not cfg.MODEL.PROMPT and not cfg.MODEL.ADAPTER
    assert len(list(visual.parameters())) == 152
    assert all(parameter.dtype == torch.float32 and parameter.requires_grad for parameter in visual.parameters())
    binding = {
        'architecture': SCHEMA, 'dataset': args.dataset, 'variant': args.variant, 'recipe': args.variant,
        'foundation_option': 'author', 'seed': 42,
        'public_clip_sha256': args.baseline_sha256, 'protocol_sha256': runner.sha256(args.protocol),
        'author_source_commit': original['author_source_commit'],
        'visual_initial_sha256': runner._module_state_sha256(visual),
        'camera_initial_sha256': clean.tensor_digest(camera),
        'shared_initializer_sha256': shared,
        'initial_model_state_sha256': runner._module_state_sha256(model),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'trainable_parameter_tensors': sum(parameter.requires_grad for parameter in model.parameters()),
        'batch_size': cfg.SOLVER.IMS_PER_BATCH, 'num_instances': cfg.DATALOADER.NUM_INSTANCE,
        'head_names': [list(names) for names in model.head_names],
        'cfg_yaml': cfg.dump(), 'entry_sha256': runner.sha256(Path(__file__)),
        'scope': 'Fresh public CLIP/camera/heads, shared adaptation and optional semantic/native roles. No ReID weights, SIM, AlignM, teacher, new loss or external data.'
    }
    return model, cfg, binding


def train_loader(args, protocol, cfg):
    return BatchOrderLoader(original_train_loader(args, protocol, cfg),
                            args.output_dir/'training_batch_order.jsonl')


def optimization(args, model, cfg):
    optimizer, loss_fn = build_author_optimizer_and_loss(cfg, model)
    if args.dataset == 'MSVR310':
        from solver.lr_scheduler310 import WarmupMultiStepLR
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        from solver.scheduler_factory import create_scheduler
        scheduler = create_scheduler(cfg, optimizer)
    # Observe the actual unscaled gradients and effective optimizer updates.
    if args.mode == 'm0':
        global M0_DIAGNOSTICS
        M0_DIAGNOSTICS = M0Diagnostics(model, optimizer)
        optimizer.register_step_post_hook(M0_DIAGNOSTICS.after_step)
    return optimizer, scheduler, loss_fn


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = original_loss_values(argparse.Namespace(**dict(vars(args), recipe='author')), output, labels, cameras, loss_fn)
    values['raw_feature_norm_mean'] = float(output['raw_fused'].detach().norm(dim=1).mean())
    values['deployment_feature_norm_mean'] = float(output['fused'].detach().norm(dim=1).mean())
    return loss, values



class M0Diagnostics:
    def __init__(self, model, optimizer):
        self.model = model
        self.detail = {name: p for name, p in model.named_parameters()
                       if '.detail_reader.' in name and p.requires_grad}
        self.initial = {name: p.detach().clone() for name, p in self.detail.items()}
        self.updates = []
        names = {id(p): name for name, p in model.named_parameters()}
        self.groups = [{'names': [names[id(p)] for p in group['params']],
                        'lr': group['lr'], 'weight_decay': group['weight_decay']}
                       for group in optimizer.param_groups]

    def after_step(self, _optimizer, _args, _kwargs):
        assert self.model.signal.training
        assert all(getattr(self.model.signal, neck).training for neck, _ in self.model.head_names)
        self.updates.append({name: {
            'unscaled_gradient_max_abs': float(p.grad.detach().abs().max()) if p.grad is not None else None,
            'parameter_delta_from_initial_max_abs': float((p.detach() - self.initial[name]).abs().max()),
        } for name, p in self.detail.items()})

    def result(self):
        assert len(self.updates) == 8
        if self.detail:
            assert len(self.detail) == 14 and sum(p.numel() for p in self.detail.values()) == 159296
            assert all(any(row[name]['unscaled_gradient_max_abs'] is not None
                           and row[name]['unscaled_gradient_max_abs'] > 0 for row in self.updates)
                       for name in self.detail)
            assert all(self.updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                       for name in self.detail)
        tracked = {neck: int(getattr(self.model.signal, neck).num_batches_tracked)
                   for neck, _ in self.model.head_names}
        assert all(value == 8 for value in tracked.values())
        return {'effective_optimizer_updates': 8, 'optimizer_groups_at_construction': self.groups,
                'author_bn_batches_tracked': tracked, 'detail_parameters': list(self.detail),
                'detail_updates': self.updates,
                'boundary': 'Engineering gradient/update support only; no evidence of retrieval benefit.'}


M0_DIAGNOSTICS = None


def train(args, protocol):
    foundation.train(args, protocol)
    if args.mode == 'm0':
        result = M0_DIAGNOSTICS.result()
        path = args.output_dir / 'training.json'
        receipt = json.loads(path.read_text())
        receipt['production_m0_diagnostics'] = result
        path.write_text(json.dumps(receipt, indent=2) + '\n')


def configure():
    clean.control.GlobalTokenTriFusion = RawFeatureSemanticTriFusion
    foundation.SCHEMA = SCHEMA
    foundation.condition = condition
    foundation.build_core = build_core
    foundation.train_loader = train_loader
    foundation.optimization = optimization
    foundation.loss_values = loss_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201', 'RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--mode', choices=('prepare', 'm0', 'train', 'evaluate'), required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--epochs', type=int, choices=(50,), default=50)
    args = parser.parse_args()
    for name in ('protocol', 'signal_source', 'clip_weight', 'initialization', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.recipe = args.variant
    args.baseline_sha256 = runner.sha256(args.clip_weight)
    configure()
    protocol = runner.read_protocol(args.protocol, args.dataset)
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        _model, _cfg, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps({
            'schema': SCHEMA, 'status': 'INITIALIZATION_VERIFIED', 'binding': binding,
            'prepared_at': datetime.now().astimezone().isoformat(),
        }, indent=2)+'\n')
    else:
        (foundation.evaluate if args.mode == 'evaluate' else train)(args, protocol)


if __name__ == '__main__':
    main()
