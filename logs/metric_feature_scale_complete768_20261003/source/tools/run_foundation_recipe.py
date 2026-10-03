"""F1: two explicit no-module public-CLIP foundation packages, full50."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_clean_clip_joint as clean

runner = clean.control.runner
SCHEMA = 'trifusion-foundation-recipe-v1'
RECIPES = ('author', 'current')


class PlainFoundation(nn.Module):
    def __init__(self, initialized, recipe):
        super().__init__()
        self.signal = initialized.backbone.signal
        self.recipe = recipe
        if recipe == 'current':
            self.neck = initialized.neck
            self.classifier = initialized.classifier
        else:
            names = ('bottleneck', 'classifier') if self.signal.direct else tuple(
                f'{kind}_{suffix}' for suffix in ('r', 'n', 't')
                for kind in ('bottleneck', 'classifier'))
            for name in names:
                module = getattr(self.signal, name)
                module.weight.requires_grad_(True)
        assert not hasattr(self.signal, 'SIM') and not hasattr(self.signal, 'AlignM')

    def forward(self, batch, *, return_aux=False):
        images = batch['images']
        if self.recipe == 'author' and return_aux:
            output = self.signal(images, cam_label=batch['camera_ids'], training=True)
            assert output[0] == 1
            heads = [(output[index], output[index + 1]) for index in range(1, len(output), 2)]
            return {'heads': heads, 'fused': torch.cat([feat for _, feat in heads], dim=1)}
        raw = self.signal(images, cam_label=batch['camera_ids'], training=False)
        fused = F.normalize(raw.float(), dim=1)
        if return_aux:
            return {'fused': fused, 'logits': self.classifier(self.neck(fused))}
        return fused


def condition(args):
    return {'recipe': args.recipe, 'seed': 42, 'epochs': 50,
            'public_clip_sha256': args.baseline_sha256,
            'initialization_sha256': runner.sha256(args.initialization)}


def build_core(args, protocol):
    values = vars(args).copy()
    values.update(readout='global_only')
    initialized, cfg, _config, original = clean.build_core(argparse.Namespace(**values), protocol)
    head_initial = {name: runner._module_state_sha256(getattr(initialized, name))
                    for name in ('neck', 'classifier')}
    model = PlainFoundation(initialized, args.recipe).cuda()
    del initialized
    cfg = cfg.clone()
    cfg.defrost()
    cfg.SOLVER.MAX_EPOCHS = 50
    cfg.SOLVER.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4
    if args.recipe == 'current':
        cfg.SOLVER.IMS_PER_BATCH = 64
        cfg.DATALOADER.NUM_INSTANCE = 8
    cfg.freeze()
    assert not cfg.MODEL.FROZEN and cfg.MODEL.NO_MARGIN
    assert not cfg.MODEL.PROMPT and not cfg.MODEL.ADAPTER
    visual = model.signal.clip_vision_encoder.base
    camera = model.signal.clip_vision_encoder.cv_embed
    binding = {'architecture': SCHEMA, 'dataset': args.dataset, 'recipe': args.recipe,
               'seed': 42, 'public_clip_sha256': args.baseline_sha256,
               'protocol_sha256': runner.sha256(args.protocol),
               'author_source_commit': original['author_source_commit'],
               'visual_initial_sha256': runner._module_state_sha256(visual),
               'camera_initial_sha256': clean.tensor_digest(camera),
               'current_head_initial_sha256': head_initial,
               'initial_model_state_sha256': runner._module_state_sha256(model),
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'trainable_parameter_tensors': sum(p.requires_grad for p in model.parameters()),
               'batch_size': cfg.SOLVER.IMS_PER_BATCH,
               'num_instances': cfg.DATALOADER.NUM_INSTANCE,
               'entry_sha256': runner.sha256(Path(__file__)),
               'scope': 'Public visual/fresh camera and heads only; no ReID weights, roles or adapters.'}
    assert len(list(visual.parameters())) == 152
    assert all(p.dtype == torch.float32 and p.requires_grad for p in visual.parameters())
    assert camera.requires_grad
    return model, cfg, binding


def build(args, protocol):
    model, cfg, binding = build_core(args, protocol)
    witness = json.loads(args.initialization.read_text())
    assert witness['schema'] == SCHEMA and witness['binding'] == binding
    return model, cfg, binding


def train_loader(args, protocol, cfg):
    if args.recipe == 'current':
        return runner.loader_for(protocol, runner.records_for(protocol, 'train'),
                                 training=True, method='PLAIN_V8', seed=42)
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from data.datasets.bases import ImageDataset
    from data.datasets.make_dataloader import RandomErasing, train_collate_fn
    from data.datasets.sampler import RandomIdentitySampler
    records = runner.records_for(protocol, 'train')
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB), T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN), T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    return DataLoader(ImageDataset(records, transform), batch_size=cfg.SOLVER.IMS_PER_BATCH,
                      sampler=RandomIdentitySampler(records, cfg.SOLVER.IMS_PER_BATCH,
                                                    cfg.DATALOADER.NUM_INSTANCE, 42),
                      num_workers=4, collate_fn=train_collate_fn)


def optimization(args, model, cfg):
    if args.recipe == 'author':
        from layers.make_loss import make_loss
        from solver.make_optimizer import make_optimizer
        loss_fn, center = make_loss(cfg, num_classes=model.signal.num_classes)
        optimizer, _unused_center_optimizer = make_optimizer(cfg, model.signal, center)
        if args.dataset == 'MSVR310':
            from solver.lr_scheduler310 import WarmupMultiStepLR
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS,
                cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            from solver.scheduler_factory import create_scheduler
            scheduler = create_scheduler(cfg, optimizer)
        return optimizer, scheduler, loss_fn
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    visual = {id(p) for p in model.signal.clip_vision_encoder.base.parameters()}
    groups = [{'params': [p for p in named.values() if id(p) not in visual], 'base_lr': 3.5e-4},
              {'params': [p for p in named.values() if id(p) in visual], 'base_lr': 5e-6}]
    return torch.optim.AdamW(groups, weight_decay=1e-4), None, None


def loss_values(args, output, labels, cameras, loss_fn):
    if args.recipe == 'author':
        components = [loss_fn(score=score, feat=feature, target=labels, target_cam=cameras)
                      for score, feature in output['heads']]
        return sum(components), {'head_losses': [float(item.detach()) for item in components]}
    identity = F.cross_entropy(output['logits'], labels, label_smoothing=0.1)
    triplet = runner._batch_hard_triplet(output['fused'], labels, margin=0.3)
    return identity + triplet, {'id': float(identity.detach()), 'triplet': float(triplet.detach())}


def frozen_parameters_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save(path, model, args, epoch, metrics):
    torch.save({'schema': SCHEMA, 'dataset': args.dataset, 'condition': condition(args),
                'protocol_sha256': runner.sha256(args.protocol), 'epoch': epoch, 'metrics': metrics,
                'state': {name: value.detach().cpu() for name, value in model.state_dict().items()}}, path)


def load(path, model, args):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == SCHEMA and payload['dataset'] == args.dataset
    assert payload['condition'] == condition(args)
    assert payload['protocol_sha256'] == runner.sha256(args.protocol)
    model.load_state_dict(payload['state'], strict=True)
    return payload


def train(args, protocol):
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    os.chdir(args.output_dir)
    model, cfg, binding = build(args, protocol)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    frozen_before = frozen_parameters_digest(model)
    optimizer, scheduler, loss_fn = optimization(args, model, cfg)
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in named.values()}
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0)
    loader = train_loader(args, protocol, cfg)
    receipt = {'schema': SCHEMA, 'status': 'RUNNING', 'dataset': args.dataset, 'recipe': args.recipe,
               'seed': 42, 'epochs': 50, 'initializer': binding, 'condition': condition(args),
               'started_at': datetime.now().astimezone().isoformat(), 'history': [],
               'cfg_yaml': cfg.dump(), 'timing_boundary': 'Training and per-epoch evaluation; excludes construction and final strict evaluation.'}
    path = args.output_dir / 'training.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    best = {'mAP': -1.0, 'epoch': None}
    live = set()
    audit_batch = None
    torch.cuda.reset_peak_memory_stats()
    with (args.output_dir / 'training_steps.jsonl').open('x') as log:
        for epoch in range(1, 2 if args.mode == 'm0' else 51):
            started = time.perf_counter()
            if scheduler is not None:
                scheduler.step(epoch)
            else:
                multiplier = runner.learning_rate_multiplier(epoch, max_epochs=50, warmup_epochs=5)
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * multiplier
            model.train()
            losses = []
            for index, raw in enumerate(loader):
                if args.mode == 'm0' and index == 8:
                    break
                batch, labels = runner._training_batch(raw)
                assert labels.shape[0] == binding['batch_size']
                if audit_batch is None:
                    audit_batch = {'images': {name: value[:2].clone() for name, value in batch['images'].items()},
                                   'camera_ids': batch['camera_ids'][:2].clone()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    loss, components = loss_values(args, output, labels, batch['camera_ids'], loss_fn)
                assert bool(torch.isfinite(loss))
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                assert all(torch.isfinite(p.grad).all() for p in named.values() if p.grad is not None)
                live.update(name for name, p in named.items() if p.grad is not None and bool(p.grad.abs().sum() > 0))
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() >= scale
                losses.append(float(loss.detach()))
                log.write(json.dumps({'epoch': epoch, 'batch': index, 'loss': losses[-1], **components,
                                      'lr': [group['lr'] for group in optimizer.param_groups]}) + '\n')
            log.flush()
            row = {'epoch': epoch, 'steps': len(losses), 'mean_loss': float(np.mean(losses)),
                   'seconds': time.perf_counter() - started}
            if args.mode == 'train':
                current_distances = args.output_dir / 'last_epoch_distances.pt'
                metrics = runner.official_metrics(model, protocol, args.signal_source, save_distances=current_distances)
                row['official_fused'] = metrics
                if metrics['mAP'] >= best['mAP']:
                    best = {'mAP': metrics['mAP'], 'epoch': epoch}
                    os.replace(current_distances, args.output_dir / 'best_epoch_distances.pt')
                    save(args.output_dir / 'best_map.pth', model, args, epoch, metrics)
            receipt['history'].append(row)
            path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'event': 'epoch', 'dataset': args.dataset, 'recipe': args.recipe, **row}), flush=True)
    assert frozen_before == frozen_parameters_digest(model)
    visual_after = runner._module_state_sha256(model.signal.clip_vision_encoder.base)
    camera_after = clean.tensor_digest(model.signal.clip_vision_encoder.cv_embed)
    assert visual_after != binding['visual_initial_sha256'] and camera_after != binding['camera_initial_sha256']
    if args.mode == 'm0':
        assert live == set(named), sorted(set(named) - live)
        assert receipt['history'][0]['steps'] == 8
        probe = args.output_dir / 'm0_reload_probe.pth'
        save(probe, model, args, 0, {})
        fresh, _cfg, fresh_binding = build(args, protocol)
        assert fresh_binding == binding
        load(probe, fresh, args)
        model.eval()
        fresh.eval()
        with torch.inference_mode():
            original, reloaded = model(audit_batch), fresh(audit_batch)
        assert torch.allclose(original, reloaded, atol=1e-5, rtol=1e-5)
        receipt['m0'] = {'nonzero_gradient_parameters': len(live), 'trainable_parameters': len(named),
                         'reload_max_abs_difference': float((original - reloaded).abs().max()),
                         'reload_probe_sha256': runner.sha256(probe)}
    receipt.update(status='M0_PASS' if args.mode == 'm0' else 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE',
                   best_epoch=best['epoch'], frozen_parameters_unchanged=True,
                   visual_parameters_changed=True, fresh_camera_parameters_changed=True,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   completed_at=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(receipt, indent=2) + '\n')


def evaluate(args, protocol):
    started = time.perf_counter()
    os.chdir(args.output_dir)
    training = json.loads((args.output_dir / 'training.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [row['epoch'] for row in training['history']] == list(range(1, 51))
    model, _cfg, binding = build(args, protocol)
    assert training['initializer'] == binding
    payload = load(args.output_dir / 'best_map.pth', model, args)
    selected = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
    assert payload['epoch'] == training['best_epoch'] == selected['epoch']
    assert payload['metrics'] == selected['official_fused']
    distances = args.output_dir / 'official_distances.pt'
    metrics = runner.official_metrics(model, protocol, args.signal_source, save_distances=distances)
    assert all(abs(metrics[name] - payload['metrics'][name]) < 1e-5 for name in metrics)
    result = {'schema': SCHEMA, 'status': 'COMPLETE', 'dataset': args.dataset, 'recipe': args.recipe,
              'seed': 42, 'training_epochs': 50, 'selected_epoch': payload['epoch'],
              'condition': condition(args), 'protocol_sha256': runner.sha256(args.protocol),
              'checkpoint_sha256': runner.sha256(args.output_dir / 'best_map.pth'),
              'distance_sha256': runner.sha256(distances),
              'training_best_distance_sha256': runner.sha256(args.output_dir / 'best_epoch_distances.pt'),
              'metrics': metrics, 'reranking': False, 'independent_upstream_metrics_equal': True,
              'final_evaluation_seconds': time.perf_counter() - started,
              'final_evaluation_timing_boundary': 'Fresh construction, strict reload and full-gallery scoring.',
              'completed_at': datetime.now().astimezone().isoformat()}
    (args.output_dir / 'official_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201', 'RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--recipe', choices=RECIPES, required=True)
    parser.add_argument('--mode', choices=('prepare', 'm0', 'train', 'evaluate'), required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--epochs', type=int, choices=(50,), default=50)
    args = parser.parse_args()
    for name in ('protocol', 'signal_source', 'clip_weight', 'initialization', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.baseline_sha256 = runner.sha256(args.clip_weight)
    protocol = runner.read_protocol(args.protocol, args.dataset)
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        _model, _cfg, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps({'schema': SCHEMA, 'status': 'INITIALIZATION_VERIFIED',
            'binding': binding, 'prepared_at': datetime.now().astimezone().isoformat()}, indent=2) + '\n')
    else:
        (evaluate if args.mode == 'evaluate' else train)(args, protocol)


if __name__ == '__main__':
    main()
