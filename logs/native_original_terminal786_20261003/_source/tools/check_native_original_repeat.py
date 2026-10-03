"""One native original/original AMP control after V5; no parameter updates."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_independent_native_evidence as entry

SCHEMA = 'trifusion-native-original-repeat-control-v1'


def comparison(x, y, tolerance):
    same_shape = x.shape == y.shape
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {'reference_shape': list(x.shape), 'candidate_shape': list(y.shape),
            'reference_dtype': str(x.dtype), 'candidate_dtype': str(y.dtype),
            'finite': finite,
            'max_absolute_difference': float((x-y).abs().max()) if same_shape else None,
            'fixed_gate_pass': same_shape and finite and bool(
                torch.allclose(x, y, atol=tolerance, rtol=tolerance))}


def hooks(model):
    return {name: [len(module._forward_hooks), len(module._forward_pre_hooks)]
            for name, module in model.named_modules()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    reference_campaign = args.reference_campaign.resolve()
    output_dir = args.output_dir.resolve()
    assert output_dir.is_dir()
    assert torch.cuda.device_count() == 2
    measured_path = output_dir/'MEASURED.json'
    pass_path = output_dir/'PASS.json'
    assert not measured_path.exists() and not pass_path.exists()
    expected = json.loads((reference_campaign/'partitioned_gradient_deltas_RGBNT201_native.json').read_text())
    initial = json.loads((reference_campaign/'initialization/RGBNT201_native.json').read_text())['binding']
    assert initial['initial_model_state_sha256'] == '7f5ff300a45faed287a510dacc68d06c006931504a4510fffd8c47e43d93b852'
    assert expected['variant'] == initial['variant'] == 'native'
    entry.configure()
    protocol_path = ROOT/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'
    protocol = entry.runner.read_protocol(protocol_path, 'RGBNT201')
    values = argparse.Namespace(dataset='RGBNT201', variant='native', recipe='native',
        mode='prepare', seed=42, epochs=50, protocol=protocol_path,
        signal_source=ROOT/'comparators/Signal-cd1b0a6',
        clip_weight=ROOT/'pertrained-model/ViT-B-16.pt',
        initialization=reference_campaign/'initialization/RGBNT201_native.json',
        output_dir=output_dir/'unused_training_output',
        baseline_sha256=initial['public_clip_sha256'])
    assert entry.runner.sha256(values.clip_weight) == values.baseline_sha256
    reference, cfg, binding = entry.build_core(values, protocol)
    candidate, other_cfg, other_binding = entry.build_core(values, protocol)
    assert cfg.dump() == other_cfg.dump() == initial['cfg_yaml']
    assert binding == other_binding
    for name in ('initial_model_state_sha256', 'visual_initial_sha256', 'camera_initial_sha256',
                 'shared_initializer_sha256', 'protocol_sha256', 'author_source_commit',
                 'head_names', 'trainable_parameters', 'trainable_parameter_tensors'):
        assert binding[name] == initial[name], name
    assert cfg.SOLVER.IMS_PER_BATCH == 64 and cfg.DATALOADER.NUM_INSTANCE == 8
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert reference_parameters.keys() == candidate_parameters.keys()
    assert all(p.device == torch.device('cuda:0') and p.dtype == torch.float32
               for model in (reference, candidate) for p in model.parameters())
    initial_parameters = {name: p.detach().cpu().clone() for name, p in reference_parameters.items()}
    assert all(torch.equal(initial_parameters[name], p.detach().cpu())
               and p.requires_grad == reference_parameters[name].requires_grad
               for name, p in candidate_parameters.items())
    trainable = [name for name, p in reference_parameters.items() if p.requires_grad]
    assert len(trainable) == 295
    original_hooks = hooks(reference)
    assert hooks(candidate) == original_hooks
    assert all(counts == [0, 0] for counts in original_hooks.values())
    raw = next(iter(entry.original_train_loader(values, protocol, cfg)))
    full_batch, full_labels = entry.runner._training_batch(raw)
    assert len(full_labels) == expected['full_author_batch'] == 64
    batch = {'images': {name: image[:32] for name, image in full_batch['images'].items()},
             'camera_ids': full_batch['camera_ids'][:32]}
    labels = full_labels[:32]
    inputs = {'labels': labels.tolist(), 'cameras': batch['camera_ids'].tolist(),
              'paths': list(raw[4][:32]),
              'images_sha256': {name: entry.clean.tensor_digest(image)
                                for name, image in batch['images'].items()}}
    assert all(inputs[name] == expected[name] for name in inputs)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    assert len(cuda_rng) == 2
    observed = []
    for model in (reference, candidate):
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        optimizer, scheduler, loss_fn = entry.optimization(values, model, cfg)
        names = {id(p): name for name, p in model.named_parameters()}
        owned = [names[id(p)] for group in optimizer.param_groups for p in group['params']]
        assert len(owned) == len(set(owned)) == 295 and set(owned) == set(trainable)
        groups = [{'names': [names[id(p)] for p in group['params']],
                   'lr': group['lr'], 'weight_decay': group['weight_decay']}
                  for group in optimizer.param_groups]
        scaler = torch.amp.GradScaler('cuda', init_scale=256.0)
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(batch, return_aux=True)
            loss, components = entry.loss_values(values, output, labels, batch['camera_ids'], loss_fn)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                     for name, p in model.named_parameters() if p.requires_grad}
        observed.append({'raw': output['raw_fused'].detach().cpu(),
            'fused': output['fused'].detach().cpu(), 'global': output['shared_global'].detach().cpu(),
            'heads': [(score.detach().cpu(), feature.detach().cpu()) for score, feature in output['heads']],
            'loss': loss.detach().cpu(), 'loss_components': components, 'gradients': gradients,
            'buffers': {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'post_state_sha256': entry.runner._module_state_sha256(model),
            'parameters_unchanged': {name: bool(torch.equal(p.detach().cpu(), initial_parameters[name]))
                                     for name, p in model.named_parameters()},
            'bn_batches_tracked': {neck: int(getattr(model.signal, neck).num_batches_tracked)
                                   for neck, _classifier in model.head_names},
            'hooks': hooks(model), 'optimizer_groups': groups, 'scale': scaler.get_scale()})
        model.zero_grad(set_to_none=True)
        del output, loss, optimizer, scheduler, loss_fn, scaler
        for device in range(2):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    a, b = observed
    gradients = {}
    for name in trainable:
        x, y = a['gradients'][name], b['gradients'][name]
        row = {'reference_none': x is None, 'candidate_none': y is None}
        if x is None or y is None:
            row['fixed_gate_pass'] = x is None and y is None
        else:
            row.update(comparison(x, y, 1e-4))
            row.update(reference_max_abs=float(x.abs().max()), candidate_max_abs=float(y.abs().max()))
        gradients[name] = row
    outputs = {name: comparison(a[name], b[name], 1e-5) for name in ('raw', 'fused', 'global', 'loss')}
    head_comparisons = [{'score': comparison(left[0], right[0], 1e-5),
                         'feature': comparison(left[1], right[1], 1e-5)}
                        for left, right in zip(a['heads'], b['heads'])]
    buffer_comparisons = {name: bool(torch.equal(value, b['buffers'][name]))
                          for name, value in a['buffers'].items()}
    gates = {'outputs': all(row['fixed_gate_pass'] for row in outputs.values()),
        'heads': len(a['heads']) == len(b['heads']) == len(reference.head_names) and all(
            row[name]['fixed_gate_pass'] for row in head_comparisons for name in ('score', 'feature')),
        'gradient_keys': set(a['gradients']) == set(b['gradients']) == set(trainable),
        'gradients': all(row['fixed_gate_pass'] for row in gradients.values()),
        'buffer_keys': a['buffers'].keys() == b['buffers'].keys(),
        'buffers': all(buffer_comparisons.values()),
        'cpu_rng': bool(torch.equal(a['cpu_rng'], b['cpu_rng'])),
        'cuda_rng': len(a['cuda_rng']) == len(b['cuda_rng']) == 2 and all(
            torch.equal(x, y) for x, y in zip(a['cuda_rng'], b['cuda_rng'])),
        'bn_one_batch': a['bn_batches_tracked'] == b['bn_batches_tracked'] and all(
            count == 1 for count in a['bn_batches_tracked'].values()),
        'post_state': a['post_state_sha256'] == b['post_state_sha256'],
        'parameters_unchanged': all(a['parameters_unchanged'].values()) and all(b['parameters_unchanged'].values()),
        'hooks_removed': a['hooks'] == b['hooks'] == original_hooks,
        'optimizer_groups': a['optimizer_groups'] == b['optimizer_groups'],
        'fixed_scale': a['scale'] == b['scale'] == 256.0}
    measurement = {'schema': SCHEMA, 'status': 'MEASURED_BEFORE_ASSERTIONS',
        'at': datetime.now().astimezone().isoformat(), 'comparison': 'native_original_vs_native_original',
        'dataset': 'RGBNT201', 'variant': 'native', 'full_author_batch': 64, 'witness_samples': 32,
        'initial_model_state_sha256': binding['initial_model_state_sha256'], 'inputs': inputs,
        'forward_atol_rtol': [1e-5, 1e-5], 'gradient_atol_rtol': [1e-4, 1e-4],
        'outputs': outputs, 'heads': head_comparisons, 'gradients': gradients,
        'buffer_comparisons': buffer_comparisons, 'gates': gates,
        'post_state_sha256': [row['post_state_sha256'] for row in observed],
        'bn_batches_tracked': [row['bn_batches_tracked'] for row in observed],
        'parameters_unchanged': [row['parameters_unchanged'] for row in observed],
        'optimizer_groups': a['optimizer_groups'], 'loss_components': [row['loss_components'] for row in observed],
        'optimizer_updates': 0, 'weights_generated': 0,
        'boundary': 'Two fresh controlled originals, each one AMP forward/backward/unscale. No updates/scoring/weights/retry. Old full gradients and pre-forward RNG were not saved; this does not reconstruct historical tensors. V5 remains FAIL even if this control passes.'}
    measured_path.write_text(json.dumps(measurement, indent=2)+'\n')
    assert all(gates.values()), [name for name, passed in gates.items() if not passed]
    pass_path.write_text(json.dumps({'schema': SCHEMA, 'status': 'NATIVE_ORIGINAL_REPEAT_CONTROL_PASS',
        'at': datetime.now().astimezone().isoformat(), 'gates': gates, 'optimizer_updates': 0,
        'weights_generated': 0, 'boundary': measurement['boundary']}, indent=2)+'\n')
    print(pass_path, flush=True)


if __name__ == '__main__':
    main()
