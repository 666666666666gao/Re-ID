"""One fixed RGBNT201 best, source-training batch, backbone input sensitivity.

No optimizer, role/head forward, mask, fitting or query/gallery evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_region_use_best as prior
from trifusion.global_relation_inputs_diagnostic import (
    GlobalInputTap, supported_global_target, patch_sensitivity,
)

SCHEMA = 'trifusion-global-relation-input-sensitivity-v1'


def select_records(protocol):
    # Canonical availability-only pilot: first8 IDs with two source cameras.
    # This is B16/K2 diagnosis, not an author-training sampler replacement.
    rows = prior.base.entry.runner.records_for(protocol, 'train')
    by_id = {}
    for record in rows:
        by_id.setdefault(record[1], {}).setdefault(record[2], []).append(record)
    selected = []
    ids = [identity for identity in sorted(by_id) if len(by_id[identity]) >= 2][:8]
    assert len(ids) == 8
    for identity in ids:
        for camera in sorted(by_id[identity])[:2]:
            selected.append(min(by_id[identity][camera], key=lambda record: tuple(record[0])))
    assert len(selected) == 16
    return selected


def describe(value):
    flat = value.flatten().float()
    return {'min': float(flat.min()), 'max': float(flat.max()),
            'mean': float(flat.mean()), 'abs_mean': float(flat.abs().mean()),
            'nonzero_elements': int((flat != 0).sum())}


def run(args):
    started = time.perf_counter()
    seal = json.loads(args.seal.read_text())
    assert seal['schema'] == SCHEMA
    for name, digest in seal['new_source_sha256'].items():
        assert prior.base.panel.base.sha(ROOT / name) == digest, name
    previous = Path(seal['prior_seal'])
    assert prior.base.panel.base.sha(previous) == seal['prior_seal_sha256']
    inputs = json.loads(previous.read_text())
    prior.require_inputs(inputs)
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.output_dir.exists()
    assert shutil.disk_usage(ROOT).free >= 500 * 1024**2
    row = next(r for r in inputs['rows']
               if (r['dataset'], r['comparison']) == ('RGBNT201', 'uniform'))
    original = Path(row['run_dir'])
    values = argparse.Namespace(dataset='RGBNT201', comparison='uniform',
        variant='semantic', recipe='semantic', mode='evaluate', seed=42, epochs=50,
        protocol=prior.base.panel.PROTOCOLS / 'RGBNT201.json',
        signal_source=prior.base.panel.SOURCE, clip_weight=prior.base.panel.WEIGHTS / 'ViT-B-16.pt',
        initialization=Path(row['initialization']), output_dir=original)
    values.baseline_sha256 = prior.producer.inner.runner.sha256(values.clip_weight)
    protocol = prior.producer.inner.runner.read_protocol(values.protocol, values.dataset)
    prior.producer.configure()
    model, _cfg, binding = prior.base.foundation.build(values, protocol)
    assert binding == row['initializer']
    payload = prior.base.foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch'] == 8
    model.eval()
    before = prior.base.entry.runner._module_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    records = select_records(protocol)
    image_pins = {path: prior.base.panel.base.sha(Path(path))
                  for record in records for path in record[0]}
    raw_batches = list(prior.base.entry.runner.loader_for(
        protocol, records, training=False, method='PLAIN_V8'))
    assert len(raw_batches) == 1
    raw = raw_batches[0]
    batch = prior.base.entry.runner._eval_batch(raw, 'RGBNT201')
    labels = torch.tensor([r[1] for r in records], device='cuda:0')
    environments = torch.tensor([r[2] for r in records], device='cuda:0')
    assert [int(value) for value in raw[1]] == labels.cpu().tolist()
    assert torch.equal(batch['camera_ids'].cpu(), environments.cpu())
    for device in range(2):
        torch.cuda.reset_peak_memory_stats(device)
    backbone = model.evidence_model.backbone
    # FP32 diagnostic, no autocast and no role/head call. Taps never alter inputs.
    with GlobalInputTap(backbone) as tap:
        stages, global_feature = backbone(batch['images'], batch['camera_ids'])
    assert global_feature.shape == (16, 1536) and bool(torch.isfinite(global_feature).all())
    target, support = supported_global_target(global_feature, labels, environments)
    assert support['supported_relations'] > 0, support
    states = tap.ordered_states()
    # The intentionally unused last input checks the post-forward stage stack.
    # Missing gradients at the nine actual inputs are an error, not filled zeros.
    gradients = torch.autograd.grad(target, states + [stages], allow_unused=True)
    assert gradients[-1] is None
    sensitivity, gradient_norm = patch_sensitivity(states, gradients[:-1])
    assert bool(torch.isfinite(sensitivity).all()) and bool(torch.isfinite(gradient_norm).all())
    after = prior.base.entry.runner._module_state_sha256(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    assert all(prior.base.panel.base.sha(Path(path)) == digest for path, digest in image_pins.items())
    prior.require_inputs(inputs)
    assert all(prior.base.panel.base.sha(ROOT / name) == digest
               for name, digest in seal['new_source_sha256'].items())
    args.output_dir.mkdir(parents=True)
    artifact = args.output_dir / 'SENSITIVITY.pt'
    torch.save({'patch_gradient_dot_activation': sensitivity,
                'patch_gradient_norm': gradient_norm}, artifact)
    result = {'schema': SCHEMA, 'status': 'COMPLETE', 'dataset': 'RGBNT201',
        'comparison': 'uniform', 'selected_epoch': 8, 'optimizer_updates': 0,
        'input_checkpoint_sha256': row['checkpoint_sha256'], 'input_records': records,
        'input_image_sha256': image_pins, 'batch_size': 16, 'instances_per_id': 2,
        'input_transform': 'existing_eval_transform_applied_to_source_train_records',
        'target': float(target.detach()), 'support': support, 'backbone_forwards': 1,
        'vjp_calls': 1, 'full_input_tensors': 9, 'poststack_gradient_absent': True,
        'layers_zero_based': list(backbone.layers), 'tensor_order': 'depth,modality,batch,patch',
        'signed_sensitivity_by_depth_modality': [[describe(value) for value in depth] for depth in sensitivity],
        'patch_gradient_norm_by_depth_modality': [[describe(value) for value in depth] for depth in gradient_norm],
        'full_input_devices': [str(state.device) for state in states],
        'full_input_dtypes': [str(state.dtype) for state in states],
        'model_state_before_sha256': before, 'model_state_after_sha256': after,
        'state_unchanged': True, 'parameter_gradients_absent': True,
        'artifact': {'path': str(artifact), 'bytes': artifact.stat().st_size,
                     'sha256': prior.base.panel.base.sha(artifact)},
        'peak_memory': {str(device): {'allocated': torch.cuda.max_memory_allocated(device),
                                     'reserved': torch.cuda.max_memory_reserved(device)} for device in range(2)},
        'elapsed_seconds': time.perf_counter() - started,
        'boundary': 'One source-label model-sensitivity proxy on a canonical B16/K2 batch. Batch gradient includes anchor/positive/negative contributions. FP32 eval-mode backbone only, no role/head/mask/newweight/classification or retrieval evaluation. Signed derivative is not full-deletion effect, semantic complementarity, causal part attribution, actual training-batch support or performance. No new teacher/module/training result.'}
    (args.output_dir / 'DIAGNOSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'support': support,
                      'backbone_forwards': 1, 'vjp_calls': 1,
                      'state_unchanged': True, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
