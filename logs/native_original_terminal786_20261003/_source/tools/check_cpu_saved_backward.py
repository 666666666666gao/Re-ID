"""Production AMP backward parity on a real author-batch subset; no updates."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_native_cpu_saved as entry
from tools import queue_native_cpu_saved as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', choices=panel.DATASETS, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    panel.configure()
    entry.configure()
    protocol_path = panel.PROTOCOLS / f'{args.dataset}.json'
    protocol = entry.runner.read_protocol(protocol_path, args.dataset)
    rows = []
    for variant in panel.VARIANTS:
        values = argparse.Namespace(dataset=args.dataset, variant=variant, recipe=variant,
            mode='prepare', seed=42, epochs=50, protocol=protocol_path,
            signal_source=panel.SOURCE, clip_weight=panel.WEIGHTS / 'ViT-B-16.pt',
            initialization=campaign / 'initialization' / f'{args.dataset}_{variant}.json',
            output_dir=ROOT / 'trained-model/cpu_saved_backward_unused',
            baseline_sha256=entry.runner.sha256(panel.WEIGHTS / 'ViT-B-16.pt'))
        reference, cfg, _binding = entry.original_build_core(values, protocol)
        candidate, other_cfg, binding = entry.build_core(values, protocol)
        assert binding == panel.expected_binding(campaign, args.dataset, variant)
        assert cfg.dump() == other_cfg.dump()
        assert entry.runner._module_state_sha256(reference) == entry.runner._module_state_sha256(candidate)
        raw = next(iter(entry.original_train_loader(values, protocol, cfg)))
        full_batch, full_labels = entry.runner._training_batch(raw)
        assert len(full_labels) == cfg.SOLVER.IMS_PER_BATCH
        batch = {'images': {name: image[:32] for name, image in full_batch['images'].items()},
                 'camera_ids': full_batch['camera_ids'][:32]}
        labels = full_labels[:32]
        assert len(labels.unique()) >= 2
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        observed = []
        for model in (reference, candidate):
            model.train()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            optimizer, _scheduler, loss_fn = entry.entry.optimization(values, model, cfg)
            scaler = torch.amp.GradScaler('cuda', init_scale=256.0)
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(batch, return_aux=True)
                loss, _components = entry.entry.loss_values(values, output, labels,
                                                          batch['camera_ids'], loss_fn)
            assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                         for name, parameter in model.named_parameters() if parameter.requires_grad}
            assert all(torch.isfinite(value).all() for value in gradients.values() if value is not None)
            observed.append({'raw': output['raw_fused'].detach().cpu(),
                'fused': output['fused'].detach().cpu(), 'global': output['shared_global'].detach().cpu(),
                'heads': [(score.detach().cpu(), feature.detach().cpu()) for score, feature in output['heads']],
                'loss': loss.detach().cpu(), 'gradients': gradients,
                'buffers': {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()})
            # Release one graph before the next production backward.
            model.zero_grad(set_to_none=True)
            del output, loss, optimizer, _scheduler, loss_fn, scaler
            torch.cuda.empty_cache()
        a, b = observed
        # Preserve all differences and input metadata even if the fixed gate fails.
        # Preserve every measured delta before assertions discard the in-memory data.
        gradient_deltas = {}
        for name, x in a['gradients'].items():
            y = b['gradients'][name]
            gradient_deltas[name] = {'reference_none': x is None, 'candidate_none': y is None}
            if x is not None and y is not None:
                gradient_deltas[name].update(
                    max_absolute_difference=float((x-y).abs().max()),
                    reference_max_abs=float(x.abs().max()), candidate_max_abs=float(y.abs().max()),
                    fixed_gate_pass=bool(torch.allclose(x, y, atol=1e-4, rtol=1e-4)))
        measured = campaign / f'cpu_saved_gradient_deltas_{args.dataset}_{variant}.json'
        assert not measured.exists()
        measured.write_text(json.dumps({'schema': entry.SCHEMA, 'dataset': args.dataset, 'variant': variant,
            'witness_samples': len(labels), 'full_author_batch': len(full_labels),
            'output_max_absolute_difference': {name: float((a[name]-b[name]).abs().max())
                                               for name in ('raw','fused','global','loss')},
            'gradient_atol_rtol': [1e-4,1e-4], 'gradients': gradient_deltas,
            'labels': labels.tolist(), 'cameras': batch['camera_ids'].tolist(),
            'paths': list(raw[4][:32]),
            'images_sha256': {name: entry.clean.tensor_digest(value) for name, value in batch['images'].items()},
            'boundary': 'Pre-gate measured differences, not a PASS receipt; no optimizer update or official scoring.'}, indent=2)+'\n')
        for name in ('raw', 'fused', 'global', 'loss'):
            assert torch.allclose(a[name], b[name], atol=1e-5, rtol=1e-5), (variant, name)
        assert all(torch.allclose(x, y, atol=1e-5, rtol=1e-5)
                   for left, right in zip(a['heads'], b['heads']) for x, y in zip(left, right))
        assert set(a['gradients']) == set(b['gradients'])
        max_gradient_difference = 0.0
        for name, x in a['gradients'].items():
            y = b['gradients'][name]
            assert (x is None) == (y is None), (variant, name)
            if x is not None:
                assert torch.allclose(x, y, atol=1e-4, rtol=1e-4), (variant, name, float((x-y).abs().max()))
                max_gradient_difference = max(max_gradient_difference, float((x-y).abs().max()))
        assert set(a['buffers']) == set(b['buffers'])
        assert all(torch.equal(value, b['buffers'][name]) for name, value in a['buffers'].items())
        assert torch.equal(a['cpu_rng'], b['cpu_rng']) and torch.equal(a['cuda_rng'], b['cuda_rng'])
        tracked = {neck: int(getattr(candidate.signal, neck).num_batches_tracked)
                   for neck, _classifier in candidate.head_names}
        assert all(count == 1 for count in tracked.values())
        assert entry.runner._module_state_sha256(reference) == entry.runner._module_state_sha256(candidate)
        blocks = candidate.signal.clip_vision_encoder.base.transformer.resblocks
        assert all(not blocks[index]._forward_hooks for index in candidate.evidence_model.backbone.layers)
        rows.append({'variant': variant, 'full_author_batch': len(full_labels), 'witness_samples': len(labels),
            'distinct_identities': len(labels.unique()),
            'labels': labels.tolist(), 'cameras': batch['camera_ids'].tolist(), 'paths': list(raw[4][:32]),
            'images_sha256': {name: entry.clean.tensor_digest(value) for name, value in batch['images'].items()},
            'forward_atol_rtol': [1e-5, 1e-5], 'gradient_atol_rtol': [1e-4, 1e-4],
            'gradient_parameters': len(a['gradients']), 'gradient_max_abs_difference': max_gradient_difference,
            'state_buffers_and_rng_exact': True, 'author_bn_batches_tracked': tracked,
            'outer_capture_hooks_removed': True})
        del reference, candidate, observed, a, b, gradients, full_batch, full_labels, batch, labels, raw
        torch.cuda.empty_cache()
    output = campaign / f'cpu_saved_backward_{args.dataset}.json'
    assert not output.exists()
    output.write_text(json.dumps({'schema': entry.SCHEMA, 'status': 'CPU_SAVED_BACKWARD_PARITY_PASS',
        'dataset': args.dataset, 'rows': rows,
        'boundary': 'Actual production models and AMP backward; first32 samples only for equivalence witness, no optimizer update or official scoring. All9 M0 and formal training still use unchanged FULL author batches, including RGBNT100 B128/K16.'}, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
