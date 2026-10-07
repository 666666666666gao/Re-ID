"""One fresh initial AMP graph; compare auxiliary VJP boundaries at 1/256."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_incremental_role_objective as entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1' and not args.output.exists()
    campaign = ROOT / 'logs/incremental_pending_m0_v1_20261007_880'
    run = ROOT / f'trained-model/{campaign.name}_m0_repair_keep_RGBNT201'
    receipt = json.loads((run / 'training.json').read_text())
    assert receipt['status'] == 'M0_PASS' and receipt['history'][0]['steps'] == 8
    values = argparse.Namespace(dataset='RGBNT201', objective='repair_keep', variant='semantic', recipe='semantic',
        mode='m0', seed=42, epochs=50, protocol=ROOT/'logs/training_feature_scale_protocols_20261002/RGBNT201.json',
        signal_source=ROOT/'comparators/Signal-cd1b0a6', clip_weight=ROOT/'pertrained-model/ViT-B-16.pt',
        initialization=campaign/'initialization/RGBNT201_repair_keep.json', output_dir=run)
    values.baseline_sha256 = entry.inner.runner.sha256(values.clip_weight)
    protocol = entry.inner.runner.read_protocol(values.protocol, values.dataset)
    for records in protocol['records'].values():
        for row in records:
            entry.ENVIRONMENTS[Path(row['paths'][0]).name] = row['camera']
    entry.configure()
    model, cfg, binding = entry.inner.foundation.build(values, protocol)
    assert binding == receipt['initializer']
    before = entry.inner.runner._module_state_sha256(model)
    buffers = {name:value.clone() for name,value in model.named_buffers()}
    loader = entry.inner.original_train_loader(values, protocol, cfg)
    raw = next(iter(loader))
    old_batch = json.loads((run / 'training_batch_order.jsonl').read_text().splitlines()[0])
    assert list(raw[4]) == old_batch['paths'] and raw[1].tolist() == old_batch['labels']
    batch, labels = entry.training_batch(raw)
    parameters = [(name, parameter) for name,parameter in model.named_parameters()
        if name.startswith(('evidence_model.roles.query_projections.', 'evidence_model.roles.key_projections.'))]
    assert len(parameters) == 6
    captured = {}
    handles = []
    modules = dict(model.named_modules())
    for name,_parameter in parameters:
        module_name = name.removesuffix('.weight')
        def capture_projection(_module, _inputs, output, *, key=module_name):
            assert key not in captured
            captured[key] = output
        handles.append(modules[module_name].register_forward_hook(capture_projection))
    def capture_evidence(_module, _inputs, output):
        assert 'role_evidence' not in captured
        captured['role_evidence'] = output
    handles.append(model.evidence_model.roles.register_forward_hook(capture_evidence))
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(batch, return_aux=True)
        loss, activity = entry.repair_keep_loss(output['shared_global'], output['fused'], labels, output['retrieval_environments'])
    for handle in handles:
        handle.remove()
    evidence = captured['role_evidence']
    boundaries = [('correction', output['correction'])] + [(name, getattr(evidence, name)) for name in ('cnn', 'transformer', 'mamba')]
    boundaries += [(name.removesuffix('.weight'), captured[name.removesuffix('.weight')]) for name,_parameter in parameters]
    targets = [output['shared_global']] + [tensor for _name,tensor in boundaries] + [parameter for _name,parameter in parameters]
    rows = []
    for scale in (1.0, 256.0):
        gradients = torch.autograd.grad(loss * scale, targets, retain_graph=True, allow_unused=True)
        assert gradients[0] is None and all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)
        def describe(gradient, target):
            return dict(unused=gradient is None, target_dtype=str(target.dtype),
                gradient_dtype=None if gradient is None else str(gradient.dtype),
                norm=0.0 if gradient is None else float((gradient.float()/scale).norm()),
                max_abs=0.0 if gradient is None else float((gradient.float()/scale).abs().max()))
        offset = 1 + len(boundaries)
        rows.append(dict(scale=scale, shared_global_gradient_absent=True,
            boundaries={name:describe(gradient,tensor) for (name,tensor),gradient in zip(boundaries,gradients[1:offset])},
            query_key_parameters={name:describe(gradient,parameter) for (name,parameter),gradient in zip(parameters,gradients[offset:])}))
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
    assert entry.inner.runner._module_state_sha256(model) == before
    value = dict(status='INITIAL_AUXILIARY_BOUNDARY_COMPARISON_COMPLETE', at=datetime.now().astimezone().isoformat(),
        initializer=binding, initial_model_state_sha256=before, batch_paths=list(raw[4]), batch_labels=raw[1].tolist(),
        isolated_loss=float(loss.detach()), activity=activity, rows=rows, model_state_restored_exact=True,
        real_model_forwards=1, isolated_vjps=2, optimizer_updates=0, checkpoint_loaded=False,
        boundary='Fresh verified public initialization, one RGBNT201 source batch and one AMP graph. Scaled gradients divided by scale before FP32 summaries. No reconstruction of old eight augmented batches, no old-gate change, no full-campaign qualification, no training or retrieval result.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
