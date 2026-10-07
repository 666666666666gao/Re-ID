"""One initial graph and identical targets; compare VJP autocast on/off."""
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
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(batch, return_aux=True)
        loss, activity = entry.repair_keep_loss(output['shared_global'], output['fused'], labels, output['retrieval_environments'])
    targets = [output['shared_global'], output['correction']] + [parameter for _name,parameter in parameters]
    rows = []
    for enabled in (True, False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=enabled):
            gradients = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
        assert gradients[0] is None and all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)
        def describe(gradient, target):
            return dict(unused=gradient is None, target_dtype=str(target.dtype),
                gradient_dtype=None if gradient is None else str(gradient.dtype),
                norm=0.0 if gradient is None else float(gradient.float().norm()),
                max_abs=0.0 if gradient is None else float(gradient.float().abs().max()))
        rows.append(dict(vjp_autocast_enabled=enabled, scale=1.0, shared_global_gradient_absent=True,
            correction=describe(gradients[1],output['correction']),
            query_key_parameters={name:describe(gradient,parameter) for (name,parameter),gradient in zip(parameters,gradients[2:])}))
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
    assert entry.inner.runner._module_state_sha256(model) == before
    value = dict(status='INITIAL_AUXILIARY_AUTOCAST_CONTEXT_COMPARISON_COMPLETE', at=datetime.now().astimezone().isoformat(),
        initializer=binding, initial_model_state_sha256=before, batch_paths=list(raw[4]), batch_labels=raw[1].tolist(),
        isolated_loss=float(loss.detach()), activity=activity, rows=rows, model_state_restored_exact=True,
        real_model_forwards=1, isolated_vjps=2, optimizer_updates=0, checkpoint_loaded=False,
        boundary='Fresh verified public initialization and one RGBNT201 AMP forward/loss graph. Identical original8 VJP targets and unit scale; only backward autocast context differs. No oldeight augmentation reconstruction, oldgate change, fullqualification or retrieval result.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
