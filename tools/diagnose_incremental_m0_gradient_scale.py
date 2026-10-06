"""One fixed failed-M0 batch: compare isolated AMP VJPs at scale 1 and 256."""
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
    campaign = ROOT / 'logs/incremental_role_objective_m0_v1_20261007_878'
    run = ROOT / f'trained-model/{campaign.name}_m0_md_batch_ratio_RGBNT201'
    training = json.loads((run / 'training.json').read_text())
    assert training['status'] == 'M0_PASS' and training['history'][0]['steps'] == 8
    probe = run / 'm0_reload_probe.pth'
    assert entry.inner.runner.sha256(probe) == training['m0']['reload_probe_sha256']
    values = argparse.Namespace(dataset='RGBNT201', objective='md_batch_ratio', variant='semantic', recipe='semantic',
        mode='m0', seed=42, epochs=50, protocol=ROOT/'logs/training_feature_scale_protocols_20261002/RGBNT201.json',
        signal_source=ROOT/'comparators/Signal-cd1b0a6', clip_weight=ROOT/'pertrained-model/ViT-B-16.pt',
        initialization=campaign/'initialization/RGBNT201_md_batch_ratio.json', output_dir=run)
    values.baseline_sha256 = entry.inner.runner.sha256(values.clip_weight)
    protocol = entry.inner.runner.read_protocol(values.protocol, values.dataset)
    for records in protocol['records'].values():
        for row in records:
            entry.ENVIRONMENTS[Path(row['paths'][0]).name] = row['camera']
    entry.configure()
    model, cfg, binding = entry.inner.foundation.build(values, protocol)
    assert binding == training['initializer']
    entry.inner.foundation.load(probe, model, values)
    before = entry.inner.runner._module_state_sha256(model)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    loader = entry.inner.original_train_loader(values, protocol, cfg)
    raw = next(iter(loader))
    original_batch = json.loads((run / 'training_batch_order.jsonl').read_text().splitlines()[0])
    assert list(raw[4]) == original_batch['paths'] and raw[1].tolist() == original_batch['labels']
    batch, labels = entry.training_batch(raw)
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(batch, return_aux=True)
        loss, activity = entry.batch_ratio_loss(output['shared_global'], output['correction'], output['raw_fused'], labels)
    parameters = [(name, p) for name, p in model.named_parameters()
        if name.startswith(('evidence_model.roles.query_projections.', 'evidence_model.roles.key_projections.'))]
    assert len(parameters) == 6
    targets = [output['shared_global'], output['correction']] + [p for _, p in parameters]
    rows = []
    for scale in (1.0, 256.0):
        gradients = torch.autograd.grad(loss * scale, targets, retain_graph=True, allow_unused=True)
        assert gradients[0] is None and all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        describe = lambda g: dict(unused=g is None, norm=0.0 if g is None else float((g.float()/scale).norm()),
            max_abs=0.0 if g is None else float((g.float()/scale).abs().max()))
        rows.append(dict(scale=scale, shared_global_gradient_absent=True, correction=describe(gradients[1]),
            query_key={name: describe(g) for (name, _), g in zip(parameters, gradients[2:])}))
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    assert entry.inner.runner._module_state_sha256(model) == before
    value = dict(status='FIXED_M0_SCALE_COMPARISON_COMPLETE', at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=training['m0']['reload_probe_sha256'], batch_paths=list(raw[4]),
        isolated_loss=float(loss.detach()), activity=activity, rows=rows, model_state_restored_exact=True,
        real_model_forwards=1, isolated_vjps=2, optimizer_updates=0,
        boundary='Single fixed retained eight-step M0 state and one source batch. Same AMP graph and same objective; scale256 matches production M0 GradScaler initialization, reported gradients divided by scale. Does not reconstruct old eight-step gradients, change their FAIL, establish all-model underflow cause or qualify a full campaign.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
