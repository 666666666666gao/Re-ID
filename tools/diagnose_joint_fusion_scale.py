#!/usr/bin/env python3
"""Compare joint-SIM scale derivatives on a full source-loader epoch, without updates."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tools.official_three_dataset_data import loader_for, records_for
from tools.official_three_dataset_model import sha256
from tools.run_official_three_dataset_roles import checkpoint_names, initialize, read_protocol
from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch
from tools.train_official_three_dataset_roles import _v27_loss


def differentiable_scale_embeddings(baseline, residuals):
    scale = baseline.norm(dim=1, keepdim=True)
    branches = {name: torch.cat((baseline, value * scale), dim=1)
                for name, value in residuals.items()}
    bank = F.normalize(torch.cat(tuple(residuals.values()), dim=1), dim=1)
    return torch.cat((baseline, bank * scale), dim=1), branches


def differentiable_scale_output(model, output):
    fused, branches = differentiable_scale_embeddings(
        output.baseline_embedding, output.residual_embeddings)
    return replace(output, fused_embedding=fused, branch_embeddings=branches,
                   fused_logits=model.fused_classifier(model.fused_neck(fused)),
                   branch_logits={name: model.branch_classifiers[name](
                       model.branch_necks[name](value)) for name, value in branches.items()})


def compare_vectors(current, reference):
    left, right = current.double().flatten(), reference.double().flatten()
    nl, nr = left.norm(), right.norm()
    return {'current_norm': float(nl), 'reference_norm': float(nr),
            'difference_norm': float((left - right).norm()),
            'relative_difference': float((left - right).norm() / nl) if nl > 0 else None,
            'cosine': float(torch.dot(left, right) / (nl * nr)) if nl > 0 and nr > 0 else None}


def gradients(loss, baseline, parameters, *, retain_graph):
    # Match the joint trainer's fixed, overflow-free loss scale at the measured endpoints.
    values = torch.autograd.grad(loss * 256, (baseline, *parameters), retain_graph=retain_graph)
    assert all(torch.isfinite(value).all() for value in values)
    return [value.float() / 256 for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-receipt', type=Path, required=True)
    parser.add_argument('--state', choices=('initial', 'final'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.training_receipt.read_text())
    assert receipt['status'] == 'FIXED_EPOCH20_TRAINING_COMPLETE'
    assert receipt['method'] in ('SIGNAL_SIM_JOINT', 'SIGNAL_SIM_JOINT_LOWLR')
    assert receipt['seed'] == 42
    assert sha256(receipt['checkpoint']) == receipt['checkpoint_sha256']
    protocol = read_protocol(Path(receipt['protocol']), receipt['dataset'])
    assert sha256(receipt['protocol']) == receipt['protocol_sha256']
    model_args = SimpleNamespace(
        dataset=receipt['dataset'], method=receipt['method'], seed=receipt['seed'],
        signal_source=ROOT / 'comparators/Signal-cd1b0a6',
        clip_weight=ROOT / 'pertrained-model/ViT-B-16.pt',
        signal_checkpoint=Path(receipt['initializer']['author_checkpoint']),
        signal_sha256=receipt['initializer']['author_checkpoint_sha256'])
    model, _, config, binding = initialize(model_args, protocol)
    assert binding == receipt['initializer']
    if args.state == 'final':
        payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
        assert payload['method'] == receipt['method'] and payload['dataset'] == receipt['dataset']
        assert set(payload['role_state_dict']) == checkpoint_names(model, receipt['method'])
        state = model.state_dict()
        state.update(payload['role_state_dict'])
        model.load_state_dict(state, strict=True)
    expected = receipt['training'][f'{args.state}_state_sha256']
    assert _module_state_sha256(model) == expected
    model.train()
    # Use training-batch normalization while leaving every saved BN buffer unchanged.
    for neck in (model.fused_neck, *model.branch_necks.values(), *model.residual_necks.values()):
        neck.track_running_stats = False
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    criterion = ExpertFormationV8Criterion(triplet_margin=.3, label_smoothing=.1).cuda()
    selected = [(name, p) for name, p in model.named_parameters()
                if name.startswith('baseline.signal.SIM.modal_interactive.') and p.requires_grad]
    assert len(selected) == 12
    parameters = [p for _, p in selected]
    _set_seed(42)
    loader = loader_for(protocol, records_for(protocol, 'train'), training=True,
                        method=receipt['method'], seed=42)
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    rows, identities, exposures = [], set(), 0
    with (args.output_dir / 'batches.jsonl').open('x') as handle:
        for index, raw in enumerate(loader, 1):
            batch, labels = _training_batch(raw)
            assert sorted(torch.unique(labels, return_counts=True)[1].tolist()) == [8] * 8
            identities.update(labels.cpu().tolist())
            exposures += len(labels)
            with torch.autocast('cuda', dtype=torch.float16):
                current = model(batch, return_aux=True)
                reference = differentiable_scale_output(model, current)
                assert torch.equal(current.fused_embedding, reference.fused_embedding)
                assert torch.equal(current.fused_logits, reference.fused_logits)
                for name in current.branch_embeddings:
                    assert torch.equal(current.branch_embeddings[name], reference.branch_embeddings[name])
                    assert torch.equal(current.branch_logits[name], reference.branch_logits[name])
                current_parts, reference_parts = criterion(current, labels), criterion(reference, labels)
                assert all(torch.equal(current_parts[key], reference_parts[key]) for key in current_parts)
                current_loss = _v27_loss(current_parts, config)
                reference_loss = _v27_loss(reference_parts, config)
            left = gradients(current_loss, current.baseline_embedding, parameters, retain_graph=True)
            repeated = gradients(current_loss, current.baseline_embedding, parameters, retain_graph=True)
            right = gradients(reference_loss, current.baseline_embedding, parameters, retain_graph=False)
            left_sim, right_sim, repeat_sim = [torch.cat([v.flatten() for v in values[1:]])
                                              for values in (left, right, repeated)]
            unit = F.normalize(current.baseline_embedding.detach().float(), dim=1)
            difference = left[0] - right[0]
            tangent = difference - (difference * unit).sum(dim=1, keepdim=True) * unit
            row = {'batch': index, 'source_exposures': exposures, 'loss': float(current_loss.detach()),
                   'forward_and_all_14_losses_equal': True,
                   'baseline_gradient': compare_vectors(left[0], right[0]),
                   'sim_gradient': compare_vectors(left_sim, right_sim),
                   'repeated_sim_gradient': compare_vectors(left_sim, repeat_sim),
                   'baseline_difference_tangent_norm': float(tangent.double().norm()),
                   'sim_parameters': {name: compare_vectors(a, b)
                                      for (name, _), a, b in zip(selected, left[1:], right[1:])}}
            rows.append(row)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
    assert rows and _module_state_sha256(model) == expected
    assert all(parameter.grad is None for parameter in model.parameters())
    relative = [row['sim_gradient']['relative_difference'] for row in rows
                if row['sim_gradient']['relative_difference'] is not None]
    result = {'schema': 'trifusion-joint-scale-source-gradient-v1', 'status': 'COMPLETE',
              'dataset': receipt['dataset'], 'method': receipt['method'], 'state': args.state,
              'completed_at': datetime.now().astimezone().isoformat(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'training_receipt': str(args.training_receipt),
              'training_receipt_sha256': sha256(args.training_receipt),
              'checkpoint_sha256': receipt['checkpoint_sha256'],
              'protocol_sha256': receipt['protocol_sha256'],
              'model_state_sha256': expected, 'model_state_unchanged': True,
              'source_loader_epochs': 1, 'source_batches': len(rows),
              'source_exposures': exposures, 'observed_identities': len(identities),
              'optimizer_updates': 0, 'official_model_forwards': 0,
              'batch_log_sha256': sha256(args.output_dir / 'batches.jsonl'),
              'sim_relative_difference_median': float(torch.tensor(relative).median()) if relative else None,
              'maximum_repeat_difference': max(row['repeated_sim_gradient']['difference_norm'] for row in rows),
              'boundary': 'Fixed-state source gradients only, not historical AdamW updates or retrieval effects. '
                          'One balanced source-loader epoch contains repeated views and may omit source records.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
