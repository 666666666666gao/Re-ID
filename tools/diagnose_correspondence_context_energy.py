#!/usr/bin/env python3
"""Read-only full-query/gallery energy diagnosis of five accepted best checkpoints."""

import argparse
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_correspondence_context_identity as entry
from tools.queue_correspondence_context_identity import CONDITIONS
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha


def statistics(values):
    assert values.ndim == 1 and torch.isfinite(values).all()
    quantiles = torch.quantile(values.float(), torch.tensor([0., .25, .5, .75, 1.]))
    return {'count': len(values), 'mean': values.double().mean().item(),
            'quantiles_min_q25_median_q75_max': quantiles.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--dataset', choices=('RGBNT201', 'MSVR310'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    assert matrix['schema'] == 'trifusion-correspondence-context-identity-verification-v1'
    rows = [row for row in matrix['rows'] if row['dataset'] == args.dataset]
    assert len(rows) == 5 and {row['variant'] for row in rows} == set(CONDITIONS)
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in rows)
    manifest = json.loads((ROOT / 'logs/correspondence_context_identity_20260929/manifest.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    args.output_dir.mkdir()
    reports = []
    for row in rows:
        query, auxiliary = CONDITIONS[row['variant']]
        entry.CONDITION.clear()
        entry.CONDITION.update(query_mode=query, auxiliary_target=auxiliary)
        entry.runner.CorrespondenceTriFusion = partial(entry.ContextIdentityTriFusion, **entry.CONDITION)
        weight, digest = BASELINES[args.dataset]
        run = Path(row['run_dir'])
        model_args = SimpleNamespace(dataset=args.dataset, signal_source=SOURCE,
            clip_weight=WEIGHTS / 'ViT-B-16.pt', baseline_checkpoint=WEIGHTS / weight,
            baseline_sha256=digest, seed=42, width=128, m1=True, m2=True, m3=False,
            pred_weight=.1, protocol=PROTOCOLS / f'{args.dataset}.json')
        protocol = entry.runner.read_protocol(model_args.protocol, args.dataset)
        training = json.loads((run / 'training.json').read_text())
        assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
        assert [item['epoch'] for item in training['history']] == list(range(1, 51))
        assert sha(run / 'best_map.pth') == row['checkpoint_sha256']
        assert sha(run / 'official_distances.pt') == row['distance_sha256']
        assert sha(run / 'official_metrics.json') == row['receipt_sha256']
        model, _cfg, _config, binding = entry.build(model_args, protocol)
        assert binding == training['initializer']
        payload = entry.load_checkpoint(run / 'best_map.pth', model, model_args, protocol)
        assert payload['epoch'] == row['best_epoch']
        model.eval()
        captured = {}
        backbone_hook = model.backbone.register_forward_hook(
            lambda _module, _inputs, output: captured.update(global_embedding=output[1]))
        role_hook = model.roles.register_forward_hook(
            lambda _module, _inputs, output: captured.update(correction=model.read_evidence(output)))
        vectors, energies = {}, {}
        maximum_forward_error = 0.
        with torch.inference_mode():
            for split in ('query', 'gallery'):
                chunks = {name: [] for name in ('fused', 'shared_global', 'joint_local')}
                scalars = {name: [] for name in ('global_norm', 'correction_norm',
                    'scaled_correction_to_global_norm', 'global_correction_cosine',
                    'global_fused_cosine', 'global_fused_l2')}
                records = entry.records_for(protocol, split)
                for raw in entry.loader_for(protocol, records, training=False, method='PLAIN_V8'):
                    fused = model(entry.runner._eval_batch(raw, args.dataset)).float()
                    global_value = captured['global_embedding'].float()
                    correction = captured['correction'].float()
                    global_unit = F.normalize(global_value, dim=1)
                    local_unit = F.normalize(correction, dim=1)
                    reconstructed = F.normalize(global_value + model.readout_gain * correction, dim=1)
                    maximum_forward_error = max(maximum_forward_error,
                        (reconstructed - fused).abs().max().item())
                    global_norm, correction_norm = global_value.norm(dim=1), correction.norm(dim=1)
                    assert (global_norm > 0).all() and (correction_norm > 0).all()
                    values = {'global_norm': global_norm, 'correction_norm': correction_norm,
                        'scaled_correction_to_global_norm': model.readout_gain.abs() * correction_norm / global_norm,
                        'global_correction_cosine': (global_unit * local_unit).sum(dim=1),
                        'global_fused_cosine': (global_unit * fused).sum(dim=1),
                        'global_fused_l2': (global_unit - fused).norm(dim=1)}
                    for name, value in values.items():
                        scalars[name].append(value.cpu())
                    for name, value in (('fused', fused), ('shared_global', global_unit), ('joint_local', local_unit)):
                        chunks[name].append(value.cpu())
                vectors[split] = {name: torch.cat(values) for name, values in chunks.items()}
                energies[split] = {name: torch.cat(values) for name, values in scalars.items()}
                assert all(value.shape == (protocol['counts'][split], 1536) for value in vectors[split].values())
                assert all(len(value) == protocol['counts'][split] for value in energies[split].values())
        backbone_hook.remove()
        role_hook.remove()
        assert maximum_forward_error < 1e-5
        saved = torch.load(run / 'official_distances.pt', map_location='cpu', weights_only=False)
        differences = {}
        for name in ('fused', 'shared_global', 'joint_local'):
            actual = entry.runner.distance_matrix(vectors['query'][name], vectors['gallery'][name])
            differences[name] = (actual - saved[name]).abs().max().item()
            assert differences[name] < 1e-5
        energy_path = args.output_dir / f"{row['variant']}_per_sample.pt"
        torch.save({'energies': energies, 'records': protocol['records'],
                    'checkpoint_sha256': row['checkpoint_sha256']}, energy_path)
        report = {'status': 'ACCEPTED_BEST_FULL_QUERY_GALLERY_ENERGY_DIAGNOSIS',
            'at': datetime.now().astimezone().isoformat(), 'dataset': args.dataset,
            'variant': row['variant'], 'best_epoch': row['best_epoch'],
            'checkpoint_sha256': row['checkpoint_sha256'], 'receipt_sha256': row['receipt_sha256'],
            'distance_sha256': row['distance_sha256'], 'readout_gain': model.readout_gain.item(),
            'maximum_forward_reconstruction_error': maximum_forward_error,
            'maximum_saved_distance_difference': differences,
            'statistics': {split: {name: statistics(value) for name, value in fields.items()}
                           for split, fields in energies.items()},
            'per_sample_path': str(energy_path), 'per_sample_sha256': sha(energy_path)}
        (args.output_dir / f"{row['variant']}.json").write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
        del model, captured, vectors, energies, saved, actual
        torch.cuda.empty_cache()
    summary = {'status': 'FIVE_ACCEPTED_BEST_ENERGY_DIAGNOSTICS_COMPLETE',
        'at': datetime.now().astimezone().isoformat(), 'matrix_sha256': sha(args.matrix),
        'source_sha256': sha(Path(__file__)), 'runtime_source_sha256': manifest['source_sha256'],
        'dataset': args.dataset, 'rows': reports,
        'boundary': 'Read-only full query/gallery normal forward of five already accepted fused-mAP-best '
                    'checkpoints; all three saved distance matrices must agree. No fitting, reweighting, '
                    'checkpoint search, retraining or new retrieval score. Official post-selection '
                    'mechanism diagnosis only; norm/cosine statistics do not prove causal influence '
                    'or optimizer update shares. Local tensors are not independently trained roles.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'dataset': args.dataset, 'endpoints': len(reports)}))


if __name__ == '__main__':
    main()
