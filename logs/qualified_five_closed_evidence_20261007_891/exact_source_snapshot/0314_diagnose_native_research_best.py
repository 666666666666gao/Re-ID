"""Read-only g/c/h/f diagnosis of the six sealed V6 role checkpoints."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_native_research as entry
from tools import queue_native_research as panel
from tools import run_foundation_recipe as foundation
from tools.analyze_correspondence_distances import camera_scores, scene_scores

SCHEMA = 'trifusion-native-fixed-best-diagnosis-v1'
DATASETS = ('RGBNT201', 'MSVR310', 'RGBNT100')
VARIANTS = ('semantic', 'native')
KEYS = ('shared_global', 'correction', 'raw_fused', 'fused')


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    for name, digest in seal['source_sha256'].items():
        assert panel.base.sha(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert panel.base.sha(Path(name)) == digest, name


def metadata(protocol):
    return {f'{split}_{plural}': np.asarray([row[field] for row in protocol['records'][split]])
            for split in ('query', 'gallery')
            for field, plural in (('identity', 'ids'), ('camera', 'cameras'), ('scene', 'scenes'))}


def score(distance, data, dataset, source):
    from utils import metrics as author
    assert Path(author.__file__).resolve() == (source / 'utils/metrics.py').resolve()
    env = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    result = scorer(distance.numpy(), data['query_ids'], data['gallery_ids'],
                    data['query_' + env], data['gallery_' + env])
    if dataset == 'MSVR310':
        cmc, ap = author.eval_func_msrv(distance.numpy(), data['query_ids'], data['gallery_ids'],
            data['query_cameras'], data['gallery_cameras'], data['query_scenes'], data['gallery_scenes'])
    else:
        cmc, ap = author.eval_func(distance.numpy(), data['query_ids'], data['gallery_ids'],
                                  data['query_cameras'], data['gallery_cameras'])
    upstream = {'mAP': float(ap) * 100, 'Rank-1': float(cmc[0]) * 100,
                'Rank-5': float(cmc[4]) * 100, 'Rank-10': float(cmc[9]) * 100}
    assert all(abs(result['metrics'][key] - value) < 1e-5 for key, value in upstream.items())
    return result


def paired(first, second, ids):
    a, b = (np.asarray(row['first_match_rank']) for row in (first, second))
    delta = np.asarray(second['average_precision']) - np.asarray(first['average_precision'])
    identities = [{'identity': int(identity), 'queries': int((ids == identity).sum()),
                   'mean_delta_ap_points': float(delta[ids == identity].mean() * 100)}
                  for identity in np.unique(ids)]
    repairs, errors = int(((a != 1) & (b == 1)).sum()), int(((a == 1) & (b != 1)).sum())
    metrics = {key: second['metrics'][key] - first['metrics'][key] for key in first['metrics']}
    assert abs(metrics['Rank-1'] - (repairs - errors) * 100 / len(delta)) < 1e-5
    return {'delta_metrics': metrics, 'rank1_repairs': repairs, 'rank1_new_errors': errors,
            'identity_macro_delta_ap_points': float(np.mean([x['mean_delta_ap_points'] for x in identities])),
            'identity_changes': identities,
            'query_changes': [{'query_index': i, 'identity': int(identity),
                'control_ap': float(first['average_precision'][i]),
                'candidate_ap': float(second['average_precision'][i]),
                'control_first_rank': int(a[i]), 'candidate_first_rank': int(b[i])}
                for i, identity in enumerate(ids)]}


def describe(values):
    values = values.float().numpy()
    assert np.isfinite(values).all()
    return {'count': len(values), 'mean': float(values.mean()),
            'min': float(values.min()), 'max': float(values.max()),
            'percentiles_0_25_50_75_95_100': np.percentile(values, [0, 25, 50, 75, 95, 100]).tolist()}


def extract(model, protocol, split, variant):
    rows = {key: [] for key in KEYS}
    detail_norms, detail_ratios = [], []
    current = []

    def detail_hook(_module, _inputs, output):
        assert not current
        current.append(output.detach())
        detail_norms.append(output.float().flatten(1).norm(dim=1).cpu())

    def cnn_hook(_module, inputs):
        assert len(current) == 1
        delta = current.pop()
        combined = inputs[0].detach().float()
        base = (combined - delta.float()).flatten(1).norm(dim=1)
        assert bool((base > 0).all())
        detail_ratios.append((delta.float().flatten(1).norm(dim=1) / base).cpu())

    handles = []
    core = model.evidence_model
    if variant == 'native':
        handles = [core.roles.detail_reader.register_forward_hook(detail_hook),
                   core.roles.output_norms[0].register_forward_pre_hook(cnn_hook)]
    with torch.inference_mode():
        records = entry.runner.records_for(protocol, split)
        for raw in entry.runner.loader_for(protocol, records, training=False, method='PLAIN_V8'):
            batch = entry.runner._eval_batch(raw, protocol['dataset'])
            output = core.forward_features(batch)
            assert not current
            for key in KEYS:
                value = output[key].float().cpu()
                assert value.ndim == 2 and value.shape[1] == 1536 and bool(torch.isfinite(value).all())
                rows[key].append(value)
    for handle in handles:
        handle.remove()
    result = {key: torch.cat(value) for key, value in rows.items()}
    assert all(value.shape == (protocol['counts'][split], 1536) for value in result.values())
    detail = {}
    if variant == 'native':
        detail = {'actual_reader_output_norm': torch.cat(detail_norms),
                  'reader_to_derived_semantic_plus_anchor_norm_ratio': torch.cat(detail_ratios)}
        assert all(len(value) == protocol['counts'][split] for value in detail.values())
    return result, detail


def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    row = next(row for row in seal['rows'] if (row['dataset'], row['variant']) == (args.dataset, args.variant))
    original = Path(row['run_dir'])
    output = args.output_dir / f'{args.dataset}_{args.variant}'
    output.mkdir()
    entry.configure()
    values = argparse.Namespace(dataset=args.dataset, variant=args.variant, recipe=args.variant,
        protocol=panel.PROTOCOLS / f'{args.dataset}.json', signal_source=panel.SOURCE,
        clip_weight=panel.WEIGHTS / 'ViT-B-16.pt', initialization=args.campaign / 'initialization' / f'{args.dataset}_{args.variant}.json',
        output_dir=original, seed=42, epochs=50)
    values.baseline_sha256 = entry.runner.sha256(values.clip_weight)
    protocol = entry.runner.read_protocol(values.protocol, args.dataset)
    model, _cfg, binding = foundation.build(values, protocol)
    assert binding == row['initializer']
    payload = foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch']
    assert all(abs(payload['metrics'][key] - value) < 1e-5 for key, value in row['metrics'].items())
    model.eval()
    before = entry.runner._module_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    query, qdetail = extract(model, protocol, 'query', args.variant)
    gallery, gdetail = extract(model, protocol, 'gallery', args.variant)
    after = entry.runner._module_state_sha256(model)
    assert before == after and all(parameter.grad is None for parameter in model.parameters())
    data = metadata(protocol)
    old = torch.load(original / 'official_distances.pt', map_location='cpu', weights_only=False)
    global_row = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (args.dataset, 'global_only'))
    independent = torch.load(Path(global_row['run_dir']) / 'official_distances.pt', map_location='cpu', weights_only=False)
    assert all(np.array_equal(data[key], old[key]) and np.array_equal(data[key], independent[key]) for key in data)
    distances, scores = {}, {}
    for name, key in (('fused', 'fused'), ('global', 'shared_global'), ('correction', 'correction')):
        q = query[key] if name == 'fused' else F.normalize(query[key], dim=1)
        g = gallery[key] if name == 'fused' else F.normalize(gallery[key], dim=1)
        distances[name] = entry.runner.distance_matrix(q, g)
        scores[name] = score(distances[name], data, args.dataset, values.signal_source)
    scores['independent_global_only'] = score(independent['fused'], data, args.dataset, values.signal_source)
    assert all(abs(scores['fused']['metrics'][k] - v) < 1e-5 for k, v in row['metrics'].items())
    assert all(abs(scores['independent_global_only']['metrics'][k] - v) < 1e-5 for k, v in global_row['metrics'].items())
    diagnostic = {}
    for split, features, detail in (('query', query, qdetail), ('gallery', gallery, gdetail)):
        g, c, h, f = (features[key] for key in KEYS)
        assert torch.allclose(h, g + gain * c, atol=1e-6, rtol=1e-6)
        assert torch.allclose(f, F.normalize(h, dim=1), atol=1e-6, rtol=1e-6)
        gnorm = g.norm(dim=1)
        assert bool((gnorm > 0).all())
        samples = {'global_norm': gnorm, 'correction_norm': c.norm(dim=1), 'fused_raw_norm': h.norm(dim=1),
            'actual_scaled_correction_to_global_norm_ratio': (gain * c).norm(dim=1) / gnorm,
            'global_to_fused_angle_degrees': torch.acos(F.cosine_similarity(g, f, dim=1).clamp(-1, 1)) * (180 / np.pi),
            **detail}
        diagnostic[split] = {key: describe(value) for key, value in samples.items()}
        torch.save({'features': features, 'per_sample_diagnostics': samples}, output / f'{split}_features.pt')
    torch.save(dict(data, **distances), output / 'diagnostic_distances.pt')
    report = {'schema': SCHEMA, 'status': 'COMPLETE', 'dataset': args.dataset, 'variant': args.variant,
        'completed_at': stamp(), 'elapsed_seconds': time.perf_counter() - started,
        'input_checkpoint_sha256': row['checkpoint_sha256'], 'selected_epoch': row['best_epoch'],
        'original_receipt_sha256': row['receipt_sha256'], 'binding': binding, 'readout_gain': gain,
        'model_state_before_sha256': before, 'model_state_after_sha256': after,
        'scores': scores, 'diagnostic': diagnostic,
        'fused_distance_max_absolute_difference_from_original': float((distances['fused'] - old['fused']).abs().max()),
        'comparisons': {
            'same_model_global_to_fused': paired(scores['global'], scores['fused'], data['query_ids']),
            'independent_global_only_to_same_model_global': paired(scores['independent_global_only'], scores['global'], data['query_ids']),
            'independent_global_only_to_fused': paired(scores['independent_global_only'], scores['fused'], data['query_ids'])},
        'artifacts': {name: {'bytes': (output / name).stat().st_size, 'sha256': panel.base.sha(output / name)}
                      for name in ('query_features.pt', 'gallery_features.pt', 'diagnostic_distances.pt')},
        'boundary': 'Fixed selected best; one eval forward per record, no AMP inference, no updates, no new selection. Same-model global is not an independent ablation. Differences are descriptive, not causal gradient attribution. Native amplitude denominator is the pre-LayerNorm CNN-plus-anchor input minus actual detail delta, not a second forward. Original evaluation/parity/M0 records unchanged.'}
    require_inputs(seal)
    write(output / 'DIAGNOSIS.json', report)
    print(json.dumps({'status': report['status'], 'dataset': args.dataset, 'variant': args.variant,
                      'seconds': report['elapsed_seconds'], 'metrics': {k: v['metrics'] for k, v in scores.items()},
                      'readout_gain': gain}), flush=True)


def coordinate(args, seal):
    assert not args.output_dir.exists()
    require_inputs(seal)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    used = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    assert all(int(line.split(',')[1]) < 500 for line in used.splitlines())
    args.output_dir.mkdir()
    state = {'schema': SCHEMA, 'status': 'RUNNING', 'started_at': stamp(), 'pid': os.getpid(),
             'seal_sha256': panel.base.sha(args.seal), 'jobs': [], 'optimizer_updates': 0,
             'physical_gpus': [0, 1], 'environment': sys.executable}
    write(args.output_dir / 'campaign.json', state)
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = {'dataset': dataset, 'variant': variant, 'started_at': stamp(), 'status': 'RUNNING'}
            state['jobs'].append(row)
            command = [sys.executable, '-B', str(Path(__file__)), '--campaign', str(args.campaign),
                '--seal', str(args.seal), '--output-dir', str(args.output_dir), '--dataset', dataset, '--variant', variant]
            with (args.output_dir / f'{dataset}_{variant}.log').open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                write(args.output_dir / 'campaign.json', state)
                code = process.wait()
            row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=stamp())
            write(args.output_dir / 'campaign.json', state)
            if code:
                state.update(status='FAILED', completed_at=stamp())
                write(args.output_dir / 'campaign.json', state)
                return code
    require_inputs(seal)
    state.update(status='COMPLETE', completed_at=stamp())
    write(args.output_dir / 'campaign.json', state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'seal', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    seal = json.loads(args.seal.read_text())
    assert (args.dataset is None) == (args.variant is None)
    if args.dataset is None:
        return coordinate(args, seal)
    diagnose(args, seal)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
