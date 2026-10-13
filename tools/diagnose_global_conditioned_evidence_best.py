"""Two closed exposure bests: actual G/C/F decomposition, no fitting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_conditioned_evidence as producer
from tools import diagnose_native_research_best as base

SCHEMA = 'trifusion-global-conditioned-best-decomposition-v1'
KEYS = ('shared_global', 'correction', 'raw_fused', 'fused')
runner = producer.inner.runner


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    for name, digest in seal['source_sha256'].items():
        assert runner.sha256(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert runner.sha256(Path(name)) == digest, name
    state = json.loads(Path(seal['campaign']).read_text())
    exit_record = json.loads(Path(seal['controller_exit']).read_text())
    assert state['status'] == 'COMPLETE_TWO_FULL50_FIRST_STRICT'
    assert exit_record['exit_code'] == 0
    assert {row['exposure'] for row in state['accepted_endpoints']} == {'gradient', 'random'}


def extract(model, protocol, split):
    rows = {key: [] for key in KEYS}
    records = runner.records_for(protocol, split)
    with torch.inference_mode():
        for raw in runner.loader_for(protocol, records, training=False, method='PLAIN_V8'):
            batch = runner._eval_batch(raw, 'RGBNT201')
            output = model.evidence_model.forward_features(batch)
            assert output['evidence_exposure'] is None
            assert model.evidence_model.roles.excluded is None
            for key in KEYS:
                rows[key].append(output[key].float().cpu())
    result = {key: torch.cat(value) for key, value in rows.items()}
    assert all(value.shape == (protocol['counts'][split], 1536)
               and torch.isfinite(value).all() for value in result.values())
    return result


def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    row = next(row for row in seal['rows'] if row['exposure'] == args.exposure)
    original = Path(row['run_dir'])
    output = args.output_dir
    assert not output.exists()
    values = argparse.Namespace(dataset='RGBNT201', variant='semantic', recipe='semantic',
        exposure=args.exposure, mode='evaluate', seed=42, epochs=50,
        protocol=Path(seal['protocol']), signal_source=Path(seal['signal_source']),
        clip_weight=Path(seal['clip_weight']), initialization=Path(row['initialization']),
        output_dir=original)
    values.baseline_sha256 = runner.sha256(values.clip_weight)
    protocol = runner.read_protocol(values.protocol, 'RGBNT201')
    producer.configure()
    model, _cfg, binding = producer.inner.foundation.build(values, protocol)
    training = json.loads((original / 'training.json').read_text())
    receipt = json.loads((original / 'official_metrics.json').read_text())
    assert binding == row['initializer'] == training['initializer']
    assert receipt['schema'] == producer.SCHEMA and receipt['status'] == 'COMPLETE'
    assert receipt['condition'] == producer.condition(values)
    assert receipt['training_epochs'] == 50
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    selected = max(training['history'], key=lambda r: (r['official_fused']['mAP'], r['epoch']))
    payload = producer.inner.foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == receipt['selected_epoch'] == selected['epoch'] == row['selected_epoch']
    assert payload['metrics'] == selected['official_fused']
    assert receipt['metrics'] == row['metrics']
    assert all(abs(payload['metrics'][key] - row['metrics'][key]) < 1e-5 for key in row['metrics'])
    assert receipt['checkpoint_sha256'] == row['checkpoint_sha256']
    assert receipt['distance_sha256'] == row['distance_sha256']
    model.eval()
    before = runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    extraction_started = time.perf_counter()
    query, gallery = [extract(model, protocol, split) for split in ('query', 'gallery')]
    extraction_seconds = time.perf_counter() - extraction_started
    after = runner._module_state_sha256(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    data = base.metadata(protocol)
    old = torch.load(original / 'official_distances.pt', map_location='cpu', weights_only=False)
    independent = torch.load(Path(seal['independent_global_distance']), map_location='cpu', weights_only=False)
    assert all(np.array_equal(data[key], old[key]) and np.array_equal(data[key], independent[key]) for key in data)
    distances = {name: runner.distance_matrix(query[key], gallery[key])
                 for name, key in (('global', 'shared_global'), ('correction', 'correction'), ('fused', 'fused'))}
    scores = {name: base.score(distance, data, 'RGBNT201', values.signal_source)
              for name, distance in distances.items()}
    scores['independent_global_only'] = base.score(independent['fused'], data, 'RGBNT201', values.signal_source)
    difference = float((distances['fused'] - old['fused']).abs().max())
    assert difference <= 1e-5
    assert all(abs(scores['fused']['metrics'][key] - receipt['metrics'][key]) < 1e-5 for key in receipt['metrics'])
    assert all(abs(scores['independent_global_only']['metrics'][key] - seal['independent_global_metrics'][key]) < 1e-5
               for key in seal['independent_global_metrics'])
    geometry = {}
    for split, features in (('query', query), ('gallery', gallery)):
        g, c, h, f = [features[key] for key in KEYS]
        assert torch.allclose(h, g + gain * c, atol=1e-6, rtol=1e-6)
        assert torch.allclose(f, F.normalize(h, dim=1), atol=1e-6, rtol=1e-6)
        assert (g.norm(dim=1) > 0).all()
        geometry[split] = {
            'scaled_correction_global_norm_ratio': base.describe((gain * c).norm(dim=1) / g.norm(dim=1)),
            'global_fused_angle_degrees': base.describe(torch.rad2deg(torch.acos(
                (F.normalize(g, dim=1) * f).sum(1).clamp(-1, 1))))}
    comparisons = {
        'same_model_global_to_fused': base.paired(scores['global'], scores['fused'], data['query_ids']),
        'independent_global_to_same_model_global': base.paired(scores['independent_global_only'], scores['global'], data['query_ids']),
        'independent_global_to_fused': base.paired(scores['independent_global_only'], scores['fused'], data['query_ids'])}
    output.mkdir(parents=True)
    for split, features in (('query', query), ('gallery', gallery)):
        torch.save(features, output / f'{split}_features.pt')
    torch.save(dict(data, **distances), output / 'diagnostic_distances.pt')
    result = {'schema': SCHEMA, 'status': 'COMPLETE', 'exposure': args.exposure, 'dataset': 'RGBNT201',
        'selected_epoch': row['selected_epoch'], 'input_checkpoint_sha256': row['checkpoint_sha256'],
        'seal_sha256': runner.sha256(args.seal), 'model_state_before_sha256': before,
        'model_state_after_sha256': after, 'readout_gain': gain, 'scores': scores,
        'comparisons': comparisons, 'geometry': geometry,
        'fused_distance_max_abs_difference': difference, 'optimizer_updates': 0,
        'inference_exposure': False, 'elapsed_seconds': time.perf_counter() - started,
        'query_gallery_extraction_seconds': extraction_seconds,
        'extraction_timing_boundary': 'Full query/gallery dataloading, one encoding per batch and CPU feature transfer; excludes construction, hashes and scoring. Single shared-server observation, not isolated kernel time.',
        'completed_at': datetime.now().astimezone().isoformat(),
        'artifacts': {name: {'bytes': (output / name).stat().st_size, 'sha256': runner.sha256(output / name)}
                      for name in ('query_features.pt', 'gallery_features.pt', 'diagnostic_distances.pt')},
        'boundary': 'One selected-best extraction per arm, each full legal query/gallery batch once. No optimizer, new weight, re-selection, local comparator or hyperparameter search. G/C/F scores are descriptive diagnostics, not independent ablations or causal root-cause proof. Same-model global is measured here, not inferred from teacher scalars.'}
    require_inputs(seal)
    base.write(output / 'DIAGNOSIS.json', result)
    print(json.dumps({'exposure': args.exposure, 'status': result['status'],
                      'metrics': {name: score['metrics'] for name, score in scores.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exposure', choices=('gradient', 'random'), required=True)
    for name in ('seal', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.seal, args.output_dir = args.seal.resolve(), args.output_dir.resolve()
    diagnose(args, json.loads(args.seal.read_text()))
