"""One fixed-best G/C/F and spectral-factor extraction per closed arm."""
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
from tools import run_joint_spectral_delta as producer
from tools import queue_joint_spectral_delta as panel
from tools import diagnose_native_research_best as base

SCHEMA = 'trifusion-joint-spectral-fixed-best-diagnosis-v1'
KEYS = ('shared_global', 'correction', 'raw_fused', 'fused')
DETAILS = ('actual_descriptor', 'reference_descriptor', 'delta_descriptor',
           'modality_factors', 'reference_factors')
runner = producer.inner.runner


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    assert seal['conditions'] == list(panel.MODES)
    for name, digest in seal['source_sha256'].items():
        assert runner.sha256(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert runner.sha256(Path(name)) == digest, name
    state = json.loads(Path(seal['campaign']).read_text())
    assert state['status'] == 'COMPLETE' and state['accepted'] == state['expected'] == 3
    assert json.loads(Path(seal['controller_exit']).read_text())['exit_code'] == 0


def moments(value):
    """Empirical vector energies, without labeling common content as noise."""
    value = value.double()
    mean = value.mean(dim=0)
    return dict(samples=value.shape[0], axes=list(value.shape[1:-1]),
                sample_mean_energy=value.square().sum(dim=-1).mean(dim=0).tolist(),
                common_mean_energy=mean.square().sum(dim=-1).tolist(),
                centered_energy=(value - mean).square().sum(dim=-1).mean(dim=0).tolist())


def extract(model, protocol, split):
    rows = {key: [] for key in KEYS}
    details = {key: [] for key in DETAILS}
    regions = {key: [] for key in ('processed_regions', 'reference_regions')}

    def region_hook(_module, inputs):
        processed, reference, _weight = inputs
        for name, group in zip(regions, (processed, reference)):
            regions[name].append(torch.stack(group, dim=1).float().cpu())

    handle = model.evidence_model.spectral_readout.register_forward_pre_hook(region_hook)
    with torch.inference_mode():
        for index, raw in enumerate(runner.loader_for(protocol, runner.records_for(protocol, split),
                                                      training=False, method='PLAIN_V8')):
            output = model.evidence_model.forward_features(runner._eval_batch(raw, 'RGBNT201'))
            assert all(len(value) == index + 1 for value in regions.values())
            for key in KEYS:
                rows[key].append(output[key].float().cpu())
            for key in DETAILS:
                details[key].append(output['spectral_diagnostics'][key].float().cpu())
    handle.remove()
    features = {key: torch.cat(value) for key, value in rows.items()}
    detail = {key: torch.cat(value) for key, value in details.items()}
    region_values = {key: torch.cat(value) for key, value in regions.items()}
    assert all(value.shape == (protocol['counts'][split], 3, 3, 16, 128)
               and bool(torch.isfinite(value).all()) for value in region_values.values())
    detail.update(region_values)
    assert all(value.shape == (protocol['counts'][split], 1536)
               and bool(torch.isfinite(value).all()) for value in features.values())
    assert all(value.shape[0] == protocol['counts'][split]
               and bool(torch.isfinite(value).all()) for value in detail.values())
    return features, detail


def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.output_dir.exists()
    row = next(row for row in seal['rows'] if row['spectral_aggregation'] == args.aggregation)
    original = Path(row['run_dir'])
    values = argparse.Namespace(dataset='RGBNT201', variant='semantic', recipe='semantic',
        mode='evaluate', seed=42, epochs=50, protocol=Path(seal['protocol']),
        signal_source=Path(seal['signal_source']), clip_weight=Path(seal['clip_weight']),
        initialization=Path(row['initialization']), output_dir=original)
    values.baseline_sha256 = runner.sha256(values.clip_weight)
    protocol = runner.read_protocol(values.protocol, 'RGBNT201')
    producer.MODE = args.aggregation
    producer.configure()
    model, _cfg, binding = producer.inner.foundation.build(values, protocol)
    training = json.loads((original / 'training.json').read_text())
    receipt = json.loads((original / 'official_metrics.json').read_text())
    assert binding == row['initializer'] == training['initializer']
    assert receipt['schema'] == producer.SCHEMA and receipt['status'] == 'COMPLETE'
    assert receipt['condition'] == producer.condition(values)
    assert [item['epoch'] for item in training['history']] == list(range(1, 51))
    selected = max(training['history'], key=lambda item: (item['official_fused']['mAP'], item['epoch']))
    payload = producer.inner.foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == receipt['selected_epoch'] == selected['epoch'] == row['best_epoch']
    assert payload['metrics'] == selected['official_fused']
    assert receipt['metrics'] == row['metrics']
    assert receipt['checkpoint_sha256'] == row['checkpoint_sha256']
    assert receipt['distance_sha256'] == row['distance_sha256']
    model.eval()
    before = runner._module_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    extraction_started = time.perf_counter()
    query, qdetail = extract(model, protocol, 'query')
    gallery, gdetail = extract(model, protocol, 'gallery')
    extraction_seconds = time.perf_counter() - extraction_started
    after = runner._module_state_sha256(model)
    assert before == after and all(parameter.grad is None for parameter in model.parameters())
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
    for split, features, detail in (('query', query, qdetail), ('gallery', gallery, gdetail)):
        g, c, h, f = [features[key] for key in KEYS]
        assert torch.allclose(h, g + gain * c, atol=1e-6, rtol=1e-6)
        assert torch.allclose(f, F.normalize(h, dim=1), atol=1e-6, rtol=1e-6)
        assert bool((g.norm(dim=1) > 0).all())
        geometry[split] = dict(
            scaled_correction_global_norm_ratio=base.describe((gain * c).norm(dim=1) / g.norm(dim=1)),
            global_fused_angle_degrees=base.describe(torch.rad2deg(torch.acos(
                (F.normalize(g, dim=1) * f).sum(1).clamp(-1, 1)))),
            correction_energy=moments(c), descriptor_energy={key: moments(value) for key, value in detail.items()})
    comparisons = {
        'same_model_global_to_fused': base.paired(scores['global'], scores['fused'], data['query_ids']),
        'independent_global_to_same_model_global': base.paired(scores['independent_global_only'], scores['global'], data['query_ids']),
        'independent_global_to_fused': base.paired(scores['independent_global_only'], scores['fused'], data['query_ids'])}
    args.output_dir.mkdir()
    for split, features, detail in (('query', query, qdetail), ('gallery', gallery, gdetail)):
        torch.save(dict(features=features, spectral=detail), args.output_dir / (split + '_features.pt'))
    torch.save(dict(data, **distances), args.output_dir / 'diagnostic_distances.pt')
    result = dict(schema=SCHEMA, status='COMPLETE', aggregation=args.aggregation, dataset='RGBNT201',
        selected_epoch=row['best_epoch'], input_checkpoint_sha256=row['checkpoint_sha256'],
        seal_sha256=runner.sha256(args.seal), model_state_before_sha256=before, model_state_after_sha256=after,
        readout_gain=gain, rank_weights=model.evidence_model.spectral_readout.rank_weights.detach().cpu().tolist(),
        role_order=['cnn', 'transformer', 'mamba'], modality_order=['RGB', 'NI', 'TI'],
        scores=scores, comparisons=comparisons, geometry=geometry, optimizer_updates=0,
        fused_distance_max_abs_difference=difference, query_gallery_extraction_seconds=extraction_seconds,
        elapsed_seconds=time.perf_counter() - started, completed_at=datetime.now().astimezone().isoformat(),
        artifacts={name: {'bytes': (args.output_dir / name).stat().st_size,
                          'sha256': runner.sha256(args.output_dir / name)}
                   for name in ('query_features.pt', 'gallery_features.pt', 'diagnostic_distances.pt')},
        boundary='One selected-best query/gallery extraction per closed arm, zero fitting/new weight/reselection. Original processed/reference region tensors are retained before slot pooling by a read-only hook for later fixed-data comparisons. G/C/F and empirical moments describe use and common/centered energy; they do not establish causality, identity information, noise, pure interaction, redundancy or novelty. No cross-spectral slot correspondence is assumed; no new ranking or test centering.')
    require_inputs(seal)
    base.write(args.output_dir / 'DIAGNOSIS.json', result)
    print(json.dumps(dict(status=result['status'],aggregation=args.aggregation,
                         scores={name: score['metrics'] for name, score in scores.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregation', choices=panel.MODES, required=True)
    for name in ('seal', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.seal, args.output_dir = args.seal.resolve(), args.output_dir.resolve()
    diagnose(args, json.loads(args.seal.read_text()))
