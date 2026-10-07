"""Five selected checkpoints: own global/correction/fused, no fitting or selection."""
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
from tools import diagnose_native_research_best as diagnosis
from tools import queue_incremental_qualified_five as panel
from tools import run_incremental_role_objective_checked as checked

SCHEMA = 'trifusion-incremental-fixed-best-diagnosis-v1'
JOBS = (('RGBNT201', 'md_batch_ratio'), ('RGBNT201', 'repair_keep'),
        ('MSVR310', 'md_batch_ratio'), ('MSVR310', 'repair_keep'), ('RGBNT100', 'md_batch_ratio'))
entry = checked.entry
entry.build_core = checked.build_core
entry.loss_values = checked.loss_values
entry.runner = entry.inner.runner
diagnosis.entry = entry


def require_inputs(seal):
    assert seal['schema'] == SCHEMA and len(seal['source_sha256']) == 424
    assert {(r['dataset'], r['variant']) for r in seal['rows']} == set(JOBS) and len(seal['rows']) == 5
    for name, digest in seal['source_sha256'].items():
        assert panel.base.sha(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert panel.base.sha(Path(name)) == digest, name
    terminal = json.loads((ROOT / 'logs/qualified_five_incremental_full_launch_20261007_888/EXIT.json').read_text())
    assert terminal['exit_code'] == 0


def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    row = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (args.dataset, args.objective))
    independent_row = next(r for r in seal['global_rows'] if r['dataset'] == args.dataset)
    original = Path(row['run_dir'])
    output = args.output_dir / f'{args.dataset}_{args.objective}'
    output.mkdir()
    values = argparse.Namespace(dataset=args.dataset, objective=args.objective, variant='semantic', recipe='semantic',
        mode='evaluate', protocol=ROOT / f'logs/training_feature_scale_protocols_20261002/{args.dataset}.json',
        signal_source=ROOT / 'comparators/Signal-cd1b0a6', clip_weight=ROOT / 'pertrained-model/ViT-B-16.pt',
        initialization=ROOT / f'logs/incremental_role_objective_m0_v1_20261007_886/initialization/{args.dataset}_{args.objective}.json',
        output_dir=original, seed=42, epochs=50)
    values.baseline_sha256 = entry.runner.sha256(values.clip_weight)
    protocol = entry.runner.read_protocol(values.protocol, args.dataset)
    for records in protocol['records'].values():
        for record in records:
            key = Path(record['paths'][0]).name
            environment = record['scene'] if args.dataset == 'MSVR310' else record['camera']
            assert key not in entry.ENVIRONMENTS or entry.ENVIRONMENTS[key] == environment
            entry.ENVIRONMENTS[key] = environment
    entry.configure()
    model, _cfg, binding = entry.inner.foundation.build(values, protocol)
    assert binding == row['initializer']
    payload = entry.inner.foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch']
    assert all(abs(payload['metrics'][k] - v) < 1e-5 for k, v in row['metrics'].items())
    model.eval()
    before = entry.runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    query, _ = diagnosis.extract(model, protocol, 'query', 'semantic')
    gallery, _ = diagnosis.extract(model, protocol, 'gallery', 'semantic')
    after = entry.runner._module_state_sha256(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    data = diagnosis.metadata(protocol)
    old = torch.load(original / 'official_distances.pt', map_location='cpu', weights_only=False)
    independent = torch.load(Path(independent_row['run_dir']) / 'official_distances.pt', map_location='cpu', weights_only=False)
    assert all(np.array_equal(data[k], old[k]) and np.array_equal(data[k], independent[k]) for k in data)
    distances, scores = {}, {}
    for name, key in (('fused', 'fused'), ('global', 'shared_global'), ('correction', 'correction')):
        q = query[key] if name == 'fused' else F.normalize(query[key], dim=1)
        g = gallery[key] if name == 'fused' else F.normalize(gallery[key], dim=1)
        distances[name] = entry.runner.distance_matrix(q, g)
        scores[name] = diagnosis.score(distances[name], data, args.dataset, values.signal_source)
    scores['independent_global_only'] = diagnosis.score(independent['fused'], data, args.dataset, values.signal_source)
    assert all(abs(scores['fused']['metrics'][k] - v) < 1e-5 for k, v in row['metrics'].items())
    assert all(abs(scores['independent_global_only']['metrics'][k] - v) < 1e-5 for k, v in independent_row['metrics'].items())
    distributions = {}
    for split, features in (('query', query), ('gallery', gallery)):
        g, c, h, f = (features[k] for k in diagnosis.KEYS)
        assert torch.allclose(h, g + gain * c, atol=1e-6, rtol=1e-6)
        assert torch.allclose(f, F.normalize(h, dim=1), atol=1e-6, rtol=1e-6)
        gnorm = g.norm(dim=1)
        assert bool((gnorm > 0).all())
        samples = {'global_norm': gnorm, 'correction_norm': c.norm(dim=1), 'fused_raw_norm': h.norm(dim=1),
            'actual_scaled_correction_to_global_norm_ratio': (gain * c).norm(dim=1) / gnorm,
            'global_to_fused_angle_degrees': torch.acos(F.cosine_similarity(g, f, dim=1).clamp(-1, 1)) * (180 / np.pi)}
        distributions[split] = {k: diagnosis.describe(v) for k, v in samples.items()}
        torch.save({'features': features, 'per_sample_diagnostics': samples}, output / f'{split}_features.pt')
    torch.save(dict(data, **distances), output / 'diagnostic_distances.pt')
    report = dict(schema=SCHEMA, status='COMPLETE', dataset=args.dataset, objective=args.objective,
        completed_at=diagnosis.stamp(), elapsed_seconds=time.perf_counter() - started,
        input_checkpoint_sha256=row['checkpoint_sha256'], selected_epoch=row['best_epoch'],
        original_receipt_sha256=row['receipt_sha256'], binding=binding, readout_gain=gain,
        model_state_before_sha256=before, model_state_after_sha256=after, scores=scores, diagnostic=distributions,
        fused_distance_max_absolute_difference_from_original=float((distances['fused'] - old['fused']).abs().max()),
        comparisons={'same_model_global_to_fused': diagnosis.paired(scores['global'], scores['fused'], data['query_ids']),
            'independent_global_only_to_same_model_global': diagnosis.paired(scores['independent_global_only'], scores['global'], data['query_ids']),
            'independent_global_only_to_fused': diagnosis.paired(scores['independent_global_only'], scores['fused'], data['query_ids'])},
        artifacts={name: {'bytes': (output / name).stat().st_size, 'sha256': panel.base.sha(output / name)}
            for name in ('query_features.pt', 'gallery_features.pt', 'diagnostic_distances.pt')},
        boundary='Fixed five originally selected mAP-best weights, no updates/new selection/loss rescue. One eval forward per record, existing full camera/scene protocol. Same-model g is a component diagnosis, not an independent ablation; corrections are normalized only for standalone diagnosis. Descriptive differences, not causal gradient attribution or seed/SOTA proof. Failed100repair and all earlier failures retained.')
    require_inputs(seal)
    diagnosis.write(output / 'DIAGNOSIS.json', report)
    print(json.dumps(dict(status='COMPLETE', dataset=args.dataset, objective=args.objective,
        seconds=report['elapsed_seconds'], metrics={k: v['metrics'] for k, v in scores.items()})), flush=True)


def coordinate(args, seal):
    assert not args.output_dir.exists()
    require_inputs(seal)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    used = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    assert all(int(line.split(',')[1]) < 500 for line in used.splitlines())
    args.output_dir.mkdir()
    state = dict(schema=SCHEMA, status='RUNNING', started_at=diagnosis.stamp(), pid=os.getpid(),
        seal_sha256=panel.base.sha(args.seal), jobs=[], optimizer_updates=0, physical_gpus=[0,1], environment=sys.executable)
    diagnosis.write(args.output_dir / 'campaign.json', state)
    for dataset, objective in JOBS:
        row = dict(dataset=dataset, objective=objective, started_at=diagnosis.stamp(), status='RUNNING')
        state['jobs'].append(row)
        command = [sys.executable, '-B', str(Path(__file__)), '--seal', str(args.seal), '--output-dir', str(args.output_dir),
            '--dataset', dataset, '--objective', objective]
        with (args.output_dir / f'{dataset}_{objective}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            diagnosis.write(args.output_dir / 'campaign.json', state)
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=diagnosis.stamp())
        diagnosis.write(args.output_dir / 'campaign.json', state)
        if code:
            state.update(status='FAILED', completed_at=diagnosis.stamp())
            diagnosis.write(args.output_dir / 'campaign.json', state)
            return code
    require_inputs(seal)
    state.update(status='COMPLETE', completed_at=diagnosis.stamp())
    diagnosis.write(args.output_dir / 'campaign.json', state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=('RGBNT201','MSVR310','RGBNT100'))
    parser.add_argument('--objective', choices=entry.OBJECTIVES)
    args = parser.parse_args()
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert (args.dataset is None) == (args.objective is None)
    seal = json.loads(args.seal.read_text())
    if args.dataset is None:
        return coordinate(args, seal)
    diagnose(args, seal)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
