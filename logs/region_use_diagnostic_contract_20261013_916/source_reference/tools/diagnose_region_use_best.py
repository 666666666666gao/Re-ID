"""Current region-use bests: five descriptive scores, no fitting."""
import argparse
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
from tools import diagnose_same_evidence_comparison as frozen
from tools import run_region_use_probe as producer
base = frozen.base
SCHEMA = 'trifusion-region-use-fixed-best-decomposition-v1'

def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    assert len(seal['rows']) == 6
    assert {(r['dataset'], r['comparison']) for r in seal['rows']} == {
        (d, c) for d in ('RGBNT201', 'MSVR310', 'RGBNT100') for c in ('uniform', 'ambiguity')}
    for name, digest in seal['source_sha256'].items():
        assert base.panel.base.sha(ROOT/name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert base.panel.base.sha(Path(name)) == digest, name
    state = json.loads((Path(seal['remaining_campaign'])/'campaign.json').read_text())
    exit_record = json.loads(Path(seal['remaining_exit']).read_text())
    assert state['status'] == 'TRAINING_AND_STRICT_EVALUATION_COMPLETE'
    assert state['jobs'][0]['status'] == 'COMPLETE' and exit_record['exit_code'] == 0

def region_scores(query, gallery):
    q = F.normalize(query.float().reshape(len(query), 9, 16, 128), dim=-1)
    g = F.normalize(gallery.float().reshape(len(gallery), 9, 16, 128), dim=-1)
    scores = torch.empty(len(q), len(g))
    with torch.autocast('cuda', enabled=False):
        for first in range(0, len(q), 16):
            a = q[first:first+16].to('cuda:0')
            for second in range(0, len(g), 128):
                b = g[second:second+128].to('cuda:0')
                pair = torch.einsum('akpc,bkrc->abkpr', a, b)
                value = 0.5*(pair.max(-1).values.mean((-1, -2))+
                             pair.max(-2).values.mean((-1, -2)))
                scores[first:first+len(a), second:second+len(b)] = value.cpu()
    assert torch.isfinite(scores).all()
    assert float(scores.min()) >= -1.00001 and float(scores.max()) <= 1.00001
    return scores

def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    row = next(r for r in seal['rows']
               if (r['dataset'], r['comparison']) == (args.dataset, args.comparison))
    original, output = Path(row['run_dir']), args.output_dir
    assert not output.exists()
    values = argparse.Namespace(dataset=args.dataset, comparison=args.comparison,
        variant='semantic', recipe='semantic', mode='evaluate', seed=42, epochs=50,
        protocol=base.panel.PROTOCOLS/f'{args.dataset}.json', signal_source=base.panel.SOURCE,
        clip_weight=base.panel.WEIGHTS/'ViT-B-16.pt',
        initialization=Path(row['initialization']), output_dir=original)
    values.baseline_sha256 = producer.inner.runner.sha256(values.clip_weight)
    protocol = producer.inner.runner.read_protocol(values.protocol, args.dataset)
    # Vehicle eval also uses the patched training_batch. Mirror the real entry.
    for records in protocol['records'].values():
        for record in records:
            key = Path(record['paths'][0]).name
            environment = record['scene'] if args.dataset == 'MSVR310' else record['camera']
            assert key not in producer.ENVIRONMENTS or producer.ENVIRONMENTS[key] == environment
            producer.ENVIRONMENTS[key] = environment
    producer.configure()
    model, _cfg, binding = base.foundation.build(values, protocol)
    training = json.loads((original/'training.json').read_text())
    receipt = json.loads((original/'official_metrics.json').read_text())
    assert binding == row['initializer'] == training['initializer']
    assert receipt['schema'] == producer.SCHEMA and receipt['status'] == 'COMPLETE'
    assert receipt['training_epochs'] == 50 and receipt['condition'] == producer.condition(values)
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    selected = max(training['history'], key=lambda r: (r['official_fused']['mAP'], r['epoch']))
    payload = base.foundation.load(original/'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch'] == receipt['selected_epoch'] == selected['epoch']
    assert payload['metrics'] == selected['official_fused'] and receipt['metrics'] == row['metrics']
    assert receipt['checkpoint_sha256'] == row['checkpoint_sha256']
    model.eval()
    before = base.entry.runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    query, gallery = [frozen.extract(model, protocol, split) for split in ('query', 'gallery')]
    after = base.entry.runner._module_state_sha256(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    for features in (query, gallery):
        assert torch.allclose(features['fused'],
            F.normalize(features['shared_global']+gain*features['correction'], dim=1),
            atol=1e-6, rtol=1e-6)
    data = base.metadata(protocol)
    old = torch.load(original/'official_distances.pt', map_location='cpu', weights_only=False)
    assert all(np.array_equal(data[key], old[key]) for key in data)
    output.mkdir(parents=True)
    os.chdir(output.resolve())  # Pinned MSVR author scorer writes re.txt here.
    distances = {name: base.entry.runner.distance_matrix(query[key], gallery[key])
                 for name, key in (('global', 'shared_global'),
                                   ('correction', 'correction'), ('fused', 'fused'))}
    regional_score = region_scores(query['regions'], gallery['regions'])
    global_score = (F.normalize(query['shared_global'].float(), dim=1) @
                    F.normalize(gallery['shared_global'].float(), dim=1).T)
    distances['regions'] = 2-2*regional_score
    # Exact fixed score from the existing auxiliary target; no coefficient fitting.
    distances['training_pairwise'] = 2-2*(0.5*(global_score+regional_score))
    scores = {name: base.score(distance, data, args.dataset, values.signal_source)
              for name, distance in distances.items()}
    difference = float((distances['fused']-old['fused']).abs().max())
    assert difference <= 1e-5
    assert all(abs(scores['fused']['metrics'][n]-v) < 1e-5 for n,v in receipt['metrics'].items())
    geometry = {}
    for split, features in (('query', query), ('gallery', gallery)):
        g, c, f = (features[n] for n in ('shared_global', 'correction', 'fused'))
        units = F.normalize(features['regions'].reshape(len(g), 9, 16, 128), dim=-1)
        cosine = units @ units.transpose(-1, -2)
        off_diagonal = cosine[:, :, ~torch.eye(16, dtype=torch.bool)].mean(-1)
        geometry[split] = dict(
            scaled_correction_global_ratio=base.describe((gain*c).norm(dim=1)/g.norm(dim=1)),
            global_fused_angle_degrees=base.describe(torch.rad2deg(torch.acos(
                (F.normalize(g, dim=1)*f).sum(1).clamp(-1, 1)))),
            mean_off_diagonal_slot_cosine_by_role_modality=off_diagonal.mean(0).reshape(3, 3).tolist())
    comparisons = {name: frozen.damage(scores['global'], score, data['query_ids'])
                   for name,score in scores.items() if name != 'global'}
    torch.save(dict(data, **distances), output/'diagnostic_distances.pt')
    result = dict(schema=SCHEMA, status='COMPLETE', completed_at=frozen.stamp(),
        dataset=args.dataset, comparison=args.comparison, selected_epoch=row['best_epoch'],
        input_checkpoint_sha256=row['checkpoint_sha256'], gain=gain, optimizer_updates=0,
        model_state_before_sha256=before, model_state_after_sha256=after,
        original_fused_distance_max_abs_difference=difference,
        scores=scores, comparisons=comparisons, geometry=geometry,
        distance_artifact=dict(bytes=(output/'diagnostic_distances.pt').stat().st_size,
                              sha256=base.panel.base.sha(output/'diagnostic_distances.pt')),
        elapsed_seconds=time.perf_counter()-started,
        boundary='Current producer schema/constructor/binding; one forward per selected Q/G batch. Five descriptive scores only, including the fixed 0.5*(global cosine+regional MaxSim) auxiliary-target score; zero fitting/newweight/best reselection. Actual final C/T/M, not keys or part GT. Regional MaxSim is pairwise diagnosis, not1536 deployment/teacher/compression proof. Full legal gallery/author scorer; consumed development benchmarks.')
    require_inputs(seal)
    frozen.write(output/'DIAGNOSIS.json', result)
    print(json.dumps(dict(dataset=args.dataset, comparison=args.comparison, status='COMPLETE',
                         metrics={n:s['metrics'] for n,s in scores.items()})), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seal', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dataset', choices=('RGBNT201', 'MSVR310', 'RGBNT100'), required=True)
    parser.add_argument('--comparison', choices=('uniform', 'ambiguity'), required=True)
    args = parser.parse_args()
    args.seal, args.output_dir = args.seal.resolve(), args.output_dir.resolve()
    diagnose(args, json.loads(args.seal.read_text()))