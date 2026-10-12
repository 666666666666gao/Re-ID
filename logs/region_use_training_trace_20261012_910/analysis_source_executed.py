"""Describe saved training-source relation counts and correction energy."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


work = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903')
intake = work / 'results908'
manifest = json.loads((intake / 'INTAKE_MANIFEST_908.json').read_text())['files']
out = work / 'training_trace910'
out.mkdir()
inputs = {}
results = []
for dataset in ('RGBNT201', 'MSVR310'):
    for condition in ('uniform', 'ambiguity'):
        run = f'trained-model/region_use_v1_20261012_906_full_{condition}_{dataset}'
        objects = {}
        for filename in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            path = intake / run / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == manifest[f'{run}/{filename}']
            inputs[str(path)] = digest
            objects[filename] = [json.loads(line) for line in path.read_text().splitlines()] if filename.endswith('.jsonl') else json.loads(path.read_text())
        history = objects['training.json']['history']
        receipt = objects['official_metrics.json']
        assert receipt['status'] == 'COMPLETE' and receipt['training_epochs'] == 50
        assert [r['epoch'] for r in history] == list(range(1, 51))
        selected = receipt['selected_epoch']
        assert selected == objects['training.json']['best_epoch']
        steps = objects['training_steps.jsonl']
        epochs = []
        for old in history:
            rows = [r for r in steps if r['epoch'] == old['epoch']]
            assert len(rows) == old['steps']
            legal = sum(r['region_use_support']['legal_triplets'] for r in rows)
            wrong = sum(r['region_use_support']['global_wrong_triplets'] for r in rows)
            assert 0 <= wrong <= legal and legal > 0
            scalar_keys = ('region_use_loss', 'actual_scaled_correction_global_ratio_mean', 'global_loss', 'fused_role_loss')
            means = {name: sum(r[name] for r in rows) / len(rows) for name in scalar_keys}
            assert all(math.isfinite(v) for v in means.values())
            epochs.append({
                'epoch': old['epoch'], 'recorded_steps': len(rows),
                'legal_triplet_exposures': legal,
                'global_nonpositive_margin_triplet_exposures': wrong,
                'fraction_global_nonpositive_margin_among_legal_exposures': wrong / legal,
                'recorded_batch_mean_averages': means,
                'official_fused': old['official_fused'],
            })
        total_legal = sum(e['legal_triplet_exposures'] for e in epochs)
        total_wrong = sum(e['global_nonpositive_margin_triplet_exposures'] for e in epochs)
        results.append({
            'dataset': dataset, 'condition': condition,
            'selected_epoch': selected,
            'total_legal_triplet_exposures': total_legal,
            'total_global_nonpositive_margin_triplet_exposures': total_wrong,
            'epochs': epochs,
            'first_selected_last': {name: epochs[epoch - 1] for name, epoch in [('first', 1), ('selected', selected), ('last', 50)]},
        })

value = {
    'status': 'FOUR_ACCEPTED_ENDPOINT_TRAINING_TRACE_DESCRIPTION_COMPLETE',
    'created_at': datetime.now().astimezone().isoformat(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs': inputs,
    'definitions': {
        'global_nonpositive_margin': 'Logged legal training relations where detached global cosine positive-minus-negative margin is <=0.',
        'correction_ratio': 'Mean per training batch of norm(raw_features-shared_global)/norm(shared_global); report unweighted mean of recorded batch means within an epoch.',
        'counts': 'Repeated training triplet/query exposures; not unique identities, independent trials or query/gallery retrieval samples.',
    },
    'scope': 'Descriptive saved source-training logs of four completed seed42 endpoints/two datasets only. Not same-checkpoint query/gallery geometry, regional standalone retrieval, exact global-margin distribution, training ambiguity weights, a cause of failure, or a justification for gain/seed rescue. No model/tensor/optimizer/SSH/GPU/production invocation. Official benchmark was consumed in development.',
    'results': results,
}
(out / 'SUMMARY.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': value['status'], 'rows': [{'dataset': r['dataset'], 'condition': r['condition'], 'first_selected_last': {k: {'epoch': e['epoch'], 'nonpositive_margin_fraction': e['fraction_global_nonpositive_margin_among_legal_exposures'], 'auxiliary_loss': e['recorded_batch_mean_averages']['region_use_loss'], 'scaled_correction_global_ratio': e['recorded_batch_mean_averages']['actual_scaled_correction_global_ratio_mean']} for k, e in r['first_selected_last'].items()}} for r in results]}))
