"""Summarize recorded F1 step evidence after the complete six-endpoint intake."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('C:/Users/gb/.codex_tmp/foundation_recipe_complete_20261002')
OUT = ROOT / 'RECORDED_TRAINING_ACTIVITY.json'
assert not OUT.exists()
intake = json.loads((ROOT / 'INTAKE.json').read_text(encoding='utf-8'))
assert intake['status'] == 'ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE'
rows = []
for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    for recipe in ('author', 'current'):
        relative = f'trained-model/foundation_recipe_20261002_v1_full_{recipe}_{dataset}/training_steps.jsonl'
        data = (ROOT / 'raw' / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == intake['text'][relative]['sha256']
        steps = [json.loads(line) for line in data.splitlines()]
        training_path = relative.replace('training_steps.jsonl', 'training.json')
        training_data = (ROOT / 'raw' / training_path).read_bytes()
        assert hashlib.sha256(training_data).hexdigest() == intake['text'][training_path]['sha256']
        training = json.loads(training_data)
        assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history']) == 50
        epochs = []
        for history in training['history']:
            epoch = history['epoch']
            batch = [step for step in steps if step['epoch'] == epoch]
            assert len(batch) == history['steps'] and [s['batch'] for s in batch] == list(range(len(batch)))
            values = [step['loss'] for step in batch]
            assert all(math.isfinite(value) for value in values)
            assert abs(sum(values) / len(values) - history['mean_loss']) < 1e-10
            rate_count = len(batch[0]['lr'])
            assert all(len(step['lr']) == rate_count for step in batch)
            rates = [batch[0]['lr'][i] for i in range(rate_count)]
            assert all(step['lr'] == rates for step in batch)
            assert all(math.isfinite(value) and value >= 0 for value in rates)
            row = {'epoch': epoch, 'steps': len(batch), 'mean_total_loss': history['mean_loss'],
                   'learning_rates': rates, 'official_metrics': history['official_fused']}
            if recipe == 'current':
                assert all(math.isfinite(s['id']) and math.isfinite(s['triplet']) and s['triplet'] >= 0 for s in batch)
                assert all(abs(s['loss'] - s['id'] - s['triplet']) < 1e-5 for s in batch)
                row.update(mean_id_loss=sum(s['id'] for s in batch) / len(batch),
                           mean_triplet_loss=sum(s['triplet'] for s in batch) / len(batch),
                           nonzero_triplet_steps=sum(s['triplet'] > 0 for s in batch))
            else:
                assert all(all(math.isfinite(value) for value in s['head_losses']) for s in batch)
                assert all(abs(s['loss'] - sum(s['head_losses'])) < 1e-5 for s in batch)
                row['logged_head_count'] = len(batch[0]['head_losses'])
                assert all(len(s['head_losses']) == row['logged_head_count'] for s in batch)
                row['mean_combined_head_losses'] = [sum(s['head_losses'][i] for s in batch) / len(batch)
                                                   for i in range(row['logged_head_count'])]
            epochs.append(row)
        assert len(steps) == sum(row['steps'] for row in epochs)
        row = {'dataset': dataset, 'recipe': recipe, 'step_source': relative,
               'step_source_sha256': intake['text'][relative]['sha256'], 'steps': len(steps), 'epochs': epochs}
        if recipe == 'current':
            row['nonzero_triplet_steps'] = sum(e['nonzero_triplet_steps'] for e in epochs)
        rows.append(row)
result = {'schema': 'trifusion-f1-recorded-training-activity-v1',
          'created_at': datetime.now().astimezone().isoformat(), 'rows': rows,
          'boundaries': ['CPU-only descriptive aggregation of all recorded training steps; no model/scorer execution or optimizer update.',
                         'Author logs record combined loss per head, not separate CE/triplet components; author triplet activity cannot be inferred.',
                         'Different package losses are not comparable scales and do not establish a causal source of retrieval changes.',
                         'Positive current hinge counts describe training-batch activity; they do not measure unseen-identity generalization.',
                         'Does not replace the once-only formal report, independent integrity review, or multi-seed evidence.']}
OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(OUT), 'rows': len(rows), 'steps': sum(r['steps'] for r in rows)}))
