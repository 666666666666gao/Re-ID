from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_closed_six_training_traces')
assert not target.exists()
archives = {
    ('RGBNT201', 'global_only'): 'native_research_first_full800_20261003',
    ('RGBNT201', 'semantic'): 'native_research_semantic801_20261004',
    ('RGBNT201', 'native'): 'native_research_first_dataset802_20261004',
    ('MSVR310', 'global_only'): 'native_research_msvr_global803_20261004',
    ('MSVR310', 'semantic'): 'native_research_msvr_semantic804_20261004',
    ('MSVR310', 'native'): 'native_research_second_dataset805_20261004',
}
rows = []
for (dataset, variant), archive in archives.items():
    prefix = f'native_research_v6_20261003_794_full_{variant}_{dataset}'
    paths = {name: repo / 'logs' / archive / f'{prefix}_{name}' for name in
             ('training.json', 'training_steps.jsonl', 'official_metrics.json')}
    bindings = {}
    for name, path in paths.items():
        data = path.read_bytes()
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(repo).as_posix()], cwd=repo)
        assert data == committed
        bindings[name] = hashlib.sha256(data).hexdigest()
    training = json.loads(paths['training.json'].read_bytes())
    official = json.loads(paths['official_metrics.json'].read_bytes())
    steps = [json.loads(line) for line in paths['training_steps.jsonl'].read_bytes().splitlines()]
    assert training['status'] == official['status'] == 'COMPLETE'
    assert [row['epoch'] for row in training['history']] == list(range(1, 51))
    assert len(steps) == sum(row['steps'] for row in training['history'])
    epochs = []
    for epoch in training['history']:
        selected = [row for row in steps if row['epoch'] == epoch['epoch']]
        assert len(selected) == epoch['steps']
        assert all(math.isfinite(row[key]) for row in selected for key in
                   ('loss', 'raw_feature_norm_mean', 'deployment_feature_norm_mean'))
        epochs.append({
            'epoch': epoch['epoch'], 'steps': len(selected),
            'mean_training_loss': epoch['mean_loss'],
            'mean_batch_raw_feature_norm': statistics.mean(row['raw_feature_norm_mean'] for row in selected),
            'deployment_norm_range': [min(row['deployment_feature_norm_mean'] for row in selected),
                                      max(row['deployment_feature_norm_mean'] for row in selected)],
            'official_fused': epoch['official_fused'],
        })
    best = max(epochs, key=lambda row: row['official_fused']['mAP'])
    assert best['official_fused'] == official['metrics']
    rows.append({'dataset': dataset, 'variant': variant, 'source_text_sha256': bindings,
                 'epochs': epochs, 'first_epoch': epochs[0], 'best_epoch': best,
                 'last_epoch': epochs[-1],
                 'best_to_last_mAP_drop': best['official_fused']['mAP'] - epochs[-1]['official_fused']['mAP']})
report = {
    'at': datetime.now().astimezone().isoformat(),
    'publication_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'status': 'PUBLISHED_SIX_ENDPOINT_TEXT_TRACES_ANALYZED', 'rows': rows,
    'boundary': 'Only completed RGBNT201/MSVR310 endpoints, all50 epochs and existing step logs. Raw norm is fused h norm, not detail-only evidence amplitude. No CE/Triplet separation or detail attention statistics were logged here. No causal mechanism, three-dataset conclusion, training-seed robustness, new formal score, or recipe change. Original9-arm CPU report remains pending.'
}
target.mkdir()
(target / 'ANALYSIS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'at': report['at'], 'head': report['publication_head'], 'rows': [
    {'dataset': row['dataset'], 'variant': row['variant'],
     'first_loss': row['first_epoch']['mean_training_loss'],
     'last_loss': row['last_epoch']['mean_training_loss'],
     'first_raw_norm': row['first_epoch']['mean_batch_raw_feature_norm'],
     'last_raw_norm': row['last_epoch']['mean_batch_raw_feature_norm'],
     'best_epoch': row['best_epoch']['epoch'], 'best_to_last_mAP_drop': row['best_to_last_mAP_drop']}
    for row in rows], 'boundary': report['boundary']}))
