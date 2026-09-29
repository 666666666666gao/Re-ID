"""Read accepted complete-gallery arrays; no model or new evaluation forward."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.train_msvr310_signal_oof import scene_scores

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
matrix_path = root / '.git/context_identity_accepted_654_20260929.json'
matrix = json.loads(matrix_path.read_text())
row = next(r for r in matrix['rows'] if r['dataset'] == 'MSVR310' and r['variant'] == 'static_none')
assert row['status'] == 'VERIFIED_COMPLETE'
run = Path(row['run_dir'])
assert sha(run / 'official_metrics.json') == row['receipt_sha256']
assert sha(run / 'official_distances.pt') == row['distance_sha256']
data = torch.load(run / 'official_distances.pt', map_location='cpu', weights_only=False)
scores = {}
for name in ('fused', 'shared_global', 'joint_local'):
    scores[name] = scene_scores(data[name].numpy(), data['query_ids'], data['gallery_ids'],
                               data['query_scenes'], data['gallery_scenes'])
    metrics = row['metrics'] if name == 'fused' else row['diagnostic_metrics'][name]
    assert all(abs(scores[name]['metrics'][key] - metrics[key]) < 1e-5 for key in metrics)
first, second = scores['shared_global'], scores['fused']
old_rank, new_rank = (np.asarray(item['first_match_rank']) for item in (first, second))
delta_ap = np.asarray(second['average_precision']) - np.asarray(first['average_precision'])
repairs = int(((old_rank != 1) & (new_rank == 1)).sum())
new_errors = int(((old_rank == 1) & (new_rank != 1)).sum())
delta_metrics = {name: second['metrics'][name] - first['metrics'][name] for name in first['metrics']}
assert abs(delta_metrics['Rank-1'] - (repairs - new_errors) * 100 / len(delta_ap)) < 1e-5
identities = [{'identity': int(identity), 'queries': int((data['query_ids'] == identity).sum()),
               'mean_delta_ap_points': float(delta_ap[data['query_ids'] == identity].mean() * 100)}
              for identity in np.unique(data['query_ids'])]
changes = np.asarray([item['mean_delta_ap_points'] for item in identities])
report = {'status': 'SAME_BEST_FULL_GALLERY_READOUT_DIAGNOSIS_COMPLETE',
          'at': datetime.now().astimezone().isoformat(), 'source_sha256': sha(Path(__file__)),
          'matrix_sha256': sha(matrix_path), 'dataset': 'MSVR310', 'variant': 'static_none',
          'checkpoint_sha256': row['checkpoint_sha256'], 'distance_sha256': row['distance_sha256'],
          'receipt_sha256': row['receipt_sha256'], 'best_epoch': row['best_epoch'],
          'metrics': {name: value['metrics'] for name, value in scores.items()},
          'global_to_fused_delta_pp': delta_metrics, 'rank1_repairs': repairs,
          'rank1_new_errors': new_errors, 'query_ap_improved': int((delta_ap > 1e-8).sum()),
          'query_ap_worsened': int((delta_ap < -1e-8).sum()),
          'identity_ap_improved': int((changes > 1e-6).sum()),
          'identity_ap_worsened': int((changes < -1e-6).sum()),
          'identity_macro_mean_delta_ap_points': float(changes.mean()), 'identity_changes': identities,
          'boundary': 'Three outputs of one accepted fused-mAP-best checkpoint, not independently '
                      'trained role baselines. Post-selection explanation only; no new forward, '
                      'reweighting, configuration selection or training. Local-only quality alone '
                      'does not determine complementarity. MSVR scene is a time-period label.'}
target = root / '.git/context_identity_readout_MSVR310_static_none_654_20260929.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'identity_changes'}))
