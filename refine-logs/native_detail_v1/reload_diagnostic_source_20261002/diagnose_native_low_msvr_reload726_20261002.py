from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools.train_msvr310_signal_oof import scene_scores
torch.set_num_threads(1)
campaign = ROOT / 'logs/native_detail_20261002_v1'
manifest = json.loads((campaign / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 238
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in manifest['source_sha256'].items())
run = ROOT / 'trained-model/native_detail_20261002_v1_clean_clip_low_MSVR310_seed42_full'
training = json.loads((run / 'training.json').read_text())
assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert [row['epoch'] for row in training['history']] == list(range(1, 51))
best = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
payload = torch.load(run / 'best_map.pth', map_location='cpu', weights_only=True)
assert payload['epoch'] == best['epoch'] == training['best_epoch']
assert payload['metrics'] == best['official_fused']
data = torch.load(run / 'official_distances.pt', map_location='cpu', weights_only=False)
assert torch.isfinite(data['fused']).all()
protocol_path = ROOT / 'logs/official_three_dataset_protocols_20260923/MSVR310.json'
protocol = json.loads(protocol_path.read_text())
for split in ('query', 'gallery'):
    for field, key in (('identity', 'ids'), ('camera', 'cameras'), ('scene', 'scenes')):
        assert np.array_equal(data[f'{split}_{key}'], np.asarray([row[field] for row in protocol['records'][split]]))
independent = scene_scores(data['fused'].numpy(), data['query_ids'], data['gallery_ids'], data['query_scenes'], data['gallery_scenes'])['metrics']
author_path = ROOT / 'comparators/Signal-cd1b0a6/utils/metrics.py'
spec = importlib.util.spec_from_file_location('native_reload_original_author_metrics', author_path)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
cmc, mean_ap = author.eval_func_msrv(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
    data['query_cameras'], data['gallery_cameras'], data['query_scenes'], data['gallery_scenes'])
actual = {'mAP': 100 * float(mean_ap), 'Rank-1': 100 * float(cmc[0]),
          'Rank-5': 100 * float(cmc[4]), 'Rank-10': 100 * float(cmc[9])}
assert all(abs(actual[name] - independent[name]) < 1e-5 for name in actual)
delta = {name: actual[name] - payload['metrics'][name] for name in actual}
assert any(abs(value) >= 1e-5 for value in delta.values())
result = {'status': 'ORIGINAL_RELOAD_ASSERTION_REPRODUCED_FROM_SAVED_CPU_DISTANCES',
    'observed_at': datetime.now().astimezone().isoformat(), 'dataset': 'MSVR310', 'variant': 'low',
    'source_count': 238, 'sources_unchanged': True, 'selected_epoch': payload['epoch'],
    'stored_training_best': payload['metrics'], 'saved_reload_distances_author_metrics': actual,
    'independent_scorer_metrics': independent, 'reload_minus_training': delta,
    'original_tolerance': 1e-5, 'original_assertion_passed': False,
    'distance_sha256': hashlib.sha256((run / 'official_distances.pt').read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256((run / 'best_map.pth').read_bytes()).hexdigest(),
    'boundary': 'CPU rescoring of original failed-evaluation distances; no new neural forward, optimizer, model change, official acceptance, threshold change or training retry.'}
output = ROOT / 'logs/native_detail_reload_diag726_20261002'
assert not output.exists()
output.mkdir()
(output / 'CPU.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
