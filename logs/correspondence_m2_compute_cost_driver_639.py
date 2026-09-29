from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('/data/gaob/Re-ID/Trifusion')
matrix_path = root / 'logs/correspondence_refinement_accepted_m2_complete_639_20260929.json'
matrix = json.loads(matrix_path.read_text())
assert matrix['verified_complete'] == matrix['expected_endpoints'] == 18
rows = [r for r in matrix['rows'] if r['phase'] == 'm2']
assert len(rows) == 15 and all(r['status'] == 'VERIFIED_COMPLETE' for r in rows)
variants = ('single_pooled', 'query_pooled', 'single_regions', 'query_regions', 'uniform_pooled')
datasets = ('RGBNT201', 'RGBNT100', 'MSVR310')
assert {(r['dataset'], r['variant']) for r in rows} == {(d, v) for d in datasets for v in variants}
panels = {}
for dataset in datasets:
    cells = {r['variant']: r for r in rows if r['dataset'] == dataset}
    panels[dataset] = {
        'sum_training_and_epoch_eval_seconds': sum(r['training_and_epoch_eval_seconds'] for r in cells.values()),
        'rows': [{k: cells[v][k] for k in ('variant', 'best_epoch', 'metrics', 'trainable_parameters',
                                         'training_and_epoch_eval_seconds', 'checkpoint_sha256',
                                         'distance_sha256', 'receipt_sha256')} for v in variants],
    }
report = {
    'status': 'ALL_FIFTEEN_M2_ENDPOINT_RECORDED_COSTS_BOUND',
    'at': datetime.now().astimezone().isoformat(),
    'matrix_sha256': hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'complete_endpoints': 15,
    'sum_training_and_epoch_eval_seconds': sum(r['training_and_epoch_eval_seconds'] for r in rows),
    'datasets': panels,
    'boundary': 'Sum of each run elapsed training and per-epoch official evaluation; concurrent load differs. '
                'Not project elapsed time, GPU-active hours, isolated latency or FLOPs. '
                'Excludes M0, final reload, queue, CPU diagnosis and upstream baseline training. '
                'Separate from the previous 27-endpoint ledger. No new training or selection.',
}
path = root / 'logs/correspondence_m2_compute_cost_639_20260929.json'
archive = root / 'logs/correspondence_m2_compute_cost_driver_639.py'
assert not path.exists() and not archive.exists()
path.write_text(json.dumps(report, indent=2) + '\n')
archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status': report['status'], 'sum_seconds': report['sum_training_and_epoch_eval_seconds'],
                  'datasets_seconds': {d: p['sum_training_and_epoch_eval_seconds'] for d, p in panels.items()},
                  'artifacts': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (path, archive)}}))
