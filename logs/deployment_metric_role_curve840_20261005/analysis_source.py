"""Describe every saved epoch of the accepted first pair; no selection or inference."""
from pathlib import Path
import csv
import hashlib
import json

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
packet = base / 'deployment_metric_RGBNT201_semantic_full839'
summary_path = packet / 'SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
training_path = packet / 'received/trained-model/deployment_metric_role_v1_20261005_837_full_semantic_RGBNT201/training.json'
training = json.loads(training_path.read_bytes())
archive = repo / 'logs/deployment_metric_role_first_full839_20261005'
manifest = json.loads((archive / 'MANIFEST.json').read_bytes())
for path in (summary_path, training_path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files']['deployment_metric_RGBNT201_semantic_full839/' + path.relative_to(packet).as_posix()]
seal_path = repo / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal = json.loads(seal_path.read_bytes())
matched = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == ('RGBNT201', 'semantic'))
control_transport_path = base / 'global_task_role_first_full826/stdout.json'
remote_name = matched['run_dir'] + '/training.json'
control_text = json.loads(control_transport_path.read_bytes())['files'][remote_name.removeprefix('/data/gaob/Re-ID/Trifusion/')]
assert hashlib.sha256(control_text.encode('utf-8')).hexdigest() == seal['artifact_sha256'][remote_name]
control = json.loads(control_text)
assert summary['status'] == 'ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED'
assert summary['actual_training_batch_order_equal']
assert [r['epoch'] for r in training['history']] == [r['epoch'] for r in control['history']] == list(range(1, 51))
assert training['best_epoch'] == control['best_epoch'] == summary['row']['best_epoch'] == 8
metrics = ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')
rows = []
for candidate, previous in zip(training['history'], control['history']):
    row = dict(epoch=candidate['epoch'])
    for metric in metrics:
        row['candidate_' + metric] = candidate['official_fused'][metric]
        row['matched_control_' + metric] = previous['official_fused'][metric]
        row['delta_' + metric] = row['candidate_' + metric] - row['matched_control_' + metric]
    rows.append(row)
selected = rows[7]
assert all(selected['delta_' + metric] == summary['deltas']['semantic']['delta_metrics'][metric] for metric in metrics)
output = base / 'deployment_metric_RGBNT201_semantic_epoch_comparison839'
assert not output.exists()
output.mkdir()
with (output / 'ALL50_MATCHED_DELTAS.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
boundary = ('Same two completed seed42 runs and consumed official benchmarks. These 50 epochs are dependent points '
            'on two training trajectories, not 50 independent experiments or seeds. Selected epoch remains original mAP-best8. '
            'No epoch reselection, new query scoring, model execution, bootstrap, p-value or modification to the fixed progression gate. '
            'The original once-only15pair all-query report remains pending until all six endpoints close.')
analysis = dict(status='CLOSED_FIRST_PAIR_ALL50_DESCRIPTIVE_COMPARISON_COMPLETE', dataset='RGBNT201', variant='semantic', epochs=50,
    primary_selected_epoch=8, primary_selected_deltas={m: selected['delta_' + m] for m in metrics}, primary_progress=False,
    counts={m: dict(positive=sum(r['delta_' + m] > 0 for r in rows), zero=sum(r['delta_' + m] == 0 for r in rows),
                   negative=sum(r['delta_' + m] < 0 for r in rows)) for m in metrics},
    mAP_positive_R1_nonnegative_epochs=sum(r['delta_mAP'] > 0 and r['delta_Rank-1'] >= 0 for r in rows),
    last_epoch_deltas={m: rows[-1]['delta_' + m] for m in metrics}, boundary=boundary,
    inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path, training_path, control_transport_path, seal_path)},
    original_control_training_sha256=seal['artifact_sha256'][remote_name])
(output / 'SUMMARY.json').write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf-8')
(output / 'README.md').write_text('# Closed first-pair trajectory comparison\n\n' + boundary + '\n\n'
    '47/50 saved epochs have higher mAP than the matched raw-role control; selected epoch8 has lower Rank-1 and Rank-5. '
    'The intervention changes retrieval behavior, while the fixed primary progression gate remains failed. '
    'All-query repair/damage and identity-level analysis must use the already registered final report, not these dependent epoch counts.\n', encoding='utf-8')
(output / 'MANIFEST.json').write_text(json.dumps(dict(files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=analysis['status'], counts=analysis['counts'], selected_deltas=analysis['primary_selected_deltas'], output=str(output))))
