"""Compare three accepted saved full50 pairs; no model or remote access."""
import csv
import hashlib
import json
import math
from pathlib import Path

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
output = base / 'deployment_metric_three_saved_trajectory845'
assert not output.exists()
seal_path = repo / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
scope_path = repo / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
seal = json.loads(seal_path.read_bytes())
scope = json.loads(scope_path.read_bytes())
inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (seal_path, scope_path)}
records, conclusions = [], []
for dataset, variant, packet_name, control_packet_name in (
    ('RGBNT201', 'semantic', 'deployment_metric_RGBNT201_semantic_full839', 'global_task_role_first_full826'),
    ('RGBNT201', 'native', 'deployment_metric_RGBNT201_native_full841', 'global_task_role_native_full827'),
    ('MSVR310', 'semantic', 'deployment_metric_MSVR310_semantic_full842', 'global_task_role_msvr_semantic_full828'),
):
    summary_path = base / packet_name / 'SUMMARY.json'
    summary = json.loads(summary_path.read_bytes())
    assert summary['status'] == 'ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED'
    assert summary['row']['dataset'] == dataset and summary['row']['variant'] == variant
    training_path = base / packet_name / 'received/trained-model' / f'deployment_metric_role_v1_20261005_837_full_{variant}_{dataset}' / 'training.json'
    training = json.loads(training_path.read_bytes())
    control = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
    transport_path = base / control_packet_name / 'stdout.json'
    transport = json.loads(transport_path.read_bytes())
    original_name = control['run_dir'] + '/training.json'
    original_raw = transport['files'][original_name.removeprefix('/data/gaob/Re-ID/Trifusion/')].encode('utf-8')
    assert hashlib.sha256(original_raw).hexdigest() == seal['artifact_sha256'][original_name]
    original = json.loads(original_raw)
    old_trajectory = transport['summary']['training_task_and_norm_trajectory']
    new_trajectory = summary['training_task_and_norm_trajectory']
    for values in (training['history'], original['history'], old_trajectory, new_trajectory):
        assert [r['epoch'] for r in values] == list(range(1, 51))
    assert training['best_epoch'] == summary['row']['best_epoch'] == original['best_epoch'] == control['best_epoch']
    rows = []
    for new_history, old_history, new, old in zip(training['history'], original['history'], new_trajectory, old_trajectory):
        assert new['steps'] == old['steps'] == new_history['steps'] == old_history['steps']
        row = dict(dataset=dataset, variant=variant, epoch=new['epoch'], steps=new['steps'],
            new_global_loss=new['global_loss'], old_global_loss=old['mean_global_loss'],
            new_global_norm=new['shared_global_norm_mean'], old_global_norm=old['shared_global_norm_mean'],
            new_role_loss=new['fused_role_loss'], old_role_loss=old['mean_fused_role_loss'],
            new_scaled_correction_global_ratio=new['actual_scaled_correction_global_ratio_mean'],
            old_scaled_correction_global_ratio=old['actual_scaled_correction_global_ratio_mean'],
            new_role_metric_norm=new['role_metric_norm_mean'])
        for metric in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'):
            row[f'new_{metric}'] = new_history['official_fused'][metric]
            row[f'old_{metric}'] = old_history['official_fused'][metric]
            row[f'delta_{metric}'] = row[f'new_{metric}'] - row[f'old_{metric}']
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
        rows.append(row)
    inputs.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path, training_path, transport_path)})
    records.extend(rows)
    best = rows[training['best_epoch'] - 1]
    count = {metric: dict(positive=sum(r[f'delta_{metric}'] > 0 for r in rows),
                        equal=sum(r[f'delta_{metric}'] == 0 for r in rows),
                        negative=sum(r[f'delta_{metric}'] < 0 for r in rows))
             for metric in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')}
    conclusions.append(dict(dataset=dataset, variant=variant, epochs=50,
        steps=sum(r['steps'] for r in rows), selected_epoch=training['best_epoch'],
        global_loss_max_abs_difference=max(abs(r['new_global_loss'] - r['old_global_loss']) for r in rows),
        global_norm_max_abs_difference=max(abs(r['new_global_norm'] - r['old_global_norm']) for r in rows),
        role_metric_norm_mean_range=[min(r['new_role_metric_norm'] for r in rows), max(r['new_role_metric_norm'] for r in rows)],
        selected_epoch_record=best, last_epoch_record=rows[-1], epoch_delta_counts=count,
        epoch_map_positive_and_rank1_nonnegative=sum(r['delta_mAP'] > 0 and r['delta_Rank-1'] >= 0 for r in rows),
        registered_progress=summary['deltas'][variant]['phase_progress'],
        matched_control_training_sha256=hashlib.sha256(original_raw).hexdigest()))

assert all(r['global_loss_max_abs_difference'] == r['global_norm_max_abs_difference'] == 0 for r in conclusions)
assert sum(r['epochs'] for r in conclusions) == 150
boundary = ('Three accepted original full50/first-strict pairs only. This is a saved-text descriptive analysis, '
    'not a new model run, full six-endpoint report, best reselection or training-seed uncertainty. '
    'Epoch counts are dependent observations. Global loss and norm equality are recorded scalar equality, '
    'not a claim of all model tensors being bitwise equal. Role losses combine classification and Triplet '
    'and change metric geometry; their absolute sizes do not directly measure relative fitting quality. '
    'Correction/global ratios are means over training batches, not query ratios or causal proof. '
    'MSVR310 native failed its original M0 and has no formal result; RGBNT100 is not included. '
    'All registered mAP-best and progress gates remain unchanged.')
report = dict(status='THREE_ACCEPTED_SAVED_FULL50_TRAJECTORIES_COMPLETE', epochs=150,
              formal_steps=sum(r['steps'] for r in records), rows=conclusions, inputs=inputs, boundary=boundary)
output.mkdir()
with (output / 'TRAJECTORY.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(output / 'ANALYSIS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], epochs=report['epochs'], formal_steps=report['formal_steps'],
    rows=[{key: row[key] for key in ('dataset', 'variant', 'selected_epoch', 'global_loss_max_abs_difference',
        'global_norm_max_abs_difference', 'epoch_delta_counts', 'epoch_map_positive_and_rank1_nonnegative', 'registered_progress')}
          for row in conclusions]), ensure_ascii=False))
