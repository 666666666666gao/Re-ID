"""Compare already accepted saved histories; no SSH or neural computation."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
output = base / 'semantic_capacity_first_two_saved_trajectory857'
assert not output.exists()
datasets = {
    'RGBNT201': ('semantic_capacity_first_full_intake856', {
        'native': ('global_task_role_native_full827', 'logs/global_task_role_rgb201_pair827_20261004/global_task_role_native_curve827/TRAJECTORY.csv'),
        'semantic': ('global_task_role_first_full826', 'logs/global_task_role_rgb201_pair827_20261004/global_task_role_semantic_curve827/TRAJECTORY.csv')}),
    'MSVR310': ('semantic_capacity_msvr_full_intake856', {
        'native': ('global_task_role_msvr_native_full828', 'logs/global_task_role_curve_intake829_20261004/msvr_pair_curves/NATIVE_TRAJECTORY.csv'),
        'semantic': ('global_task_role_msvr_semantic_full828', 'logs/global_task_role_curve_intake829_20261004/msvr_pair_curves/SEMANTIC_TRAJECTORY.csv')})}
inputs, paired, contrasts = {}, [], []

def bind(path):
    body = path.read_bytes()
    inputs[str(path)] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    return body

for dataset, (packet, controls) in datasets.items():
    received = base / packet
    record = json.loads(bind(received / 'stdout.json'))
    run = f'trained-model/semantic_capacity_control_v1_20261005_856_full_native_{dataset}'
    data = {}
    for name in ('training.json', 'training_steps.jsonl'):
        relative = run + '/' + name
        body = bind(received / 'received' / relative)
        info = record['files'][relative]
        assert len(body) == info['bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']
        data[name] = body
    training = json.loads(data['training.json'])
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert len(training['history']) == 50
    steps = [json.loads(line) for line in data['training_steps.jsonl'].splitlines()]
    capacity = []
    for epoch in training['history']:
        selected = [s for s in steps if s['epoch'] == epoch['epoch']]
        assert len(selected) == epoch['steps']
        capacity.append(dict(epoch=epoch['epoch'], steps=epoch['steps'], **epoch['official_fused'],
            mean_global_loss=statistics.fmean(s['global_loss'] for s in selected),
            shared_global_norm_mean=statistics.fmean(s['shared_global_norm_mean'] for s in selected),
            mean_fused_role_loss=statistics.fmean(s['fused_role_loss'] for s in selected),
            actual_scaled_correction_global_ratio_mean=statistics.fmean(s['actual_scaled_correction_global_ratio_mean'] for s in selected)))
    for variant, (old_packet, csv_relative) in controls.items():
        old_record = json.loads(bind(base / old_packet / 'stdout.json'))
        relative = f'trained-model/global_task_role_v1_20261004_824_full_{variant}_{dataset}/training.json'
        old_body = bind(base / old_packet / 'received' / relative)
        captured_body = old_record['files'][relative].encode('utf-8')
        # These legacy downloads were written through Windows text-mode I/O.
        assert old_body.replace(b'\r\n', b'\n') == captured_body
        inputs[str(base / old_packet / 'received' / relative)]['captured_utf8_sha256'] = hashlib.sha256(captured_body).hexdigest()
        inputs[str(base / old_packet / 'received' / relative)]['known_local_crlf_translation'] = True
        old_training = json.loads(old_body)
        assert old_training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
        assert len(old_training['history']) == 50
        csv_body = bind(repo / csv_relative)
        assert csv_body == subprocess.check_output(['git', 'show', '60e1c4dec1f21a772cde7c703cb29ad2b5ee61bd:' + csv_relative], cwd=repo)
        old_curve = list(csv.DictReader(csv_body.decode('utf-8').splitlines()))
        assert len(old_curve) == 50
        differences = []
        for candidate, control, old_epoch in zip(capacity, old_curve, old_training['history']):
            assert candidate['epoch'] == int(control['epoch']) == old_epoch['epoch']
            assert candidate['steps'] == int(control['steps']) == old_epoch['steps']
            assert all(float(control[k]) == old_epoch['official_fused'][k] for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
            row = dict(dataset=dataset, control='raw_' + variant, epoch=candidate['epoch'])
            for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'mean_global_loss', 'shared_global_norm_mean',
                        'mean_fused_role_loss', 'actual_scaled_correction_global_ratio_mean'):
                row[key + '_capacity'] = candidate[key]
                row[key + '_control'] = float(control[key])
                row['delta_' + key] = candidate[key] - float(control[key])
            differences.append(row)
            paired.append(row)
        result = dict(dataset=dataset, control='raw_' + variant,
            global_loss_epoch_mean_max_abs_difference=max(abs(r['delta_mean_global_loss']) for r in differences),
            global_norm_epoch_mean_max_abs_difference=max(abs(r['delta_shared_global_norm_mean']) for r in differences),
            capacity_best_epoch=training['best_epoch'], control_best_epoch=old_training['best_epoch'])
        for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'):
            values = [r['delta_' + key] for r in differences]
            result[key] = dict(higher=sum(v > 0 for v in values), equal=sum(v == 0 for v in values),
                               lower=sum(v < 0 for v in values), minimum=min(values), maximum=max(values),
                               mean=statistics.fmean(values))
        for name, index in (('capacity_best', training['best_epoch'] - 1), ('last', 49)):
            result[name] = {k: differences[index][k] for k in (
                'epoch', 'actual_scaled_correction_global_ratio_mean_capacity',
                'actual_scaled_correction_global_ratio_mean_control', 'mean_fused_role_loss_capacity',
                'mean_fused_role_loss_control')}
        contrasts.append(result)

assert len(paired) == 200 and len(contrasts) == 4
output.mkdir()
with (output / 'PAIRED_50EPOCH_TRAJECTORIES.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
summary = dict(status='FIRST_TWO_ACCEPTED_SAVED_TEXT_TRAJECTORIES_ANALYZED',
    at=datetime.now().astimezone().isoformat(), datasets=2, underlying_model_histories=6,
    underlying_epoch_observations=300, paired_epoch_rows=200, contrasts=contrasts, inputs=inputs,
    boundary='Only accepted original 50-epoch histories and published raw control CSV; no SSH, model execution, new scoring, checkpoint selection or intervention. Epochs are dependent, not seeds. Matching global scalar epoch means do not prove identical model state or query features. No RGBNT100/complete-report conclusion or causal/SOTA assertion.')
(output / 'SUMMARY.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in summary.items() if k != 'inputs'}))
