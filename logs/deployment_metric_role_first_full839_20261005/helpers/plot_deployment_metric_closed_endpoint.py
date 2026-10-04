"""Plot accepted full50 evidence and the sealed, matched role control."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--packet', required=True)
parser.add_argument('--control-packet', type=Path, required=True)
args = parser.parse_args()
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
source = base / args.packet
summary_path = source / 'SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['status'] == 'ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED'
row = summary['row']
dataset, variant = row['dataset'], row['variant']
training_path = source / 'received/trained-model' / f'deployment_metric_role_v1_20261005_837_full_{variant}_{dataset}' / 'training.json'
training = json.loads(training_path.read_bytes())
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
seal_path = repo / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
seal = json.loads(seal_path.read_bytes())
control = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
control_remote_path = control['run_dir'] + '/training.json'
control_transport_path = args.control_packet / 'stdout.json'
control_relative_path = control_remote_path.removeprefix('/data/gaob/Re-ID/Trifusion/')
control_text = json.loads(control_transport_path.read_bytes())['files'][control_relative_path]
control_bytes = control_text.encode('utf-8')
assert hashlib.sha256(control_bytes).hexdigest() == seal['artifact_sha256'][control_remote_path]
previous = json.loads(control_bytes)
history, trajectory = training['history'], summary['training_task_and_norm_trajectory']
assert [r['epoch'] for r in history] == [r['epoch'] for r in previous['history']] == [r['epoch'] for r in trajectory] == list(range(1, 51))
assert training['best_epoch'] == row['best_epoch']
assert sum(r['steps'] for r in history) == summary['formal_steps']
output = source / 'curves'
assert not output.exists()
output.mkdir()
rows = [{**r['official_fused'], **t, 'mean_total_loss': r['mean_loss'], 'training_loop_seconds': r['seconds'],
         'matched_control_mAP': old['official_fused']['mAP'], 'matched_control_Rank-1': old['official_fused']['Rank-1']}
        for r, t, old in zip(history, trajectory, previous['history'])]
with (output / 'TRAJECTORY.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
epochs = list(range(1, 51))
for key, label, color, style in (
    ('mAP', 'Deployment-metric role mAP', '#147d92', '-'),
    ('Rank-1', 'Deployment-metric role Rank-1', '#d46728', '-'),
    ('matched_control_mAP', 'Matched raw-role control mAP', '#147d92', ':'),
    ('matched_control_Rank-1', 'Matched raw-role control Rank-1', '#d46728', ':')):
    axes[0].plot(epochs, [r[key] for r in rows], label=label, color=color, linestyle=style)
axes[0].axhline(summary['deltas']['global_only']['control_metrics']['mAP'], color='#444444', linestyle='--', label='Independent global-only best mAP')
axes[0].set_ylabel('Official fused score (%)')
axes[0].legend(ncol=2, frameon=False, fontsize=9)
axes[0].set_title(f'{dataset} {variant}: deployed role metric versus matched control, seed 42')
axes[1].plot(epochs, [r['global_loss'] for r in rows], color='#147d92', label='Global author objective')
axes[1].plot(epochs, [r['fused_role_loss'] for r in rows], color='#d46728', label='Fused role objective')
axes[1].set_ylabel('Mean training objective')
axes[1].legend(frameon=False)
axes[2].plot(epochs, [100 * r['actual_scaled_correction_global_ratio_mean'] for r in rows], color='#845ba6')
axes[2].set_ylabel('Scaled correction / global (%)')
axes[2].set_xlabel('Completed training epoch')
for axis in axes:
    axis.axvline(row['best_epoch'], color='#666666', linestyle='--', linewidth=1)
    axis.grid(axis='y', alpha=.2)
    axis.set_xlim(1, 50)
fig.suptitle(f"All 50 original epochs; vertical line = selected mAP-best epoch {row['best_epoch']}", fontsize=11)
fig.savefig(output / 'FULL50_MATCHED_CONTROL.svg', metadata={'Date': None})
plt.close(fig)
boundary = ('Descriptive saved trajectories, seed42 consumed benchmarks; no model execution, smoothing or epoch reselection. '
            'Control and independent global-only are separately trained models. Head losses combine identity and soft-triplet terms. '
            'Norm ratios are batch means, not causal attribution. Training_loop_seconds excludes subsequent epoch evaluation and saving.')
manifest = dict(status='ACCEPTED_FULL50_MATCHED_CURVES_COMPLETE', dataset=dataset, variant=variant, epochs=50,
    selected_epoch=row['best_epoch'], selected_metrics=row['metrics'], matched_control_best_epoch=control['best_epoch'],
    last_metrics=history[-1]['official_fused'], best_to_last_map_drop=summary['best_to_last_map_drop'],
    inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path, training_path, control_transport_path, seal_path)},
    matched_control_training_sha256=hashlib.sha256(control_bytes).hexdigest(),
    outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}, boundary=boundary)
(output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
(output / 'README.md').write_text('# Accepted full50 matched trajectories\n\n' + boundary + '\n', encoding='utf-8')
print(json.dumps(dict(status=manifest['status'], output=str(output), selected_epoch=row['best_epoch'])))
