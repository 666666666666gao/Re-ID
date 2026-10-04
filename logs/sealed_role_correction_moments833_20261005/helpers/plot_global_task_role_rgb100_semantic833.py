"""Plot original accepted semantic trajectories; no remote or model execution."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
source = base / 'global_task_role_rgb100_semantic_full829'
summary_path = source / 'SUMMARY.json'
training_path = source / 'received/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT100/training.json'
summary = json.loads(summary_path.read_bytes())
training = json.loads(training_path.read_bytes())
output = base / 'global_task_role_rgb100_semantic_curve833'
assert not output.exists()
history = training['history']
trajectory = summary['training_task_and_norm_trajectory']
assert [r['epoch'] for r in history] == [r['epoch'] for r in trajectory] == list(range(1, 51))
assert sum(r['steps'] for r in history) == summary['formal_steps'] == 3129
best = summary['row']['best_epoch']
assert best == training['best_epoch'] == 5
assert all(abs(history[best-1]['official_fused'][k] - v) <= 1e-5 for k, v in summary['row']['metrics'].items())
output.mkdir()
rows = [{**r['official_fused'], **t, 'mean_total_loss': r['mean_loss'],
         'training_loop_seconds': r['seconds']} for r, t in zip(history, trajectory)]
with (output / 'TRAJECTORY.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
epochs = [r['epoch'] for r in rows]
for metric, color in (('mAP', '#147d92'), ('Rank-1', '#d46728'), ('Rank-5', '#845ba6'), ('Rank-10', '#47794d')):
    axes[0].plot(epochs, [r[metric] for r in rows], label=metric, color=color, linewidth=1.5)
global_map = summary['pairs']['previous_global_only']['metrics']['previous_global_only']['mAP']
axes[0].axhline(global_map, color='#444444', linewidth=1, linestyle=':', label='Independent global-only best mAP')
axes[0].set_ylabel('Official fused score (%)')
axes[0].set_ylim(20, 100)
axes[0].legend(ncol=3, frameon=False, loc='lower right')
axes[0].set_title('RGBNT100 semantic: global-task / role-head ownership control, seed 42')
axes[1].plot(epochs, [r['mean_global_loss'] for r in rows], color='#147d92', label='Global author objective')
axes[1].plot(epochs, [r['mean_fused_role_loss'] for r in rows], color='#d46728', linestyle='--', label='Fused role author objective')
axes[1].set_yscale('log')
axes[1].set_ylabel('Mean training objective (log scale)')
axes[1].legend(frameon=False)
axes[2].plot(epochs, [100*r['actual_scaled_correction_global_ratio_mean'] for r in rows], color='#845ba6')
axes[2].set_ylabel('Mean scaled correction / global (%)')
axes[2].set_xlabel('Completed training epoch')
for axis in axes:
    axis.axvline(best, color='#666666', linestyle='--', linewidth=1)
    axis.grid(axis='y', alpha=.2)
    axis.set_xlim(1, 50)
fig.suptitle('All 50 original epochs; dashed vertical line = selected mAP-best epoch 5', fontsize=11)
fig.savefig(output / 'SEMANTIC_50_EPOCHS.svg', metadata={'Date': None})
plt.close(fig)
boundary = ('Descriptive curves from one completed seed42 experiment on consumed official benchmarks. '
            'No smoothing, reranking, epoch reselection or model execution. The independent global-only reference '
            'is a separately trained control, not this model\'s global trajectory. '
            'Head objectives combine author identity/soft-triplet terms; they do not show individual loss activity. '
            'Norm ratios are training-batch mean per-sample ratios, not a causal contribution measure.')
analysis = dict(status='ORIGINAL50_TRAJECTORY_ARTIFACT_COMPLETE', dataset='RGBNT100', variant='semantic',
    epochs=50, formal_steps=3129, selected_epoch=best, selected_metrics=summary['row']['metrics'],
    last_metrics=history[-1]['official_fused'], best_to_last_map_drop=summary['best_to_last_map_drop'],
    inputs={str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path, training_path)},
    source_checkpoint_sha256=summary['row']['checkpoint_sha256'], boundary=boundary,
    outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
(output / 'MANIFEST.json').write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf-8')
(output / 'README.md').write_text('# Original RGBNT100 semantic training curves\n\n' + boundary + '\n\n'
    'The selected checkpoint remains epoch 5. Training is complete and first strict evaluation is accepted. '
    'The 1.8082-point best-to-final mAP drop accompanies lower identity objectives; the curves do not identify a unique cause.\n', encoding='utf-8')
print(json.dumps(dict(status=analysis['status'], output=str(output), epochs=50, selected_epoch=best)))
