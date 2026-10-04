"""Compare all original MSVR epochs from accepted text receipts; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
output = base / 'global_task_role_msvr_pair_curve'
assert not output.exists()
datasets = {}
inputs = {}
for variant in ('semantic', 'native'):
    packet = base / f'global_task_role_msvr_{variant}_full828'
    summary_path = packet / 'SUMMARY.json'
    training_path = packet / f'received/trained-model/global_task_role_v1_20261004_824_full_{variant}_MSVR310/training.json'
    summary = json.loads(summary_path.read_bytes())
    training = json.loads(training_path.read_bytes())
    history = training['history']
    trajectory = summary['training_task_and_norm_trajectory']
    assert [r['epoch'] for r in history] == [r['epoch'] for r in trajectory] == list(range(1, 51))
    assert sum(r['steps'] for r in history) == summary['formal_steps'] == 706
    assert summary['row']['best_epoch'] == training['best_epoch'] == 38
    assert all(abs(history[37]['official_fused'][k] - v) <= 1e-5 for k, v in summary['row']['metrics'].items())
    rows = [{**r['official_fused'], **t, 'mean_total_loss': r['mean_loss'],
             'training_loop_seconds': r['seconds']} for r, t in zip(history, trajectory)]
    datasets[variant] = dict(summary=summary, rows=rows)
    inputs.update({str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path, training_path)})
global_equal = all(a['mean_global_loss'] == b['mean_global_loss'] for a, b in zip(datasets['semantic']['rows'], datasets['native']['rows']))
global_norm_equal = all(a['shared_global_norm_mean'] == b['shared_global_norm_mean'] for a, b in zip(datasets['semantic']['rows'], datasets['native']['rows']))
output.mkdir()
for variant, data in datasets.items():
    rows = data['rows']
    with (output / f'{variant.upper()}_TRAJECTORY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
colors = {'semantic': '#147d92', 'native': '#d46728'}
for variant, data in datasets.items():
    rows = data['rows']
    epochs = [r['epoch'] for r in rows]
    for axis, metric in ((axes[0, 0], 'mAP'), (axes[0, 1], 'Rank-1')):
        axis.plot(epochs, [r[metric] for r in rows], color=colors[variant], label=variant, linewidth=1.7)
        axis.scatter([38], [rows[37][metric]], color=colors[variant], s=22, zorder=4)
    axes[1, 0].plot(epochs, [r['mean_fused_role_loss'] for r in rows], color=colors[variant], linestyle='--', label=f'{variant} fused objective')
    axes[1, 1].plot(epochs, [100*r['actual_scaled_correction_global_ratio_mean'] for r in rows], color=colors[variant], label=variant)
    if variant == 'semantic':
        axes[1, 0].plot(epochs, [r['mean_global_loss'] for r in rows], color='#444444', label='Global objective (both logs equal)')
        ref = data['summary']['pairs']['previous_global_only']['metrics']['previous_global_only']
        for axis, metric in ((axes[0, 0], 'mAP'), (axes[0, 1], 'Rank-1')):
            axis.axhline(ref[metric], color='#444444', linestyle=':', label='Independent global-only best')
axes[0, 0].set_ylabel('Official fused mAP (%)')
axes[0, 1].set_ylabel('Official fused Rank-1 (%)')
axes[1, 0].set_ylabel('Mean author objective (log scale)')
axes[1, 0].set_yscale('log')
axes[1, 1].set_ylabel('Mean scaled correction / global (%)')
for axis in axes.flat:
    axis.axvline(38, color='#666666', linestyle='--', linewidth=1)
    axis.grid(axis='y', alpha=.2)
    axis.set_xlim(1, 50)
    axis.set_xlabel('Completed training epoch')
    axis.legend(frameon=False, fontsize=8)
fig.suptitle('MSVR310 task ownership control, seed 42: all 50 original epochs\nBoth selected at epoch 38; native minus semantic = +0.0101 mAP', fontsize=12)
svg = output / 'MSVR310_PAIR_50_EPOCHS.svg'
fig.savefig(svg, metadata={'Date': None})
preview = base / 'global_task_role_msvr_pair_curve_preview.png'
assert not preview.exists()
fig.savefig(preview, dpi=130)
plt.close(fig)
ET.parse(svg)
boundary = ('Original completed seed42 results on consumed official benchmarks; no smoothing, reranking, epoch reselection, new inference or optimizer updates. '
            'Independent global-only is separately trained, not the same model global trajectory. Author objectives combine classification and soft-triplet terms; their separate activity is not logged. '
            'Equal mean global loss/norm logs are aggregate facts, not proof of identical full model states or per-query global features. Norm ratios are not causal contribution estimates.')
analysis = dict(status='ORIGINAL_MSVR_PAIR50_CURVES_COMPLETE', epochs=50, formal_steps_per_endpoint=706,
    inputs=inputs, global_loss_all50_exact_equal=global_equal, shared_global_norm_all50_exact_equal=global_norm_equal,
    endpoints={v:dict(selected_epoch=d['summary']['row']['best_epoch'], selected_metrics=d['summary']['row']['metrics'],
        last_metrics=d['rows'][-1], best_to_last_map_drop=d['summary']['best_to_last_map_drop'],
        checkpoint_sha256=d['summary']['row']['checkpoint_sha256']) for v,d in datasets.items()},
    boundary=boundary, outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
    private_preview=str(preview))
(output / 'MANIFEST.json').write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf-8')
(output / 'README.md').write_text('# MSVR310 original full training pair\n\n' + boundary + '\n\n'
    'Both original best epochs are 38. Best-to-final mAP declines are 0.060976 for semantic and 0.067221 for native. '
    'These curves show modest late declines; they do not support a large late-collapse claim for this pair.\n', encoding='utf-8')
print(json.dumps(dict(status=analysis['status'], output=str(output), preview=str(preview),
    selected_epoch=38, global50_equal=global_equal, global_norm50_equal=global_norm_equal)))
