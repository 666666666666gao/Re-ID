"""Render already-accepted saved trajectories; no model or remote access."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deployment_metric_three_saved_trajectory845')
report = json.loads((output / 'ANALYSIS.json').read_bytes())
assert report['status'] == 'THREE_ACCEPTED_SAVED_FULL50_TRAJECTORIES_COMPLETE'
assert not (output / 'SAVED_TRAINING_COMPARISON.svg').exists()
with (output / 'TRAJECTORY.csv').open(encoding='utf-8') as stream:
    data = list(csv.DictReader(stream))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, constrained_layout=True)
for column, result in enumerate(report['rows']):
    rows = [r for r in data if (r['dataset'], r['variant']) == (result['dataset'], result['variant'])]
    assert len(rows) == 50
    epochs = [int(r['epoch']) for r in rows]
    for mode, label, color, style in (
        ('old', 'Matched raw-role control', '#147d92', ':'),
        ('new', 'Deployment-metric role', '#d46728', '-'),
    ):
        axes[0, column].plot(epochs, [float(r[f'{mode}_role_loss']) for r in rows],
                             label=label, color=color, linestyle=style)
        axes[1, column].plot(epochs, [100 * float(r[f'{mode}_scaled_correction_global_ratio']) for r in rows],
                             color=color, linestyle=style)
    axes[0, column].set_title(f"{result['dataset']} {result['variant']}")
    axes[1, column].set_xlabel('Completed training epoch')
    for row in range(2):
        axes[row, column].axvline(result['selected_epoch'], color='#666666', linestyle='--', linewidth=.8)
        axes[row, column].set_xlim(1, 50)
        axes[row, column].grid(axis='y', alpha=.2)
axes[0, 0].set_ylabel('Combined role training objective')
axes[1, 0].set_ylabel('Scaled correction / global (%)')
axes[0, 0].legend(frameon=False)
fig.suptitle('Saved full50 comparisons, seed42; objective scales changed and are not directly comparable', fontsize=11)
fig.savefig(output / 'SAVED_TRAINING_COMPARISON.svg', metadata={'Date': None})
plt.close(fig)

text = '# Three accepted full50 trajectory pairs\n\n'
text += 'This local saved-text analysis covers 150 completed epochs and 6,004 formal optimizer steps. It runs no neural model and does not inspect the current training process.\n\n'
text += '| Dataset / variant | Selected epoch | Selected ΔmAP / ΔR1 | Scaled correction/global at selected epoch (old → new) | Epochs with ΔmAP > 0 | Epochs with ΔR1 < 0 |\n'
text += '|---|---:|---:|---:|---:|---:|\n'
for result in report['rows']:
    selected = result['selected_epoch_record']
    counts = result['epoch_delta_counts']
    text += (f"| {result['dataset']} / {result['variant']} | {result['selected_epoch']} | "
        f"{selected['delta_mAP']:+.6f} / {selected['delta_Rank-1']:+.6f} | "
        f"{100 * selected['old_scaled_correction_global_ratio']:.4f}% → "
        f"{100 * selected['new_scaled_correction_global_ratio']:.4f}% | "
        f"{counts['mAP']['positive']}/50 | {counts['Rank-1']['negative']}/50 |\n")
text += '\nAll 50 recorded epoch means of the global objective and shared-global feature norm match their respective raw-role control exactly for all three endpoints. This supports the intended isolation at the recorded-scalar level; it does not establish bitwise equality of every model tensor.\n\n'
text += ('The two RGBNT201 endpoints show mAP improvement across most saved epochs, while Rank-5 and Rank-10 '
    'decrease in most epochs. MSVR310 semantic has lower mAP in 49/50 epochs. These are descriptive training '
    'trajectories; dependent epochs do not replace independent training seeds. Each registered selected best '
    'and the original mAP +0.5 / Rank-1 non-decrease gate remain unchanged; all three selected pairs fail that gate.\n\n')
text += ('Changing the role metric changes its training loss scale and geometry. The displayed role objective combines '
    'identity and Triplet terms; absolute old/new loss values cannot alone show worse fitting, an inactive Triplet, '
    'or a gradient conflict. Larger correction norms are observed behavior, not a demonstrated cause of the CMC changes.\n\n')
text += report['boundary'] + '\n'
(output / 'README.md').write_text(text, encoding='utf-8')
manifest = dict(status='THREE_SAVED_PAIR_VISUALIZATION_COMPLETE',
    inputs={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ('ANALYSIS.json', 'TRAJECTORY.csv')},
    outputs={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ('README.md', 'SAVED_TRAINING_COMPARISON.svg')},
    boundary=report['boundary'])
(output / 'VISUALIZATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(text)
