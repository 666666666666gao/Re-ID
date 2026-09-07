"""Plot complete, verified freshness diagnostics; never reads image/model arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'source'], required=True)
    args = parser.parse_args()
    data_path = args.input_dir / 'local_epoch_age_reaggregation.json'
    data = json.loads(data_path.read_bytes())
    assert data['status'] == 'PASS_ALL_REGISTERED_EPOCH_AGE_TEXT_REAGGREGATION'
    assert data['mode'] == args.mode
    expected_epochs = 1 if args.mode == 'preflight' else 20
    assert data['total_steps'] == (72 if args.mode == 'preflight' else 1560)
    cpu_path = args.input_dir / args.mode / 'cpu_verification.json'
    assert hashlib.sha256(cpu_path.read_bytes()).hexdigest() == data['cpu_sha256']
    ends = ['control', 'instance_memory']
    roles = ['cnn', 'transformer', 'mamba']
    epochs = sorted(data['epochs'], key=lambda r: (r['fold'], r['endpoint'], r['epoch']))
    role_rows = data['all_role_epochs']
    expected = {(f, e, ep) for f in range(3) for e in ends for ep in range(1, expected_epochs + 1)}
    assert len(epochs) == len(expected)
    assert {(r['fold'], r['endpoint'], r['epoch']) for r in epochs} == expected
    assert len(role_rows) == 3 * len(expected)
    assert {(r['fold'], r['endpoint'], r['epoch'], r['role']) for r in role_rows} == {
        (*key, role) for key in expected for role in roles}
    role_index = {(r['fold'], r['endpoint'], r['epoch'], r['role']): r for r in role_rows}
    points = []
    for row in epochs:
        count = row['history_steps']
        assert row['anchor_exposures'] == 64 * count
        assert row['fresh_minus_stale_loss']['count'] == count
        point = {k: row[k] for k in ['fold', 'endpoint', 'epoch', 'history_steps', 'anchor_exposures']}
        point['distance_error'] = row['distance_error_pair_weighted_mean']
        point['loss_change'] = row['fresh_minus_stale_loss'].get('mean')
        point['negative_changes_percent'] = 100 * row['negative_winner_changes'] / row['anchor_exposures'] if count else None
        for role in roles:
            gradient = role_index[(row['fold'], row['endpoint'], row['epoch'], role)]
            assert gradient['fresh_cosine']['count'] + gradient['undefined_cosine'] == count
            point[role + '_cosine'] = gradient['fresh_cosine'].get('mean')
            point[role + '_defined_count'] = gradient['fresh_cosine']['count']
            point[role + '_undefined_count'] = gradient['undefined_cosine']
        points.append(point)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), layout='constrained')
    keys = ['distance_error', 'loss_change', 'negative_changes_percent'] + [r + '_cosine' for r in roles]
    labels = ['Mean absolute\ndistance error', 'Mean hinge change\n(fresh − stale)',
              'Hard-negative choice\nchanged (%)', 'CNN mean\ngradient cosine',
              'Transformer mean\ngradient cosine', 'Mamba mean\ngradient cosine']
    colors = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']
    handles = []
    legend_labels = []
    for panel, (ax, key, label) in enumerate(zip(axes.flat, keys, labels, strict=True)):
        for fold in range(3):
            for end_i, end in enumerate(ends):
                series = [p for p in points if p['fold'] == fold and p['endpoint'] == end]
                y = [np.nan if p[key] is None else p[key] for p in series]
                if args.mode == 'preflight':
                    x = [fold + (-.07 if end_i == 0 else .07)]
                    color, marker = colors[end_i], ['o', 's'][end_i]
                    style = 'None'
                else:
                    x = [p['epoch'] for p in series]
                    color, marker = colors[fold], markers[fold]
                    style = '-' if end_i == 0 else '--'
                line, = ax.plot(x, y, linestyle=style, color=color, marker=marker,
                                markerfacecolor=color if end_i == 0 else 'white',
                                markersize=4, linewidth=1.1)
                if panel == 0 and (args.mode == 'source' or fold == 0):
                    handles.append(line)
                    end_label = 'Control' if end_i == 0 else 'Stale-memory updates'
                    legend_labels.append((f'Fold {fold}: ' if args.mode == 'source' else '') + end_label)
        ax.set_ylabel(label)
        ax.text(-.20, 1.03, f'({chr(97 + panel)})', transform=ax.transAxes, fontweight='bold')
        if args.mode == 'preflight':
            ax.set_xlabel('Fold')
            ax.set_xticks(range(3))
            ax.set_xlim(-.35, 2.35)
        else:
            ax.set_xlabel('Source training epoch')
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlim(.5, 20.5)
        if panel in [0, 2]:
            ax.set_ylim(bottom=0)
        elif panel == 1:
            ax.axhline(0, color='.55', linewidth=.6, zorder=0)
        else:
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color='.75', linewidth=.6, zorder=0)
    fig.legend(handles, legend_labels, loc='outside upper center',
               ncol=2, frameon=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'msvr310_freshness_' + args.mode
    for extension in ['pdf', 'png']:
        fig.savefig(args.output_dir / (stem + '.' + extension))
    plt.close(fig)
    counts = {f'fold_{f}_{e}': sum(p['history_steps'] for p in points if (p['fold'], p['endpoint']) == (f, e))
              for f in range(3) for e in ends}
    undefined = {r: sum(p[r + '_undefined_count'] for p in points) for r in roles}
    scope = ('Complete short preflight only: six endpoints, 12 updates per endpoint, '
             'two warmup updates, one first-cache-fill update, and nine updates with historical candidates; full learning rate. '
             'This is a rendering and diagnostic check, not a full-source training trend.' if args.mode == 'preflight' else
             'Complete source diagnostic: six endpoints, 260 updates in 20 epochs per endpoint, '
             '65 warmup updates; every registered epoch is retained.')
    caption = (scope + ' Control and stale-memory labels identify the actual training update rule; '
               'within each trajectory the same anchors, historical records, augmented views and current parameters '
               'are compared using cached versus freshly re-encoded historical coordinates. Fresh-coordinate '
               'losses never update the model. The compared metric term uses within-batch candidates for '
               'Control updates and cached expanded candidates for stale-memory updates; other training losses '
               'are unchanged. (a) Pair-weighted absolute difference of Euclidean distances '
               'between unit-normalized fused features for anchor--historical-candidate pairs only '
               '(64 times the historical candidate exposures). (b) Mean fresh minus stale batch-hard hinge loss. '
               '(c) Changed hard-negative selections divided by 64 times the number of updates with history. '
               '(d–f) Update-wise mean cosine between runtime-measured diagnostic gradients of the stale '
               'and fresh expanded losses on the same role '
               'encoder parameter block. Current peers retain gradients; historical candidates are detached '
               'in both comparisons. Gradient witnesses are runtime measurements, not independent CPU '
               'gradient recomputations. No-history metrics and undefined cosines are omitted, never set to zero. '
               'Folds are shown individually, without confidence intervals or a pooled-fold mean. '
               'These are source diagnostics, not held-out retrieval results. '
               'Historical update counts (control / stale-memory) for folds 0, 1, 2: '
               + '; '.join(f"{counts[f'fold_{f}_control']} / {counts[f'fold_{f}_instance_memory']}" for f in range(3))
               + '. Undefined gradient cosine counts (CNN / Transformer / Mamba): '
               + ' / '.join(str(undefined[r]) for r in roles) + '.')
    (args.output_dir / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    tex_caption = caption.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    tex = ('\\begin{figure*}[t]\n\\centering\n'
           f'\\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n'
           f'\\caption{{{tex_caption}}}\n\\label{{fig:{stem}}}\n\\end{{figure*}}\n')
    (args.output_dir / 'latex_includes.tex').write_text(tex, encoding='utf-8')
    receipt = dict(status='RENDERED_COMPLETE_VERIFIED_DIAGNOSTIC_DATA', mode=args.mode,
                   input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                   cpu_sha256=data['cpu_sha256'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   total_updates=data['total_steps'], historical_updates=counts,
                   undefined_cosines=undefined, plotted_points=points,
                   artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir()
                              if p.name in [stem + '.pdf', stem + '.png', 'caption.txt', 'latex_includes.tex']})
    (args.output_dir / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'plotted_points'}))


if __name__ == '__main__':
    main()
