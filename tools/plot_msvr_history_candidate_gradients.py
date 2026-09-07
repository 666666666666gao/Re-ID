"""Render complete fixed-state gradient text; no model or optimizer execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(run, analysis, output):
    summary_path = run / 'summary.json'
    cpu_path = run / 'cpu_verification.json'
    aggregate_path = analysis / 'complete_text_reaggregation.json'
    csv_path = analysis / 'all_role_history_steps.csv'
    summary = json.loads(summary_path.read_bytes())
    cpu = json.loads(cpu_path.read_bytes())
    aggregate = json.loads(aggregate_path.read_bytes())
    assert summary['status'] == 'PASS_COMPLETE_FIXED_STATE_PROBE'
    assert cpu['status'] == 'PASS_COMPLETE_FIXED_STATE_PROBE_CPU'
    assert aggregate['status'] == 'PASS_COMPLETE_TEXT_REAGGREGATION'
    assert cpu['summary_sha256'] == aggregate['summary_sha256'] == sha(summary_path)
    mode = summary['mode']
    assert mode in ('preflight', 'source')
    expected_batches = 72 if mode == 'preflight' else 2340
    assert cpu['batches'] == aggregate['batches'] == expected_batches
    states = ('initial', 'control', 'fresh_memory')
    roles = ('cnn', 'transformer', 'mamba')
    keys = [(fold, state) for fold in range(3) for state in states]
    indexed = {(s['fold'], s['state']): s for s in aggregate['states']}
    assert len(aggregate['states']) == len(keys) and set(indexed) == set(keys)
    with csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == aggregate['role_history_rows']
    row_keys = [(int(r['fold']), r['state'], int(r['step']), r['role']) for r in rows]
    assert len(set(row_keys)) == len(row_keys)
    assert {(f, s, r) for f, s, _, r in row_keys} == {(*k, r) for k in keys for r in roles}
    fields = ('history_to_current_ratio', 'current_history_cosine',
              'current_both_cosine', 'task_total_both_cosine')
    points = []
    for fold, state in keys:
        for role in roles:
            own = [r for r in rows if (int(r['fold']), r['state'], r['role']) == (fold, state, role)]
            for field in fields:
                values = [float(r[field]) for r in own if r[field] != '']
                assert all(math.isfinite(v) for v in values)
                mean = statistics.fmean(values) if values else None
                saved = indexed[(fold, state)]['roles'][role][field]
                assert saved['defined'] == len(values) and saved['undefined'] == len(own)-len(values)
                if values:
                    assert math.isclose(mean, saved['mean'], rel_tol=1e-12, abs_tol=1e-12)
                else:
                    assert saved['mean'] is None
                points.append(dict(fold=fold, state=state, role=role, metric=field,
                                   mean=mean, defined=len(values), undefined=len(own)-len(values)))
    point_index = {(p['fold'], p['state'], p['role'], p['metric']): p for p in points}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 180})
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), layout='constrained')
    names = {'initial': 'Initial', 'control': 'Stale endpoint', 'fresh_memory': 'Fresh endpoint'}
    labels = [f'F{f} | {names[s]}' for f, s in keys]
    titles = [r'(a) Historical/current norm: $\|g_V\|/\|g_U\|$',
              r'(b) Current vs historical: $\cos(g_U,g_V)$',
              r'(c) Current vs both: $\cos(g_U,g_U+g_V)$',
              r'(d) Task total: $\cos(G,G+\lambda g_V)$']
    for ax, field, title in zip(axes.flat, fields, titles, strict=True):
        panel = [[point_index[(f, s, r, field)] for r in roles] for f, s in keys]
        values = np.asarray([[np.nan if p['mean'] is None else p['mean'] for p in row] for row in panel])
        cmap = plt.get_cmap('YlGnBu' if field == fields[0] else 'RdBu').copy()
        cmap.set_bad('#ededed')
        low, high = (0, float(np.nanmax(values))) if field == fields[0] else (-1, 1)
        artist = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect='auto')
        for y, row in enumerate(panel):
            for x, point in enumerate(row):
                mean = point['mean']
                if mean is None:
                    text, color = 'Undefined\n(n=0)', '#222222'
                else:
                    rgb = cmap(artist.norm(mean))[:3]
                    color = 'white' if sum(a*b for a, b in zip(rgb, (.299, .587, .114))) < .5 else '#111111'
                    text = f'{mean:.3f}\n(n={point["defined"]})'
                ax.text(x, y, text, ha='center', va='center', fontsize=8, color=color)
        ax.set_xticks(range(3), ['CNN', 'Transformer', 'Mamba'])
        ax.set_yticks(range(9), labels)
        ax.set_title(title, fontsize=11, pad=10)
        ax.tick_params(length=0)
        for boundary in (2.5, 5.5):
            ax.axhline(boundary, color='white', linewidth=1.5)
        fig.colorbar(artist, ax=ax, shrink=.88, pad=.025,
                     label='Mean ratio' if field == fields[0] else 'Mean cosine')
    label = 'SHORT PREFLIGHT' if mode == 'preflight' else 'COMPLETE SOURCE DIAGNOSTIC'
    fig.suptitle('Historical-candidate parameter gradients | ' + label, fontsize=14)
    fig.supxlabel('Fixed weights; same-role parameter blocks; n = defined history batches. No optimizer updates.', fontsize=10)
    output.mkdir()
    stem = 'msvr310_history_candidate_gradients_' + mode
    fig.savefig(output / (stem + '.pdf'))
    fig.savefig(output / (stem + '.png'))
    plt.close(fig)
    caption = (
        f'{label}: all nine fixed model states, {expected_batches} replayed B64 source batches. '
        'Initial means the original role initialization; stale/fresh endpoint labels refer to checkpoints '
        'from the sealed coordinate-update comparison, not to any updates performed in this diagnostic. '
        'g_U is the partial expanded-triplet parameter gradient from the current batch, including its '
        'candidate-peer paths; g_V is the historical-candidate side reconstructed by grouped vector-Jacobian '
        'products at the same current parameters and historical views. G is the recorded gradient of all '
        '14 existing task losses and lambda is the configured fused-triplet weight (1 in this experiment). '
        'Each cell is an arithmetic mean of the displayed batch-wise ratio or cosine on one role encoder '
        'parameter block. Ratios are means of ratios, not ratios of mean norms. Folds remain separate. '
        'n counts defined history batches for that metric; missing cosines/ratios are omitted and never '
        'filled with zero. The JSON receipt retains defined and undefined counts. '
        'Negative cos(g_U,g_V) alone is not evidence of harmful task conflict; these are two partial '
        'derivatives of the same scalar metric objective. These directions are not measured AdamW update '
        'directions and do not establish held-out retrieval gains. The plot checks text arithmetic and '
        'bindings, not independent model backpropagation. The direct-graph proof in the diagnostic is '
        'limited to the first single historical group per state. '
        + ('This short preflight has five history batches per state and is not a complete-source result.'
           if mode == 'preflight' else 'All registered fixed-source replay batches are retained.')
    )
    (output / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    receipt = dict(status='RENDERED_VERIFIED_COMPLETE_GRADIENT_TEXT', mode=mode,
                   summary_sha256=sha(summary_path), cpu_sha256=sha(cpu_path),
                   aggregate_sha256=sha(aggregate_path), csv_sha256=sha(csv_path),
                   plot_script_sha256=sha(Path(__file__)), batches=expected_batches,
                   role_history_rows=len(rows), plotted_cells=len(points), points=points,
                   contributing_metric_values=sum(p['defined'] for p in points),
                   undefined_metric_values=sum(p['undefined'] for p in points),
                   matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                   artifacts={p.name: sha(p) for p in output.iterdir()},
                   scope='Verified text arithmetic and figure rendering only; no new model forward or backward.')
    (output / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('points', 'artifacts')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plot(args.run_dir, args.analysis_dir, args.output_dir)
