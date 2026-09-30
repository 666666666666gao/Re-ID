"""Summarize bound six-end receipts and complete diagnostics, without inference."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_correspondence_context_identity_losses import audit

MATRIX = ROOT / 'logs/patch_memory_roles_recovery_20261001/accepted_matrix.json'
OUTPUT = ROOT / 'results/patch_memory_complete_20261001'
DATASETS = ('RGBNT201', 'RGBNT100', 'MSVR310')
MODES = ('local_memory', 'full_memory')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    matrix = json.loads(MATRIX.read_text())
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == 6
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    sources = {str(MATRIX.relative_to(ROOT)): sha(MATRIX)}
    rows, histories, pairs, slots = [], {}, [], []
    for row in matrix['rows']:
        run = Path(row['run_dir'])
        assert row['status'] == 'VERIFIED_COMPLETE'
        assert sha(run / 'official_metrics.json') == row['receipt_sha256']
        losses = audit(run, 'none')
        histories[(row['dataset'], row['variant'])] = losses['history']
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            sources[str((run / name).relative_to(ROOT))] = sha(run / name)
        post_best = [item for item in losses['history'] if item['epoch'] > row['best_epoch']]
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        rows.append({**row, 'final_metrics': losses['last']['official_fused'],
                     'final_minus_best_map_pp': losses['last']['official_fused']['mAP'] - row['metrics']['mAP'],
                     'final_minus_best_rank1_pp': losses['last']['official_fused']['Rank-1'] - row['metrics']['Rank-1'],
                     'best_mean_loss': losses['best']['loss'], 'final_mean_loss': losses['last']['loss'],
                     'after_best_steps': sum(item['steps'] for item in post_best),
                     'after_best_positive_triplet_steps': sum(step['triplet'] > 0 for step in steps
                                                               if step['epoch'] > row['best_epoch'])})
    for dataset in DATASETS:
        local, full = (next(row for row in rows if row['dataset'] == dataset and row['variant'] == mode)
                       for mode in MODES)
        assert local['initial_model_state_sha256'] == full['initial_model_state_sha256']
        assert local['trainable_parameters'] == full['trainable_parameters']
        deltas = {key: full['metrics'][key] - local['metrics'][key] for key in local['metrics']}
        required_map = 0.5 if dataset in ('RGBNT201', 'MSVR310') else 0.0
        passes = deltas['mAP'] > 0 and deltas['mAP'] >= required_map and deltas['Rank-1'] >= 0
        path = (ROOT / 'logs/patch_memory_100_diagnosis_intake_20261001/cpu/RGBNT100.json'
                if dataset == 'RGBNT100' else ROOT / f'logs/patch_memory_pair_diagnosis_20261001/{dataset}.json')
        diagnosis = json.loads(path.read_text())
        assert all(abs(diagnosis['delta_metrics'][key] - deltas[key]) < 1e-5 for key in deltas)
        sources[str(path.relative_to(ROOT))] = sha(path)
        pairs.append({'dataset': dataset, 'delta_metrics': deltas, 'registered_gate_passed': passes,
                      'required_positive_map_pp': required_map, 'paired_diagnosis': diagnosis})
        for mode in MODES:
            path = (ROOT / f'logs/patch_memory_100_diagnosis_intake_20261001/slots/{mode}.json'
                    if dataset == 'RGBNT100' else ROOT / f'logs/patch_memory_slot_diagnosis_20261001/{dataset}/{mode}.json')
            measure = json.loads(path.read_text())
            assert measure['model_state_unchanged']
            assert max(measure['full_gallery_metric_difference_pp'].values()) < 1e-5
            sources[str(path.relative_to(ROOT))] = sha(path)
            means = np.asarray(measure['statistics']['query']['role_modality_mean']).mean(axis=(0, 1))
            slots.append({'dataset': dataset, 'mode': mode,
                          'query_rows': measure['statistics']['query']['rows'],
                          'gallery_rows': measure['statistics']['gallery']['rows'],
                          'query_role_modality_mean': dict(zip(measure['statistic_names'], means.tolist()))})
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 2, figsize=(9.5, 8.5), layout='constrained')
    for index, dataset in enumerate(DATASETS):
        for mode, color in zip(MODES, ('#D55E00', '#0072B2')):
            history = histories[(dataset, mode)]
            selected = next(row for row in rows if row['dataset'] == dataset and row['variant'] == mode)['best_epoch']
            epochs = [item['epoch'] for item in history]
            maps = [item['official_fused']['mAP'] for item in history]
            axes[index, 0].plot(epochs, maps, color=color, label=mode.replace('_memory', ''), linewidth=1.5)
            axes[index, 0].scatter(selected, maps[selected - 1], color=color, s=22, zorder=3)
            axes[index, 1].plot(epochs, [item['loss'] for item in history], color=color, linewidth=1.5)
        axes[index, 0].set_title(dataset + ': official mAP (dot = selected best)')
        axes[index, 1].set_title(dataset + ': mean training loss')
        axes[index, 0].set_ylabel('mAP (%)')
        axes[index, 1].set_ylabel('ID + Triplet')
        for axis in axes[index]:
            axis.set_xlabel('Role-stage epoch')
            axis.set_xlim(1, 50)
            axis.grid(alpha=0.2)
        axes[index, 0].legend(frameon=False)
    figure.savefig(OUTPUT / 'trajectories.png', dpi=180)
    figure.savefig(OUTPUT / 'trajectories.svg')
    plt.close(figure)
    report = {'schema': 'patch-memory-six-end-complete-analysis-v1',
              'analyzed_at': datetime.now().astimezone().isoformat(), 'source_sha256': sha(Path(__file__)),
              'accepted': 6, 'rows': rows, 'pairs': pairs, 'slot_statistics': slots,
              'registered_advancement_gate': 'PASS' if all(pair['registered_gate_passed'] for pair in pairs) else 'FAIL',
              'source_artifacts_sha256': sources,
              'figure_sha256': {name: sha(OUTPUT / name) for name in ('trajectories.png', 'trajectories.svg')},
              'boundaries': ['One development seed, official data used for epoch selection.',
                             'Within-checkpoint global/local outputs are not separately trained controls.',
                             'Sampled-content similarity is not physical correspondence or unique causal proof.',
                             'Task scalar activity is not optimizer update share.',
                             'Prior failed hardware campaign remains preserved; no failed receipt is rewritten.']}
    (OUTPUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'accepted': 6, 'gate': report['registered_advancement_gate'],
                      'paired_deltas': [{'dataset': pair['dataset'], 'metrics': pair['delta_metrics']}
                                        for pair in pairs], 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
