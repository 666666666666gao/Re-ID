"""Report the completed six-end visual-initialization comparison from fixed arrays."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.audit_correspondence_context_identity_losses import audit
from tools.diagnose_patch_memory_readout import analyze

DATASETS = ('RGBNT201', 'RGBNT100', 'MSVR310')
VARIANTS = ('reid_visual', 'public_visual')


def elapsed(start, finish):
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    matrix_path = campaign / 'accepted_matrix.json'
    manifest_path = campaign / 'manifest.json'
    state_path = campaign / 'campaign.json'
    matrix, manifest, state = (json.loads(path.read_text()) for path in (matrix_path, manifest_path, state_path))
    assert matrix['schema'] == 'trifusion-visual-start-verification-v1'
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == 6
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in matrix['rows'])
    assert {(row['dataset'], row['variant']) for row in matrix['rows']} == {(d, v) for d in DATASETS for v in VARIANTS}
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 6
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert manifest['schema'] == 'trifusion-visual-start-panel-v1'
    assert manifest['seed'] == 42 and manifest['epochs'] == 50
    assert 'tools/run_visual_start_roles.py' in manifest['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in manifest['source_sha256'].items())
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    sources = {str(path): sha(path) for path in (matrix_path, manifest_path, state_path)}
    reused_sources = {name: sha(ROOT / 'tools' / name) for name in (
        'analyze_correspondence_distances.py', 'audit_correspondence_context_identity_losses.py',
        'diagnose_patch_memory_readout.py', 'train_msvr310_signal_oof.py', 'train_rgbnt100_signal_oof.py')}
    witness_path = Path(manifest['initialization_witness_path'])
    inputs_path = Path(manifest['inputs_path'])
    assert sha(witness_path) == manifest['initialization_witness_sha256']
    assert sha(inputs_path) == manifest['inputs_sha256']
    witness = json.loads(witness_path.read_text())
    assert witness['status'] == 'MATCHED_TRAINABLE_INITIALIZATION_PASS'
    expected_initializers = {row['dataset']: row['initializers'] for row in witness['rows']}
    sources.update({str(witness_path): sha(witness_path), str(inputs_path): sha(inputs_path)})
    rows, histories = [], {}
    for row in matrix['rows']:
        run = Path(row['run_dir'])
        child_path = Path(row['campaign_dir']) / 'campaign.json'
        child = json.loads(child_path.read_text())
        assert child['status'] == 'COMPLETE' and child['dataset'] == row['dataset'] and child['variant'] == row['variant']
        assert [stage['mode'] for stage in child['jobs']] == ['m0', 'train', 'evaluate']
        assert all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in child['jobs'])
        training = json.loads((run / 'training.json').read_text())
        assert training['initializer']['token_mode'] == 'static'
        assert training['initializer']['visual_start'] == row['variant']
        assert training['initializer']['initial_model_state_sha256'] == row['initial_model_state_sha256']
        assert row['initial_model_state_sha256'] == expected_initializers[row['dataset']][row['variant']]['initial_model_state_sha256']
        losses = audit(run, 'none')
        readout = analyze(row)
        histories[row['dataset'], row['variant']] = losses['history']
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        m0_run = Path(child['jobs'][0]['output_dir'])
        for path in (child_path, run / 'training.json', run / 'training_steps.jsonl', run / 'official_metrics.json',
                     m0_run / 'training.json', m0_run / 'training_steps.jsonl'):
            sources[str(path)] = sha(path)
        cost = {stage['mode'] + '_stage_wall_seconds': elapsed(stage['started_at'], stage['completed_at'])
                for stage in child['jobs']}
        cost.update(cpu_verification_wall_seconds=elapsed(child['jobs'][-1]['completed_at'], child['completed_at']),
                    endpoint_wall_seconds=elapsed(child['started_at'], child['completed_at']),
                    training_step_seconds=sum(item['seconds'] for item in training['history']),
                    boundary='Actual subprocess-stage intervals; step seconds omit epoch evaluation; upstream ReID and queue wait excluded.')
        rows.append({**row, 'readout': readout,
                     'readout_semantics': 'joint_local is normalized total role correction, not independently trained or guaranteed pure-local.',
                     'actual_cost': cost, 'final_metrics': losses['last']['official_fused'],
                     'final_minus_best_metrics_pp': {key: losses['last']['official_fused'][key] - value for key, value in row['metrics'].items()},
                     'best_mean_loss': losses['best']['loss'], 'final_mean_loss': losses['last']['loss'],
                     'after_best_steps': sum(item['epoch'] > row['best_epoch'] for item in steps),
                     'after_best_positive_triplet_steps': sum(item['epoch'] > row['best_epoch'] and item['triplet'] > 0 for item in steps)})
    pairs = []
    for dataset in DATASETS:
        endpoints = [row for row in rows if row['dataset'] == dataset]
        assert len({row['initial_model_state_sha256'] for row in endpoints}) == 2
        assert len({row['trainable_parameters'] for row in endpoints}) == 1
        query_count = endpoints[0]['readout']['query_count']
        for control in ('reid_visual',):
            paired = compare(matrix, dataset, control, 'public_visual')
            paired['boundary'] = ('Matched trainable/nonvisual initial state; different frozen visual tensors; same parameter count, seed42/full50 and one official mAP best. '
                                  'Trained camera/nonvisual state retained; not a wholly public-only system or a new architecture. '
                                  'Official post-selection fixed-model diagnosis; identity bootstrap is not seed variance or untouched-test significance.')
            required = 0.5 if dataset in ('RGBNT201', 'MSVR310') else 0.0
            delta = paired['delta_metrics']
            pairs.append({'dataset': dataset, 'control': control, 'candidate': 'public_visual',
                          'registered_required_map_pp': required,
                          'registered_gate_passed': delta['mAP'] > 0 and delta['mAP'] >= required and delta['Rank-1'] >= 0,
                          'negative_flip_rate_percent_of_all_queries': paired['rank1_new_errors'] * 100 / query_count,
                          'paired_diagnosis': paired})
    assert all(sha(Path(path)) == digest for path, digest in sources.items())
    assert all(sha(ROOT / 'tools' / name) == digest for name, digest in reused_sources.items())
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    args.output_dir.mkdir()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 2, figsize=(10, 8.5), layout='constrained')
    for index, dataset in enumerate(DATASETS):
        for variant, color in zip(VARIANTS, ('#D55E00', '#0072B2')):
            history = histories[dataset, variant]
            selected = next(row for row in rows if row['dataset'] == dataset and row['variant'] == variant)['best_epoch']
            epochs = [item['epoch'] for item in history]
            maps = [item['official_fused']['mAP'] for item in history]
            axes[index, 0].plot(epochs, maps, color=color, label=variant, linewidth=1.4)
            axes[index, 0].scatter(selected, maps[selected - 1], color=color, s=22, zorder=3)
            axes[index, 1].plot(epochs, [item['loss'] for item in history], color=color, linewidth=1.4)
        axes[index, 0].set_title(dataset + ': official mAP; dot = selected best')
        axes[index, 1].set_title(dataset + ': mean ID + Triplet loss')
        axes[index, 0].set_ylabel('mAP (%)')
        axes[index, 1].set_ylabel('Training loss')
        for axis in axes[index]:
            axis.set_xlabel('Role-stage epoch')
            axis.set_xlim(1, 50)
            axis.grid(alpha=0.2)
        axes[index, 0].legend(frameon=False)
    for name in ('trajectories.png', 'trajectories.svg'):
        figure.savefig(args.output_dir / name, dpi=180)
    plt.close(figure)
    gate = 'PASS' if all(pair['registered_gate_passed'] for pair in pairs) else 'FAIL'
    report = {'schema': 'visual-start-six-end-complete-analysis-v1',
              'analyzed_at': datetime.now().astimezone().isoformat(), 'source_sha256': sha(Path(__file__)),
              'accepted': 6, 'rows': rows, 'pairs': pairs, 'registered_advancement_gate': gate,
              'campaign_wall_seconds': elapsed(state['started_at'], state['completed_at']),
              'summed_endpoint_wall_seconds': sum(row['actual_cost']['endpoint_wall_seconds'] for row in rows),
              'source_artifacts_sha256': sources, 'reused_analysis_source_sha256': reused_sources,
              'figure_sha256': {name: sha(args.output_dir / name) for name in ('trajectories.png', 'trajectories.svg')},
              'boundaries': ['All six fresh endpoints and all three registered candidate-control comparisons are required.',
                             'joint_local is the total role correction; no pure-local or independent-control claim.',
                             'Within-checkpoint shared_global also received joint training; not a separately trained global-only control.',
                             'Parameter equality is not equal effective capacity or computation.',
                             'Scalar task support is not optimizer update share or unique cause.',
                             'One seed with official-best development selection cannot establish stability, novelty or SOTA.',
                             'No model forward, optimizer, new inference rule or test-based retuning occurs in this CPU analysis.']}
    (args.output_dir / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Complete frozen visual initialization comparison', '', 'Seed42; full50; one official fused-mAP-best per row.', '',
             '| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Endpoint hours |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda item: (DATASETS.index(item['dataset']), VARIANTS.index(item['variant']))):
        scores = row['metrics']
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {scores['mAP']:.4f} | {scores['Rank-1']:.4f} | {scores['Rank-5']:.4f} | {scores['Rank-10']:.4f} | {row['actual_cost']['endpoint_wall_seconds'] / 3600:.3f} |")
    lines += ['', '| Dataset | ReID visual to public visual | Delta mAP | Delta R1 | Repairs | New errors | Identity macro delta AP | Gate |', '|---|---|---:|---:|---:|---:|---:|---|']
    for pair in pairs:
        item = pair['paired_diagnosis']
        lines.append(f"| {pair['dataset']} | {pair['control']} to public visual | {item['delta_metrics']['mAP']:+.4f} | {item['delta_metrics']['Rank-1']:+.4f} | {item['rank1_repairs']} | {item['rank1_new_errors']} | {item['identity_macro_mean_delta_ap_points']:+.4f} | {'PASS' if pair['registered_gate_passed'] else 'FAIL'} |")
    lines += ['', 'Registered advancement gate: **' + gate + '**. The full three-dataset baseline/SOTA goal is not established by this gate.', '', *['- ' + line for line in report['boundaries']]]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'accepted': 6, 'registered_gate': gate, 'output': str(args.output_dir),
                      'paired_deltas': [{'dataset': pair['dataset'], 'control': pair['control'], 'delta': pair['paired_diagnosis']['delta_metrics']} for pair in pairs]}), flush=True)


if __name__ == '__main__':
    main()