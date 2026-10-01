"""Analyze all twelve fixed visual/readout controls from saved real-GT distances."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.collect_visual_update_control import CONDITIONS, DATASETS


def elapsed(start, finish):
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def gate(pair, dataset):
    delta = pair['delta_metrics']
    minimum = 0.5 if dataset in ('RGBNT201', 'MSVR310') else 0.0
    return delta['mAP'] > 0 and delta['mAP'] >= minimum and delta['Rank-1'] >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    files = [campaign / name for name in ('accepted_matrix.json', 'manifest.json', 'campaign.json')]
    matrix, manifest, state = [json.loads(path.read_text()) for path in files]
    assert matrix['schema'] == 'trifusion-visual-update-verification-v1'
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == 12
    assert {(row['dataset'], row['variant']) for row in matrix['rows']} == {
        (dataset, variant) for dataset in DATASETS for variant in CONDITIONS}
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in matrix['rows'])
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 12
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert manifest['schema'] == 'trifusion-visual-update-panel-v1'
    assert manifest['seed'] == 42 and manifest['epochs'] == 50
    assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
    for name in ('initialization_witness', 'preflight'):
        path = Path(manifest[name + '_path'])
        assert sha(path) == manifest[name + '_sha256']
        files.append(path)
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    sources = {str(path): sha(path) for path in files}
    reused = {name: sha(ROOT / 'tools' / name) for name in (
        'analyze_correspondence_distances.py', 'collect_visual_update_control.py',
        'train_msvr310_signal_oof.py', 'train_rgbnt100_signal_oof.py')}
    rows, histories, output_paths = [], {}, set()
    for row in matrix['rows']:
        run = Path(row['run_dir'])
        child_path = Path(row['campaign_dir']) / 'campaign.json'
        child = json.loads(child_path.read_text())
        assert child['status'] == 'COMPLETE' and child['dataset'] == row['dataset']
        assert child['variant'] == row['variant']
        assert [stage['mode'] for stage in child['jobs']] == ['m0', 'train', 'evaluate']
        assert all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in child['jobs'])
        training = json.loads((run / 'training.json').read_text())
        assert [item['epoch'] for item in training['history']] == list(range(1, 51))
        assert training['initializer']['common_initializer_sha256'] == row['common_initializer_sha256']
        assert training['initializer']['initial_model_state_sha256'] == row['initial_model_state_sha256']
        assert training['frozen_signal_state_unchanged']
        assert training['visual_parameters_changed'] == (CONDITIONS[row['variant']][0] == 'low_lr')
        best = max(training['history'], key=lambda item: (item['official_fused']['mAP'], item['epoch']))
        assert best['epoch'] == row['best_epoch']
        assert all(abs(best['official_fused'][key] - value) < 1e-5 for key, value in row['metrics'].items())
        for filename, key in (('best_map.pth', 'checkpoint_sha256'),
                              ('official_distances.pt', 'distance_sha256'),
                              ('official_metrics.json', 'receipt_sha256')):
            assert sha(run / filename) == row[key]
        m0_run = Path(child['jobs'][0]['output_dir'])
        output_paths.update((run, m0_run))
        for path in (child_path, run / 'training.json', run / 'training_steps.jsonl',
                     run / 'official_metrics.json', m0_run / 'training.json', m0_run / 'training_steps.jsonl'):
            sources[str(path)] = sha(path)
        histories[row['dataset'], row['variant']] = training['history']
        cost = {stage['mode'] + '_parent_observed_wall_seconds': elapsed(stage['started_at'], stage['completed_at'])
                for stage in child['jobs']}
        cost.update(endpoint_wall_seconds=elapsed(child['started_at'], child['completed_at']),
                    training_step_seconds=sum(item['seconds'] for item in training['history']),
                    m0_origin=child['jobs'][0].get('origin', 'worker'),
                    boundary='Two M0s reused from preflight; their intervals are parent-observed and precede formal worker start. '
                             'Endpoint intervals exclude those preflight stages; all timing excludes upstream ReID training. '
                             'Training-step seconds omit epoch evaluation; equal epochs do not imply equal GPU computation.')
        rows.append({**row, 'actual_cost': cost, 'best_mean_loss': best['mean_loss'],
                     'final_mean_loss': training['history'][-1]['mean_loss'],
                     'final_minus_best_metrics_pp': {key: row['final_metrics'][key] - value
                                                    for key, value in row['metrics'].items()}})
    visual, roles = [], []
    for dataset in DATASETS:
        bindings = [row['common_initializer_sha256'] for row in rows if row['dataset'] == dataset]
        assert all(binding == bindings[0] for binding in bindings)
        for readout in ('global_only', 'roles'):
            pair = compare(matrix, dataset, 'frozen_' + readout, 'low_lr_' + readout)
            visual.append({'dataset': dataset, 'readout': readout, 'gate_passed': gate(pair, dataset),
                           'paired_diagnosis': pair})
        for update in ('frozen', 'low_lr'):
            pair = compare(matrix, dataset, update + '_global_only', update + '_roles')
            roles.append({'dataset': dataset, 'visual_update': update,
                          'registered_gate_condition': update == 'low_lr',
                          'gate_passed': gate(pair, dataset), 'paired_diagnosis': pair})
    interactions = []
    for dataset in DATASETS:
        effects = {item['readout']: item['paired_diagnosis']['delta_metrics']
                   for item in visual if item['dataset'] == dataset}
        interactions.append({'dataset': dataset, 'roles_minus_global_visual_update_effect_pp': {
            key: effects['roles'][key] - value for key, value in effects['global_only'].items()}})
    visual_pass = all(item['gate_passed'] for item in visual)
    role_pass = all(item['gate_passed'] for item in roles if item['registered_gate_condition'])
    storage = {'observed_at': datetime.now().astimezone().isoformat(),
               'output_directory_logical_bytes': sum(path.stat().st_size for folder in output_paths
                                                      for path in folder.rglob('*') if path.is_file()),
               'output_directories': sorted(str(path) for path in output_paths),
               'free_disk_bytes': shutil.disk_usage(ROOT).free,
               'boundary': 'Current logical bytes of the12 M0 and12 full output directories; excludes inputs and report. '
                           'Not historical peak storage, allocated filesystem blocks or total project storage.'}
    assert all(sha(Path(path)) == digest for path, digest in sources.items())
    assert all(sha(ROOT / 'tools' / name) == digest for name, digest in reused.items())
    assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
    args.output_dir.mkdir()
    figure, axes = plt.subplots(3, 2, figsize=(11, 8.5), layout='constrained')
    colors = ('#0072B2', '#009E73', '#D55E00', '#CC79A7')
    for index, dataset in enumerate(DATASETS):
        for variant, color in zip(CONDITIONS, colors):
            history = histories[dataset, variant]
            epochs = [item['epoch'] for item in history]
            maps = [item['official_fused']['mAP'] for item in history]
            selected = next(row['best_epoch'] for row in rows
                            if row['dataset'] == dataset and row['variant'] == variant)
            axes[index, 0].plot(epochs, maps, color=color, label=variant, linewidth=1.3)
            axes[index, 0].scatter(selected, maps[selected - 1], color=color, s=20)
            axes[index, 1].plot(epochs, [item['mean_loss'] for item in history], color=color, linewidth=1.3)
        axes[index, 0].set_title(dataset + ': official mAP; dot = selected best')
        axes[index, 1].set_title(dataset + ': mean ID + Triplet loss')
        axes[index, 0].set_ylabel('mAP (%)')
        axes[index, 1].set_ylabel('Training loss')
        axes[index, 0].legend(frameon=False, fontsize=7)
        for axis in axes[index]:
            axis.set_xlabel('Adaptation-stage epoch')
            axis.set_xlim(1, 50)
            axis.grid(alpha=0.2)
    for name in ('trajectories.png', 'trajectories.svg'):
        figure.savefig(args.output_dir / name, dpi=180)
    plt.close(figure)
    report = {'schema': 'visual-update-twelve-end-complete-analysis-v1',
              'analyzed_at': datetime.now().astimezone().isoformat(), 'source_sha256': sha(Path(__file__)),
              'accepted': 12, 'rows': rows, 'visual_update_comparisons': visual,
              'independent_role_comparisons': roles, 'interactions': interactions,
              'registered_visual_update_gate': 'PASS' if visual_pass else 'FAIL',
              'registered_role_gate': 'PASS' if role_pass else 'FAIL',
              'registered_joint_gate': 'PASS' if visual_pass and role_pass else 'FAIL',
              'campaign_wall_seconds': elapsed(state['started_at'], state['completed_at']),
              'summed_endpoint_wall_seconds': sum(row['actual_cost']['endpoint_wall_seconds'] for row in rows),
              'disk_observation': storage, 'source_artifacts_sha256': sources,
              'reused_analysis_source_sha256': reused,
              'figure_sha256': {name: sha(args.output_dir / name) for name in ('trajectories.png', 'trajectories.svg')},
              'boundaries': ['All12 fresh full50 endpoints required; one official-mAP-best checkpoint supplies all metrics.',
                             'All four conditions use FP32 visual storage and matched common initialization; trained camera/nonvisual state retained.',
                             'Global-only is independently trained, not a same-checkpoint slice; roles have different parameters/cost.',
                             'Only global-only visual improvement is partial C1 evidence; the overall visual gate still fails unless both readouts qualify.',
                             'Ordinary fine-tuning is a control, not novelty. No gate, learning rate, seed or epoch rule is changed by this report.',
                             'One seed and official-best selection do not establish stability, untouched-test significance or SOTA.',
                             'Identity bootstrap resamples fixed-model identities, not training seeds; all flips/AP are post-selection real-GT diagnoses.',
                             'Memory is peak allocated usage after initialization, excluding initialization transients and reserved memory.',
                             'This report loads fixed CPU arrays; it performs no neural forward, optimizer or inference-rule change.']}
    (args.output_dir / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Complete visual-update and independent-role controls', '',
             '| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda item: (DATASETS.index(item['dataset']), tuple(CONDITIONS).index(item['variant']))):
        scores = row['metrics']
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {scores['mAP']:.4f} | {scores['Rank-1']:.4f} | {scores['Rank-5']:.4f} | {scores['Rank-10']:.4f} |")
    lines += ['', '| Dataset | Comparison | Delta mAP | Delta R1 | Repairs | New errors | Identity macro AP delta | Gate |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for item in (*visual, *roles):
        pair = item['paired_diagnosis']
        lines.append(f"| {item['dataset']} | {pair['control']} to {pair['candidate']} | {pair['delta_metrics']['mAP']:+.4f} | {pair['delta_metrics']['Rank-1']:+.4f} | {pair['rank1_repairs']} | {pair['rank1_new_errors']} | {pair['identity_macro_mean_delta_ap_points']:+.4f} | {'PASS' if item['gate_passed'] else 'FAIL'} |")
    lines += ['', 'Frozen-role comparisons are diagnostic; the registered role gate uses only low_lr.', '',
              'Registered visual-update gate: **' + report['registered_visual_update_gate'] + '**.',
              'Registered role gate: **' + report['registered_role_gate'] + '**.',
              'Registered joint gate: **' + report['registered_joint_gate'] + '**.', '',
              *['- ' + line for line in report['boundaries']], '',
              'The full three-dataset baseline/SOTA goal is not established by this gate.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'accepted': 12,
                      'registered_visual_update_gate': report['registered_visual_update_gate'],
                      'registered_role_gate': report['registered_role_gate'],
                      'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
