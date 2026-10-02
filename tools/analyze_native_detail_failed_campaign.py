"""CPU closeout of the original N1 failure, after all six training arms end."""
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
from tools import queue_native_detail as native
from tools.analyze_correspondence_distances import compare, sha
from tools.report_clean_clip_joint_complete import elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    native.install()
    campaign = args.campaign.resolve()
    assert campaign == ROOT / 'logs/native_detail_20261002_v1'
    assert args.output_dir.resolve() == ROOT / 'results/native_detail_failed_closeout_20261002'
    assert not args.output_dir.exists()
    manifest = native.panel.require_sources(campaign)
    assert len(manifest['source_sha256']) == 238
    state = json.loads((campaign / 'campaign.json').read_text())
    waiter_path = ROOT / 'logs/native_detail_analysis_waiter_20261002.json'
    waiter = json.loads(waiter_path.read_text())
    assert state['status'] == 'FAILED' and state['completed_at']
    assert len(state['jobs']) == 6
    assert {(j['dataset'], j['variant']) for j in state['jobs']} == {
        (d, v) for d in native.DATASETS for v in native.CONDITIONS}
    assert {(j['dataset'], j['variant']) for j in state['jobs'] if j['status'] == 'FAILED'} == {('MSVR310', 'low')}
    assert all(j['status'] in ('COMPLETE', 'FAILED') for j in state['jobs'])
    assert not Path(f"/proc/{state['controller_pid']}/cmdline").exists()
    assert waiter['status'] == 'CAMPAIGN_FAILED_NO_REPORT' and waiter['invocations'] == 0
    assert not (campaign / 'accepted_matrix.json').exists()
    assert not (ROOT / 'results/native_detail_complete_20261002').exists()
    assert sha(native.SUMMARY) == native.SUMMARY_SHA
    old_report = json.loads(native.SUMMARY.read_text())
    old_matrix_path = native.PREDECESSOR / 'accepted_matrix.json'
    old_matrix = json.loads(old_matrix_path.read_text())
    assert old_matrix['accepted'] == old_matrix['expected'] == len(old_matrix['rows']) == 6
    for old in old_matrix['rows']:
        saved = next(r for r in old_report['rows'] if (r['dataset'], r['variant']) == (old['dataset'], old['variant']))
        assert old == {k: saved[k] for k in old}
    torch.set_num_threads(1)
    inputs = [campaign / 'campaign.json', campaign / 'manifest.json', waiter_path,
        Path(manifest['preflight_path']), *map(Path, manifest['initialization_sha256']),
        native.SUMMARY, old_matrix_path, Path(__file__).resolve(),
        ROOT / 'tools/analyze_correspondence_distances.py']
    bindings = {str(p): sha(p) for p in inputs}
    rows, accepted, histories = [], [], {}
    for job in state['jobs']:
        child_dir = native.queue.child_campaign(campaign, 'clean_clip', job['dataset'], job['variant'])
        child = json.loads((child_dir / 'campaign.json').read_text())
        assert child['status'] == job['status']
        assert [s['mode'] for s in child['jobs']] == ['m0', 'train', 'evaluate']
        m0, train, evaluation = child['jobs']
        assert m0['status'] == train['status'] == 'COMPLETE'
        assert m0['exit_code'] == train['exit_code'] == 0
        run = Path(train['output_dir'])
        training = json.loads((run / 'training.json').read_text())
        assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
        history = training['history']
        assert [h['epoch'] for h in history] == list(range(1, 51))
        steps = sum(1 for _ in (run / 'training_steps.jsonl').open())
        assert steps == sum(h['steps'] for h in history)
        best = max(history, key=lambda h: (h['official_fused']['mAP'], h['epoch']))
        assert best['epoch'] == training['best_epoch']
        official_path = run / 'official_metrics.json'
        row = {'dataset': job['dataset'], 'variant': job['variant'],
            'training_epochs': 50, 'formal_steps': steps, 'run_dir': str(run),
            'training_selected_epoch': best['epoch'],
            'trainable_parameters': training['initializer']['trainable_parameters'],
            'best_mean_loss': best['mean_loss'], 'final_mean_loss': history[-1]['mean_loss'],
            'training_trajectory_final_minus_best': {
                k: history[-1]['official_fused'][k] - best['official_fused'][k] for k in best['official_fused']},
            'training_and_epoch_eval_seconds': elapsed(training['started_at'], training['completed_at']),
            'peak_training_and_epoch_eval_allocated_bytes': training['peak_training_and_epoch_eval_allocated_bytes']}
        if job['status'] == 'COMPLETE':
            assert job['exit_code'] == evaluation['exit_code'] == 0 and evaluation['status'] == 'COMPLETE'
            actual = native.panel.verify(run, Path(m0['output_dir']), job['dataset'], job['variant'],
                Path(manifest['preflight_path']).parent)
            assert actual == child['verification']
            accepted.append(actual)
            row.update(status='VERIFIED_COMPLETE', metrics=actual['metrics'],
                best_epoch=actual['best_epoch'], checkpoint_sha256=actual['checkpoint_sha256'],
                distance_sha256=actual['distance_sha256'], receipt_sha256=actual['receipt_sha256'])
            bindings[str(official_path)] = sha(official_path)
        else:
            assert job['exit_code'] == evaluation['exit_code'] == 1 and evaluation['status'] == 'FAILED'
            assert not official_path.exists()
            row.update(status='FAILED_EVALUATION_AFTER50', metrics=None, best_epoch=None,
                boundary='Training trajectory and epoch selection are unaccepted deployment diagnostics. Original strict reload failed; no formal metrics or replacement receipt.')
        for path in (child_dir / 'campaign.json', child_dir / 'evaluate.log', run / 'training.json',
                     run / 'training_steps.jsonl', Path(m0['output_dir']) / 'training.json',
                     run / 'best_map.pth', run / 'official_distances.pt'):
            bindings[str(path)] = sha(path)
        assert run == Path(evaluation['output_dir'])
        assert not Path(f"/proc/{job['pid']}/cmdline").exists()
        histories[job['dataset'], job['variant']] = history
        rows.append(row)
    assert len(accepted) == 5
    combined = {'rows': old_matrix['rows'] + accepted}
    pairs = []
    for dataset in native.DATASETS:
        for control, gate in (('roles', 'N1-A'), ('low', 'N1-B'), ('global_only', None)):
            if dataset == 'MSVR310' and control == 'low':
                pairs.append({'dataset': dataset, 'control': control, 'candidate': 'high', 'gate': gate,
                    'status': 'UNAVAILABLE_ORIGINAL_CONTROL_EVALUATION_FAILED', 'gate_passed': None,
                    'paired_diagnosis': None})
                continue
            diagnosis = compare(combined, dataset, control, 'high')
            if control == 'low':
                diagnosis['boundary'] = ('Official post-selection fixed-model diagnosis; matched parameters and '
                    'initial states, different input computation. Low upsampling does not restore lost detail. '
                    'Identity bootstrap is not training-seed variance.')
            pair = {'dataset': dataset, 'control': control, 'candidate': 'high', 'gate': gate,
                    'status': 'VALID_FIXED_DISTANCE_COMPARISON', 'paired_diagnosis': diagnosis}
            if gate:
                delta = diagnosis['delta_metrics']
                floor = 0.5 if gate == 'N1-A' and dataset in ('RGBNT201', 'MSVR310') else 0.0
                pair.update(minimum_map_points=floor,
                    gate_passed=delta['mAP'] > 0 and delta['mAP'] >= floor and delta['Rank-1'] >= 0)
            pairs.append(pair)
    assert all(p['gate_passed'] is False for p in pairs if p['gate'] == 'N1-A')
    assert next(p for p in pairs if p['dataset'] == 'RGBNT201' and p['gate'] == 'N1-B')['gate_passed'] is False
    for row in combined['rows']:
        for name in ('official_metrics.json', 'official_distances.pt'):
            path = Path(row['run_dir']) / name
            bindings[str(path)] = sha(path)
    assert native.panel.require_sources(campaign) == manifest
    assert all(sha(Path(p)) == digest for p, digest in bindings.items())
    boundaries = ['Original six-end acceptance remains FAILED: five valid endpoints and one invalid strict-reload endpoint.',
        'Original success-report waiter exited without invocation; this separate CPU closeout does not replace it or create an accepted matrix.',
        'All six original training arms reached50; failed deployment metrics remain null, diagnostic reruns never substitute.',
        'N1-A failed on all three datasets; N1-B already fails201 and its MSVR comparison remains unavailable.',
        'N1 replaces CNN semantic values with image-native values while retaining semantic keys; it does not test preserving both evidence sources.',
        'N1-A jointly changes CNN value source, candidate grid from128 to512 positions, and capacity by93,248 parameters.',
        'N1-B matches parameters, initialization and512 candidates while changing input-detail computation; low upsampling does not restore lost detail.',
        'Old clean controls are pinned original results, not rerun or reselected. Global-only comparisons are descriptive, without a new gate.',
        'All valid formal metrics use real GT/full gallery/original environment filtering/no rerank and one mAP-best checkpoint per50epoch run.',
        'Official benchmarks were consumed in development/epoch selection; one seed and fixed-model identity bootstrap do not prove unbiased or multi-seed stability.',
        'Training costs include epoch evaluations; exclude construction, final reload, prior controls and separate failure diagnostics.',
        'No N2/N3 implementation, individual-role necessity, baseline/SOTA achievement or evidence of a unique failure cause.']
    report = {'schema': 'trifusion-native-detail-failed-campaign-closeout-v1',
        'status': 'FAILED_CAMPAIGN_CLOSEOUT_COMPLETE', 'created_at': datetime.now().astimezone().isoformat(),
        'campaign': str(campaign), 'original_campaign_status': 'FAILED', 'original_contract': 'FAILED',
        'accepted': 5, 'failed_evaluation': 1, 'formal_epochs': 300,
        'formal_steps': sum(r['formal_steps'] for r in rows), 'original_success_report_invocations': 0,
        'registered_gates': {'N1-A': 'FAIL', 'N1-B': 'FAIL_WITH_MSVR_COMPARISON_UNAVAILABLE'},
        'registered_both_gates': 'FAIL', 'rows': rows, 'prior_control_rows': old_report['rows'],
        'pairs': pairs, 'report_inputs_sha256': bindings,
        'campaign_parent_observed_wall_seconds': elapsed(state['started_at'], state['completed_at']),
        'sum_training_and_epoch_eval_seconds': sum(r['training_and_epoch_eval_seconds'] for r in rows),
        'disk_free_bytes_at_report': shutil.disk_usage(ROOT).free, 'boundaries': boundaries}
    args.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for index, dataset in enumerate(native.DATASETS):
        for variant in native.CONDITIONS:
            history = histories[dataset, variant]
            label = variant + (' (reload failed)' if (dataset, variant) == ('MSVR310', 'low') else '')
            epochs = [h['epoch'] for h in history]
            axes[index, 0].plot(epochs, [h['mean_loss'] for h in history], label=label)
            axes[index, 1].plot(epochs, [h['official_fused']['mAP'] for h in history], label=label)
        for column, label in enumerate(('Mean training loss', 'Recorded epoch mAP (%)')):
            axes[index, column].set(title=dataset, xlabel='Epoch', ylabel=label)
            axes[index, column].grid(alpha=.2)
            axes[index, column].legend()
    for suffix in ('png', 'svg'):
        figure.savefig(args.output_dir / f'TRAINING_CURVES.{suffix}', dpi=160)
    plt.close(figure)
    report['figure_sha256'] = {n: sha(args.output_dir / n) for n in ('TRAINING_CURVES.png', 'TRAINING_CURVES.svg')}
    (args.output_dir / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Original N1 failed campaign closeout', '',
        '**Original acceptance: FAILED; accepted5/6; invalid1; original success-report invocations0.**', '',
        '| Dataset | Variant | Status | Best epoch | mAP | R1 | R5 | R10 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values = ['—'] * 4
        if row['metrics'] is not None:
            values[:2] = [f"{row['metrics'][k]:.4f}" for k in ('mAP', 'Rank-1')]
            if row['dataset'] == 'RGBNT201':
                values[2:] = [f"{row['metrics'][k]:.4f}" for k in ('Rank-5', 'Rank-10')]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['status']} | {row['best_epoch'] or '—'} | {' | '.join(values)} |")
    lines += ['', '| Dataset | High minus | ΔmAP | ΔR1 | Repairs/new errors | Gate |',
        '|---|---|---:|---:|---|---|']
    for pair in pairs:
        d = pair['paired_diagnosis']
        if d is None:
            lines.append(f"| {pair['dataset']} | {pair['control']} | — | — | — | UNAVAILABLE: original reload failure |")
        else:
            gate = 'Descriptive only' if pair['gate'] is None else pair['gate'] + (': PASS' if pair['gate_passed'] else ': FAIL')
            lines.append(f"| {pair['dataset']} | {pair['control']} | {d['delta_metrics']['mAP']:+.4f} | {d['delta_metrics']['Rank-1']:+.4f} | {d['rank1_repairs']}/{d['rank1_new_errors']} | {gate} |")
    lines += ['', *['- ' + b for b in boundaries]]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'accepted': 5, 'failed_evaluation': 1,
        'original_success_report_invocations': 0, 'registered_gates': report['registered_gates']}), flush=True)


if __name__ == '__main__':
    main()
