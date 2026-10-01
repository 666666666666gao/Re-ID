"""One complete CPU report for the six matched clean-public CLIP controls."""
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
from tools import queue_clean_clip_joint as panel
from tools.analyze_correspondence_distances import compare, sha


def elapsed(start, finish):
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = panel.require_sources(campaign)
    state = json.loads((campaign/'campaign.json').read_text())
    matrix = json.loads((campaign/'accepted_matrix.json').read_text())
    assert matrix['schema'] == panel.SCHEMA and matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 6
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert {(row['dataset'],row['variant']) for row in matrix['rows']} == {
        (dataset,variant) for dataset in panel.DATASETS for variant in panel.CONDITIONS}
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    files = [campaign/name for name in ('manifest.json','campaign.json','accepted_matrix.json')]
    files += [Path(manifest['preflight_path']), *map(Path,manifest['initialization_sha256'])]
    sources = {str(path):sha(path) for path in files}
    sources[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    analyzer = ROOT/'tools/analyze_correspondence_distances.py'
    sources[str(analyzer)] = sha(analyzer)
    initialization_dir = Path(manifest['preflight_path']).parent
    rows, histories, outputs = [], {}, set()
    for row in matrix['rows']:
        run = Path(row['run_dir'])
        child_dir = panel.queue.child_campaign(campaign,'clean_clip',row['dataset'],row['variant'])
        child = json.loads((child_dir/'campaign.json').read_text())
        assert child['status'] == 'COMPLETE' and child['dataset'] == row['dataset'] and child['variant'] == row['variant']
        assert [stage['mode'] for stage in child['jobs']] == ['m0','train','evaluate']
        assert all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in child['jobs'])
        actual = panel.verify(run,Path(row['m0_dir']),row['dataset'],row['variant'],initialization_dir)
        assert actual == row
        training = json.loads((run/'training.json').read_text())
        history = training['history']
        step_count = sum(1 for _ in (run/'training_steps.jsonl').open())
        assert step_count == sum(item['steps'] for item in history)
        best = history[row['best_epoch']-1]
        for path in (child_dir/'campaign.json',run/'training.json',run/'training_steps.jsonl',
                     run/'official_metrics.json',Path(row['m0_dir'])/'training.json',
                     Path(row['m0_dir'])/'training_steps.jsonl'):
            sources[str(path)] = sha(path)
        outputs.update((run,Path(row['m0_dir'])))
        histories[row['dataset'],row['variant']] = history
        rows.append({**row,'formal_steps':step_count,'trainable_parameters':row['initializer']['trainable_parameters'],
                     'best_mean_loss':best['mean_loss'],'final_mean_loss':history[-1]['mean_loss'],
                     'final_minus_best_metrics':{name:history[-1]['official_fused'][name]-row['metrics'][name]
                                                for name in row['metrics']},
                     'endpoint_parent_observed_wall_seconds':elapsed(child['started_at'],child['completed_at']),
                     'm0_origin':child['jobs'][0].get('origin','worker')})
    pairs = []
    for dataset in panel.DATASETS:
        diagnosis = compare(matrix,dataset,'global_only','roles')
        delta = diagnosis['delta_metrics']
        floor = 0.5 if dataset in ('RGBNT201','MSVR310') else 0.0
        passed = delta['mAP'] > 0 and delta['mAP'] >= floor and delta['Rank-1'] >= 0
        pairs.append({'dataset':dataset,'minimum_map_points':floor,'gate_passed':passed,
                      'paired_diagnosis':diagnosis})
    assert panel.require_sources(campaign) == manifest
    assert all(sha(Path(path)) == digest for path,digest in sources.items())
    boundaries = [
        'Clean public CLIP visual weights; fresh trainable camera, adapters and heads. No trained ReID state loaded.',
        'Global-only includes the same shared adapters; it is an independent trained control, not raw or zero-shot CLIP.',
        'Roles add capacity and may consume randomness differently. Common initial states were actually compared bitwise.',
        'All six ends completed50 epochs; one highest official fused-mAP checkpoint per end, tied epochs resolved later.',
        'Full real-GT query/gallery and original camera/time-block exclusions; gallery-only distractors retained; no reranking.',
        'Official benchmarks already participated in epoch and method development. Fixed-model identity bootstrap is not training-seed stability.',
        'Training/epoch-evaluation intervals exclude construction and upstream costs. Two RGBNT201 M0s precede the formal controller.',
        'Trainable counts are reported from initializer witnesses; equal output width/epochs do not imply equal capacity or computation.',
        'Warm ReID comparisons are historical and differ in camera training and initialization; no single-factor causal attribution.',
        'J1 is an intermediate role-contribution gate, not the full three-dataset baseline/SOTA goal. N1/N2/N3 are not tested here.',
    ]
    report = {'schema':'trifusion-clean-public-clip-complete-report-v1','status':'COMPLETE',
              'created_at':datetime.now().astimezone().isoformat(),'campaign':str(campaign),
              'accepted':6,'formal_epochs':300,'formal_steps':sum(row['formal_steps'] for row in rows),
              'registered_J1_gate':'PASS' if all(item['gate_passed'] for item in pairs) else 'FAIL',
              'registered_J1_components_passed':sum(item['gate_passed'] for item in pairs),
              'rows':rows,'pairs':pairs,'report_inputs_sha256':sources,
              'campaign_parent_observed_wall_seconds':elapsed(state['started_at'],state['completed_at']),
              'sum_training_and_epoch_eval_seconds':sum(row['training_and_epoch_eval_seconds'] for row in rows),
              'logical_m0_and_full_output_bytes':sum(path.stat().st_size for directory in outputs
                                                    for path in directory.rglob('*') if path.is_file()),
              'disk_free_bytes_at_report':shutil.disk_usage(ROOT).free,'boundaries':boundaries}
    args.output_dir.mkdir(parents=True)
    figure,axes = plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for index,dataset in enumerate(panel.DATASETS):
        for variant in panel.CONDITIONS:
            history = histories[dataset,variant]
            epochs = [item['epoch'] for item in history]
            axes[index,0].plot(epochs,[item['mean_loss'] for item in history],label=variant)
            axes[index,1].plot(epochs,[item['official_fused']['mAP'] for item in history],label=variant)
        for column,label in enumerate(('Mean training loss','Official fused mAP (%)')):
            axes[index,column].set(title=dataset,xlabel='Epoch',ylabel=label)
            axes[index,column].grid(alpha=.2)
            axes[index,column].legend()
    for suffix in ('png','svg'):
        figure.savefig(args.output_dir/f'TRAINING_CURVES.{suffix}',dpi=160)
    plt.close(figure)
    report['figure_sha256'] = {name:sha(args.output_dir/name) for name in ('TRAINING_CURVES.png','TRAINING_CURVES.svg')}
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Complete clean-public CLIP joint controls','',
             '| Dataset | Readout | Best epoch | mAP | R1 | R5 | R10 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda item:(panel.DATASETS.index(item['dataset']),panel.CONDITIONS.index(item['variant']))):
        scores = row['metrics']
        tail = [f"{scores[name]:.4f}" if row['dataset'] == 'RGBNT201' else '—' for name in ('Rank-5','Rank-10')]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {scores['mAP']:.4f} | {scores['Rank-1']:.4f} | {' | '.join(tail)} |")
    lines += ['', '| Dataset | Roles minus global mAP | R1 | Repairs | New errors | Identity macro AP delta | J1 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for item in pairs:
        pair = item['paired_diagnosis']
        lines.append(f"| {item['dataset']} | {pair['delta_metrics']['mAP']:+.4f} | {pair['delta_metrics']['Rank-1']:+.4f} | {pair['rank1_repairs']} | {pair['rank1_new_errors']} | {pair['identity_macro_mean_delta_ap_points']:+.4f} | {'PASS' if item['gate_passed'] else 'FAIL'} |")
    lines += ['', 'Registered J1: **'+report['registered_J1_gate']+'**.', '',
              *['- '+line for line in boundaries]]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'COMPLETE','accepted':6,'registered_J1_gate':report['registered_J1_gate'],
                      'output_dir':str(args.output_dir)}),flush=True)


if __name__ == '__main__':
    main()
