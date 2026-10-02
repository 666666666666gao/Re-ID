"""One CPU paired report after the six complete F1 full-gallery endpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_foundation_recipe as panel
from tools.analyze_correspondence_distances import compare, sha

HISTORICAL = ROOT / 'logs/clean_clip_complete721_20261002/raw/results/clean_clip_joint_complete_20261002/SUMMARY.json'
HISTORICAL_SHA = '784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    panel.require_sources(campaign)
    state = json.loads((campaign / 'campaign.json').read_text())
    matrix = json.loads((campaign / 'accepted_matrix.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1
    assert len(state['jobs']) == 12 and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert matrix['schema'] == panel.SCHEMA and matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert {(row['dataset'], row['variant']) for row in matrix['rows']} == {
        (dataset, recipe) for dataset in panel.DATASETS for recipe in panel.RECIPES}
    assert sha(HISTORICAL) == HISTORICAL_SHA
    historical = json.loads(HISTORICAL.read_text())
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    rows, pairs, references = [], [], []
    for row in matrix['rows']:
        assert panel.verify(campaign, row['dataset'], row['variant']) == row
        output = Path(row['run_dir'])
        training = json.loads((output / 'training.json').read_text())
        evaluation = json.loads((output / 'official_metrics.json').read_text())
        history = training['history']
        steps = sum(1 for _ in (output / 'training_steps.jsonl').open())
        assert steps == sum(item['steps'] for item in history)
        seconds = (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds()
        rows.append({**row, 'formal_steps': steps, 'history': history,
            'trainable_parameters': row['initializer']['trainable_parameters'],
            'training_and_epoch_evaluation_seconds': seconds,
            'final_evaluation_seconds': evaluation['final_evaluation_seconds'],
            'peak_allocated_bytes': training['peak_allocated_bytes'],
            'final_minus_best_metrics': {name: history[-1]['official_fused'][name] - row['metrics'][name] for name in row['metrics']}})
    for dataset in panel.DATASETS:
        diagnosis = compare(matrix, dataset, 'current', 'author')
        delta = diagnosis['delta_metrics']
        pairs.append({'dataset': dataset, 'comparison': 'author minus current',
                      'phase_progress': delta['mAP'] >= 0.5 and delta['Rank-1'] >= 0,
                      'paired_diagnosis': diagnosis})
        old = next(row for row in historical['rows'] if row['dataset'] == dataset and row['variant'] == 'global_only')
        current = next(row for row in rows if row['dataset'] == dataset and row['variant'] == 'current')
        for key, old_key in (('visual_initial_sha256', 'visual_initial_sha256'),
                             ('camera_initial_sha256', 'fresh_camera_initial_sha256')):
            assert current['initializer'][key] == old['initializer'][old_key]
        assert current['initializer']['current_head_initial_sha256'] == {
            name: old['initializer']['common_initializer_sha256'][name] for name in ('neck', 'classifier')}
        references.append({'dataset': dataset, 'historical_shared_global': old['metrics'],
            'historical_best_epoch': old['best_epoch'],
            'shared_global_minus_current': {name: old['metrics'][name] - current['metrics'][name] for name in current['metrics']},
            'boundary': 'Historical separate adapter control; matching initial public visual/camera/current head, not capacity matched or a new concurrent arm.'})
    report = {'schema': 'trifusion-foundation-recipe-complete-report-v1', 'status': 'COMPLETE',
        'created_at': datetime.now().astimezone().isoformat(), 'campaign': str(campaign),
        'accepted': 6, 'formal_epochs': 300, 'formal_steps': sum(row['formal_steps'] for row in rows),
        'rows': rows, 'pairs': pairs, 'historical_references': references,
        'historical_summary_sha256': HISTORICAL_SHA,
        'boundaries': ['Foundation package comparison, not a new model contribution or isolated hyperparameter causality.',
            'Full official galleries and original camera/time filtering; all metrics from one mAP-best checkpoint per endpoint.',
            'Seed42 only, consumed official benchmark selection; identity bootstrap is fixed-model diagnosis, not training-seed significance.',
            'Author RGBNT100 schedule extended from30 to50; common workers4/AMPscale256/evaluation path differ from the original entry.',
            'Reported training interval includes epoch evaluation but excludes construction/M0/final evaluation; different B/K means unequal update counts.',
            'Phase-progress threshold is not a SOTA or significance gate; overall objective remains ACTIVE_UNMET.']}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# F1 six-endpoint foundation-package report', '',
             '| Dataset | Recipe | Best epoch | mAP | R1 | R5 | R10 | Steps |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        metrics = row['metrics']
        values = [f'{metrics[name]:.4f}' if row['dataset'] == 'RGBNT201' or name in ('mAP', 'Rank-1') else '-' for name in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {' | '.join(values)} | {row['formal_steps']} |")
    lines += ['', *report['boundaries']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'accepted': 6, 'output_dir': str(args.output_dir)}), flush=True)


if __name__ == '__main__':
    main()
