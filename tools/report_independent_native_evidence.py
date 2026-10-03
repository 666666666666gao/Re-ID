"""One terminal CPU report: matched role contribution and additive native evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_independent_native_evidence as panel
from tools.analyze_correspondence_distances import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    panel.configure()
    panel.base.require_sources(campaign)
    state = json.loads((campaign / 'campaign.json').read_text())
    matrix = json.loads((campaign / 'accepted_matrix.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1
    assert len(state['jobs']) == 18 and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 9
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    rows, pairs = [], []
    for row in matrix['rows']:
        assert panel.base.verify(campaign, row['dataset'], row['variant']) == row
        output = Path(row['run_dir'])
        training = json.loads((output / 'training.json').read_text())
        steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(item['steps'] for item in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        rows.append({**row, 'formal_steps': len(steps), 'history': training['history'],
            'training_and_epoch_evaluation_seconds': (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds(),
            'peak_allocated_bytes': training['peak_allocated_bytes'],
            'final_minus_best_metrics': {name: training['history'][-1]['official_fused'][name] - row['metrics'][name] for name in row['metrics']}})
    for dataset in panel.DATASETS:
        selected = {row['variant']: row for row in rows if row['dataset'] == dataset}
        orders = [(Path(selected[name]['run_dir']) / 'training_batch_order.jsonl').read_bytes() for name in panel.VARIANTS]
        assert orders[0] == orders[1] == orders[2], dataset
        for control, candidate in (('global_only', 'semantic'), ('semantic', 'native'), ('global_only', 'native')):
            diagnosis = compare(matrix, dataset, control, candidate)
            delta = diagnosis['delta_metrics']
            pairs.append({'dataset': dataset, 'comparison': f'{candidate} minus {control}',
                'actual_training_batch_order_equal': True,
                'phase_progress': delta['mAP'] >= 0.5 and delta['Rank-1'] >= 0,
                'paired_diagnosis': diagnosis})
    report = {'schema': panel.SCHEMA, 'status': 'COMPLETE', 'created_at': datetime.now().astimezone().isoformat(),
        'accepted': 9, 'formal_epochs': 450, 'formal_steps': sum(row['formal_steps'] for row in rows),
        'rows': rows, 'pairs': pairs,
        'boundary': 'One seed42, consumed official mAP selection, matched author package. Native adds159296 parameters; no capacity-only causal claim, three-new-module success, seed robustness or SOTA. Identity bootstrap is not training-seed uncertainty.'}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Independent native evidence report', '', '| Dataset | Variant | Epoch | mAP | R1 | Steps |',
        '|---|---|---:|---:|---:|---:|']
    lines += [f"| {row['dataset']} | {row['variant']} | {row['best_epoch']} | {row['metrics']['mAP']:.4f} | {row['metrics']['Rank-1']:.4f} | {row['formal_steps']} |" for row in rows]
    lines += ['', report['boundary']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'accepted': 9, 'formal_steps': report['formal_steps']}), flush=True)


if __name__ == '__main__':
    main()
