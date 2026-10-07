"""Once-only all-query CPU comparison against sealed task-ownership controls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_deployment_metric_role as panel
from tools.analyze_correspondence_distances import compare, camera_scores, scene_scores


def paired(combined, dataset, control, candidate):
    result = compare(combined, dataset, control, candidate)
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    env = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scores = {}
    for name in (control, candidate):
        row = next(r for r in combined['rows'] if (r['dataset'], r['variant']) == (dataset, name))
        data = torch.load(Path(row['run_dir']) / 'official_distances.pt', map_location='cpu', weights_only=False)
        scores[name] = scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                              data['query_' + env], data['gallery_' + env])
    result['query_changes'] = [dict(query_index=i, identity=int(identity),
        control_ap=float(scores[control]['average_precision'][i]), candidate_ap=float(scores[candidate]['average_precision'][i]),
        control_first_rank=int(scores[control]['first_match_rank'][i]), candidate_first_rank=int(scores[candidate]['first_match_rank'][i]))
        for i, identity in enumerate(data['query_ids'])]
    delta = result['delta_metrics']
    return dict(dataset=dataset, variant=candidate, control=control,
                phase_progress=delta['mAP'] >= .5 and delta['Rank-1'] >= 0, paired_diagnosis=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    panel.configure()
    panel.base.require_sources(args.campaign)
    controls = panel.require_controls()
    state = json.loads((args.campaign / 'campaign.json').read_text())
    matrix = json.loads((args.campaign / 'accepted_matrix.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1
    assert len(state['jobs']) == 12 and all(r['status'] == 'COMPLETE' and r['exit_code'] == 0 for r in state['jobs'])
    assert matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    combined = dict(rows=[dict(r, variant='control_' + r['variant']) for r in controls['rows']] + matrix['rows'])
    rows, pairs = [], []
    for row in matrix['rows']:
        dataset, variant = row['dataset'], row['variant']
        assert panel.accepted_row(args.campaign, dataset, variant) == row
        run = Path(row['run_dir'])
        training = json.loads((run / 'training.json').read_text())
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (run / 'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(item['steps'] for item in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        control = next(r for r in controls['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
        assert (run / 'training_batch_order.jsonl').read_bytes() == (Path(control['run_dir']) / 'training_batch_order.jsonl').read_bytes()
        rows.append({**row, 'formal_steps': len(steps), 'history': training['history'],
                     'training_and_epoch_evaluation_seconds': (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds()})
        for control_name in ('control_' + variant, 'control_global_only'):
            pair = paired(combined, dataset, control_name, variant)
            pair['actual_training_batch_order_equal'] = True
            pairs.append(pair)
    for dataset in panel.DATASETS:
        pairs.append(paired(combined, dataset, 'semantic', 'native'))
    result = dict(schema=panel.SCHEMA, status='COMPLETE', created_at=datetime.now().astimezone().isoformat(),
                  accepted=6, formal_epochs=300, formal_steps=sum(r['formal_steps'] for r in rows), rows=rows, pairs=pairs,
                  primary_matched_role_progress_count=sum(p['phase_progress'] for p in pairs if p['control'] == 'control_' + p['variant']),
                  independent_global_progress_count=sum(p['phase_progress'] for p in pairs if p['control'] == 'control_global_only'),
                  boundary='Seed42 consumed-benchmark development. Role Triplet uses joint1536 L2(h); vehicle geometry also changes modal to joint. Raw classifier logits, global task, original head-loss sum, state/capacity/recipe unchanged. Strict verification occurred before each exact probe retirement; immutable acceptance artifacts verified here. No retired M0 or historical report replay, new inference, reranking, seed search or SOTA claim.')
    panel.require_controls()
    panel.base.require_sources(args.campaign)
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir / 'SUMMARY.json', result)
    lines = ['# 角色度量部署几何对照', '', '| 数据集 | 条件 | best轮 | mAP | R1 | 正式步数 |',
             '|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["dataset"]} | {r["variant"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.4f} | {r["metrics"]["Rank-1"]:.4f} | {r["formal_steps"]} |' for r in rows]
    lines += ['', result['boundary']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='COMPLETE', accepted=6, formal_steps=result['formal_steps'], pairs=len(pairs))), flush=True)


if __name__ == '__main__':
    main()
