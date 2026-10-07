"""All-query report for six training-only increment objective controls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_full as panel
from tools.analyze_correspondence_distances import compare, camera_scores, scene_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    panel.base.source_map = panel.source_map
    panel.base.require_sources(args.campaign)
    controls = json.loads(panel.m0.CONTROLS.read_text())
    assert all(panel.base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    state = json.loads((args.campaign / 'campaign.json').read_text())
    matrix = json.loads((args.campaign / 'accepted_matrix.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1
    assert len(state['jobs']) == 6 and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    assert matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    combined = dict(rows=[dict(row, variant='raw_'+row['variant']) for row in controls['rows']] + matrix['rows'])
    rows, pairs = [], []
    for row in matrix['rows']:
        dataset, objective = row['dataset'], row['variant']
        assert panel.accepted_row(args.campaign, dataset, objective) == row
        run = Path(row['run_dir'])
        training = json.loads((run / 'training.json').read_text())
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (run / 'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(item['steps'] for item in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        rows.append(dict(row, formal_steps=len(steps), history=training['history'],
            training_and_epoch_eval_seconds=(datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds()))
        comparisons = ['raw_semantic', 'raw_global_only']
        if objective == 'repair_keep':
            comparisons.append('md_batch_ratio')
        for control in comparisons:
            result = compare(combined, dataset, control, objective)
            capacity = 'Global-only omits semantic roles.' if control == 'raw_global_only' else 'Same semantic model parameter set; training objective differs.'
            result['boundary'] = 'Official post-selection diagnosis, not new inference or selection. ' + capacity + ' Bootstrap resamples fixed-model identities, not training seeds.'
            scorer = scene_scores if dataset == 'MSVR310' else camera_scores
            environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
            scores = {}
            for label in (control, objective):
                selected = next(r for r in combined['rows'] if (r['dataset'], r['variant']) == (dataset, label))
                data = torch.load(Path(selected['run_dir']) / 'official_distances.pt', map_location='cpu', weights_only=False)
                scores[label] = scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                    data['query_'+environment], data['gallery_'+environment])
            result['query_changes'] = [dict(query_index=i, identity=int(identity),
                control_ap=float(scores[control]['average_precision'][i]), candidate_ap=float(scores[objective]['average_precision'][i]),
                control_first_rank=int(scores[control]['first_match_rank'][i]), candidate_first_rank=int(scores[objective]['first_match_rank'][i]))
                for i, identity in enumerate(data['query_ids'])]
            delta = result['delta_metrics']
            pairs.append(dict(dataset=dataset, objective=objective, control=control,
                phase_progress=delta['mAP'] >= 0.5 and delta['Rank-1'] >= 0, paired_diagnosis=result))
    value = dict(schema=panel.m0.SCHEMA, status='COMPLETE', created_at=datetime.now().astimezone().isoformat(),
        accepted=6, formal_epochs=300, formal_steps=sum(row['formal_steps'] for row in rows), rows=rows, pairs=pairs,
        boundary='Training-only mechanism comparison on consumed official benchmarks. No new parameters or inference changes. MD ratio is an adaptation, not full author reproduction; repair/keep formula and upstream responsibility are not novel. Complete curves/E50 and all legal queries retained; project gate is not significance, multi-seed stability, P1/P2/P3 or SOTA proof.')
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir / 'SUMMARY.json', value)
    lines = ['# 新增证据目标对照', '', '| 数据集 | 目标 | best轮 | mAP | R1 | 正式更新 |', '|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["dataset"]} | {r["variant"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.4f} | {r["metrics"]["Rank-1"]:.4f} | {r["formal_steps"]} |' for r in rows]
    lines += ['', value['boundary']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE', accepted=6, pairs=len(pairs), formal_steps=value['formal_steps'])), flush=True)


if __name__ == '__main__':
    main()
