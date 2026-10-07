"""Once-only all-query patch/mean and sealed RAW-control comparisons."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_region_reconstruction as panel
from tools.analyze_correspondence_distances import compare, camera_scores, scene_scores


def query_comparison(combined, dataset, control, candidate):
    result = compare(combined, dataset, control, candidate)
    if control == 'mean':
        result['boundary'] = ('Official post-selection diagnosis, not new inference or selection. '
                              'Same active parameters and initial state; query routing differs. '
                              'Bootstrap resamples fixed-model identities, not training seeds.')
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scores = {}
    for label in (control, candidate):
        selected = next(r for r in combined['rows'] if (r['dataset'], r['variant']) == (dataset, label))
        data = torch.load(Path(selected['run_dir'])/'official_distances.pt', map_location='cpu', weights_only=False)
        scores[label] = scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                              data['query_'+environment], data['gallery_'+environment])
    result['query_changes'] = [dict(query_index=i, identity=int(identity),
        control_ap=float(scores[control]['average_precision'][i]),
        candidate_ap=float(scores[candidate]['average_precision'][i]),
        control_first_rank=int(scores[control]['first_match_rank'][i]),
        candidate_first_rank=int(scores[candidate]['first_match_rank'][i]))
        for i, identity in enumerate(data['query_ids'])]
    delta = result['delta_metrics']
    return dict(dataset=dataset, candidate=candidate, control=control,
                actual_training_batch_order_equal=True,
                phase_progress=delta['mAP'] >= 0.5 and delta['Rank-1'] >= 0,
                paired_diagnosis=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    panel.configure()
    panel.base.require_sources(args.campaign)
    controls = panel.previous.require_controls()
    state = json.loads((args.campaign/'campaign.json').read_text())
    matrix = json.loads((args.campaign/'accepted_matrix.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1
    assert len(state['jobs']) == 12 and all(r['status'] == 'COMPLETE' and r['exit_code'] == 0 for r in state['jobs'])
    assert matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    combined = dict(rows=[dict(r, variant='raw_'+r['variant']) for r in controls['rows']]+
                         [dict(r, variant=panel.QUERY_MODES[r['variant']]) for r in matrix['rows']])
    rows, pairs = [], []
    for row in matrix['rows']:
        dataset, variant = row['dataset'], row['variant']
        assert panel.previous.accepted_row(args.campaign, dataset, variant) == row
        run = Path(row['run_dir'])
        training = json.loads((run/'training.json').read_text())
        steps = [json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(r['steps'] for r in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        for original in (r for r in combined['rows'] if r['dataset'] == dataset):
            assert (run/'training_batch_order.jsonl').read_bytes() == (Path(original['run_dir'])/'training_batch_order.jsonl').read_bytes()
        mode = panel.QUERY_MODES[variant]
        rows.append(dict(row, query_mode=mode, formal_steps=len(steps), history=training['history'],
            training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-
                datetime.fromisoformat(training['started_at'])).total_seconds()))
        for control in (('mean', 'raw_semantic', 'raw_global_only') if mode == 'patch'
                        else ('raw_semantic', 'raw_global_only')):
            pairs.append(query_comparison(combined, dataset, control, mode))
    result = dict(schema=panel.SCHEMA, status='COMPLETE', created_at=datetime.now().astimezone().isoformat(),
        accepted=6, formal_epochs=300, formal_steps=sum(r['formal_steps'] for r in rows), rows=rows, pairs=pairs,
        active_reconstruction_parameters=105232, active_reconstruction_tensors=15,
        primary_patch_vs_mean_progress_count=sum(r['phase_progress'] for r in pairs if r['control'] == 'mean'),
        boundary='Single seed42 on consumed official benchmarks. Same active parameters and original RAW-semantic pipeline; patch-specific routing versus broadcast mean context. No text resources or full SAGA reproduction; no seed stability, true-part correspondence or SOTA claim.')
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir/'SUMMARY.json', result)
    lines = ['# 区域候选重建：patch查询与均值广播对照', '', '| 数据集 | 查询 | best轮 | mAP | R1 | 正式步数 |',
             '|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["dataset"]} | {r["query_mode"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.4f} | {r["metrics"]["Rank-1"]:.4f} | {r["formal_steps"]} |' for r in rows]
    lines += ['', result['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE', accepted=6, pairs=len(pairs))), flush=True)


if __name__ == '__main__':
    main()
