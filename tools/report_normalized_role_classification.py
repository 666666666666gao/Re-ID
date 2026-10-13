"""One CPU saved-distance report after the original201 strict completion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_normalized_role_classification as panel
from tools.analyze_correspondence_distances import camera_scores, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    panel.base.SCHEMA = panel.SCHEMA
    panel.base.source_map = panel.source_map
    panel.base.output_dir = panel.previous.output_dir
    panel.base.require_sources(args.campaign)
    state = json.loads((args.campaign/'campaign.json').read_text())
    assert state['status'] == 'COMPLETE'
    assert [row['mode'] for row in state['steps']] == ['prepare', 'm0', 'train', 'evaluate']
    assert all(row['exit_code'] == 0 and row['status'] == 'COMPLETE' for row in state['steps'])
    accepted = json.loads((args.campaign/'ACCEPTED_RESULT.json').read_text())
    assert panel.base.verify(args.campaign, 'RGBNT201', 'semantic') == accepted
    candidate = dict(accepted, variant='normalized_role_ce')
    controls = json.loads(panel.CONTROL.read_text())
    control_batches = json.loads((ROOT/'refine-logs/normalized_role_training_20261013_938/REPORT_CONTROL_BATCHES_940.json').read_text())
    selected = [row for row in controls['rows'] if row['dataset'] == 'RGBNT201'
                and row['variant'] in ('semantic', 'global_only')]
    assert len(selected) == 2
    matrix = {'rows': [dict(row, variant='raw_'+row['variant']) for row in selected]+[candidate]}
    run = Path(accepted['run_dir'])
    training = json.loads((run/'training.json').read_text())
    steps = [json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
    batches = [json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
    assert len(steps) == len(batches) == sum(row['steps'] for row in training['history'])
    assert [(row['epoch'], row['batch']) for row in steps] == [(row['epoch'], row['batch']) for row in batches]
    data = torch.load(run/'official_distances.pt', map_location='cpu', weights_only=False)
    scores = camera_scores(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                           data['query_cameras'], data['gallery_cameras'])
    assert all(abs(scores['metrics'][key]-accepted['metrics'][key]) < 1e-5 for key in accepted['metrics'])
    pairs = []
    for control in ('raw_semantic', 'raw_global_only'):
        old = next(row for row in matrix['rows'] if row['variant'] == control)
        old_batch = Path(old['run_dir'])/'training_batch_order.jsonl'
        assert panel.base.sha(old_batch) == control_batches['files'][str(old_batch)]['sha256']
        assert old_batch.read_bytes() == (run/'training_batch_order.jsonl').read_bytes()
        paired = compare(matrix, 'RGBNT201', control, 'normalized_role_ce')
        previous = torch.load(Path(old['run_dir'])/'official_distances.pt', map_location='cpu', weights_only=False)
        previous_scores = camera_scores(previous['fused'].numpy(), previous['query_ids'], previous['gallery_ids'],
                                         previous['query_cameras'], previous['gallery_cameras'])
        paired['query_changes'] = [dict(query_index=index, identity=int(identity),
            control_ap=float(previous_scores['average_precision'][index]), candidate_ap=float(scores['average_precision'][index]),
            control_first_rank=int(previous_scores['first_match_rank'][index]),
            candidate_first_rank=int(scores['first_match_rank'][index])) for index, identity in enumerate(data['query_ids'])]
        delta = paired['delta_metrics']
        pairs.append(dict(control=control, batch_order_equal=True, phase_progress=delta['mAP'] >= .5 and delta['Rank-1'] >= 0,
                          paired_diagnosis=paired))
    result = dict(schema=panel.SCHEMA, status='COMPLETE_SAVED_DISTANCE_DIAGNOSIS', at=datetime.now().astimezone().isoformat(),
        accepted_result=accepted, formal_epochs=50, formal_steps=len(steps), history=training['history'], pairs=pairs,
        full_training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
        full_signal_numeric_reference={'mAP':80.3029,'Rank-1':85.1675,'source':'Existing local complete Signal evaluation; different model/initialization, not this single-factor control'},
        boundary='First fixed seed42 development comparison on consumed official benchmarks. Primary raw semantic and independent shared-adapter global controls reuse sealed saved distances. Existing camera/gallery rules intact; no NN, new ranking rule, seed average, test centering, tuning or causal claim. Identity bootstrap in reused compare is fixed-model descriptive only, not training-seed uncertainty.')
    args.output_dir.mkdir()
    panel.base.queue.write(args.output_dir/'SUMMARY.json', result)
    lines = ['# 角色分类入口单项完整结果', '',
             f'201 seed42，同一mAP-best epoch{accepted["best_epoch"]}，正式50轮/{len(steps)}步。', '',
             '| 指标 | mAP | R1 | R5 | R10 |', '|---|---:|---:|---:|---:|',
             '| candidate | '+' | '.join(f'{accepted["metrics"][key]:.4f}' for key in ('mAP','Rank-1','Rank-5','Rank-10'))+' |', '',
             '| 对照 | ΔmAP | ΔR1 | 修复 | 新增错误 | 身份宏APΔ | 推进 |', '|---|---:|---:|---:|---:|---:|---|']
    for pair in pairs:
        value = pair['paired_diagnosis'];delta = value['delta_metrics']
        lines.append(f'| {pair["control"]} | {delta["mAP"]:+.4f} | {delta["Rank-1"]:+.4f} | {value["rank1_repairs"]} | {value["rank1_new_errors"]} | {value["identity_macro_mean_delta_ap_points"]:+.4f} | {pair["phase_progress"]} |')
    lines += ['', result['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],metrics=accepted['metrics'],formal_steps=len(steps),pairs=len(pairs))))


if __name__ == '__main__':
    main()
