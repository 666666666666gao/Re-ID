"""Saved-distance attribution report after both first strict completions."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_pretrained_value_delta as panel
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
    state = json.loads((args.campaign / 'campaign.json').read_text())
    assert state['status'] == 'COMPLETE' and state['accepted'] == state['expected'] == 2
    controls = json.loads(panel.CONTROL.read_text())
    batches = json.loads((ROOT / 'refine-logs/normalized_role_training_20261013_938/REPORT_CONTROL_BATCHES_940.json').read_text())
    rows = [dict(row, variant='raw_' + row['variant']) for row in controls['rows']
            if row['dataset'] == 'RGBNT201' and row['variant'] in ('semantic', 'global_only')]
    assert len(rows) == 2
    history, batch_bytes = {}, None
    for mode, arm in zip(panel.MODES, state['arms']):
        child = Path(arm['campaign'])
        assert arm['status'] == 'COMPLETE' and arm['value_reference'] == mode
        assert [row['mode'] for row in arm['steps']] == ['prepare', 'm0', 'train', 'evaluate']
        assert all(row['exit_code'] == 0 and row['status'] == 'COMPLETE' for row in arm['steps'])
        panel.base.require_sources(child)
        accepted = panel.base.verify(child, 'RGBNT201', 'semantic')
        assert dict(accepted, value_reference=mode) == arm['result']
        run = Path(accepted['run_dir'])
        receipt = json.loads((run / 'training.json').read_text())
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        order = (run / 'training_batch_order.jsonl').read_bytes()
        order_rows = [json.loads(line) for line in order.splitlines()]
        assert len(steps) == len(order_rows) == sum(row['steps'] for row in receipt['history'])
        assert [(row['epoch'], row['batch']) for row in steps] == [(row['epoch'], row['batch']) for row in order_rows]
        if batch_bytes is None:
            batch_bytes = order
        assert batch_bytes == order
        data = torch.load(run / 'official_distances.pt', map_location='cpu', weights_only=False)
        scores = camera_scores(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                               data['query_cameras'], data['gallery_cameras'])
        assert all(abs(scores['metrics'][key] - accepted['metrics'][key]) < 1e-5 for key in accepted['metrics'])
        rows.append(dict(accepted, variant=mode))
        history[mode] = dict(history=receipt['history'], formal_steps=len(steps),
            full_training_and_epoch_evaluation_seconds=(datetime.fromisoformat(receipt['completed_at']) - datetime.fromisoformat(receipt['started_at'])).total_seconds())
    for row in rows[:2]:
        path = Path(row['run_dir']) / 'training_batch_order.jsonl'
        assert panel.base.sha(path) == batches['files'][str(path)]['sha256']
        assert path.read_bytes() == batch_bytes
    comparisons = [('uniform_reference', 'semantic_reference')]
    comparisons += [(control, candidate) for control in ('raw_semantic', 'raw_global_only') for candidate in panel.MODES]
    paired = []
    matrix = dict(rows=rows)
    for control, candidate in comparisons:
        value = compare(matrix, 'RGBNT201', control, candidate)
        values = []
        for name in (control, candidate):
            row = next(row for row in rows if row['variant'] == name)
            data = torch.load(Path(row['run_dir']) / 'official_distances.pt', map_location='cpu', weights_only=False)
            scores = camera_scores(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                                   data['query_cameras'], data['gallery_cameras'])
            values.append((data, scores))
        old, new = values
        value['query_changes'] = [dict(query_index=index, identity=int(identity),
            control_ap=float(old[1]['average_precision'][index]), candidate_ap=float(new[1]['average_precision'][index]),
            control_first_rank=int(old[1]['first_match_rank'][index]), candidate_first_rank=int(new[1]['first_match_rank'][index]))
            for index, identity in enumerate(new[0]['query_ids'])]
        delta = value['delta_metrics']
        paired.append(dict(control=control, candidate=candidate, batch_order_equal=True,
                           phase_progress=delta['mAP'] >= .5 and delta['Rank-1'] >= 0, paired_diagnosis=value))
    result = dict(schema=panel.SCHEMA, status='COMPLETE_SAVED_DISTANCE_DIAGNOSIS',
                  at=datetime.now().astimezone().isoformat(), rows=rows, pairs=paired, histories=history,
                  formal_epochs_per_arm=50, formal_conditions=2,
                  full_signal_numeric_reference={'mAP':80.3029, 'Rank-1':85.1675,
                    'source':'Existing complete Signal local evaluation; different system/initialization, not a single-factor comparison'},
                  boundary='First fixed seed42 consumed-benchmark development comparison. Two same-parameter value reading references, semantic versus uniform; not a novelty proof. Value source, readout and parameter count differ from old semantic. Original camera/gallery rules; no NN/new ranking/test centering/tuning. Fixed-model identity bootstrap is descriptive, not training-seed uncertainty.')
    args.output_dir.mkdir()
    panel.base.queue.write(args.output_dir / 'SUMMARY.json', result)
    lines = ['# 预训练value增量读取：两端完整结果', '',
             '| 条件 | best | mAP | R1 | R5 | R10 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["best_epoch"]} | ' + ' | '.join(f'{row["metrics"][key]:.4f}' for key in ('mAP','Rank-1','Rank-5','Rank-10')) + ' |')
    lines += ['', '| 对照→候选 | ΔmAP | ΔR1 | 修复 | 新增错误 | 身份宏APΔ | 推进 |', '|---|---:|---:|---:|---:|---:|---|']
    for pair in paired:
        value = pair['paired_diagnosis']; delta = value['delta_metrics']
        lines.append(f'| {pair["control"]}→{pair["candidate"]} | {delta["mAP"]:+.4f} | {delta["Rank-1"]:+.4f} | {value["rank1_repairs"]} | {value["rank1_new_errors"]} | {value["identity_macro_mean_delta_ap_points"]:+.4f} | {pair["phase_progress"]} |')
    lines += ['', result['boundary']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'],conditions=2,pairs=len(paired))))


if __name__ == '__main__':
    main()
