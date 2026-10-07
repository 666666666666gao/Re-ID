"""Full-query H2a report: matched baseline and same-weight global decomposition."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_prepool_dense_correspondence as panel
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def paired(dataset, control, candidate):
    arrays, scores = {}, {}
    environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    for row in (control, candidate):
        path = Path(row['distance_path'])
        assert panel.base.sha(path) == row['distance_sha256']
        data = torch.load(path, map_location='cpu', weights_only=False)
        arrays[row['name']] = data
        value = scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                       data['query_' + environment], data['gallery_' + environment])
        assert all(abs(value['metrics'][name] - row['metrics'][name]) < 1e-5 for name in row['metrics'])
        scores[row['name']] = value
    first_array, second_array = arrays[control['name']], arrays[candidate['name']]
    for name in ('query_ids', 'gallery_ids', 'query_cameras', 'gallery_cameras', 'query_scenes', 'gallery_scenes'):
        assert np.array_equal(first_array[name], second_array[name])
    assert first_array['fused'].shape == second_array['fused'].shape
    first, second = scores[control['name']], scores[candidate['name']]
    first_rank, second_rank = (np.asarray(value['first_match_rank']) for value in (first, second))
    delta_ap = np.asarray(second['average_precision']) - np.asarray(first['average_precision'])
    repairs = int(((first_rank != 1) & (second_rank == 1)).sum())
    new_errors = int(((first_rank == 1) & (second_rank != 1)).sum())
    delta = {name: second['metrics'][name] - first['metrics'][name] for name in first['metrics']}
    assert abs(delta['Rank-1'] - (repairs - new_errors) * 100 / len(delta_ap)) < 1e-5
    identities = first_array['query_ids']
    changes = [dict(identity=int(identity), queries=int((identities == identity).sum()),
        mean_delta_ap_points=float(delta_ap[identities == identity].mean() * 100)) for identity in np.unique(identities)]
    return dict(dataset=dataset, control=control['name'], candidate=candidate['name'],
        metrics={control['name']: first['metrics'], candidate['name']: second['metrics']},
        delta_metrics=delta, rank1_repairs=repairs, rank1_new_errors=new_errors,
        query_ap_improved=int((delta_ap > 1e-8).sum()), query_ap_worsened=int((delta_ap < -1e-8).sum()),
        identity_macro_mean_delta_ap_points=float(np.mean([row['mean_delta_ap_points'] for row in changes])),
        identity_changes=changes,
        query_changes=[dict(query_index=index, identity=int(identity),
            control_ap=float(first['average_precision'][index]), candidate_ap=float(second['average_precision'][index]),
            control_first_rank=int(first_rank[index]), candidate_first_rank=int(second_rank[index]))
            for index, identity in enumerate(identities)],
        boundary='Complete legal query/gallery ground truth, same camera/scene filtering. Post-selection fixed-model diagnosis, not training seed uncertainty or causal correspondence proof.')


def descriptor(row, name, own=False):
    return dict(name=name, metrics=row['own_global_metrics'] if own else row['metrics'],
        distance_path=str(Path(row['run_dir']) / ('own_global_distances.pt' if own else 'official_distances.pt')),
        distance_sha256=row['own_global_distance_sha256'] if own else row['distance_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    panel.base.source_map = panel.source_map
    manifest = panel.base.require_sources(args.campaign)
    controls = json.loads(panel.CONTROLS.read_text())
    assert panel.base.sha(panel.CONTROLS) == manifest['control_seal_sha256']
    assert all(panel.base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    state = json.loads((args.campaign / 'campaign.json').read_text())
    matrix = json.loads((args.campaign / 'accepted_matrix.json').read_text())
    assert state['status'] in ('COMPLETE', 'COMPLETE_WITH_MISSING') and state['report_invocations'] == 1
    assert len(state['jobs']) == matrix['expected'] == 3
    assert len(matrix['rows']) == matrix['accepted']
    assert {job['dataset'] for job in state['jobs']} == set(panel.DATASETS)
    assert len(matrix['rows']) + len(matrix['missing']) == 3
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    rows, pairs = [], []
    for row in matrix['rows']:
        dataset = row['dataset']
        assert panel.accepted_row(args.campaign, dataset) == row
        run = Path(row['run_dir'])
        training = json.loads((run / 'training.json').read_text())
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (run / 'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(value['steps'] for value in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        traces = []
        for epoch in range(1, 51):
            selected = [step for step in steps if step['epoch'] == epoch]
            assert selected
            traces.append(dict(epoch=epoch,
                auxiliary_mean=float(np.mean([step['dense_correspondence_loss'] for step in selected])),
                match_mass_by_role=np.mean([step['dense_match_mass_by_role'] for step in selected], axis=0).tolist(),
                actual_full_read_entropy_by_role=np.mean([step['actual_read_entropy_by_role'] for step in selected], axis=0).tolist(),
                actual_visible_mass_mapped_l1_by_role=np.mean([step['actual_read_mapped_l1_by_role'] for step in selected], axis=0).tolist()))
        rows.append(dict(row, formal_steps=len(steps), history=training['history'], auxiliary_traces=traces,
            production_training_interval_seconds=(datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds(),
            partition_peak_memory=training['partition_peak_memory']))
        raw = next(value for value in controls['rows'] if value['dataset'] == dataset and value['variant'] == 'semantic')
        independent = next(value for value in controls['rows'] if value['dataset'] == dataset and value['variant'] == 'global_only')
        dense, own = descriptor(row, 'dense'), descriptor(row, 'own_global', own=True)
        raw, independent = descriptor(raw, 'raw_semantic'), descriptor(independent, 'independent_global')
        for first, second, primary in ((raw, dense, True), (independent, dense, False),
                                        (own, dense, False), (independent, own, False)):
            result = paired(dataset, first, second)
            result['primary_pair'] = primary
            result['phase_progress'] = result['delta_metrics']['mAP'] >= 0.5 and result['delta_metrics']['Rank-1'] >= 0
            pairs.append(result)
    assert len(pairs) == len(rows) * 4
    value = dict(schema=panel.SCHEMA, status='COMPLETE', created_at=datetime.now().astimezone().isoformat(),
        accepted=len(rows), expected=3, formal_epochs=50 * len(rows), formal_steps=sum(row['formal_steps'] for row in rows),
        rows=rows, pairs=pairs, missing=matrix['missing'],
        primary_advance=sum(pair['primary_pair'] and pair['phase_progress'] for pair in pairs),
        boundary='Fixed same-image same-modality geometry auxiliary package; zero new model parameters, original raw tasks/global ownership/1536 inference. Full50 on consumed official development benchmarks. Correspondence/entropy are proxies, not identity mechanism proof. Missing endpoints have no substituted score; no full-pipeline seeds, cross-spectral true parts, formula novelty or SOTA claim.')
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir / 'SUMMARY.json', value)
    lines = ['# 同模态几何辅助目标：三端结果', '',
             '| 数据集 | best轮 | mAP | R1 | own-global mAP | 末轮mAP | 正式更新 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f'| {row["dataset"]} | {row["best_epoch"]} | {row["metrics"]["mAP"]:.4f} | {row["metrics"]["Rank-1"]:.4f} | {row["own_global_metrics"]["mAP"]:.4f} | {row["history"][-1]["official_fused"]["mAP"]:.4f} | {row["formal_steps"]} |' for row in rows]
    lines += ['', f'主要配对过线：{value["primary_advance"]}/3；正式端完成：{len(rows)}/3。', '', value['boundary']]
    for item in matrix['missing']:
        lines += ['', f'缺失：{item["dataset"]}，停止阶段 {item["failed_phase"]}，无替代分数。']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='COMPLETE', accepted=len(rows), expected=3,
        primary_advance=value['primary_advance'], pairs=len(pairs), formal_steps=value['formal_steps'])), flush=True)


if __name__ == '__main__':
    main()
