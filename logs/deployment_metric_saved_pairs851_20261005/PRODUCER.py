"""Compare three completed matched pairs using their sealed saved texts only."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

BASE = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
OUTPUT = BASE / 'deployment_metric_saved_pair_cross_dataset_20261005'
REMOTE = '/data/gaob/Re-ID/Trifusion/'
METRICS = ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')
PAIRS = (
    ('RGBNT201', 'semantic', 'deployment_metric_RGBNT201_semantic_full839', 'global_task_role_first_full826', 2649),
    ('RGBNT201', 'native', 'deployment_metric_RGBNT201_native_full841', 'global_task_role_native_full827', 2649),
    ('MSVR310', 'semantic', 'deployment_metric_MSVR310_semantic_full842', 'global_task_role_msvr_semantic_full828', 706),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    seal_path = REPO / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
    assert sha(seal_path) == '9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
    seal = json.loads(seal_path.read_bytes())
    panel, tables = [], {}
    inputs = {str(seal_path): sha(seal_path)}
    for dataset, variant, new_name, old_name, steps in PAIRS:
        packet = BASE / new_name
        summary_path = packet / 'SUMMARY.json'
        summary = json.loads(summary_path.read_bytes())
        assert summary['status'] == 'ORIGINAL_FRESH50_FIRST_STRICT_AND_PROBE_RETIREMENT_VERIFIED'
        assert summary['actual_training_batch_order_equal'] is True
        row = summary['row']
        assert (row['dataset'], row['variant']) == (dataset, variant)
        training_path = packet / 'received' / row['run_dir'].removeprefix(REMOTE) / 'training.json'
        acceptance_path = packet / 'received/logs/deployment_metric_role_v1_20261005_837/acceptance' / f'{dataset}_{variant}.json'
        acceptance = json.loads(acceptance_path.read_bytes())
        assert sha(acceptance_path) == summary['full_acceptance_sha256']
        assert sha(training_path) == acceptance['artifact_sha256'][row['run_dir'] + '/training.json']
        new_training = json.loads(training_path.read_bytes())
        control = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
        transport_path = BASE / old_name / 'stdout.json'
        transport = json.loads(transport_path.read_bytes())
        old_name_abs = control['run_dir'] + '/training.json'
        old_bytes = transport['files'][old_name_abs.removeprefix(REMOTE)].encode('utf-8')
        assert hashlib.sha256(old_bytes).hexdigest() == seal['artifact_sha256'][old_name_abs]
        old_training = json.loads(old_bytes)
        new_tasks = summary['training_task_and_norm_trajectory']
        old_tasks = transport['summary']['training_task_and_norm_trajectory']
        for series in (new_training['history'], old_training['history'], new_tasks, old_tasks):
            assert [r['epoch'] for r in series] == list(range(1, 51))
        best = row['best_epoch']
        assert best == new_training['best_epoch'] == old_training['best_epoch'] == control['best_epoch']
        records = []
        for new_h, old_h, new_t, old_t in zip(new_training['history'], old_training['history'], new_tasks, old_tasks):
            assert new_h['steps'] == old_h['steps'] == new_t['steps'] == old_t['steps']
            record = dict(dataset=dataset, variant=variant, epoch=new_t['epoch'], steps=new_t['steps'],
                new_global_loss=new_t['global_loss'], old_global_loss=old_t['mean_global_loss'],
                new_global_norm=new_t['shared_global_norm_mean'], old_global_norm=old_t['shared_global_norm_mean'],
                new_role_loss=new_t['fused_role_loss'], old_role_loss=old_t['mean_fused_role_loss'],
                new_correction_norm=new_t['correction_norm_mean'], old_correction_norm=old_t['correction_norm_mean'],
                new_scaled_correction_norm=new_t['actual_scaled_correction_norm_mean'], old_scaled_correction_norm=old_t['actual_scaled_correction_norm_mean'],
                new_correction_global_ratio=new_t['actual_scaled_correction_global_ratio_mean'], old_correction_global_ratio=old_t['actual_scaled_correction_global_ratio_mean'],
                new_role_metric_norm=new_t['role_metric_norm_mean'])
            for metric in METRICS:
                record['new_' + metric] = new_h['official_fused'][metric]
                record['old_' + metric] = old_h['official_fused'][metric]
                record['delta_' + metric] = record['new_' + metric] - record['old_' + metric]
            assert all(math.isfinite(v) for k, v in record.items() if k not in ('dataset', 'variant'))
            records.append(record)
        assert sum(r['steps'] for r in records) == steps == summary['formal_steps']
        for metric in METRICS:
            assert records[best - 1]['new_' + metric] == row['metrics'][metric]
            assert records[best - 1]['old_' + metric] == control['metrics'][metric]
        key = dataset + '_' + variant
        tables[key] = records
        panel.append(dict(dataset=dataset, variant=variant, formal_epochs=50, formal_steps=steps,
            selected_epoch=best, selected_new_metrics=row['metrics'], selected_control_metrics=control['metrics'],
            registered_gate=summary['deltas'][variant]['phase_progress'],
            global_loss_max_abs_difference=max(abs(r['new_global_loss']-r['old_global_loss']) for r in records),
            global_norm_max_abs_difference=max(abs(r['new_global_norm']-r['old_global_norm']) for r in records),
            epoch_delta_counts={m: dict(positive=sum(r['delta_' + m] > 0 for r in records), equal=sum(r['delta_' + m] == 0 for r in records), negative=sum(r['delta_' + m] < 0 for r in records)) for m in METRICS},
            epoch_map_positive_rank1_nonnegative=sum(r['delta_mAP'] > 0 and r['delta_Rank-1'] >= 0 for r in records),
            new_best_to_last_map_drop=records[best-1]['new_mAP']-records[-1]['new_mAP'],
            old_best_to_last_map_drop=records[best-1]['old_mAP']-records[-1]['old_mAP'],
            selected_epoch_record=records[best-1], last_epoch_record=records[-1],
            role_metric_norm_range=[min(r['new_role_metric_norm'] for r in records), max(r['new_role_metric_norm'] for r in records)],
            matched_control_training_sha256=hashlib.sha256(old_bytes).hexdigest()))
        for path in (summary_path, training_path, acceptance_path, transport_path):
            inputs[str(path)] = sha(path)
    boundary = ('Three completed matched pairs, two datasets, 150 dependent epochs and 6004 current formal steps; '
        'source/receipts and original best checkpoints unchanged. Local sealed text only: no SSH, PyTorch, NN, '
        'optimizer, native-progress query, scientific-source change or post-hoc epoch/seed selection. '
        'Recorded global objective/norm equality is aggregate scalar equality, not full-state or retrieval-feature identity. '
        'Correction ratios describe training batches, not query evidence or a causal explanation. '
        'Role losses aggregate weighted CE and Triplet and change metric geometry; their magnitudes do not compare fitting quality. '
        'Dependent epochs are not independent seeds or statistical significance. '
        'RGBNT100 semantic is separately archived in section849; its native is still pending. '
        'Original MSVR310 native M0 failure remains missing, not retried or assigned a score. '
        'Finish the available formal panel and fixed-best decomposition before selecting the next intervention.')
    analysis = dict(status='THREE_COMPLETED_MATCHED_SAVED_PAIRS_ANALYSIS_COMPLETE',
        at=datetime.now().astimezone().isoformat(), producer_sha256=sha(Path(__file__)),
        accepted_pairs=3, datasets=2, epochs=150, formal_steps=6004, pairs=panel, inputs=inputs, boundary=boundary)
    OUTPUT.mkdir()
    for key, records in tables.items():
        with (OUTPUT / (key + '.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (OUTPUT / 'ANALYSIS.json').write_bytes((json.dumps(analysis, indent=2)+'\n').encode('utf-8'))
    lines = ['# Completed deployment-metric pairs: saved full50 trajectories', '', boundary, '',
        '| Dataset/variant | New best-to-last mAP drop | Raw best-to-last drop | New c/g at selected epoch | Raw c/g at selected epoch | New c/g E50 | Raw c/g E50 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in panel:
        best, last = item['selected_epoch_record'], item['last_epoch_record']
        lines.append(f"| {item['dataset']}/{item['variant']} | {item['new_best_to_last_map_drop']:.6f} | {item['old_best_to_last_map_drop']:.6f} | {best['new_correction_global_ratio']:.6f} | {best['old_correction_global_ratio']:.6f} | {last['new_correction_global_ratio']:.6f} | {last['old_correction_global_ratio']:.6f} |")
    (OUTPUT / 'README.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(status=analysis['status'], at=analysis['at'], pairs=[{
        k: item[k] for k in ('dataset','variant','global_loss_max_abs_difference','global_norm_max_abs_difference',
            'epoch_delta_counts','new_best_to_last_map_drop','old_best_to_last_map_drop')
        } for item in panel]), ensure_ascii=False))


if __name__ == '__main__':
    main()
