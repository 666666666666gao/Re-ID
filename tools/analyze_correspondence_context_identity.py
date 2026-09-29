#!/usr/bin/env python3
"""Analyze the complete context/local panel; preserve explicitly supplied sealed pairs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.collect_correspondence_roles import METRICS
from tools.queue_correspondence_context_identity import CONDITIONS
from tools.queue_correspondence_refinement import DATASETS

CONTRASTS = (
    ('static_none', 'context_none', 'context_queries_without_auxiliary_identity'),
    ('static_local', 'context_local', 'context_queries_with_local_identity'),
    ('static_none', 'static_local', 'local_identity_with_static_queries'),
    ('context_none', 'context_local', 'local_identity_with_context_queries'),
    ('context_global', 'context_local', 'local_vs_global_extra_identity'),
)


def factor_metrics(rows):
    metrics = {row['variant']: row['metrics'] for row in rows}
    result = {name: {} for name in ('mean_query_effect', 'mean_local_identity_effect',
                                    'query_local_identity_interaction', 'local_vs_global_identity')}
    for name in METRICS:
        sn, cn, sl, cl, cg = (metrics[variant][name] for variant in CONDITIONS)
        result['mean_query_effect'][name] = ((cn - sn) + (cl - sl)) / 2
        result['mean_local_identity_effect'][name] = ((sl - sn) + (cl - cn)) / 2
        result['query_local_identity_interaction'][name] = (cl - sl) - (cn - sn)
        result['local_vs_global_identity'][name] = cl - cg
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--sealed-pair', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
    assert matrix['schema'] == 'trifusion-correspondence-context-identity-verification-v1'
    assert matrix['expected_endpoints'] == matrix['verified_complete'] == len(matrix['rows']) == 15
    assert matrix['seed'] == 42
    rows = {(row['dataset'], row['variant']): row for row in matrix['rows']}
    assert set(rows) == {(dataset, variant) for dataset in DATASETS for variant in CONDITIONS}
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in rows.values())
    for dataset in DATASETS:
        group = [rows[(dataset, variant)] for variant in CONDITIONS]
        assert len({row['initial_model_state_sha256'] for row in group}) == 1
        trainings = [json.loads((Path(row['run_dir']) / 'training.json').read_text()) for row in group]
        first = trainings[0]
        for row, training in zip(group, trainings):
            assert sha(Path(row['run_dir']) / 'best_map.pth') == row['checkpoint_sha256']
            assert sha(Path(row['run_dir']) / 'official_metrics.json') == row['receipt_sha256']
            assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
            assert training['epochs'] == 50 and training['seed'] == 42
            assert [item['epoch'] for item in training['history']] == list(range(1, 51))
            for name in ('dataset', 'protocol_sha256', 'checkpoint_policy'):
                assert training[name] == first[name]
            for name in ('author_checkpoint_sha256', 'width', 'fused_width', 'm1', 'm2', 'm3',
                         'learning_rate', 'weight_decay', 'initial_model_state_sha256',
                         'model_source_sha256', 'entry_sha256', 'context_source_sha256',
                         'evidence_source_sha256', 'auxiliary_id_weight', 'query_context_width'):
                assert training['initializer'][name] == first['initializer'][name]
            query, target = CONDITIONS[row['variant']]
            assert row['condition'] == training['initializer']['condition'] == {
                'query_mode': query, 'auxiliary_target': target}
        assert group[0]['trainable_parameters'] == group[1]['trainable_parameters']
        assert len({row['trainable_parameters'] for row in group[2:]}) == 1
    sealed = {}
    for path in args.sealed_pair:
        report = json.loads(path.read_text(encoding='utf-8'))
        key = (report['dataset'], report['control'], report['candidate'])
        assert key not in sealed
        assert report['status'] == 'CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE'
        assert (key[1], key[2]) in {(control, candidate) for control, candidate, _ in CONTRASTS}
        for variant in key[1:]:
            row = rows[(key[0], variant)]
            assert report['bindings'][variant] == {name: row[name] for name in
                ('best_epoch', 'checkpoint_sha256', 'distance_sha256', 'receipt_sha256')}
            assert all(abs(report['metrics'][variant][name] - row['metrics'][name]) < 1e-5
                       for name in METRICS)
        sealed[key] = (path, report)
    args.output_dir.mkdir()
    summary = {'status': 'COMPLETE_CONTEXT_LOCAL_15_ENDPOINT_DIAGNOSIS',
               'at': datetime.now().astimezone().isoformat(),
               'matrix': str(args.matrix), 'matrix_sha256': sha(args.matrix),
               'source_sha256': sha(Path(__file__)),
               'comparison_source_sha256': sha(ROOT / 'tools/analyze_correspondence_distances.py'),
               'datasets': {}, 'pairs': [],
               'boundary': 'All fifteen full50/single mAP-best/reload endpoints required. Explicitly supplied '
                           'sealed pairs retain original bytes and bindings and are not recomputed. '
                           'Fixed single seed and official post-selection diagnosis; bootstrap uses fixed-model '
                           'identities, not training seeds or independent tests. Equal stored query parameters '
                           'do not imply equal effective capacity. Local supervision can also update shared '
                           'adapters and context queries. No new inference, GPU, weights, seed or configuration selection.'}
    for dataset in DATASETS:
        group = [rows[(dataset, variant)] for variant in CONDITIONS]
        summary['datasets'][dataset] = {'factor_metrics_pp': factor_metrics(group),
            'cells': [{name: row[name] for name in ('variant', 'best_epoch', 'metrics',
                      'diagnostic_metrics', 'trainable_parameters', 'training_and_epoch_eval_seconds')}
                      for row in group]}
        for control, candidate, question in CONTRASTS:
            key = (dataset, control, candidate)
            if key in sealed:
                path, report = sealed[key]
                origin = 'sealed_report_reused'
            else:
                report = compare(matrix, dataset, control, candidate)
                path = args.output_dir / f'{dataset}_{control}_to_{candidate}.json'
                path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                origin = 'first_cpu_distance_comparison'
            summary['pairs'].append({**{name: value for name, value in report.items() if name != 'identity_changes'},
                                     'question': question, 'origin': origin,
                                     'report': str(path), 'report_sha256': sha(path)})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'at': summary['at'], 'datasets': 3,
                      'pairs': len(summary['pairs']), 'sealed_reused': len(sealed)}))


if __name__ == '__main__':
    main()
