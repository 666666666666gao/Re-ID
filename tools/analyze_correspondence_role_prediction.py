#!/usr/bin/env python3
"""Analyze four accepted M3 cells using saved complete-gallery distances."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_correspondence_distances import compare, sha
from tools.collect_correspondence_roles import METRICS
from tools.queue_correspondence_role_prediction import CONDITIONS
from tools.queue_correspondence_refinement import DATASETS


CONTRASTS = (
    ('own_direct', 'matched_direct', 'address_matching_with_direct_regression'),
    ('own_predictor', 'matched_predictor', 'address_matching_with_predictors'),
    ('own_direct', 'own_predictor', 'predictors_with_own_addresses'),
    ('matched_direct', 'matched_predictor', 'predictors_with_matched_addresses'),
    ('original_111', 'own_direct', 'unchanged_forward_and_loss_replication'),
    ('global_only', 'own_direct', 'role_path_beyond_independent_global_control'),
)


def factor_metrics(rows):
    metrics = {row['variant']: row['metrics'] for row in rows}
    result = {name: {} for name in ('mean_address_effect', 'mean_predictor_effect', 'address_predictor_interaction')}
    for name in METRICS:
        od, md, op, mp = (metrics[v][name] for v in CONDITIONS)
        result['mean_address_effect'][name] = ((md - od) + (mp - op)) / 2
        result['mean_predictor_effect'][name] = ((op - od) + (mp - md)) / 2
        result['address_predictor_interaction'][name] = (mp - op) - (md - od)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--prior-matrix', type=Path, required=True)
    parser.add_argument('--original-matrix', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    matrix, prior, original = (json.loads(p.read_text()) for p in
                               (args.matrix, args.prior_matrix, args.original_matrix))
    assert matrix['schema'] == 'trifusion-correspondence-m3-panel-verification-v1'
    assert matrix['expected_endpoints'] == len(matrix['rows']) == 12
    assert prior['expected_endpoints'] == prior['verified_complete'] == len(prior['rows']) == 18
    assert original['expected_endpoints'] == original['verified_complete'] == len(original['rows']) == 24
    rows = [r for r in matrix['rows'] if r['dataset'] == args.dataset and r['phase'] == 'm3']
    assert len(rows) == 4 and {r['variant'] for r in rows} == set(CONDITIONS)
    assert all(r['status'] == 'VERIFIED_COMPLETE' for r in rows)
    ordered = {r['variant']: r for r in rows}
    rows = [ordered[v] for v in CONDITIONS]
    trainings = [json.loads((Path(r['run_dir']) / 'training.json').read_text()) for r in rows]
    first = trainings[0]
    for row, training in zip(rows, trainings):
        for name in ('dataset', 'protocol_sha256', 'seed', 'epochs', 'checkpoint_policy'):
            assert training[name] == first[name]
        for name in ('author_checkpoint_sha256', 'width', 'fused_width', 'm1', 'm2', 'm3',
                     'prediction_weight', 'learning_rate', 'weight_decay', 'model_source_sha256',
                     'initial_model_state_sha256', 'entry_sha256', 'prediction_source_sha256',
                     'reused_entry_sha256', 'prediction_head_parameters'):
            assert training['initializer'][name] == first['initializer'][name]
        address, prediction = CONDITIONS[row['variant']]
        assert training['initializer']['prediction'] == {'address_mode': address, 'prediction_mode': prediction}
        assert row['trainable_parameters'] == training['initializer']['trainable_parameters']
    for direct, predictor in (('own_direct', 'own_predictor'), ('matched_direct', 'matched_predictor')):
        assert ordered[predictor]['trainable_parameters'] - ordered[direct]['trainable_parameters'] == 100608
    assert ordered['own_direct']['trainable_parameters'] == ordered['matched_direct']['trainable_parameters']
    old = next(r for r in original['rows'] if r['dataset'] == args.dataset and r['variant'] == '111')
    global_row = next(r for r in prior['rows'] if r['dataset'] == args.dataset and r['phase'] == 'global')
    assert old['status'] == global_row['status'] == 'VERIFIED_COMPLETE'
    controls = [{**old, 'variant': 'original_111'}, {**global_row, 'variant': 'global_only'}]
    for row in controls:
        training = json.loads((Path(row['run_dir']) / 'training.json').read_text())
        for name in ('dataset', 'protocol_sha256', 'seed', 'epochs', 'checkpoint_policy'):
            assert training[name] == first[name]
        assert training['initializer']['author_checkpoint_sha256'] == first['initializer']['author_checkpoint_sha256']
    summary = {'status': 'COMPLETE_FOUR_CELL_M3_FACTOR_DIAGNOSIS', 'at': datetime.now().astimezone().isoformat(),
               'dataset': args.dataset, 'source_sha256': sha(Path(__file__)),
               'matrix': str(args.matrix), 'matrix_sha256': sha(args.matrix),
               'prior_matrix': str(args.prior_matrix), 'prior_matrix_sha256': sha(args.prior_matrix),
               'original_matrix': str(args.original_matrix), 'original_matrix_sha256': sha(args.original_matrix),
               'factor_metrics_pp': factor_metrics(rows),
               'cells': [{k: r[k] for k in ('variant', 'best_epoch', 'metrics', 'trainable_parameters',
                                           'training_and_epoch_eval_seconds')} for r in rows],
               'delta_from_global_only_pp': {r['variant']: {name: r['metrics'][name] - global_row['metrics'][name]
                                                           for name in METRICS} for r in rows},
               'pairs': [],
               'boundary': 'Only a dataset with all four accepted full50/best/reload endpoints is analyzed. '
                           'Exploratory single-seed official post-selection comparison; not independent test '
                           'or training-seed uncertainty. Each cell keeps all metrics from its own one mAP-best. '
                           'Address matching also changes local-mask support and removes prediction gradients '
                           'to student offsets. Predictors add 100608 trainable parameters. Global/original '
                           'controls have different capacity or saved state layout. Identity bootstrap uses '
                           'fixed-model identities; no averaged confidence bounds or significance claim. '
                           'No reweighting, new inference, training or configuration selection.'}
    args.output_dir.mkdir()
    for control, candidate, question in CONTRASTS:
        result = compare({'rows': rows + controls}, args.dataset, control, candidate)
        result['question'] = question
        path = args.output_dir / f'{control}_to_{candidate}.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        summary['pairs'].append({**{k: v for k, v in result.items() if k != 'identity_changes'},
                                 'report': str(path), 'report_sha256': sha(path)})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
