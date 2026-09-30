"""Explain the readout of the closed six-end panel from unchanged CPU arrays."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_correspondence_roles import PROTOCOLS
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(row):
    run = Path(row['run_dir'])
    distance_path = run / 'official_distances.pt'
    receipt_path = run / 'official_metrics.json'
    checkpoint = run / 'best_map.pth'
    assert sha(distance_path) == row['distance_sha256']
    assert sha(receipt_path) == row['receipt_sha256']
    assert sha(checkpoint) == row['checkpoint_sha256']
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['training_epochs'] == 50
    assert receipt['selected_epoch'] == row['best_epoch']
    protocol_path = PROTOCOLS / (row['dataset'] + '.json')
    assert sha(protocol_path) == receipt['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    arrays = torch.load(distance_path, map_location='cpu', weights_only=False)
    for split in ('query', 'gallery'):
        records = protocol['records'][split]
        assert len(records) == protocol['counts'][split]
        for field, name in (('identity', 'ids'), ('camera', 'cameras'), ('scene', 'scenes')):
            assert np.array_equal(np.asarray(arrays[split + '_' + name]),
                                  np.asarray([record[field] for record in records]))
    environment = 'scenes' if row['dataset'] == 'MSVR310' else 'cameras'
    scorer = scene_scores if row['dataset'] == 'MSVR310' else camera_scores
    scores, parity = {}, {}
    for name in ('shared_global', 'fused', 'joint_local'):
        distances = arrays[name].numpy()
        assert distances.shape == (protocol['counts']['query'], protocol['counts']['gallery'])
        assert np.isfinite(distances).all()
        scores[name] = scorer(distances, arrays['query_ids'], arrays['gallery_ids'],
                              arrays['query_' + environment], arrays['gallery_' + environment])
        expected = receipt['metrics'] if name == 'fused' else receipt['diagnostic_metrics'][name]
        parity[name] = {key: abs(value - expected[key])
                        for key, value in scores[name]['metrics'].items()}
        assert max(parity[name].values()) < 1e-5
    ranks = {name: np.asarray(value['first_match_rank']) for name, value in scores.items()}
    ap = {name: np.asarray(value['average_precision']) for name, value in scores.items()}
    correct = {name: value == 1 for name, value in ranks.items()}
    global_correct, fused_correct, local_correct = (correct[name] for name in
                                                   ('shared_global', 'fused', 'joint_local'))
    repaired = ~global_correct & fused_correct
    damaged = global_correct & ~fused_correct
    delta_ap = (ap['fused'] - ap['shared_global']) * 100
    query_ids = np.asarray(arrays['query_ids'])
    identities = []
    for identity in np.unique(query_ids):
        mask = query_ids == identity
        identities.append({'identity': int(identity), 'queries': int(mask.sum()),
                           'mean_delta_ap_pp': float(delta_ap[mask].mean()),
                           'rank1_repairs': int(repaired[mask].sum()),
                           'rank1_new_errors': int(damaged[mask].sum())})
    counts = {'global_and_fused_correct': int((global_correct & fused_correct).sum()),
              'global_correct_fused_wrong': int(damaged.sum()),
              'global_wrong_fused_correct': int(repaired.sum()),
              'global_and_fused_wrong': int((~global_correct & ~fused_correct).sum()),
              'joint_local_correct_global_wrong': int((local_correct & ~global_correct).sum()),
              'joint_local_correct_fused_wrong': int((local_correct & ~fused_correct).sum()),
              'joint_local_wrong_on_fused_repairs': int((~local_correct & repaired).sum())}
    assert sum(counts[key] for key in (
        'global_and_fused_correct', 'global_correct_fused_wrong',
        'global_wrong_fused_correct', 'global_and_fused_wrong')) == protocol['counts']['query']
    delta_metrics = {key: scores['fused']['metrics'][key] - scores['shared_global']['metrics'][key]
                     for key in scores['fused']['metrics']}
    assert abs(delta_metrics['Rank-1'] - (repaired.sum() - damaged.sum()) /
               protocol['counts']['query'] * 100) < 1e-10
    return {'dataset': row['dataset'], 'variant': row['variant'], 'best_epoch': row['best_epoch'],
            'bindings': {'checkpoint_sha256': row['checkpoint_sha256'],
                         'distance_sha256': row['distance_sha256'],
                         'receipt_sha256': row['receipt_sha256'],
                         'protocol_sha256': sha(protocol_path)},
            'query_count': protocol['counts']['query'], 'gallery_count': protocol['counts']['gallery'],
            'metrics': {name: value['metrics'] for name, value in scores.items()},
            'fused_minus_global_metrics_pp': delta_metrics, 'rank1_evidence_counts': counts,
            'identity_macro_delta_ap_pp': float(np.mean([item['mean_delta_ap_pp'] for item in identities])),
            'identity_changes': identities,
            'query_scores': [{'query_index': index, 'identity': int(query_ids[index]),
                              'average_precision': {name: float(value[index]) for name, value in ap.items()},
                              'first_match_rank': {name: int(value[index]) for name, value in ranks.items()}}
                             for index in range(protocol['counts']['query'])],
            'full_gallery_metric_difference_pp': parity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    assert sha(args.matrix) == '35067bce6690e86e43a08f963ddc259ade704549d899f8b2ce58cfb1dcb3aa56'
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == 6
    assert {(row['dataset'], row['variant']) for row in matrix['rows']} == {
        (dataset, variant) for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310')
        for variant in ('local_memory', 'full_memory')}
    assert all(row['status'] == 'VERIFIED_COMPLETE' for row in matrix['rows'])
    assert not args.output.exists()
    torch.set_num_threads(1)
    rows = [analyze(row) for row in matrix['rows']]
    report = {'status': 'CLOSED_PANEL_FULL_GALLERY_READOUT_DIAGNOSIS_COMPLETE',
              'completed_at': datetime.now().astimezone().isoformat(),
              'matrix_sha256': sha(args.matrix), 'source_sha256': sha(Path(__file__)),
              'scorer_sha256': {name: sha(ROOT / 'tools' / name) for name in (
                  'train_msvr310_signal_oof.py', 'train_rgbnt100_signal_oof.py')},
              'rows': rows,
              'boundary': 'Within-checkpoint global/fused/local outputs, not independent training. '
                          'All official queries and complete gallery, original filtering and checkpoint. '
                          'Label-based local-correct counts are post-selection explanations, not a '
                          'deployable selector, oracle score, evidence of optimal mixing or a causal claim. '
                          'No model forward, GPU, optimizer, new inference rule or current training read.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'rows': [{key: row[key] for key in (
        'dataset', 'variant', 'query_count', 'fused_minus_global_metrics_pp', 'rank1_evidence_counts')}
        for row in rows]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
