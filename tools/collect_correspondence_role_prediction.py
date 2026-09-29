#!/usr/bin/env python3
"""Verify full M3 panel endpoints and recompute their complete gallery scores."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_correspondence_roles import METRICS, read, sha
from tools.queue_correspondence_role_prediction import CONDITIONS
from tools.queue_correspondence_refinement import DATASETS, child_campaign
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, WEIGHTS


def verify(run, m0_run, dataset, variant, manifest):
    import numpy as np
    import torch
    from tools.train_msvr310_signal_oof import scene_scores
    from tools.train_rgbnt100_signal_oof import camera_scores

    training, official, m0 = (read(p) for p in (run / 'training.json', run / 'official_metrics.json',
                                               m0_run / 'training.json'))
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert official['status'] == 'COMPLETE' and m0['status'] == 'M0_PASS'
    protocol_path = PROTOCOLS / f'{dataset}.json'
    protocol = read(protocol_path)
    address, prediction = CONDITIONS[variant]
    condition = {'address_mode': address, 'prediction_mode': prediction}
    _, baseline_sha = BASELINES[dataset]
    for receipt in (training, m0):
        assert receipt['dataset'] == dataset and receipt['seed'] == 42 and receipt['epochs'] == 50
        assert receipt['checkpoint_policy'] == 'best_official_map' and receipt['protocol_sha256'] == sha(protocol_path)
        assert receipt['initializer']['author_checkpoint_sha256'] == baseline_sha
    binding = training['initializer']
    assert binding == m0['initializer']
    assert binding['architecture'] == 'correspondence_role_prediction_v1' and binding['prediction'] == condition
    assert binding['width'] == 128 and binding['fused_width'] == 1536 and binding['prediction_weight'] == 0.1
    assert binding['m1'] and binding['m2'] and binding['m3'] and binding['prediction_head_parameters'] == 100608
    assert binding['prediction_address_gradient'] == ('student' if address == 'own' else 'detached_teacher')
    for field, path in (('model_source_sha256', 'modeling/trifusion/correspondence_roles.py'),
                        ('entry_sha256', 'tools/run_correspondence_role_prediction.py'),
                        ('reused_entry_sha256', 'tools/run_correspondence_roles.py'),
                        ('prediction_source_sha256', 'modeling/trifusion/correspondence_role_prediction.py')):
        assert binding[field] == manifest['source_sha256'][path] == sha(ROOT / path)
    assert m0['m0']['frozen_signal_unchanged']
    assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters']
    assert m0['m0']['reload_max_abs_difference'] <= 1e-5
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    best = max(training['history'], key=lambda r: (r['official_fused']['mAP'], r['epoch']))
    checkpoint, distances_path = run / 'best_map.pth', run / 'official_distances.pt'
    assert training['checkpoint'] == str(checkpoint)
    assert sha(checkpoint) == official['checkpoint_sha256'] and sha(distances_path) == official['distance_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-correspondence-role-prediction-v1'
    assert payload['prediction'] == condition and payload['variants'] == {'m1': True, 'm2': True, 'm3': True}
    assert payload['dataset'] == official['dataset'] == dataset and payload['seed'] == official['seed'] == 42
    assert payload['protocol_sha256'] == official['protocol_sha256'] == sha(protocol_path)
    assert payload['baseline_sha256'] == official['baseline_sha256'] == baseline_sha
    assert payload['epoch'] == official['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert official['training_epochs'] == 50 and not official['reranking']
    assert official['independent_upstream_metrics_equal']
    for name in METRICS:
        assert abs(payload['metrics'][name] - official['metrics'][name]) < 1e-5
        assert abs(best['official_fused'][name] - official['metrics'][name]) < 1e-5
    arrays = torch.load(distances_path, map_location='cpu', weights_only=False)
    for split in ('query', 'gallery'):
        records = protocol['records'][split]
        assert len(records) == protocol['counts'][split]
        for key, field in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
            assert np.array_equal(arrays[f'{split}_{key}'], np.asarray([r[field] for r in records]))
    distances = arrays['fused'].numpy()
    assert distances.shape == (protocol['counts']['query'], protocol['counts']['gallery'])
    assert np.isfinite(distances).all()
    field = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scores = (scene_scores if dataset == 'MSVR310' else camera_scores)(
        distances, arrays['query_ids'], arrays['gallery_ids'], arrays[f'query_{field}'], arrays[f'gallery_{field}'])
    differences = {name: abs(scores['metrics'][name] - official['metrics'][name]) for name in METRICS}
    assert max(differences.values()) < 1e-5
    return {'status': 'VERIFIED_COMPLETE', 'best_epoch': best['epoch'], 'metrics': official['metrics'],
            'trainable_parameters': binding['trainable_parameters'],
            'training_and_epoch_eval_seconds': (datetime.fromisoformat(training['completed_at']) -
                                                datetime.fromisoformat(training['started_at'])).total_seconds(),
            'checkpoint_sha256': official['checkpoint_sha256'], 'distance_sha256': official['distance_sha256'],
            'receipt_sha256': sha(run / 'official_metrics.json'),
            'full_gallery_recomputed_metric_difference_pp': differences}


def collect(campaign):
    manifest = read(campaign / 'manifest.json')
    assert manifest['schema'] == 'trifusion-correspondence-m3-panel-v1'
    expected = [(d, v) for v in CONDITIONS for d in DATASETS]
    assert [(r['dataset'], r['variant']) for r in manifest['jobs']] == expected
    assert manifest['seed'] == 42 and manifest['epochs'] == 50
    assert sha(Path(manifest['prior_matrix'])) == manifest['prior_matrix_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    for weight, digest in BASELINES.values():
        assert sha(WEIGHTS / weight) == digest
    rows = []
    for dataset, variant in expected:
        folder = child_campaign(campaign, 'm3', dataset, variant)
        row = {'phase': 'm3', 'dataset': dataset, 'variant': variant, 'campaign_dir': str(folder)}
        if not (folder / 'campaign.json').exists():
            row['status'] = 'PENDING'
        else:
            state = read(folder / 'campaign.json')
            if state['status'] == 'COMPLETE':
                assert state['dataset'] == dataset and state['variant'] == variant
                assert [job['mode'] for job in state['jobs']] == ['m0', 'train', 'evaluate']
                assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in state['jobs'])
                m0_run, run = (Path(state['jobs'][i]['output_dir']) for i in (0, 2))
                assert state['jobs'][1]['output_dir'] == str(run)
                row['run_dir'] = str(run)
                row.update(verify(run, m0_run, dataset, variant, manifest))
            else:
                row.update(status='UNACCEPTED', campaign_status=state['status'])
        rows.append(row)
    return {'schema': 'trifusion-correspondence-m3-panel-verification-v1',
            'collected_at': datetime.now().astimezone().isoformat(), 'seed': 42,
            'expected_endpoints': 12, 'verified_complete': sum(r['status'] == 'VERIFIED_COMPLETE' for r in rows),
            'scope': 'Full50, single mAP best, strict reload and CPU complete gallery; no new inference', 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = collect(args.campaign.resolve())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('collected_at', 'expected_endpoints', 'verified_complete')}))
