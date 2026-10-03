"""Verify full50 shared/private evidence flows and recompute complete-gallery metrics."""
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_correspondence_roles import METRICS, read, sha
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, WEIGHTS
from tools.check_shared_private_initialization import CONDITIONS, DATASETS
from tools.run_visual_update_control import VISUAL_PREFIX, VISUAL_LR
from tools.run_shared_private_evidence import SCHEMA
from tools.queue_correspondence_refinement import child_campaign
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def verify_m0(folder, dataset, variant, manifest):
    receipt = read(folder / 'training.json')
    update, readout = CONDITIONS[variant]
    expected = {'visual_update': update, 'readout': readout,
                'visual_lr': VISUAL_LR, 'visual_parameter_dtype': 'float32', 'evidence_flow': variant}
    assert receipt['schema'] == SCHEMA and receipt['status'] == 'M0_PASS'
    assert receipt['dataset'] == dataset and receipt['seed'] == 42 and receipt['epochs'] == 50
    assert receipt['condition'] == expected and receipt['checkpoint_policy'] == 'best_official_map'
    assert receipt['protocol_sha256'] == sha(PROTOCOLS / f'{dataset}.json')
    witness = read(Path(manifest['initialization_witness_path']))
    row = next(item for item in witness['rows'] if (item['dataset'], item['variant']) == (dataset, variant))
    assert receipt['initializer'] == row['binding']
    assert receipt['initializer']['entry_sha256'] == manifest['source_sha256']['tools/run_shared_private_evidence.py']
    assert receipt['initializer']['role_source_sha256'] == manifest['source_sha256']['modeling/trifusion/role_global_tokens.py']
    assert receipt['initializer']['evidence_flow_source_sha256'] == manifest['source_sha256']['modeling/trifusion/shared_private_evidence.py']
    assert receipt['m0']['frozen_signal_state_unchanged'] and receipt['frozen_signal_state_unchanged']
    assert receipt['m0']['visual_parameters_changed'] == (update == 'low_lr')
    assert receipt['visual_parameters_changed'] == (update == 'low_lr')
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == row['trainable_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    steps = [json.loads(line) for line in (folder / 'training_steps.jsonl').read_text().splitlines()]
    assert [item['batch'] for item in steps] == list(range(8))
    assert all(item['epoch'] == 1 for item in steps)
    assert abs(np.mean([item['loss'] for item in steps]) - receipt['history'][0]['mean_loss']) < 1e-5
    assert sha(folder / 'm0_reload_probe.pth') == receipt['m0']['reload_probe_sha256']
    return receipt


def verify(run, m0_run, dataset, variant, manifest):
    m0 = verify_m0(m0_run, dataset, variant, manifest)
    training, official = read(run / 'training.json'), read(run / 'official_metrics.json')
    assert training['schema'] == official['schema'] == SCHEMA
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status'] == 'COMPLETE'
    assert training['initializer'] == m0['initializer']
    assert training['condition'] == official['condition'] == m0['condition']
    assert [item['epoch'] for item in training['history']] == list(range(1, 51))
    assert training['frozen_signal_state_unchanged']
    update = CONDITIONS[variant][0]
    assert training['visual_parameters_changed'] == (update == 'low_lr')
    best = max(training['history'], key=lambda item: (item['official_fused']['mAP'], item['epoch']))
    for item in (training, official):
        assert item['dataset'] == dataset and item['seed'] == 42
        assert item['protocol_sha256'] == sha(PROTOCOLS / f'{dataset}.json')
    filename, baseline_sha = BASELINES[dataset]
    assert sha(WEIGHTS / filename) == official['baseline_sha256'] == baseline_sha
    assert official['training_epochs'] == training['epochs'] == 50
    assert official['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert official['independent_upstream_metrics_equal'] and not official['reranking']
    assert sha(run / 'best_map.pth') == official['checkpoint_sha256']
    assert sha(run / 'official_distances.pt') == official['distance_sha256']
    payload = torch.load(run / 'best_map.pth', map_location='cpu', weights_only=True)
    assert payload['schema'] == SCHEMA and payload['condition'] == official['condition']
    assert payload['dataset'] == dataset and payload['seed'] == 42
    assert payload['baseline_sha256'] == baseline_sha and payload['protocol_sha256'] == official['protocol_sha256']
    assert payload['epoch'] == best['epoch']
    visual_prefix = 'backbone.signal.' + VISUAL_PREFIX
    visual_values = [value for key, value in payload['state'].items() if key.startswith(visual_prefix)]
    assert len(visual_values) == 152 and all(value.dtype == torch.float32 for value in visual_values)
    assert all(torch.isfinite(value).all() for value in payload['state'].values())
    assert any(key.startswith('backbone.signal.') and not key.startswith(visual_prefix) for key in payload['state'])
    for name in METRICS:
        assert abs(official['metrics'][name] - best['official_fused'][name]) < 1e-5
        assert abs(official['metrics'][name] - payload['metrics'][name]) < 1e-5
    del payload
    protocol = read(PROTOCOLS / f'{dataset}.json')
    arrays = torch.load(run / 'official_distances.pt', map_location='cpu', weights_only=False)
    for split in ('query', 'gallery'):
        records = protocol['records'][split]
        assert len(records) == protocol['counts'][split]
        for name, field in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
            assert np.array_equal(arrays[f'{split}_{name}'], np.asarray([row[field] for row in records]))
    distances = arrays['fused'].numpy()
    assert distances.shape == (protocol['counts']['query'], protocol['counts']['gallery'])
    assert np.isfinite(distances).all()
    field = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    scores = scorer(distances, arrays['query_ids'], arrays['gallery_ids'],
                    arrays[f'query_{field}'], arrays[f'gallery_{field}'])
    differences = {name: abs(scores['metrics'][name] - official['metrics'][name]) for name in METRICS}
    assert max(differences.values()) < 1e-5
    steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
    for row in training['history']:
        epoch_steps = [item for item in steps if item['epoch'] == row['epoch']]
        assert [item['batch'] for item in epoch_steps] == list(range(row['steps']))
        assert abs(np.mean([item['loss'] for item in epoch_steps]) - row['mean_loss']) < 1e-5
    assert len(steps) == sum(row['steps'] for row in training['history'])
    assert all(np.isfinite([item['loss'], item['id'], item['triplet']]).all() for item in steps)
    assert max(abs(item['loss'] - item['id'] - item['triplet']) for item in steps) < 1e-5
    return {'status': 'VERIFIED_COMPLETE', 'metrics': official['metrics'],
            'best_epoch': best['epoch'], 'checkpoint_sha256': official['checkpoint_sha256'],
            'distance_sha256': official['distance_sha256'], 'receipt_sha256': sha(run / 'official_metrics.json'),
            'common_initializer_sha256': training['initializer']['common_initializer_sha256'],
            'initial_model_state_sha256': training['initializer']['initial_model_state_sha256'],
            'trainable_parameters': training['initializer']['trainable_parameters'],
            'training_and_epoch_eval_seconds': (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds(),
            'peak_training_and_epoch_eval_allocated_bytes': training['peak_training_and_epoch_eval_allocated_bytes'],
            'full_gallery_recomputed_metric_difference_pp': differences,
            'logged_steps': len(steps), 'positive_triplet_steps': sum(item['triplet'] > 0 for item in steps),
            'final_metrics': training['history'][-1]['official_fused']}


def collect(campaign):
    manifest = read(campaign / 'manifest.json')
    assert manifest['schema'] == 'trifusion-shared-private-panel-v1'
    expected = [(dataset, variant) for variant in CONDITIONS for dataset in DATASETS]
    assert [(job['dataset'], job['variant']) for job in manifest['jobs']] == expected
    assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
    assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
    rows = []
    for dataset, variant in expected:
        folder = child_campaign(campaign, 'shared_private', dataset, variant)
        state = read(folder / 'campaign.json')
        assert state['status'] == 'COMPLETE' and state['dataset'] == dataset and state['variant'] == variant
        assert [job['mode'] for job in state['jobs']] == ['m0', 'train', 'evaluate']
        assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in state['jobs'])
        m0_run, run = (Path(state['jobs'][index]['output_dir']) for index in (0, 2))
        assert state['jobs'][1]['output_dir'] == str(run)
        rows.append({'dataset': dataset, 'variant': variant, 'run_dir': str(run),
                     'campaign_dir': str(folder), **verify(run, m0_run, dataset, variant, manifest)})
    for dataset in DATASETS:
        bindings = [row['common_initializer_sha256'] for row in rows if row['dataset'] == dataset]
        assert all(binding == bindings[0] for binding in bindings)
        role_rows = [row for row in rows if row['dataset'] == dataset and row['variant'] != 'global_only']
        assert len(role_rows) == 2
        assert role_rows[0]['initial_model_state_sha256'] == role_rows[1]['initial_model_state_sha256']
        assert role_rows[0]['trainable_parameters'] == role_rows[1]['trainable_parameters']
    return {'schema': 'trifusion-shared-private-verification-v1', 'expected_endpoints': 9,
            'verified_complete': len(rows), 'rows': rows, 'collected_at': datetime.now().astimezone().isoformat()}
