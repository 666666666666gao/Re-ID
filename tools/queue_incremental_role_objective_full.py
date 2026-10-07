"""Six fresh full50s after all real isolated-gradient M0 acceptances."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_m0 as m0

base = m0.base
SCOPE = ROOT / 'refine-logs/incremental_role_objective_v1/FULL_SOURCE_SCOPE.json'
STORAGE_BYTES = 6 * 384 * 1024**2 + 600 * 1024**2 + 2 * 1024**3
REPORT_ENTRY = ROOT / 'tools/report_incremental_role_objective.py'


def source_map():
    sources = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def require_m0(campaign):
    state = json.loads((campaign / 'campaign.json').read_text())
    assert state['status'] == 'M0_ONLY_COMPLETE' and state['accepted'] == state['expected'] == 6
    assert len(state['jobs']) == 6
    for job in state['jobs']:
        assert job['status'] == 'COMPLETE' and job['exit_code'] == 0
        acceptance = json.loads((campaign / 'acceptance' / f'{job["dataset"]}_{job["objective"]}.json').read_text())
        assert acceptance == job['acceptance']
        assert acceptance['status'] == 'REAL_M0_AND_ISOLATED_INCREMENT_ACTIVITY_ACCEPTED'
        assert acceptance['correction_gradient_norm_sum'] > 0 and len(acceptance['query_key_gradient_norm_sums']) == 6
        assert all(value > 0 for value in acceptance['query_key_gradient_norm_sums'].values())
        assert all(base.sha(Path(name)) == digest for name, digest in acceptance['artifact_sha256'].items())
        assert not Path(acceptance['probe']['path']).exists()
    return state


def accepted_row(campaign, dataset, objective):
    output = ROOT / f'trained-model/{campaign.name}_full_{objective}_{dataset}'
    manifest = json.loads((campaign / 'manifest.json').read_text())
    initializer = Path(manifest['m0_campaign']) / 'initialization' / f'{dataset}_{objective}.json'
    binding = json.loads(initializer.read_text())['binding']
    training = json.loads((output / 'training.json').read_text())
    result = json.loads((output / 'official_metrics.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and result['status'] == 'COMPLETE'
    assert training['schema'] == result['schema'] == m0.SCHEMA and training['initializer'] == binding
    assert training['condition'] == result['condition'] and result['condition']['incremental_objective'] == objective
    assert [row['epoch'] for row in training['history']] == list(range(1, 51))
    best = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
    assert result['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert all(abs(result['metrics'][name] - best['official_fused'][name]) < 1e-5 for name in result['metrics'])
    assert result['training_epochs'] == 50 and result['seed'] == 42
    assert result['protocol_sha256'] == binding['protocol_sha256']
    assert result['checkpoint_sha256'] == base.sha(output / 'best_map.pth')
    assert result['distance_sha256'] == base.sha(output / 'official_distances.pt')
    assert result['training_best_distance_sha256'] == base.sha(output / 'best_epoch_distances.pt')
    assert result['independent_upstream_metrics_equal'] and not result['reranking']
    controls = json.loads(m0.CONTROLS.read_text())
    old = next(row for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
    assert (output / 'training_batch_order.jsonl').read_bytes() == (Path(old['run_dir']) / 'training_batch_order.jsonl').read_bytes()
    return dict(dataset=dataset, variant=objective, status='VERIFIED_COMPLETE', run_dir=str(output),
        initializer=binding, best_epoch=best['epoch'], metrics=result['metrics'],
        checkpoint_sha256=result['checkpoint_sha256'], distance_sha256=result['distance_sha256'],
        receipt_sha256=base.sha(output / 'official_metrics.json'), actual_training_batch_order_equal=True)


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert shutil.disk_usage(ROOT).free >= STORAGE_BYTES
    m0_state = require_m0(args.m0_campaign)
    controls = json.loads(m0.CONTROLS.read_text())
    assert all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    sources = source_map()
    initial = {str(args.m0_campaign / 'initialization' / f'{job["dataset"]}_{job["objective"]}.json'):
        base.sha(args.m0_campaign / 'initialization' / f'{job["dataset"]}_{job["objective"]}.json') for job in m0_state['jobs']}
    jobs = [dict(dataset=d, objective=o, phase='full', status='PENDING') for d in m0.DATASETS for o in m0.OBJECTIVES]
    args.campaign.mkdir(parents=True)
    manifest = dict(schema=m0.SCHEMA, source_sha256=sources, initialization_sha256=initial,
        jobs=jobs, m0_campaign=str(args.m0_campaign), control_seal_sha256=base.sha(m0.CONTROLS),
        physical_gpus=[0, 1], max_parallel_jobs=1, epochs=50, seed=42, storage_required_bytes=STORAGE_BYTES,
        boundary='Six fresh public initializations, never M0 probe inheritance. Same semantic model/raw author tasks/deployment; only training increment differs.')
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(), jobs=jobs, report_invocations=0)
    base.source_map = source_map
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for job in jobs:
        dataset, objective = job['dataset'], job['objective']
        output = ROOT / f'trained-model/{args.campaign.name}_full_{objective}_{dataset}'
        job['steps'] = []
        for mode in ('train', 'evaluate'):
            step = dict(mode=mode, command=m0.command(dataset, objective, mode, args.m0_campaign, output))
            job['steps'].append(step)
            if m0.previous.run_logged(args.campaign, state, step, f'{dataset}_{objective}_{mode}.log'):
                job.update(status='FAILED', exit_code=step['exit_code'])
                base.queue.write(args.campaign / 'campaign.json', state)
                return 1
        job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(), result=accepted_row(args.campaign, dataset, objective))
        base.queue.write(args.campaign / 'campaign.json', state)
    rows = [job['result'] for job in jobs]
    base.queue.write(args.campaign / 'accepted_matrix.json', dict(schema=m0.SCHEMA, accepted=6, expected=6, rows=rows))
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(REPORT_ENTRY),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'm0-campaign', 'report-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.campaign, args.m0_campaign, args.report_dir = args.campaign.resolve(), args.m0_campaign.resolve(), args.report_dir.resolve()
    raise SystemExit(coordinate(args))
