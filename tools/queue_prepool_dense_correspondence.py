"""Three fixed H2a endpoints; each real M0 precedes its fresh full50."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_m0 as inherited

base, previous = inherited.base, inherited.previous
DATASETS = ('RGBNT201', 'MSVR310', 'RGBNT100')
SCHEMA = 'trifusion-prepool-dense-correspondence-v1'
SCOPE = ROOT / 'refine-logs/prepool_dense_correspondence_v1/SOURCE_SCOPE.json'
CONTROLS = inherited.CONTROLS
STORAGE_BYTES = 4 * 384 * 1024**2 + 2 * 1024**3


def source_map():
    sources = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def command(dataset, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT / 'tools/run_prepool_dense_correspondence.py'),
        '--dataset', dataset, '--variant', 'semantic', '--mode', mode,
        '--protocol', str(previous.PROTOCOLS / f'{dataset}.json'),
        '--signal-source', str(previous.SOURCE), '--clip-weight', str(previous.WEIGHTS / 'ViT-B-16.pt'),
        '--initialization', str(campaign / 'initialization' / f'{dataset}.json'),
        '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def accept_initialization(campaign, dataset):
    witness = json.loads((campaign / 'initialization' / f'{dataset}.json').read_text())
    assert witness['schema'] == SCHEMA and witness['status'] == 'INITIALIZATION_VERIFIED'
    binding = witness['binding']
    controls = json.loads(CONTROLS.read_text())
    old = next(row['initializer'] for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
    assert all(binding[key] == value for key, value in old.items() if key not in ('architecture', 'entry_sha256', 'scope'))
    path = campaign / 'acceptance' / f'{dataset}_initialization.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    base.queue.write(path, dict(status='MATCHED_INITIALIZATION_ACCEPTED', dataset=dataset,
        initializer=binding, accepted_at=base.queue.stamp()))


def accept_m0(campaign, dataset):
    output = ROOT / f'trained-model/{campaign.name}_m0_{dataset}'
    binding = json.loads((campaign / 'initialization' / f'{dataset}.json').read_text())['binding']
    receipt = json.loads((output / 'training.json').read_text())
    assert receipt['schema'] == SCHEMA and receipt['status'] == 'M0_PASS'
    assert receipt['initializer'] == binding and binding['added_model_parameters'] == 0
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert receipt['frozen_parameters_unchanged'] and receipt['visual_parameters_changed'] and receipt['fresh_camera_parameters_changed']
    assert receipt['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert all(value == 8 for value in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8
    assert all(row['teacher_state_rng_checked'] and row['dense_isolated_shared_global_gradient_absent'] for row in steps)
    assert all(all(value > 0 for value in row['dense_supported_rows_by_modality']) for row in steps)
    assert all(all(value > 0 for value in row['dense_supported_columns_by_modality']) for row in steps)
    assert all(math.isfinite(row['dense_correspondence_loss']) for row in steps)
    names = set(steps[0]['dense_isolated_key_gradient_norms'])
    assert len(names) == 3
    assert all(set(row['dense_isolated_key_gradient_norms']) == names for row in steps)
    accumulated = {}
    for name in sorted(names):
        values = [row['dense_isolated_key_gradient_norms'][name] for row in steps]
        assert all(math.isfinite(value) and value >= 0 for value in values) and sum(values) > 0, name
        accumulated[name] = sum(values)
    controls = json.loads(CONTROLS.read_text())
    old = next(row for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
    batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()]
    old_batches = [json.loads(line) for line in (Path(old['m0_dir']) / 'training_batch_order.jsonl').read_text().splitlines()]
    assert len(batches) == 8 and batches == old_batches[:8]
    probe = output / 'm0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT / 'trained-model').resolve())
    assert {p.name for p in output.glob('*.pth')} == {'m0_reload_probe.pth'}
    assert base.sha(probe) == receipt['m0']['reload_probe_sha256']
    accepted = dict(status='REAL_M0_AND_ISOLATED_DENSE_ACTIVITY_ACCEPTED', dataset=dataset,
        accepted_at=base.queue.stamp(), output_dir=str(output), initializer=binding,
        auxiliary_key_gradient_norm_sums=accumulated, source_raw_m0_batch_order_equal=True,
        teacher_state_rng_checked=True,
        artifact_sha256={str(p): base.sha(p) for p in output.iterdir() if p.is_file() and p != probe},
        probe=dict(path=str(probe), bytes=probe.stat().st_size, sha256=base.sha(probe)))
    path = campaign / 'acceptance' / f'{dataset}.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    base.queue.write(path, accepted)
    # This probe's verification consumer has closed; full50 builds fresh state.
    probe.unlink()
    with (campaign / 'probe_retirement.jsonl').open('a') as log:
        log.write(json.dumps(dict(retired_at=base.queue.stamp(), acceptance=str(path), **accepted['probe'])) + '\n')
    return accepted


def accepted_row(campaign, dataset):
    output = ROOT / f'trained-model/{campaign.name}_full_{dataset}'
    binding = json.loads((campaign / 'initialization' / f'{dataset}.json').read_text())['binding']
    training = json.loads((output / 'training.json').read_text())
    result = json.loads((output / 'official_metrics.json').read_text())
    own = json.loads((output / 'own_global_metrics.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert result['status'] == own['status'] == 'COMPLETE'
    assert training['schema'] == result['schema'] == own['schema'] == SCHEMA
    assert training['initializer'] == binding and binding['added_model_parameters'] == 0
    assert training['condition'] == result['condition'] and result['condition']['auxiliary_weight'] == 1.0
    assert [row['epoch'] for row in training['history']] == list(range(1, 51))
    best = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
    assert result['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert all(abs(result['metrics'][name] - best['official_fused'][name]) < 1e-5 for name in result['metrics'])
    assert result['training_epochs'] == 50 and result['seed'] == 42
    assert result['protocol_sha256'] == binding['protocol_sha256']
    for key, name in (('checkpoint_sha256', 'best_map.pth'), ('distance_sha256', 'official_distances.pt'),
                      ('training_best_distance_sha256', 'best_epoch_distances.pt')):
        assert result[key] == base.sha(output / name)
    assert result['independent_upstream_metrics_equal'] and not result['reranking']
    assert own['same_original_fused_forward'] and own['new_global_NN_forwards'] == 0
    assert own['epoch_selection'] == 'same_fused_mAP_best'
    assert own['distance_sha256'] == base.sha(output / 'own_global_distances.pt')
    controls = json.loads(CONTROLS.read_text())
    old = next(row for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
    assert (output / 'training_batch_order.jsonl').read_bytes() == (Path(old['run_dir']) / 'training_batch_order.jsonl').read_bytes()
    return dict(dataset=dataset, variant='dense', status='VERIFIED_COMPLETE', run_dir=str(output),
        initializer=binding, best_epoch=best['epoch'], metrics=result['metrics'], own_global_metrics=own['metrics'],
        checkpoint_sha256=result['checkpoint_sha256'], distance_sha256=result['distance_sha256'],
        receipt_sha256=base.sha(output / 'official_metrics.json'),
        own_global_distance_sha256=own['distance_sha256'],
        own_global_receipt_sha256=base.sha(output / 'own_global_metrics.json'), actual_training_batch_order_equal=True)


def accept_full(campaign, dataset):
    row = accepted_row(campaign, dataset)
    path = campaign / 'acceptance' / f'{dataset}_full.json'
    assert not path.exists()
    base.queue.write(path, row)


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert shutil.disk_usage(ROOT).free >= STORAGE_BYTES
    controls = json.loads(CONTROLS.read_text())
    assert len(controls['rows']) == 6
    assert all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    sources = source_map()
    base.source_map = source_map
    args.campaign.mkdir(parents=True)
    jobs = [dict(dataset=dataset, status='PENDING', steps=[]) for dataset in DATASETS]
    manifest = dict(schema=SCHEMA, source_sha256=sources, initialization_sha256={}, jobs=jobs,
        control_seal_sha256=base.sha(CONTROLS), physical_gpus=[0, 1], max_parallel_jobs=1,
        storage_required_bytes=STORAGE_BYTES, epochs=50, seed=42,
        boundary='Three fixed same-modal geometry auxiliary endpoints. Each M0 then fresh50; no old retraining, auto retry, score-driven tuning or blanket parity prerequisite.')
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(), jobs=jobs, report_invocations=0)
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for job in jobs:
        dataset = job['dataset']
        prepare = dict(mode='prepare', command=command(dataset, 'prepare', args.campaign,
            ROOT / f'trained-model/{args.campaign.name}_prepare_unused'))
        job['steps'].append(prepare)
        state['status'] = 'RUNNING'
        if previous.run_logged(args.campaign, state, prepare, f'{dataset}_prepare.log'):
            job.update(status='FAILED', failed_phase='prepare')
            continue
        initializer = args.campaign / 'initialization' / f'{dataset}.json'
        manifest['initialization_sha256'][str(initializer)] = base.sha(initializer)
        base.queue.write(args.campaign / 'manifest.json', manifest)
        for mode in ('accept-initialization', 'm0', 'accept-m0', 'train', 'evaluate', 'accept-full'):
            if mode.startswith('accept-'):
                invocation = [sys.executable, '-B', str(Path(__file__).resolve()), '--accept', mode.removeprefix('accept-'),
                    '--campaign', str(args.campaign), '--dataset', dataset]
            else:
                phase = 'm0' if mode == 'm0' else 'full'
                output = ROOT / f'trained-model/{args.campaign.name}_{phase}_{dataset}'
                invocation = command(dataset, mode, args.campaign, output)
            step = dict(mode=mode, command=invocation)
            job['steps'].append(step)
            state['status'] = 'RUNNING'
            if previous.run_logged(args.campaign, state, step, f'{dataset}_{mode}.log'):
                job.update(status='FAILED', failed_phase=mode, exit_code=step['exit_code'])
                break
            if mode == 'accept-m0':
                job['m0_acceptance'] = json.loads((args.campaign / 'acceptance' / f'{dataset}.json').read_text())
        else:
            job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(), result=json.loads((args.campaign / 'acceptance' / f'{dataset}_full.json').read_text()))
        base.queue.write(args.campaign / 'campaign.json', state)
    rows = [job['result'] for job in jobs if job['status'] == 'COMPLETE']
    missing = [dict(dataset=job['dataset'], failed_phase=job['failed_phase']) for job in jobs if job['status'] == 'FAILED']
    base.queue.write(args.campaign / 'accepted_matrix.json', dict(schema=SCHEMA, accepted=len(rows), expected=3, rows=rows, missing=missing))
    state.update(status='COMPLETE' if len(rows) == 3 else 'COMPLETE_WITH_MISSING', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_prepool_dense_correspondence.py'),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return result.returncode if result.returncode else int(bool(missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--accept', choices=('initialization', 'm0', 'full'))
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.accept:
        assert args.dataset is not None
        source_map()
        {'initialization': accept_initialization, 'm0': accept_m0, 'full': accept_full}[args.accept](args.campaign, args.dataset)
    else:
        assert args.report_dir is not None and args.dataset is None
        args.report_dir = args.report_dir.resolve()
        raise SystemExit(coordinate(args))
