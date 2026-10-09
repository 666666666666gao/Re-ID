"""Six registered text-package endpoints; each M0 precedes its own fresh50."""
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
PACKAGES = ('pretrained', 'random')
SCHEMA = 'trifusion-text-semantic-prior-v1'
SCOPE = ROOT / 'refine-logs/text_semantic_prior_v1/PRODUCTION_SOURCE_SCOPE.json'
CONTROLS = inherited.CONTROLS
STORAGE_BYTES = 7 * 512 * 1024**2 + 2 * 1024**3


def source_map():
    sources = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def output_dir(campaign, phase, dataset, package):
    return ROOT / f'trained-model/{campaign.name}_{phase}_{package}_{dataset}'


def command(dataset, package, mode, campaign):
    return [sys.executable, '-B', str(ROOT / 'tools/run_text_semantic_prior.py'),
        '--dataset', dataset, '--text-package', package, '--mode', mode,
        '--protocol', str(previous.PROTOCOLS / f'{dataset}.json'),
        '--signal-source', str(previous.SOURCE), '--clip-weight', str(previous.WEIGHTS / 'ViT-B-16.pt'),
        '--initialization', str(campaign / 'initialization' / f'{dataset}_{package}.json'),
        '--output-dir', str(output_dir(campaign, 'm0' if mode == 'm0' else 'full', dataset, package))]


def binding_for(campaign, dataset, package):
    value = json.loads((campaign / 'initialization' / f'{dataset}_{package}.json').read_text())
    assert value['schema'] == SCHEMA and value['status'] == 'INITIALIZATION_VERIFIED'
    binding = value['binding']
    assert (binding['dataset'], binding['variant'], binding['text_package']) == (dataset, 'semantic', package)
    assert binding['added_trainable_parameters'] == 592000 and binding['added_trainable_tensors'] == 5
    return binding


def accept_initialization(campaign, dataset, package):
    binding = binding_for(campaign, dataset, package)
    old = next(row['initializer'] for row in json.loads(CONTROLS.read_text())['rows']
               if row['dataset'] == dataset and row['variant'] == 'semantic')
    changed = {'architecture', 'entry_sha256', 'scope', 'initial_model_state_sha256',
               'trainable_parameters', 'trainable_parameter_tensors', 'parameter_placement'}
    assert all(binding[key] == value for key, value in old.items() if key not in changed)
    assert binding['common_initial_model_state_sha256'] == old['initial_model_state_sha256']
    assert binding['trainable_parameters'] == old['trainable_parameters'] + 592000
    assert binding['trainable_parameter_tensors'] == old['trainable_parameter_tensors'] + 5
    path = campaign / 'acceptance' / f'{dataset}_{package}_initialization.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    base.queue.write(path, dict(status='MATCHED_COMMON_INITIALIZATION_ACCEPTED', initializer=binding,
                               accepted_at=base.queue.stamp()))


def accept_m0(campaign, dataset, package):
    output = output_dir(campaign, 'm0', dataset, package)
    binding = binding_for(campaign, dataset, package)
    receipt = json.loads((output / 'training.json').read_text())
    assert receipt['schema'] == SCHEMA and receipt['status'] == 'M0_PASS'
    assert receipt['initializer'] == binding and receipt['condition']['text_package'] == package
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert receipt['frozen_parameters_unchanged'] and receipt['visual_parameters_changed'] and receipt['fresh_camera_parameters_changed']
    assert receipt['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert all(value == 8 for value in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    observed = receipt['text_training_diagnostics']
    assert observed['status'] == 'ACTUAL_TEXT_TRAINING_OBSERVATIONS_COMPLETE'
    assert observed['effective_optimizer_updates'] == 8 and observed['full_frozen_text_state_unchanged']
    assert observed['frozen_text_initial_sha256'] == observed['frozen_text_final_sha256'] == binding['text_state_sha256']
    activity = receipt['text_m0_diagnostics']
    assert activity['status'] == 'TEXT_INPUT_AUTOGRAD_ACTIVITY_ACCEPTED' and activity['updates'] == 8
    assert activity['steps'][0]['context_gradient_max_abs'] > 0 and activity['steps'][0]['context_delta_max_abs'] > 0
    names = set(activity['steps'][0]['parameters'])
    assert len(names) == 5
    assert all(set(row['parameters']) == names for row in activity['steps'])
    assert all(row['gradient_max_abs'] == 0 for row in activity['steps'][0]['parameters'].values())
    assert all(any(row['parameters'][name]['gradient_max_abs'] > 0 for row in activity['steps'][1:]) for name in names)
    assert all(row['cumulative_delta_max_abs'] > 0 for row in activity['steps'][-1]['parameters'].values())
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8 and all(row['text_role_shared_global_gradient_absent'] for row in steps)
    assert all(row['text_isolated_vjp_scale'] == 256.0 for row in steps)
    assert all(set(row['text_isolated_task_gradient_max_abs']) == names for row in steps)
    assert all(math.isfinite(value) and value >= 0 for row in steps
               for value in row['text_isolated_task_gradient_max_abs'].values())
    assert all(len(row['role_attention_diagnostics']) == 3 and row['text_context_diagnostics'] for row in steps)
    assert all(any(row['text_isolated_task_gradient_max_abs'][name] > 0 for row in steps[1:]) for name in names)
    old = next(row for row in json.loads(CONTROLS.read_text())['rows']
               if row['dataset'] == dataset and row['variant'] == 'semantic')
    batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()]
    old_batches = [json.loads(line) for line in (Path(old['m0_dir']) / 'training_batch_order.jsonl').read_text().splitlines()]
    assert len(batches) == 8 and batches == old_batches[:8]
    probe = output / 'm0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT / 'trained-model').resolve())
    assert {p.name for p in output.glob('*.pth')} == {'m0_reload_probe.pth'}
    assert base.sha(probe) == receipt['m0']['reload_probe_sha256'] and probe.stat().st_size <= 512 * 1024**2
    accepted = dict(status='REAL_M0_TEXT_AUTOGRAD_AND_RELOAD_ACCEPTED', dataset=dataset, package=package,
        initializer=binding, accepted_at=base.queue.stamp(), actual_raw_control_batch_order_equal=True,
        text_activity=activity, artifact_sha256={str(p): base.sha(p) for p in output.iterdir() if p.is_file() and p != probe},
        probe=dict(path=str(probe), bytes=probe.stat().st_size, sha256=base.sha(probe)))
    path = campaign / 'acceptance' / f'{dataset}_{package}_m0.json'
    assert not path.exists()
    base.queue.write(path, accepted)
    # The strict-reload consumer closed; full50 constructs fresh state.
    probe.unlink()
    with (campaign / 'probe_retirement.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(retired_at=base.queue.stamp(), acceptance=str(path), **accepted['probe'])) + '\n')


def accepted_row(campaign, dataset, package):
    output = output_dir(campaign, 'full', dataset, package)
    binding = binding_for(campaign, dataset, package)
    training = json.loads((output / 'training.json').read_text())
    result = json.loads((output / 'official_metrics.json').read_text())
    own = json.loads((output / 'own_global_metrics.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert result['status'] == own['status'] == 'COMPLETE'
    assert training['schema'] == result['schema'] == own['schema'] == SCHEMA
    assert training['initializer'] == binding and training['condition'] == result['condition']
    assert training['condition']['text_package'] == package
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
    assert own['epoch_selection'] == 'same_fused_mAP_best' and own['distance_sha256'] == base.sha(output / 'own_global_distances.pt')
    observed = training['text_training_diagnostics']
    assert observed['status'] == 'ACTUAL_TEXT_TRAINING_OBSERVATIONS_COMPLETE' and observed['full_frozen_text_state_unchanged']
    assert observed['frozen_text_initial_sha256'] == observed['frozen_text_final_sha256'] == binding['text_state_sha256']
    assert observed['effective_optimizer_updates'] == sum(row['steps'] for row in training['history'])
    old = next(row for row in json.loads(CONTROLS.read_text())['rows']
               if row['dataset'] == dataset and row['variant'] == 'semantic')
    assert (output / 'training_batch_order.jsonl').read_bytes() == (Path(old['run_dir']) / 'training_batch_order.jsonl').read_bytes()
    return dict(dataset=dataset, package=package, status='VERIFIED_COMPLETE', run_dir=str(output),
        initializer=binding, best_epoch=best['epoch'], metrics=result['metrics'], own_global_metrics=own['metrics'],
        checkpoint_sha256=result['checkpoint_sha256'], distance_sha256=result['distance_sha256'],
        receipt_sha256=base.sha(output / 'official_metrics.json'),
        own_global_distance_sha256=own['distance_sha256'], own_global_receipt_sha256=base.sha(output / 'own_global_metrics.json'))


def accept_full(campaign, dataset, package):
    path = campaign / 'acceptance' / f'{dataset}_{package}_full.json'
    assert not path.exists()
    base.queue.write(path, accepted_row(campaign, dataset, package))


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert shutil.disk_usage(ROOT).free >= STORAGE_BYTES
    controls = json.loads(CONTROLS.read_text())
    assert len(controls['rows']) == 6 and all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    sources = source_map()
    prior = json.loads((args.prior_campaign / 'campaign.json').read_text())
    prior_matrix = json.loads((args.prior_campaign / 'accepted_matrix.json').read_text())
    prior_manifest = json.loads((args.prior_campaign / 'manifest.json').read_text())
    assert prior['status'] == 'COMPLETE_WITH_MISSING' and prior['report_exit_code'] == 0
    assert prior_matrix['accepted'] == 0 and prior_matrix['expected'] == 6
    expected = {(d, p) for d in DATASETS for p in PACKAGES}
    assert {(j['dataset'], j['package']) for j in prior['jobs']} == expected
    assert {(j['dataset'], j['package']) for j in prior_matrix['missing']} == expected
    assert all(j['status'] == 'FAILED' and j['failed_phase'] == 'accept-m0' for j in prior['jobs'])
    assert set(prior_manifest['source_sha256']) == set(sources)
    changed = {name for name, digest in sources.items() if prior_manifest['source_sha256'][name] != digest}
    assert changed == {'tools/run_text_semantic_prior.py', 'tools/queue_text_semantic_prior.py'}
    qualified_prefix = args.prior_campaign / 'prefix.json'
    prefix_receipt = json.loads(qualified_prefix.read_text())
    assert prefix_receipt['status'] == 'PREFIX_OUTPUT_AND_INPUT_VJP_PASS' and len(prefix_receipt['rows']) == 4
    assert prefix_receipt['clip_sha256'] == base.sha(previous.WEIGHTS / 'ViT-B-16.pt')
    assert prior_manifest['initialization_sha256'][str(qualified_prefix)] == base.sha(qualified_prefix)
    base.source_map = source_map
    args.campaign.mkdir(parents=True)
    jobs = [dict(dataset=d, package=p, status='PENDING', steps=[]) for d in DATASETS for p in PACKAGES]
    manifest = dict(schema=SCHEMA, source_sha256=sources, initialization_sha256={}, jobs=jobs,
        control_seal_sha256=base.sha(CONTROLS), physical_gpus=[0, 1], max_parallel_jobs=1,
        storage_required_bytes=STORAGE_BYTES, epochs=50, seed=42,
        continuation_from=str(args.prior_campaign), prior_accepted=0, prior_missing=6,
        boundary='Six fixed frozen text-package necessity endpoints; per-end real8M0 then fresh50, no score-driven tuning/retry or old control retraining.')
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(), jobs=jobs, report_invocations=0)
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    prefix = dict(mode='qualified-prefix-reuse', status='REUSED_PASS', source=str(qualified_prefix),
                  source_sha256=base.sha(qualified_prefix), new_neural_forwards=0, new_input_vjps=0)
    state['prefix'] = prefix
    prefix_path = args.campaign / 'prefix.json'
    prefix_path.write_bytes(qualified_prefix.read_bytes())
    manifest['initialization_sha256'][str(prefix_path)] = base.sha(prefix_path)
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for job in jobs:
        dataset, package = job['dataset'], job['package']
        for mode in ('prepare', 'accept-initialization', 'm0', 'accept-m0', 'train', 'evaluate', 'accept-full'):
            invocation = ([sys.executable, '-B', str(Path(__file__).resolve()), '--accept', mode.removeprefix('accept-'),
                '--campaign', str(args.campaign), '--dataset', dataset, '--text-package', package]
                if mode.startswith('accept-') else command(dataset, package, mode, args.campaign))
            step = dict(mode=mode, command=invocation)
            job['steps'].append(step)
            state['status'] = 'RUNNING'
            if previous.run_logged(args.campaign, state, step, f'{dataset}_{package}_{mode}.log'):
                job.update(status='FAILED', failed_phase=mode, exit_code=step['exit_code'])
                break
            if mode == 'prepare':
                path = args.campaign / 'initialization' / f'{dataset}_{package}.json'
                manifest['initialization_sha256'][str(path)] = base.sha(path)
                base.queue.write(args.campaign / 'manifest.json', manifest)
        else:
            job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(),
                result=json.loads((args.campaign / 'acceptance' / f'{dataset}_{package}_full.json').read_text()))
        base.queue.write(args.campaign / 'campaign.json', state)
    rows = [job['result'] for job in jobs if job['status'] == 'COMPLETE']
    missing = [dict(dataset=job['dataset'], package=job['package'], failed_phase=job['failed_phase']) for job in jobs if job['status'] == 'FAILED']
    base.queue.write(args.campaign / 'accepted_matrix.json', dict(schema=SCHEMA, accepted=len(rows), expected=6, rows=rows, missing=missing))
    state.update(status='COMPLETE' if len(rows) == 6 else 'COMPLETE_WITH_MISSING', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as stream:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_text_semantic_prior.py'),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return result.returncode if result.returncode else int(bool(missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--prior-campaign', type=Path)
    parser.add_argument('--accept', choices=('initialization', 'm0', 'full'))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--text-package', choices=PACKAGES)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.accept:
        assert args.dataset is not None and args.text_package is not None
        source_map()
        {'initialization': accept_initialization, 'm0': accept_m0, 'full': accept_full}[args.accept](args.campaign, args.dataset, args.text_package)
    else:
        assert args.report_dir is not None and args.prior_campaign is not None and args.dataset is None and args.text_package is None
        args.report_dir = args.report_dir.resolve()
        args.prior_campaign = args.prior_campaign.resolve()
        raise SystemExit(coordinate(args))
