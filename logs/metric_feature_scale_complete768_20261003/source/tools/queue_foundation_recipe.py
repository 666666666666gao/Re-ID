"""F1: all six M0s precede six fresh full50 runs; one terminal CPU report."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_correspondence_refinement as queue
from tools.queue_correspondence_roles import PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha

SCHEMA = 'trifusion-foundation-recipe-v1'
DATASETS = ('RGBNT201', 'RGBNT100', 'MSVR310')
RECIPES = ('author', 'current')
PREDECESSOR = ROOT / 'refine-logs/foundation_recipe_v1/SEALED_SOURCE243.json'
SOURCES = ('tools/run_foundation_recipe.py', 'tools/queue_foundation_recipe.py',
           'tools/report_foundation_recipe.py',
           'refine-logs/foundation_recipe_v1/SEALED_SOURCE243.json',
           'refine-logs/foundation_recipe_v1/EXPERIMENT_PLAN.md',
           'refine-logs/foundation_recipe_v1/EXPERIMENT_CODE_REVIEW.md')


def source_map():
    old = json.loads(PREDECESSOR.read_text())['source_sha256']
    assert len(old) == 243
    result = {}
    protocols = {str((PROTOCOLS / f'{name}.json').relative_to(ROOT)) for name in DATASETS}
    for name, digest in old.items():
        actual = sha(ROOT / name)
        if name in protocols:
            original = ROOT / '.aris/compute/source2026_protocols' / Path(name).name
            assert sha(original) == digest
            before = json.loads(original.read_text())
            after = json.loads((ROOT / name).read_text())
            assert after['dataset_root'] == f'/data2/gb/Re-ID/dataset/{after["dataset"]}'
            assert {k: v for k, v in before.items() if k != 'dataset_root'} == {
                k: v for k, v in after.items() if k != 'dataset_root'}
        else:
            assert actual == digest, name
        result[name] = actual
    return dict(result, **{name: sha(ROOT / name) for name in SOURCES})


def command(dataset, recipe, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT / 'tools/run_foundation_recipe.py'),
            '--dataset', dataset, '--recipe', recipe, '--mode', mode,
            '--protocol', str(PROTOCOLS / f'{dataset}.json'),
            '--signal-source', str(SOURCE), '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'),
            '--initialization', str(campaign / 'initialization' / f'{dataset}_{recipe}.json'),
            '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def expected_binding(campaign, dataset, recipe):
    witness = json.loads((campaign / 'initialization' / f'{dataset}_{recipe}.json').read_text())
    assert witness['schema'] == SCHEMA and witness['status'] == 'INITIALIZATION_VERIFIED'
    binding = witness['binding']
    assert (binding['dataset'], binding['recipe'], binding['seed']) == (dataset, recipe, 42)
    assert binding['public_clip_sha256'] == sha(WEIGHTS / 'ViT-B-16.pt')
    return binding


def output_dir(campaign, phase, dataset, recipe):
    return ROOT / f'trained-model/{campaign.name}_{phase}_{recipe}_{dataset}'


def verify_m0(campaign, dataset, recipe):
    output = output_dir(campaign, 'm0', dataset, recipe)
    value = json.loads((output / 'training.json').read_text())
    binding = expected_binding(campaign, dataset, recipe)
    assert value['schema'] == SCHEMA and value['status'] == 'M0_PASS'
    assert value['initializer'] == binding and value['dataset'] == dataset and value['recipe'] == recipe
    assert len(value['history']) == 1 and value['history'][0]['steps'] == 8
    assert value['m0']['nonzero_gradient_parameters'] == value['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert value['m0']['reload_max_abs_difference'] <= 1e-5
    assert value['m0']['reload_probe_sha256'] == sha(output / 'm0_reload_probe.pth')
    assert value['frozen_parameters_unchanged'] and value['visual_parameters_changed'] and value['fresh_camera_parameters_changed']
    return value


def verify(campaign, dataset, recipe):
    verify_m0(campaign, dataset, recipe)
    output = output_dir(campaign, 'full', dataset, recipe)
    training = json.loads((output / 'training.json').read_text())
    result = json.loads((output / 'official_metrics.json').read_text())
    binding = expected_binding(campaign, dataset, recipe)
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and result['status'] == 'COMPLETE'
    assert training['schema'] == result['schema'] == SCHEMA and training['initializer'] == binding
    assert training['condition'] == result['condition']
    assert [row['epoch'] for row in training['history']] == list(range(1, 51))
    best = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
    assert result['selected_epoch'] == training['best_epoch'] == best['epoch']
    assert all(abs(result['metrics'][name] - best['official_fused'][name]) < 1e-5 for name in result['metrics'])
    assert result['training_epochs'] == 50 and result['seed'] == 42
    assert result['protocol_sha256'] == binding['protocol_sha256']
    assert result['checkpoint_sha256'] == sha(output / 'best_map.pth')
    assert result['distance_sha256'] == sha(output / 'official_distances.pt')
    assert result['training_best_distance_sha256'] == sha(output / 'best_epoch_distances.pt')
    assert result['independent_upstream_metrics_equal'] and not result['reranking']
    return {'dataset': dataset, 'variant': recipe, 'status': 'VERIFIED_COMPLETE',
            'run_dir': str(output), 'm0_dir': str(output_dir(campaign, 'm0', dataset, recipe)),
            'metrics': result['metrics'], 'best_epoch': best['epoch'], 'initializer': binding,
            'checkpoint_sha256': result['checkpoint_sha256'], 'distance_sha256': result['distance_sha256'],
            'receipt_sha256': sha(output / 'official_metrics.json')}


def require_sources(campaign):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert manifest['source_sha256'] == source_map()
    for name, digest in manifest['initialization_sha256'].items():
        assert sha(Path(name)) == digest
    return manifest


def worker(args):
    require_sources(args.campaign)
    child = queue.child_campaign(args.campaign, args.phase, args.dataset, args.variant)
    assert not child.exists()
    child.mkdir()
    output = output_dir(args.campaign, args.phase, args.dataset, args.variant)
    assert not output.exists()
    state = {'status': 'RUNNING', 'dataset': args.dataset, 'variant': args.variant,
             'phase': args.phase, 'controller_pid': os.getpid(), 'gpu': args.gpu,
             'started_at': queue.stamp(), 'jobs': []}
    for mode in (('m0',) if args.phase == 'm0' else ('train', 'evaluate')):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        row = {'mode': mode, 'status': 'RUNNING', 'started_at': queue.stamp(),
               'output_dir': str(output), 'command': command(args.dataset, args.variant, mode, args.campaign, output)}
        with (child / f'{mode}.log').open('x') as log:
            process = subprocess.Popen(row['command'], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            queue.write(child / 'campaign.json', state)
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=queue.stamp())
        state['status'] = 'FAILED' if code else 'RUNNING'
        queue.write(child / 'campaign.json', state)
        if code:
            return code
    checked = verify_m0(args.campaign, args.dataset, args.variant) if args.phase == 'm0' else verify(args.campaign, args.dataset, args.variant)
    state.update(status='COMPLETE', verification=checked, completed_at=queue.stamp())
    queue.write(child / 'campaign.json', state)
    return 0


def require_complete(child, dataset):
    state = json.loads((child / 'campaign.json').read_text())
    assert state['status'] == 'COMPLETE' and state['dataset'] == dataset
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
    return state['verification']


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
            '--phase', job['phase'], '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def coordinate(args):
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    sources = source_map()
    initial = {}
    queue.write(args.campaign / 'campaign.json', {'status': 'INITIALIZING', 'controller_pid': os.getpid(), 'jobs': []})
    for dataset in DATASETS:
        for recipe in RECIPES:
            with (args.campaign / f'prepare_{recipe}_{dataset}.log').open('x') as log:
                subprocess.run(command(dataset, recipe, 'prepare', args.campaign, ROOT / 'trained-model/f1_prepare_unused'),
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
            path = args.campaign / 'initialization' / f'{dataset}_{recipe}.json'
            initial[str(path)] = sha(path)
        a, b = (expected_binding(args.campaign, dataset, recipe) for recipe in RECIPES)
        assert all(a[key] == b[key] for key in ('visual_initial_sha256', 'camera_initial_sha256', 'current_head_initial_sha256'))
    assert sources == source_map()
    jobs = [{'phase': phase, 'dataset': dataset, 'variant': recipe, 'status': 'PENDING'}
            for phase in ('m0', 'full') for recipe in RECIPES for dataset in DATASETS]
    manifest = {'schema': SCHEMA, 'seed': 42, 'epochs': 50, 'poll_seconds': 240,
                'source_sha256': sources, 'initialization_sha256': initial,
                'jobs': jobs, 'boundary': 'Six package controls; no adapters, roles, rescue or seed selection.'}
    queue.write(args.campaign / 'manifest.json', manifest)
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': queue.stamp(),
             'jobs': jobs, 'report_invocations': 0}
    queue.start_command, queue.require_complete = start_command, require_complete
    for phase in ('m0', 'full'):
        code = queue.run_phase(args.campaign, state, phase)
        if code:
            state.update(status='FAILED', completed_at=queue.stamp())
            queue.write(args.campaign / 'campaign.json', state)
            return code
        if phase == 'm0':
            assert len([row for row in state['jobs'] if row['phase'] == 'm0' and row['status'] == 'COMPLETE']) == 6
            for dataset in DATASETS:
                for recipe in RECIPES:
                    verify_m0(args.campaign, dataset, recipe)
    rows = [verify(args.campaign, dataset, recipe) for recipe in RECIPES for dataset in DATASETS]
    queue.write(args.campaign / 'accepted_matrix.json', {'schema': SCHEMA, 'accepted': 6, 'expected': 6, 'rows': rows})
    state.update(status='COMPLETE', completed_at=queue.stamp(), report_invocations=1)
    queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        process = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_foundation_recipe.py'),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=process.returncode, report_completed_at=queue.stamp())
    queue.write(args.campaign / 'campaign.json', state)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--phase', choices=('m0', 'full'))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--variant', choices=RECIPES)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.phase and args.dataset and args.variant
        return worker(args)
    assert args.report_dir
    args.report_dir = args.report_dir.resolve()
    assert not args.report_dir.exists()
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    assert dict((int(line.split(',')[0]), int(line.split(',')[1])) for line in used.splitlines())[args.gpu] < 500
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
