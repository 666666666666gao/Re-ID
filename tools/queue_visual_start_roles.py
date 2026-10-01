"""Run six fresh fixed-architecture visual-initialization endpoints on free GPUs."""
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
from tools.queue_role_global_tokens import SOURCE_PATHS as REUSED_SOURCES
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha
from tools.collect_visual_start_roles import CONDITIONS, DATASETS, collect, verify

SOURCES = (*REUSED_SOURCES, 'tools/run_visual_start_roles.py', 'tools/queue_visual_start_roles.py',
           'tools/collect_visual_start_roles.py', 'tools/prepare_visual_start_inputs.py',
           'tools/check_visual_start_initialization.py', 'refine-logs/visual_start_roles_v1/EXPERIMENT_PLAN.md')


def require_sources(campaign):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert sha(Path(manifest['inputs_path'])) == manifest['inputs_sha256']
    assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    return manifest


def command(manifest, dataset, variant, mode, output):
    inputs = json.loads(Path(manifest['inputs_path']).read_text())
    weight = inputs['datasets'][dataset][variant]
    return [sys.executable, '-B', str(ROOT / 'tools/run_visual_start_roles.py'),
            '--visual-start', variant, '--visual-start-inputs', manifest['inputs_path'],
            '--dataset', dataset, '--mode', mode, '--protocol', str(PROTOCOLS / f'{dataset}.json'),
            '--signal-source', str(SOURCE), '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'),
            '--baseline-checkpoint', weight['path'], '--baseline-sha256', weight['sha256'],
            '--output-dir', str(output), '--seed', '42', '--epochs', '50', '--width', '128',
            '--pred-weight', '0.1', '--m1', '--m2', '--no-m3', '--query-mode', 'context',
            '--auxiliary-target', 'none', '--token-mode', 'static']


def worker(args):
    manifest = require_sources(args.campaign)
    child = queue.child_campaign(args.campaign, 'visual_start', args.dataset, args.variant)
    assert not child.exists()
    child.mkdir()
    state = {'status': 'RUNNING', 'dataset': args.dataset, 'variant': args.variant,
             'gpu': args.gpu, 'controller_pid': os.getpid(), 'started_at': queue.stamp(), 'jobs': []}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for mode in ('m0', 'train', 'evaluate'):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        suffix = 'm0' if mode == 'm0' else 'full'
        output = ROOT / f'trained-model/{child.name}_seed42_{suffix}'
        row = {'mode': mode, 'status': 'RUNNING', 'started_at': queue.stamp(),
               'output_dir': str(output), 'command': command(manifest, args.dataset, args.variant, mode, output)}
        with (child / f'{mode}.log').open('x') as log:
            process = subprocess.Popen(row['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            queue.write(child / 'campaign.json', state)
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=queue.stamp())
        state['status'] = 'FAILED' if code else 'RUNNING'
        queue.write(child / 'campaign.json', state)
        if code:
            return code
        receipt = json.loads((output / ('official_metrics.json' if mode == 'evaluate' else 'training.json')).read_text())
        assert receipt['status'] == {'m0': 'M0_PASS', 'train': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 'evaluate': 'COMPLETE'}[mode]
        if mode == 'm0':
            assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
            assert receipt['m0']['frozen_signal_unchanged']
            assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    result = verify(output, Path(state['jobs'][0]['output_dir']), args.dataset, args.variant, manifest)
    state.update(status='COMPLETE', completed_at=queue.stamp(), verification=result)
    queue.write(child / 'campaign.json', state)
    return 0


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
            '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def coordinate(args):
    previous = json.loads((args.after_campaign / 'campaign.json').read_text())
    prior_manifest = json.loads((args.after_campaign / 'manifest.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['jobs']) == 9
    assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in previous['jobs'])
    assert sha(args.after_campaign / 'accepted_matrix.json') == args.after_matrix_sha256
    assert all(sha(ROOT / name) == digest for name, digest in prior_manifest['source_sha256'].items())
    witness = json.loads(args.initialization_witness.read_text())
    assert witness['status'] == 'MATCHED_TRAINABLE_INITIALIZATION_PASS'
    assert witness['inputs_sha256'] == sha(args.inputs)
    assert witness['source_sha256'] == sha(ROOT / 'tools/check_visual_start_initialization.py')
    assert {row['dataset'] for row in witness['rows']} == set(DATASETS)
    assert all(row['trainable_states_bitwise_equal'] and row['frozen_visual_states_differ'] for row in witness['rows'])
    for filename, digest in BASELINES.values():
        assert sha(WEIGHTS / filename) == digest
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase': 'visual_start', 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
            for variant in CONDITIONS for dataset in DATASETS]
    upstream = [p.relative_to(ROOT).as_posix() for p in sorted(SOURCE.rglob('*.py'))]
    local = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'modeling/trifusion').rglob('*.py'))]
    inputs = [p.relative_to(ROOT).as_posix() for dataset in DATASETS
              for p in (PROTOCOLS / f'{dataset}.json', SOURCE / 'configs' / dataset / 'Signal.yml')]
    manifest = {'schema': 'trifusion-visual-start-panel-v1', 'seed': 42, 'epochs': 50,
                'poll_seconds': queue.POLL_SECONDS, 'inputs_path': str(args.inputs), 'inputs_sha256': sha(args.inputs),
                'initialization_witness_path': str(args.initialization_witness),
                'initialization_witness_sha256': sha(args.initialization_witness),
                'after_campaign': str(args.after_campaign), 'after_matrix_sha256': args.after_matrix_sha256,
                'jobs': jobs, 'source_sha256': {name: sha(ROOT / name) for name in (*SOURCES, *local, *upstream, *inputs)},
                'boundary': 'Exactly152 frozen visual tensors differ; all other initial states match. Fixed STATIC role architecture, full50 and official-mAP-best. No auto-retry.'}
    queue.write(args.campaign / 'manifest.json', manifest)
    state = {'status': 'RUNNING', 'phase': 'visual_start', 'controller_pid': os.getpid(), 'started_at': queue.stamp(), 'jobs': jobs}
    queue.start_command = start_command
    code = queue.run_phase(args.campaign, state, 'visual_start')
    if code == 0:
        require_sources(args.campaign)
        accepted = collect(args.campaign)
        assert accepted['verified_complete'] == accepted['expected_endpoints'] == 6
        queue.write(args.campaign / 'accepted_matrix.json', accepted)
    state.update(status='FAILED' if code else 'COMPLETE', completed_at=queue.stamp())
    queue.write(args.campaign / 'campaign.json', state)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--variant', choices=CONDITIONS)
    parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--initialization-witness', type=Path)
    parser.add_argument('--after-campaign', type=Path)
    parser.add_argument('--after-matrix-sha256')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return worker(args)
    assert args.inputs and args.initialization_witness and args.after_campaign and args.after_matrix_sha256
    args.inputs, args.initialization_witness, args.after_campaign = (p.resolve() for p in (args.inputs, args.initialization_witness, args.after_campaign))
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
