"""Run the matched nine-end shared/private evidence-flow panel without retries."""
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
from tools.queue_visual_update_control import SOURCES as PRIOR_SOURCES
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS
from tools.collect_correspondence_roles import sha
from tools.collect_shared_private_evidence import CONDITIONS, DATASETS, collect, verify, verify_m0

SOURCES = (*PRIOR_SOURCES, 'tools/run_shared_private_evidence.py',
           'tools/queue_shared_private_evidence.py', 'tools/collect_shared_private_evidence.py',
           'tools/check_shared_private_initialization.py', 'tools/preflight_shared_private_evidence.py',
           'refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md')


def source_map():
    upstream = [path.relative_to(ROOT).as_posix() for path in sorted(SOURCE.rglob('*.py'))]
    local = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / 'modeling/trifusion').rglob('*.py'))]
    inputs = [path.relative_to(ROOT).as_posix() for dataset in DATASETS
              for path in (PROTOCOLS / f'{dataset}.json', SOURCE / 'configs' / dataset / 'Signal.yml')]
    return {path: sha(ROOT / path) for path in (*SOURCES, *local, *upstream, *inputs)}


def require_sources(campaign):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
    assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
    assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
    return manifest


def command(dataset, variant, mode, output):
    filename, digest = BASELINES[dataset]
    return [sys.executable, '-B', str(ROOT / 'tools/run_shared_private_evidence.py'),
            '--dataset', dataset, '--mode', mode, '--evidence-flow', variant,
            '--protocol', str(PROTOCOLS / f'{dataset}.json'), '--signal-source', str(SOURCE),
            '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'), '--baseline-checkpoint', str(WEIGHTS / filename),
            '--baseline-sha256', digest, '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def worker(args):
    manifest = require_sources(args.campaign)
    child = queue.child_campaign(args.campaign, 'shared_private', args.dataset, args.variant)
    assert not child.exists()
    child.mkdir()
    state = {'status': 'RUNNING', 'dataset': args.dataset, 'variant': args.variant,
             'gpu': args.gpu, 'controller_pid': os.getpid(), 'started_at': queue.stamp(), 'jobs': []}
    preflight = json.loads(Path(manifest['preflight_path']).read_text())
    reused = [row for row in preflight['jobs'] if (row['dataset'], row['variant']) == (args.dataset, args.variant)]
    assert len(reused) == (1 if args.dataset == 'RGBNT201' else 0)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for mode in ('m0', 'train', 'evaluate'):
        require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
        if mode == 'm0' and reused:
            row = dict(reused[0], origin='verified_preflight')
            verify_m0(Path(row['output_dir']), args.dataset, args.variant, manifest)
            state['jobs'].append(row)
            queue.write(child / 'campaign.json', state)
            continue
        output = ROOT / f"trained-model/{child.name}_seed42_{'m0' if mode == 'm0' else 'full'}"
        row = {'mode': mode, 'status': 'RUNNING', 'started_at': queue.stamp(),
               'output_dir': str(output), 'command': command(args.dataset, args.variant, mode, output)}
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
        if mode == 'm0':
            verify_m0(output, args.dataset, args.variant, manifest)
    result = verify(output, Path(state['jobs'][0]['output_dir']), args.dataset, args.variant, manifest)
    state.update(status='COMPLETE', completed_at=queue.stamp(), verification=result)
    queue.write(child / 'campaign.json', state)
    return 0


def require_complete(child, dataset):
    # Verification runs inside the worker, so its failure is a nonzero worker exit
    # that the existing scheduler handles while draining its other active jobs.
    state = json.loads((child / 'campaign.json').read_text())
    return state['verification']


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
            '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def coordinate(args):
    previous = json.loads((args.after_campaign / 'campaign.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['jobs']) == 12
    assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in previous['jobs'])
    summary = json.loads(args.after_summary.read_text())
    assert sha(args.after_summary) == args.after_summary_sha256
    assert summary['schema'] == 'visual-update-twelve-end-complete-analysis-v1' and summary['accepted'] == 12
    assert all(summary['source_artifacts_sha256'][str(path.resolve())] == sha(path)
               for path in (args.after_campaign / name for name in
                            ('campaign.json', 'manifest.json', 'accepted_matrix.json')))
    witness = json.loads(args.initialization_witness.read_text())
    assert witness['status'] == 'MATCHED_COMMON_INITIALIZATION_PASS'
    assert witness['source_sha256'] == sha(ROOT / 'tools/check_shared_private_initialization.py')
    assert witness['entry_sha256'] == sha(ROOT / 'tools/run_shared_private_evidence.py')
    expected = [(dataset, variant) for variant in CONDITIONS for dataset in DATASETS]
    assert {(row['dataset'], row['variant']) for row in witness['rows']} == set(expected)
    assert len(witness['rows']) == 9 and all(row['common_states_bitwise_equal'] for row in witness['rows'])
    preflight = json.loads(args.preflight.read_text())
    assert preflight['status'] == 'COMPLETE'
    assert [(row['dataset'], row['variant']) for row in preflight['jobs']] == [
        ('RGBNT201', variant) for variant in CONDITIONS]
    assert all(row['mode'] == 'm0' and row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in preflight['jobs'])
    sources = source_map()
    assert preflight['source_sha256'] == witness['source_snapshot_sha256'] == sources
    assert witness['preflight_sha256'] == sha(args.preflight)
    for filename, digest in BASELINES.values():
        assert sha(WEIGHTS / filename) == digest
    assert not args.campaign.exists()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase': 'shared_private', 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
            for dataset, variant in expected]
    manifest = {'schema': 'trifusion-shared-private-panel-v1', 'seed': 42, 'epochs': 50,
                'poll_seconds': queue.POLL_SECONDS, 'jobs': jobs,
                'initialization_witness_path': str(args.initialization_witness),
                'initialization_witness_sha256': sha(args.initialization_witness),
                'preflight_path': str(args.preflight), 'preflight_sha256': sha(args.preflight),
                'after_campaign': str(args.after_campaign), 'after_summary_path': str(args.after_summary),
                'after_summary_sha256': args.after_summary_sha256,
                'source_sha256': sources,
                'boundary': 'Fresh full50; three evidence flows, FP32 low-LR visual storage, matched shared initialization and coupled/separated capacity; no retry.'}
    for row in preflight['jobs']:
        verify_m0(Path(row['output_dir']), row['dataset'], row['variant'], manifest)
    queue.write(args.campaign / 'manifest.json', manifest)
    state = {'status': 'RUNNING', 'phase': 'shared_private', 'controller_pid': os.getpid(),
             'started_at': queue.stamp(), 'jobs': jobs}
    queue.start_command = start_command
    queue.require_complete = require_complete
    code = queue.run_phase(args.campaign, state, 'shared_private')
    if code == 0:
        accepted = collect(args.campaign)
        assert accepted['verified_complete'] == accepted['expected_endpoints'] == 9
        queue.write(args.campaign / 'accepted_matrix.json', accepted)
    state.update(status='FAILED' if code else 'COMPLETE', completed_at=queue.stamp())
    queue.write(args.campaign / 'campaign.json', state)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--variant', choices=tuple(CONDITIONS))
    parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--initialization-witness', type=Path)
    parser.add_argument('--after-campaign', type=Path)
    parser.add_argument('--after-summary', type=Path)
    parser.add_argument('--after-summary-sha256')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.worker:
        assert args.dataset is not None and args.variant is not None and args.gpu is not None
        return worker(args)
    assert args.preflight and args.initialization_witness and args.after_campaign and args.after_summary and args.after_summary_sha256
    args.preflight, args.initialization_witness, args.after_campaign, args.after_summary = (
        path.resolve() for path in (args.preflight, args.initialization_witness, args.after_campaign, args.after_summary))
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
