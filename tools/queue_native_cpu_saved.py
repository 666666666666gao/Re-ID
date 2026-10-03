"""Nine matched author-package runs, 2026 physical GPU0-3 only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_foundation_recipe as base

SCHEMA = 'trifusion-independent-native-cpu-saved-v4'
DATASETS = base.DATASETS
VARIANTS = ('global_only', 'semantic', 'native')
SOURCE, WEIGHTS = base.SOURCE, base.WEIGHTS
PROTOCOLS = ROOT / 'logs/training_feature_scale_protocols_20261002'
PREVIOUS = ROOT / 'logs/independent_native_evidence_20261003_v3'
SOURCES = (
    'modeling/trifusion/cpu_saved_evidence_clip.py',
    'tools/check_cpu_saved_backward.py',
    'modeling/trifusion/image_native_evidence.py',
    'modeling/trifusion/independent_native_roles.py',
    'modeling/trifusion/evidence_author_heads.py',
    'tools/run_native_cpu_saved.py',
    'tools/check_cpu_saved_native_pair.py',
    'tools/queue_native_cpu_saved.py',
    'tools/report_native_cpu_saved.py',
    'refine-logs/native_cpu_saved_v4/EXPERIMENT_PLAN.md',
    'refine-logs/native_cpu_saved_v4/EXPERIMENT_CODE_REVIEW.md',
)
# F3's fixed 10GiB per-stage guard actually interrupted a finished training.
# Budget this NEW nine-arm campaign once; do not change F3's original guard.
RESERVE_BYTES = 2 * 1024**3
CAMPAIGN_STORAGE_BYTES = 18 * 384 * 1024**2 + 600 * 1024**2 + 256 * 1024**2
OUTPUT_ROOT = Path('/home/gaob/trifusion-native-evidence-v4')


def source_map():
    old = json.loads((ROOT / 'logs/native_original_repeat_result776_20261003/terminal/INTAKE.json').read_text())['source_sha256']
    assert len(old) == 293
    assert all(base.sha(ROOT / name) == digest for name, digest in old.items())
    return dict(old, **{name: base.sha(ROOT / name) for name in SOURCES})


def command(dataset, variant, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT / 'tools/run_native_cpu_saved.py'),
        '--dataset', dataset, '--variant', variant, '--mode', mode,
        '--protocol', str(PROTOCOLS / f'{dataset}.json'), '--signal-source', str(SOURCE),
        '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'),
        '--initialization', str(campaign / 'initialization' / f'{dataset}_{variant}.json'),
        '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def expected_binding(campaign, dataset, variant):
    return base.expected_binding(campaign, dataset, variant)


def output_dir(campaign, phase, dataset, variant):
    return OUTPUT_ROOT / f'{campaign.name}_{phase}_{variant}_{dataset}'


def start_command(campaign, job, gpu):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--campaign', str(campaign),
        '--phase', job['phase'], '--dataset', job['dataset'], '--variant', job['variant'], '--gpu', str(gpu)]


def verify_m0(campaign, dataset, variant):
    value = base.verify_m0(campaign, dataset, variant)
    probe = value['production_m0_diagnostics']
    assert probe['effective_optimizer_updates'] == 8
    assert all(count == 8 for count in probe['author_bn_batches_tracked'].values())
    if variant == 'native':
        assert len(probe['detail_parameters']) == 14 and len(probe['detail_updates']) == 8
        assert all(probe['detail_updates'][-1][name]['parameter_delta_from_initial_max_abs'] > 0
                   for name in probe['detail_parameters'])
    return value


def worker(args):
    base.require_sources(args.campaign)
    child = base.queue.child_campaign(args.campaign, args.phase, args.dataset, args.variant)
    assert not child.exists()
    child.mkdir()
    output = base.output_dir(args.campaign, args.phase, args.dataset, args.variant)
    assert not output.exists()
    state = {'status': 'RUNNING', 'dataset': args.dataset, 'variant': args.variant,
        'phase': args.phase, 'controller_pid': os.getpid(), 'gpu': args.gpu,
        'started_at': base.queue.stamp(), 'jobs': []}
    for mode in (('m0',) if args.phase == 'm0' else ('train', 'evaluate')):
        base.require_sources(args.campaign)
        assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES
        assert shutil.disk_usage(OUTPUT_ROOT).free >= RESERVE_BYTES
        row = {'mode': mode, 'status': 'RUNNING', 'started_at': base.queue.stamp(),
            'output_dir': str(output), 'command': command(args.dataset, args.variant, mode, args.campaign, output)}
        with (child / f'{mode}.log').open('x') as log:
            process = subprocess.Popen(row['command'], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['jobs'].append(row)
            base.queue.write(child / 'campaign.json', state)
            code = process.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=base.queue.stamp())
        state['status'] = 'FAILED' if code else 'RUNNING'
        base.queue.write(child / 'campaign.json', state)
        if code:
            return code
    checked = verify_m0(args.campaign, args.dataset, args.variant) if args.phase == 'm0' else base.verify(args.campaign, args.dataset, args.variant)
    state.update(status='COMPLETE', verification=checked, completed_at=base.queue.stamp())
    base.queue.write(child / 'campaign.json', state)
    return 0


def run_phase(campaign, state, phase):
    pending = [job for job in state['jobs'] if job['phase'] == phase]
    active, failed = [], False
    state.update(status='RUNNING', phase=phase, updated_at=base.queue.stamp())
    while pending or active:
        for job, process in list(active):
            code = process.poll()
            if code is None:
                continue
            if code == 0:
                child = base.queue.child_campaign(campaign, phase, job['dataset'], job['variant'])
                job['result'] = base.require_complete(child, job['dataset'])
            job.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=base.queue.stamp())
            failed |= code != 0
            active.remove((job, process))
        if not failed:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                '--format=csv,noheader,nounits'], text=True)
            occupied = {job['gpu'] for job, _ in active}
            available = [int(row.split(',')[0]) for row in memory.splitlines()
                if int(row.split(',')[0]) in range(4) and int(row.split(',')[0]) not in occupied
                and int(row.split(',')[1]) < 500]
            for gpu in available:
                if not pending:
                    break
                assert len(active) < 4 and shutil.disk_usage(ROOT).free >= RESERVE_BYTES
                assert shutil.disk_usage(OUTPUT_ROOT).free >= RESERVE_BYTES
                job = pending.pop(0)
                job.update(gpu=gpu, status='RUNNING', started_at=base.queue.stamp(), command=start_command(campaign, job, gpu))
                with (campaign / f"{phase}_{job['variant']}_{job['dataset']}.log").open('x') as log:
                    process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                active.append((job, process))
        state['updated_at'] = base.queue.stamp()
        if failed:
            state['status'] = 'FAILED'
        base.queue.write(campaign / 'campaign.json', state)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(240)
    return 0


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    previous = json.loads((PREVIOUS / 'campaign.json').read_text())
    assert previous['status'] == 'INITIALIZING' and previous['jobs'] == []
    assert not (PREVIOUS / 'manifest.json').exists()
    assert not (Path('/proc') / str(previous['controller_pid'])).exists()
    repeat = json.loads((ROOT / 'logs/native_original_backward_repeat_20261003_v1/JOB.json').read_text())
    assert repeat['status'] == 'COMPLETE' and repeat['exit_code'] == 0
    assert not (Path('/proc') / str(repeat['controller_pid'])).exists()
    assert json.loads((ROOT / 'logs/native_original_backward_repeat_20261003_v1/original_backward_repeat_RGBNT201.json').read_text())['status'] == 'ORIGINAL_BACKWARD_REPEAT_PARITY_PASS'
    assert not args.campaign.exists()
    assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES
    assert not OUTPUT_ROOT.exists()
    OUTPUT_ROOT.mkdir()
    assert shutil.disk_usage(OUTPUT_ROOT).free >= CAMPAIGN_STORAGE_BYTES + RESERVE_BYTES
    args.campaign.mkdir(parents=True)
    sources, initial = source_map(), {}
    base.queue.write(args.campaign / 'campaign.json', {'status': 'INITIALIZING', 'controller_pid': os.getpid(), 'jobs': []})
    for dataset in DATASETS:
        for variant in VARIANTS:
            with (args.campaign / f'prepare_{variant}_{dataset}.log').open('x') as log:
                subprocess.run(command(dataset, variant, 'prepare', args.campaign, ROOT / 'trained-model/native_prepare_unused'),
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
            path = args.campaign / 'initialization' / f'{dataset}_{variant}.json'
            initial[str(path)] = base.sha(path)
        with (args.campaign / f'initial_forward_pair_{dataset}.log').open('x') as log:
            subprocess.run([sys.executable, '-B', str(ROOT / 'tools/check_cpu_saved_native_pair.py'),
                '--campaign', str(args.campaign), '--dataset', dataset], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
        path = args.campaign / f'initial_forward_pair_{dataset}.json'
        initial[str(path)] = base.sha(path)
        with (args.campaign / f'cpu_saved_backward_{dataset}.log').open('x') as log:
            subprocess.run([sys.executable, '-B', str(ROOT / 'tools/check_cpu_saved_backward.py'),
                '--campaign', str(args.campaign), '--dataset', dataset], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
        path = args.campaign / f'cpu_saved_backward_{dataset}.json'
        initial[str(path)] = base.sha(path)
    assert sources == source_map()
    jobs = [{'phase': phase, 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
        for phase in ('m0', 'full') for variant in VARIANTS for dataset in DATASETS]
    base.queue.write(args.campaign / 'manifest.json', {'schema': SCHEMA, 'seed': 42, 'epochs': 50,
        'poll_seconds': 240, 'source_sha256': sources, 'initialization_sha256': initial, 'jobs': jobs,
        'storage_estimate_bytes': CAMPAIGN_STORAGE_BYTES, 'free_reserve_bytes': RESERVE_BYTES,
        'output_root': str(OUTPUT_ROOT),
        'boundary': 'Computation-only v4 after preserved V1 OOM and V2/V3 gradient-gate failures plus one actual original-repeat PASS; original graph and author batches with visual saved tensors on CPU, no checkpointing; global-only shared adapters, semantic roles and independent additive native detail; no auxiliary loss or F3 rescue;2026GPU0-3/max4.'})
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': base.queue.stamp(),
        'jobs': jobs, 'report_invocations': 0}
    for phase in ('m0', 'full'):
        if run_phase(args.campaign, state, phase):
            state.update(status='FAILED', completed_at=base.queue.stamp())
            base.queue.write(args.campaign / 'campaign.json', state)
            return 1
        if phase == 'm0':
            for dataset in DATASETS:
                for variant in VARIANTS:
                    verify_m0(args.campaign, dataset, variant)
    rows = [base.verify(args.campaign, dataset, variant) for variant in VARIANTS for dataset in DATASETS]
    base.queue.write(args.campaign / 'accepted_matrix.json', {'schema': SCHEMA, 'accepted': 9, 'expected': 9, 'rows': rows})
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        process = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_native_cpu_saved.py'),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=process.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return process.returncode


def configure():
    base.SCHEMA, base.RECIPES = SCHEMA, VARIANTS
    base.source_map, base.command, base.start_command = source_map, command, start_command
    base.output_dir = output_dir
    base.worker, base.coordinate = worker, coordinate


if __name__ == '__main__':
    configure()
    raise SystemExit(base.main())
