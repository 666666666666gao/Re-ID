"""Nine matched endpoints: each real M0 precedes its own fresh full50 run."""
import argparse
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
from tools.queue_native_partitioned import verify_m0

SCHEMA = 'trifusion-independent-native-research-v6'
DATASETS = ('RGBNT201', 'MSVR310', 'RGBNT100')
VARIANTS = ('global_only', 'semantic', 'native')
SOURCE, WEIGHTS = base.SOURCE, base.WEIGHTS
PROTOCOLS = ROOT / 'logs/training_feature_scale_protocols_20261002'
SOURCE_SCOPE = ROOT / 'refine-logs/native_research_v6/SOURCE_SCOPE.json'
RESERVE_BYTES = 2 * 1024**3
CAMPAIGN_STORAGE_BYTES = 18 * 384 * 1024**2 + 600 * 1024**2 + 256 * 1024**2


def source_map():
    sources = json.loads(SOURCE_SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def output_dir(campaign, phase, dataset, variant):
    return ROOT / f'trained-model/{campaign.name}_{phase}_{variant}_{dataset}'


def expected_binding(campaign, dataset, variant):
    return base.expected_binding(campaign, dataset, variant)


def command(dataset, variant, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT / 'tools/run_native_research.py'),
        '--dataset', dataset, '--variant', variant, '--mode', mode,
        '--protocol', str(PROTOCOLS / f'{dataset}.json'), '--signal-source', str(SOURCE),
        '--clip-weight', str(WEIGHTS / 'ViT-B-16.pt'),
        '--initialization', str(campaign / 'initialization' / f'{dataset}_{variant}.json'),
        '--output-dir', str(output), '--seed', '42', '--epochs', '50']


def wait_for_devices(campaign, state):
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in rows.splitlines()}
        if used[0] < 500 and used[1] < 500:
            state.update(status='RUNNING', updated_at=base.queue.stamp())
            return
        state.update(status='WAITING_FREE_GPUS', observed_gpu_memory_mib=used, updated_at=base.queue.stamp())
        base.queue.write(campaign / 'campaign.json', state)
        time.sleep(240)


def run_logged(campaign, state, row, log_name):
    base.require_sources(campaign)
    assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES
    wait_for_devices(campaign, state)
    with (campaign / log_name).open('x') as log:
        process = subprocess.Popen(row['command'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1'), stdout=log, stderr=subprocess.STDOUT)
        row.update(status='RUNNING', pid=process.pid, started_at=base.queue.stamp(), gpus=[0, 1])
        state['active_command'] = row
        base.queue.write(campaign / 'campaign.json', state)
        code = process.wait()
    row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=base.queue.stamp())
    state.pop('active_command')
    if code:
        state.update(status='FAILED', completed_at=base.queue.stamp())
    base.queue.write(campaign / 'campaign.json', state)
    return code


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert shutil.disk_usage(ROOT).free >= CAMPAIGN_STORAGE_BYTES + RESERVE_BYTES
    sources = source_map()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase': phase, 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
            for dataset in DATASETS for variant in VARIANTS for phase in ('m0', 'full')]
    manifest = {'schema': SCHEMA, 'seed': 42, 'epochs': 50, 'source_sha256': sources,
                'initialization_sha256': {}, 'jobs': jobs, 'physical_gpus': [0, 1], 'max_parallel_jobs': 1,
                'poll_seconds': 240, 'storage_estimate_bytes': CAMPAIGN_STORAGE_BYTES,
                'boundary': 'User selected per-endpoint research training. Full author batch/AMP/model placement reused; no additional backward-repeatability prerequisite. Historical parity FAIL remains sealed.'}
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': base.queue.stamp(),
             'jobs': jobs, 'preparation': [], 'report_invocations': 0}
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = {'dataset': dataset, 'variant': variant, 'mode': 'prepare',
                   'command': command(dataset, variant, 'prepare', args.campaign, ROOT / 'trained-model/native_research_prepare_unused')}
            state['preparation'].append(row)
            if run_logged(args.campaign, state, row, f'prepare_{dataset}_{variant}.log'):
                return 1
            path = args.campaign / 'initialization' / f'{dataset}_{variant}.json'
            manifest['initialization_sha256'][str(path)] = base.sha(path)
            base.queue.write(args.campaign / 'manifest.json', manifest)
        row = {'dataset': dataset, 'mode': 'initial_forward_pair', 'command': [sys.executable, '-B',
               str(ROOT / 'tools/check_native_research_pair.py'), '--campaign', str(args.campaign), '--dataset', dataset]}
        state['preparation'].append(row)
        if run_logged(args.campaign, state, row, f'initial_forward_pair_{dataset}.log'):
            return 1
        path = args.campaign / f'initial_forward_pair_{dataset}.json'
        manifest['initialization_sha256'][str(path)] = base.sha(path)
        base.queue.write(args.campaign / 'manifest.json', manifest)
        for variant in VARIANTS:
            for phase in ('m0', 'full'):
                if phase == 'full':
                    verify_m0(args.campaign, dataset, variant)
                job = next(row for row in jobs if (row['dataset'], row['variant'], row['phase']) == (dataset, variant, phase))
                job['steps'] = []
                for mode in (('m0',) if phase == 'm0' else ('train', 'evaluate')):
                    row = {'mode': mode, 'command': command(dataset, variant, mode, args.campaign,
                                                          output_dir(args.campaign, phase, dataset, variant))}
                    job['steps'].append(row)
                    if run_logged(args.campaign, state, row, f'{dataset}_{variant}_{mode}.log'):
                        job.update(status='FAILED', exit_code=row['exit_code'])
                        base.queue.write(args.campaign / 'campaign.json', state)
                        return 1
                job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(),
                           result=verify_m0(args.campaign, dataset, variant) if phase == 'm0' else base.verify(args.campaign, dataset, variant))
                base.queue.write(args.campaign / 'campaign.json', state)
    rows = [base.verify(args.campaign, dataset, variant) for dataset in DATASETS for variant in VARIANTS]
    base.queue.write(args.campaign / 'accepted_matrix.json', {'schema': SCHEMA, 'accepted': 9, 'expected': 9, 'rows': rows})
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_native_research.py'),
                                 '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return result.returncode


def configure():
    base.SCHEMA, base.RECIPES = SCHEMA, VARIANTS
    base.source_map, base.command, base.output_dir = source_map, command, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    args.campaign, args.report_dir = args.campaign.resolve(), args.report_dir.resolve()
    configure()
    return coordinate(args)


if __name__ == '__main__':
    raise SystemExit(main())
