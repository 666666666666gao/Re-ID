"""Six fresh objective-ownership interventions with sealed formal controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_native_research as previous

SCHEMA = 'trifusion-global-task-role-v1'
DATASETS = previous.DATASETS
VARIANTS = ('semantic', 'native')
SOURCE_SCOPE = ROOT / 'refine-logs/global_task_role_v1/SOURCE_SCOPE.json'
CONTROLS = ROOT / 'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
HISTORICAL_CONTROLS = ROOT / 'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
base = previous.base
SOURCE, WEIGHTS, PROTOCOLS = previous.SOURCE, previous.WEIGHTS, previous.PROTOCOLS


def source_map():
    sources = json.loads(SOURCE_SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def require_controls():
    controls = json.loads(CONTROLS.read_text())
    assert controls['schema'] == 'trifusion-role-input-detach-fixed-best-diagnosis-v1'
    assert len(controls['rows']) == 9
    assert all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    historical = json.loads(HISTORICAL_CONTROLS.read_text())
    assert historical['schema'] == 'trifusion-native-fixed-best-diagnosis-v1' and len(historical['rows']) == 9
    assert all(base.sha(Path(name)) == digest for name, digest in historical['artifact_sha256'].items())
    controls['original_rows'] = historical['rows']
    return controls


def command(dataset, variant, mode, campaign, output):
    result = previous.command(dataset, variant, mode, campaign, output)
    result[2] = str(ROOT / 'tools/run_global_task_role.py')
    return result


def configure():
    base.SCHEMA, base.RECIPES = SCHEMA, VARIANTS
    base.source_map, base.command, base.output_dir = source_map, command, previous.output_dir


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert shutil.disk_usage(ROOT).free >= 6 * 2 * 384 * 1024**2 + 2 * 1024**3
    controls = require_controls()
    sources = source_map()
    args.campaign.mkdir(parents=True)
    jobs = [{'phase': phase, 'dataset': dataset, 'variant': variant, 'status': 'PENDING'}
            for dataset in DATASETS for variant in VARIANTS for phase in ('m0', 'full')]
    manifest = {'schema': SCHEMA, 'seed': 42, 'epochs': 50, 'source_sha256': sources,
        'initialization_sha256': {}, 'jobs': jobs, 'physical_gpus': [0, 1], 'max_parallel_jobs': 1,
        'poll_seconds': 240, 'control_seal_sha256': base.sha(CONTROLS),
        'historical_control_seal_sha256': base.sha(HISTORICAL_CONTROLS),
        'boundary': 'Same inference/state/capacity/recipe and detached role inputs; author global objective updates shared/head, stateless frozen author fused head trains roles/readout/gain. Both historical families preserved, no retired M0 verifier or parity repair.'}
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': base.queue.stamp(),
             'jobs': jobs, 'preparation': [], 'report_invocations': 0}
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = {'dataset': dataset, 'variant': variant, 'mode': 'prepare',
                'command': command(dataset, variant, 'prepare', args.campaign, ROOT / 'trained-model/global_task_role_prepare_unused')}
            state['preparation'].append(row)
            if previous.run_logged(args.campaign, state, row, f'prepare_{dataset}_{variant}.log'):
                return 1
            witness = args.campaign / 'initialization' / f'{dataset}_{variant}.json'
            binding = base.expected_binding(args.campaign, dataset, variant)
            original = next(r['initializer'] for r in controls['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
            assert binding['role_input_gradient_policy'] == 'detach_stages_context_shared_global_at_role_read_only'
            assert binding['objective_gradient_policy'] == 'author_global_loss_to_shared_and_heads_fused_loss_to_roles_only'
            excluded = ('architecture', 'entry_sha256', 'scope', 'objective_gradient_policy')
            assert {k: v for k, v in binding.items() if k not in excluded} == {
                k: v for k, v in original.items() if k not in excluded}
            manifest['initialization_sha256'][str(witness)] = base.sha(witness)
            base.queue.write(args.campaign / 'manifest.json', manifest)
            for phase in ('m0', 'full'):
                if phase == 'full':
                    previous.verify_m0(args.campaign, dataset, variant)
                job = next(r for r in jobs if (r['dataset'], r['variant'], r['phase']) == (dataset, variant, phase))
                job['steps'] = []
                for mode in (('m0',) if phase == 'm0' else ('train', 'evaluate')):
                    row = {'mode': mode, 'command': command(dataset, variant, mode, args.campaign,
                            previous.output_dir(args.campaign, phase, dataset, variant))}
                    job['steps'].append(row)
                    if previous.run_logged(args.campaign, state, row, f'{dataset}_{variant}_{mode}.log'):
                        job.update(status='FAILED', exit_code=row['exit_code'])
                        base.queue.write(args.campaign / 'campaign.json', state)
                        return 1
                job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(),
                    result=previous.verify_m0(args.campaign, dataset, variant) if phase == 'm0' else base.verify(args.campaign, dataset, variant))
                base.queue.write(args.campaign / 'campaign.json', state)
    require_controls()
    rows = [base.verify(args.campaign, dataset, variant) for dataset in DATASETS for variant in VARIANTS]
    base.queue.write(args.campaign / 'accepted_matrix.json', {'schema': SCHEMA, 'accepted': 6, 'expected': 6, 'rows': rows})
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_global_task_role.py'),
            '--campaign', str(args.campaign), '--output-dir', str(args.report_dir)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode, report_completed_at=base.queue.stamp())
    base.queue.write(args.campaign / 'campaign.json', state)
    return result.returncode


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
