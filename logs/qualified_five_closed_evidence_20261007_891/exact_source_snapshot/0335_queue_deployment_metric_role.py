"""Six role-metric endpoints; retire each probe only after strict acceptance."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_global_task_role as previous

SCHEMA = 'trifusion-deployment-metric-role-v1'
METRIC_POLICY = 'role_triplet_joint_1536_l2_h_global_triplet_original_raw_parts'
DATASETS, VARIANTS = previous.DATASETS, previous.VARIANTS
base = previous.base
SOURCE_SCOPE = ROOT / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
CONTROLS = ROOT / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'


def source_map():
    sources = json.loads(SOURCE_SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def require_controls():
    controls = json.loads(CONTROLS.read_text())
    assert controls['schema'] == 'trifusion-global-task-role-fixed-best-diagnosis-v1'
    assert len(controls['rows']) == 9 and len(controls['artifact_sha256']) == 187
    assert all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    return controls


def command(dataset, variant, mode, campaign, output):
    result = previous.command(dataset, variant, mode, campaign, output)
    result[2] = str(ROOT / 'tools/run_deployment_metric_role.py')
    return result


def configure():
    previous.configure()
    base.SCHEMA = SCHEMA
    base.source_map, base.command = source_map, command


def accept_and_retire_probe(campaign, dataset, variant):
    # Original strict verification still needs the probe and runs BEFORE unlink.
    row = base.verify(campaign, dataset, variant)
    m0_dir, run = Path(row['m0_dir']), Path(row['run_dir'])
    probe = m0_dir / 'm0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT / 'trained-model').resolve())
    assert probe.parent.name == f'{campaign.name}_m0_{variant}_{dataset}'
    m0 = json.loads((m0_dir / 'training.json').read_text())
    assert base.sha(probe) == m0['m0']['reload_probe_sha256']
    files = [p for folder in (m0_dir, run) for p in folder.iterdir() if p.is_file() and p != probe]
    assert {p.name for p in run.glob('*.pth')} == {'best_map.pth'}
    assert {p.name for p in m0_dir.glob('*.pth')} == {'m0_reload_probe.pth'}
    receipt = dict(status='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT',
                   verified_at=base.queue.stamp(), row=row,
                   artifact_sha256={str(p): base.sha(p) for p in files},
                   probe=dict(path=str(probe), bytes=probe.stat().st_size, sha256=base.sha(probe)))
    path = campaign / 'acceptance' / f'{dataset}_{variant}.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    base.queue.write(path, receipt)
    probe.unlink()
    assert not probe.exists()
    with (campaign / 'probe_retirement.jsonl').open('a') as log:
        log.write(json.dumps(dict(dataset=dataset, variant=variant, retired_at=base.queue.stamp(),
                                 acceptance_sha256=base.sha(path), **receipt['probe'])) + '\n')
    assert accepted_row(campaign, dataset, variant) == row
    return row


def accepted_row(campaign, dataset, variant):
    # Validate immutable acceptance artifacts, not the now-retired probe binary.
    path = campaign / 'acceptance' / f'{dataset}_{variant}.json'
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
    assert all(base.sha(Path(name)) == digest for name, digest in receipt['artifact_sha256'].items())
    records = [json.loads(line) for line in (campaign / 'probe_retirement.jsonl').read_text().splitlines()]
    matching = [item for item in records if (item['dataset'], item['variant']) == (dataset, variant)]
    assert len(matching) == 1 and matching[0]['acceptance_sha256'] == base.sha(path)
    assert all(matching[0][key] == value for key, value in receipt['probe'].items())
    assert not Path(receipt['probe']['path']).exists()
    assert receipt['row']['initializer'] == base.expected_binding(campaign, dataset, variant)
    return receipt['row']


def coordinate(args):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert not args.campaign.exists() and not args.report_dir.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    # Six final bests, at most one live M0 probe, plus unchanged 2GiB reserve.
    assert shutil.disk_usage(ROOT).free >= 7 * 384 * 1024**2 + 2 * 1024**3
    controls, sources = require_controls(), source_map()
    args.campaign.mkdir(parents=True)
    jobs = [dict(phase=phase, dataset=d, variant=v, status='PENDING')
            for d in DATASETS for v in VARIANTS for phase in ('m0', 'full')]
    manifest = dict(schema=SCHEMA, seed=42, epochs=50, source_sha256=sources, initialization_sha256={},
                    jobs=jobs, physical_gpus=[0, 1], max_parallel_jobs=1, poll_seconds=240,
                    control_seal_sha256=base.sha(CONTROLS), role_metric_policy=METRIC_POLICY,
                    storage_policy='strict_accept_each_full_then_hash_seal_and_retire_its_m0_probe',
                    boundary='Role metric input only; raw BN logits, original global author loss, state, recipe and deployment unchanged. Vehicles change both L2 and joint geometry. No power/temperature action.')
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(),
                 jobs=jobs, preparation=[], report_invocations=0)
    base.queue.write(args.campaign / 'manifest.json', manifest)
    base.queue.write(args.campaign / 'campaign.json', state)
    for dataset in DATASETS:
        for variant in VARIANTS:
            prep = dict(dataset=dataset, variant=variant, mode='prepare',
                        command=command(dataset, variant, 'prepare', args.campaign, ROOT / 'trained-model/deployment_metric_prepare_unused'))
            state['preparation'].append(prep)
            if previous.previous.run_logged(args.campaign, state, prep, f'prepare_{dataset}_{variant}.log'):
                return 1
            witness = args.campaign / 'initialization' / f'{dataset}_{variant}.json'
            binding = base.expected_binding(args.campaign, dataset, variant)
            original = next(r['initializer'] for r in controls['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
            assert binding['role_metric_policy'] == METRIC_POLICY
            assert binding['objective_gradient_policy'] == 'author_global_loss_to_shared_and_heads_fused_loss_to_roles_only'
            excluded = ('architecture', 'entry_sha256', 'scope', 'role_metric_policy')
            assert {k: v for k, v in binding.items() if k not in excluded} == {k: v for k, v in original.items() if k not in excluded}
            manifest['initialization_sha256'][str(witness)] = base.sha(witness)
            base.queue.write(args.campaign / 'manifest.json', manifest)
            for phase in ('m0', 'full'):
                if phase == 'full':
                    previous.previous.verify_m0(args.campaign, dataset, variant)
                job = next(r for r in jobs if (r['dataset'], r['variant'], r['phase']) == (dataset, variant, phase))
                job['steps'] = []
                for mode in (('m0',) if phase == 'm0' else ('train', 'evaluate')):
                    step = dict(mode=mode, command=command(dataset, variant, mode, args.campaign,
                                base.output_dir(args.campaign, phase, dataset, variant)))
                    job['steps'].append(step)
                    if previous.previous.run_logged(args.campaign, state, step, f'{dataset}_{variant}_{mode}.log'):
                        job.update(status='FAILED', exit_code=step['exit_code'])
                        base.queue.write(args.campaign / 'campaign.json', state)
                        return 1
                row = previous.previous.verify_m0(args.campaign, dataset, variant) if phase == 'm0' else accept_and_retire_probe(args.campaign, dataset, variant)
                job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(), result=row)
                base.queue.write(args.campaign / 'campaign.json', state)
    require_controls()
    rows = [accepted_row(args.campaign, d, v) for d in DATASETS for v in VARIANTS]
    base.queue.write(args.campaign / 'accepted_matrix.json', dict(schema=SCHEMA, accepted=6, expected=6, rows=rows))
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=1)
    base.queue.write(args.campaign / 'campaign.json', state)
    with (args.campaign / 'report.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_deployment_metric_role.py'),
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
