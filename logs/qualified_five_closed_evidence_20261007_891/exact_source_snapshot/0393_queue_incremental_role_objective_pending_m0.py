"""Run only the five originally pending M0s; original first FAIL is immutable."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_m0 as original

SCOPE = ROOT / 'refine-logs/incremental_role_objective_v1/PENDING_M0_SOURCE_SCOPE.json'
PARENT = ROOT / 'logs/incremental_role_objective_m0_v1_20261007_878'
PARENT_JOURNAL = ROOT / 'logs/incremental_role_objective_m0_launch_20261007_878'


def source_map():
    source = json.loads(SCOPE.read_text())['source_sha256']
    assert all(original.base.sha(ROOT / name) == digest for name, digest in source.items())
    return source


def coordinate(campaign):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1' and str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert not campaign.exists() and shutil.disk_usage(ROOT).free >= original.M0_STORAGE_BYTES
    parent = json.loads((PARENT / 'campaign.json').read_text())
    assert json.loads((PARENT_JOURNAL / 'EXIT.json').read_text())['exit_code'] == 1
    jobs = [dict(dataset=row['dataset'], objective=row['objective'], status='PENDING', phase='m0')
        for row in parent['jobs'] if row['status'] == 'PENDING']
    assert len(jobs) == 5 and not any((row['dataset'], row['objective']) == ('RGBNT201', 'md_batch_ratio') for row in jobs)
    controls = json.loads(original.CONTROLS.read_text())
    assert all(original.base.sha(Path(n)) == d for n, d in controls['artifact_sha256'].items())
    source = source_map()
    campaign.mkdir(parents=True)
    manifest = dict(schema=original.SCHEMA, source_sha256=source, initialization_sha256={}, jobs=jobs,
        original_failed_campaign=str(PARENT), original_exit_sha256=original.base.sha(PARENT_JOURNAL / 'EXIT.json'),
        physical_gpus=[0, 1], max_parallel_jobs=1,
        boundary='Five never-run conditions only. Same first-eight unscaled gate; original first failure remains, so complete six-end formal stage is not eligible. Probes retired after real production strict receipt and activity classification, no future probe consumers registered.')
    state = dict(status='RUNNING', started_at=original.base.queue.stamp(), controller_pid=os.getpid(), jobs=jobs, preparation=[])
    original.base.source_map = source_map
    original.base.queue.write(campaign / 'manifest.json', manifest)
    original.base.queue.write(campaign / 'campaign.json', state)
    for job in jobs:
        dataset, objective = job['dataset'], job['objective']
        output = ROOT / f'trained-model/{campaign.name}_m0_{objective}_{dataset}'
        prepare = dict(mode='prepare', command=original.command(dataset, objective, 'prepare', campaign, output))
        state['preparation'].append(prepare)
        if original.previous.run_logged(campaign, state, prepare, f'{dataset}_{objective}_prepare.log'):
            return 1
        initializer = campaign / 'initialization' / f'{dataset}_{objective}.json'
        binding = json.loads(initializer.read_text())['binding']
        old = next(row['initializer'] for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
        for key, value in old.items():
            if key not in ('architecture', 'entry_sha256', 'scope'):
                assert binding[key] == value, key
        manifest['initialization_sha256'][str(initializer)] = original.base.sha(initializer)
        original.base.queue.write(campaign / 'manifest.json', manifest)
        job['steps'] = []
        step = dict(mode='m0', command=original.command(dataset, objective, 'm0', campaign, output))
        job['steps'].append(step)
        if original.previous.run_logged(campaign, state, step, f'{dataset}_{objective}_m0.log'):
            return 1
        gate = dict(mode='activity_gate', command=[sys.executable, '-B', str(ROOT / 'tools/accept_incremental_m0_support.py'),
            '--campaign', str(campaign), '--dataset', dataset, '--objective', objective])
        job['steps'].append(gate)
        code = original.previous.run_logged(campaign, state, gate, f'{dataset}_{objective}_activity_gate.log')
        if code not in (0, 2):
            return code
        job.update(status='QUALIFIED' if code == 0 else 'ACTIVITY_GATE_FAIL', exit_code=code, completed_at=original.base.queue.stamp())
        state['status'] = 'RUNNING'
        original.base.queue.write(campaign / 'campaign.json', state)
    state.update(status='M0_SCREENING_COMPLETE_WITH_ORIGINAL_FAILURE', completed_at=original.base.queue.stamp(),
        new_qualified=sum(job['status'] == 'QUALIFIED' for job in jobs), new_failed=sum(job['status'] == 'ACTIVITY_GATE_FAIL' for job in jobs),
        original_failed=1, formal_eligible=False)
    original.base.queue.write(campaign / 'campaign.json', state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    raise SystemExit(coordinate(parser.parse_args().campaign.resolve()))
