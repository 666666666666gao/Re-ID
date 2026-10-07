"""Six paired real M0s; accept isolated increment activity before fresh full50."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_native_research as previous

base = previous.base
DATASETS = previous.DATASETS
OBJECTIVES = ('md_batch_ratio', 'repair_keep')
SCHEMA = 'trifusion-incremental-role-objective-v1'
SCOPE = ROOT / 'refine-logs/incremental_role_objective_v1/SOURCE_SCOPE.json'
CONTROLS = ROOT / 'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json'
M0_STORAGE_BYTES = 2 * 1024**3 + 512 * 1024**2


def source_map():
    sources = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in sources.items())
    return sources


def command(dataset, objective, mode, campaign, output):
    return [sys.executable, '-B', str(ROOT / 'tools/run_incremental_role_objective.py'),
        '--dataset', dataset, '--objective', objective, '--mode', mode,
        '--protocol', str(previous.PROTOCOLS / f'{dataset}.json'),
        '--signal-source', str(previous.SOURCE), '--clip-weight', str(previous.WEIGHTS / 'ViT-B-16.pt'),
        '--initialization', str(campaign / 'initialization' / f'{dataset}_{objective}.json'),
        '--output-dir', str(output)]


def accept_m0(campaign, dataset, objective, output, binding):
    receipt = json.loads((output / 'training.json').read_text())
    assert receipt['schema'] == SCHEMA and receipt['status'] == 'M0_PASS'
    assert receipt['initializer'] == binding and receipt['condition']['incremental_objective'] == objective
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert receipt['frozen_parameters_unchanged'] and receipt['visual_parameters_changed'] and receipt['fresh_camera_parameters_changed']
    assert receipt['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert all(value == 8 for value in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8
    assert all(row['incremental_isolated_shared_global_gradient_absent'] for row in steps)
    names = set(steps[0]['incremental_isolated_query_key_gradient_norms'])
    assert len(names) == 6
    assert all(set(row['incremental_isolated_query_key_gradient_norms']) == names for row in steps)
    norms = [row['incremental_isolated_correction_gradient_norm'] for row in steps]
    assert all(math.isfinite(value) and value >= 0 for value in norms) and sum(norms) > 0
    accumulated = {}
    for name in sorted(names):
        values = [row['incremental_isolated_query_key_gradient_norms'][name] for row in steps]
        assert all(math.isfinite(value) and value >= 0 for value in values) and sum(values) > 0, name
        accumulated[name] = sum(values)
    batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()][:8]
    old_dir = ROOT / f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{dataset}'
    old_batches = [json.loads(line) for line in (old_dir / 'training_batch_order.jsonl').read_text().splitlines()][:8]
    assert len(batches) == len(old_batches) == 8 and batches == old_batches
    probe = output / 'm0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT / 'trained-model').resolve())
    assert probe.parent.name == f'{campaign.name}_m0_{objective}_{dataset}'
    assert {p.name for p in output.glob('*.pth')} == {'m0_reload_probe.pth'}
    assert base.sha(probe) == receipt['m0']['reload_probe_sha256']
    accepted = dict(status='REAL_M0_AND_ISOLATED_INCREMENT_ACTIVITY_ACCEPTED', dataset=dataset,
        objective=objective, accepted_at=base.queue.stamp(), output_dir=str(output),
        correction_gradient_norm_sum=sum(norms), query_key_gradient_norm_sums=accumulated,
        source_raw_m0_batch_order_equal=True, initializer=binding,
        artifact_sha256={str(p): base.sha(p) for p in output.iterdir() if p.is_file() and p != probe},
        probe=dict(path=str(probe), bytes=probe.stat().st_size, sha256=base.sha(probe)))
    path = campaign / 'acceptance' / f'{dataset}_{objective}.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    base.queue.write(path, accepted)
    probe.unlink()
    with (campaign / 'probe_retirement.jsonl').open('a') as log:
        log.write(json.dumps(dict(retired_at=base.queue.stamp(), acceptance=str(path), **accepted['probe'])) + '\n')
    return accepted


def coordinate(campaign):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not campaign.exists() and shutil.disk_usage(ROOT).free >= M0_STORAGE_BYTES
    controls = json.loads(CONTROLS.read_text())
    assert len(controls['rows']) == 6
    assert all(base.sha(Path(name)) == digest for name, digest in controls['artifact_sha256'].items())
    sources = source_map()
    campaign.mkdir(parents=True)
    jobs = [dict(dataset=d, objective=o, phase='m0', status='PENDING') for d in DATASETS for o in OBJECTIVES]
    manifest = dict(schema=SCHEMA, source_sha256=sources, initialization_sha256={}, jobs=jobs,
        control_seal_sha256=base.sha(CONTROLS), physical_gpus=[0, 1], max_parallel_jobs=1,
        storage_required_bytes=M0_STORAGE_BYTES, storage_policy='Retire each accepted real M0 engineering probe; fresh full50 never inherits it.',
        boundary='M0 only. Six isolated increment activity gates; no automatic full training or retrieval claim.')
    state = dict(status='RUNNING', started_at=base.queue.stamp(), controller_pid=os.getpid(), jobs=jobs, preparation=[])
    base.source_map = source_map
    base.queue.write(campaign / 'manifest.json', manifest)
    base.queue.write(campaign / 'campaign.json', state)
    for dataset in DATASETS:
        old = next(row['initializer'] for row in controls['rows'] if row['dataset'] == dataset and row['variant'] == 'semantic')
        for objective in OBJECTIVES:
            output = ROOT / f'trained-model/{campaign.name}_m0_{objective}_{dataset}'
            prep = dict(mode='prepare', dataset=dataset, objective=objective,
                command=command(dataset, objective, 'prepare', campaign, output))
            state['preparation'].append(prep)
            if previous.run_logged(campaign, state, prep, f'{dataset}_{objective}_prepare.log'):
                return 1
            initializer = campaign / 'initialization' / f'{dataset}_{objective}.json'
            witness = json.loads(initializer.read_text())
            assert witness['schema'] == SCHEMA and witness['status'] == 'INITIALIZATION_VERIFIED'
            binding = witness['binding']
            for key, value in old.items():
                if key not in ('architecture', 'entry_sha256', 'scope'):
                    assert binding[key] == value, key
            manifest['initialization_sha256'][str(initializer)] = base.sha(initializer)
            base.queue.write(campaign / 'manifest.json', manifest)
            job = next(row for row in jobs if (row['dataset'], row['objective']) == (dataset, objective))
            job['command'] = command(dataset, objective, 'm0', campaign, output)
            if previous.run_logged(campaign, state, job, f'{dataset}_{objective}_m0.log'):
                return 1
            job['acceptance'] = accept_m0(campaign, dataset, objective, output, binding)
            base.queue.write(campaign / 'campaign.json', state)
    state.update(status='M0_ONLY_COMPLETE', completed_at=base.queue.stamp(), accepted=6, expected=6)
    base.queue.write(campaign / 'campaign.json', state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    raise SystemExit(coordinate(parser.parse_args().campaign.resolve()))
