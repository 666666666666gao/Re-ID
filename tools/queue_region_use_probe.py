"""Six registered region-value probes; each accepted M0 precedes its fresh50."""
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

base = previous.base
SCHEMA = 'trifusion-region-use-supervision-v1'
CONDITIONS = ('uniform', 'ambiguity')
SCOPE = ROOT/'refine-logs/region_use_supervision_v1/INPUT_SEAL.json'
RESERVE_BYTES = 2*1024**3
NEXT_ARTIFACT_BYTES = 512*1024**2


def source_map():
    seal = json.loads(SCOPE.read_text())
    sources = seal['source_sha256']
    assert all(base.sha(ROOT/name) == digest for name, digest in sources.items())
    return sources


def output_dir(campaign, phase, dataset, comparison):
    return ROOT/f'trained-model/{campaign.name}_{phase}_{comparison}_{dataset}'


def command(campaign, dataset, comparison, mode):
    phase = 'm0' if mode == 'm0' else 'full'
    return [sys.executable, '-B', str(ROOT/'tools/run_region_use_probe.py'),
        '--dataset', dataset, '--comparison', comparison, '--mode', mode,
        '--protocol', str(previous.PROTOCOLS/f'{dataset}.json'),
        '--signal-source', str(previous.SOURCE), '--clip-weight', str(previous.WEIGHTS/'ViT-B-16.pt'),
        '--initialization', str(campaign/'initialization'/f'{dataset}_{comparison}.json'),
        '--output-dir', str(output_dir(campaign, phase, dataset, comparison))]


def accept_initialization(campaign, dataset, comparison, seal):
    path = campaign/'initialization'/f'{dataset}_{comparison}.json'
    value = json.loads(path.read_text())
    assert value['schema'] == SCHEMA and value['status'] == 'INITIALIZATION_VERIFIED'
    old = next(row['initializer'] for row in seal['rows'] if row['dataset'] == dataset)
    binding = value['binding']
    assert all(binding[key] == expected for key, expected in old.items()
               if key not in ('architecture', 'entry_sha256', 'scope'))
    base.queue.write(campaign/'acceptance'/f'{dataset}_{comparison}_initialization.json',
        dict(status='EXACT_OLD_SEMANTIC_INITIAL_STATE_ACCEPTED', initializer=binding,
             initialization_sha256=base.sha(path), accepted_at=base.queue.stamp()))
    return binding


def accept_m0(campaign, dataset, comparison, binding):
    output = output_dir(campaign, 'm0', dataset, comparison)
    receipt = json.loads((output/'training.json').read_text())
    assert receipt['schema'] == SCHEMA and receipt['initializer'] == binding
    assert receipt['condition']['region_use_comparison'] == comparison
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert receipt['frozen_parameters_unchanged'] and receipt['visual_parameters_changed'] and receipt['fresh_camera_parameters_changed']
    assert all(count == 8 for count in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    old = ROOT/f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{dataset}/training_batch_order.jsonl'
    batches = [json.loads(line) for line in (output/'training_batch_order.jsonl').read_text().splitlines()]
    old_batches = [json.loads(line) for line in old.read_text().splitlines()][:8]
    assert len(batches) == 8 and batches == old_batches
    target = campaign/'acceptance'/f'{dataset}_{comparison}_m0.json'
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/accept_region_use_m0.py'),
                    '--m0-dir', str(output), '--output', str(target)], check=True)
    probe = output/'m0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT/'trained-model').resolve())
    assert {p.name for p in output.glob('*.pth')} == {'m0_reload_probe.pth'}
    assert base.sha(probe) == receipt['m0']['reload_probe_sha256']
    retired = dict(path=str(probe), bytes=probe.stat().st_size, sha256=base.sha(probe),
                   retired_at=base.queue.stamp(), acceptance=str(target))
    probe.unlink()
    with (campaign/'probe_retirement.jsonl').open('a') as log:
        log.write(json.dumps(retired)+'\n')


def coordinate(campaign):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not campaign.exists()
    seal = json.loads(SCOPE.read_text())
    assert all(base.sha(Path(path)) == digest for path, digest in seal['artifact_sha256'].items())
    sources = source_map()
    assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES+NEXT_ARTIFACT_BYTES
    campaign.mkdir(parents=True)
    (campaign/'acceptance').mkdir()
    jobs = [dict(dataset=d, comparison=c, status='PENDING', stages=[])
            for d in previous.DATASETS for c in CONDITIONS]
    state = dict(schema=SCHEMA, status='RUNNING', controller_pid=os.getpid(),
                 started_at=base.queue.stamp(), jobs=jobs, physical_gpus=[0,1],
                 optimizer_updates_per_m0=8, epochs_per_full=50, source_sha256=sources,
                 initialization_sha256={}, input_seal_sha256=base.sha(SCOPE),
                 report_status='PENDING_SEPARATE_TERMINAL_ANALYSIS')
    base.source_map = source_map
    base.queue.write(campaign/'manifest.json', dict(state))
    base.queue.write(campaign/'campaign.json', state)
    for job in jobs:
        dataset, comparison = job['dataset'], job['comparison']
        for mode in ('prepare','m0','train','evaluate'):
            assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES+NEXT_ARTIFACT_BYTES
            row = dict(mode=mode, command=command(campaign,dataset,comparison,mode))
            job['stages'].append(row)
            if previous.run_logged(campaign,state,row,f'{dataset}_{comparison}_{mode}.log'):
                job['status'] = 'FAILED'
                base.queue.write(campaign/'campaign.json',state)
                return 1
            if mode == 'prepare':
                binding = accept_initialization(campaign,dataset,comparison,seal)
                path = campaign/'initialization'/f'{dataset}_{comparison}.json'
                state['initialization_sha256'][str(path)] = base.sha(path)
                base.queue.write(campaign/'manifest.json',dict(state))
            elif mode == 'm0':
                accept_m0(campaign,dataset,comparison,binding)
            elif mode == 'evaluate':
                result = json.loads((output_dir(campaign,'full',dataset,comparison)/'official_metrics.json').read_text())
                assert result['schema'] == SCHEMA and result['status'] == 'COMPLETE'
                assert result['training_epochs'] == 50 and result['condition']['region_use_comparison'] == comparison
                job['result'] = result
            base.queue.write(campaign/'campaign.json',state)
        job.update(status='COMPLETE',completed_at=base.queue.stamp())
        base.queue.write(campaign/'campaign.json',state)
    state.update(status='TRAINING_AND_STRICT_EVALUATION_COMPLETE',completed_at=base.queue.stamp())
    base.queue.write(campaign/'campaign.json',state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    raise SystemExit(coordinate(parser.parse_args().campaign.resolve()))
