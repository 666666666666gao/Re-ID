"""Finish the unattempted RGBNT100 endpoint after its CPU disk-wait queue closes."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registration', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.registration.read_bytes())
    assert plan['schema'] == 'trifusion-f1-monotone100-disk-continuation-v1'
    assert ROOT == Path('/data/gaob/Re-ID/Trifusion') and plan['gpu'] == 1
    assert sha(ROOT / plan['original_registration']) == plan['original_plan_sha256']
    original = json.loads((ROOT / plan['original_campaign']).read_bytes())
    assert all(next(x for x in original['jobs'] if x['dataset'] == ds)['status'] == 'COMPLETE'
               for ds in ('RGBNT201', 'MSVR310'))
    assert next(x for x in original['jobs'] if x['dataset'] == 'RGBNT100')['status'] == 'WAITING_FOR_DISK'
    stopped = json.loads((ROOT / plan['original_stop_record']).read_bytes())
    assert stopped['status'] == 'STOPPED_WAITING_FOR_DISK_BEFORE_RGBNT100_NN'
    assert stopped['pid'] == plan['original_controller_pid'] == 564175
    assert not Path('/proc/564175').exists()
    old = json.loads((ROOT / plan['after_original_f1_100']).read_bytes())
    assert old['status'] == 'COMPLETE'
    job = ROOT / plan['job_directory']
    run_directory = ROOT / plan['training_directory']
    campaign = ROOT / plan['campaign']
    assert not job.exists() and not run_directory.exists() and not campaign.exists()
    assert shutil.disk_usage(ROOT).free > plan['minimum_free_bytes'] == 2 * 1024 ** 3
    memory = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    assert int(memory.strip()) < 500
    campaign.mkdir()
    state = dict(schema=plan['schema'], status='STARTING', dataset='RGBNT100', gpu=1, seed=42,
                 method='F1_MONOTONE_AP', started_at=datetime.now().astimezone().isoformat(),
                 plan_sha256=sha(args.registration), phase_seconds={})

    def save():
        (campaign / 'campaign.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')

    def verify_sources():
        assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())

    def phase(mode, directory):
        verify_sources()
        command = [sys.executable, '-B', str(ROOT / 'tools/run_f1_monotone_roles.py'),
                   '--dataset', 'RGBNT100', '--mode', mode,
                   '--protocol', str(ROOT / 'logs/official_three_dataset_protocols_20260923/RGBNT100.json'),
                   '--signal-source', str(ROOT / 'comparators/Signal-cd1b0a6'),
                   '--clip-weight', str(ROOT / 'pertrained-model/ViT-B-16.pt'),
                   '--foundation-manifest', str(ROOT / plan['foundation_manifest']),
                   '--signal-references', str(ROOT / plan['signal_references']),
                   '--output-dir', str(directory), '--seed', '42']
        state.update(status=mode.upper(), phase_started_at=datetime.now().astimezone().isoformat(), command=command)
        save()
        start = time.perf_counter()
        with (campaign / (mode + '.log')).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='1',
                                                                  PYTHONNOUSERSITE='1'),
                                       stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            save()
            returncode = process.wait()
        state['phase_seconds'][mode] = time.perf_counter() - start
        state['child_returncode'] = returncode
        if returncode:
            state.update(status='FAILED', failed_phase=mode)
            save()
            raise RuntimeError(f'RGBNT100 {mode} failed: {returncode}')
        save()

    verify_sources()
    job.mkdir()
    phase('prepare', job / 'prepare')
    prepared = json.loads((job / 'prepare/initialization.json').read_bytes())
    assert prepared['status'] == 'F1_INITIALIZATION_AND_FORWARD_VERIFIED'
    assert prepared['initializer']['initial_role_state_sha256'] == plan['matched_initial_role_state_sha256']
    phase('m0', job / 'm0')
    m0 = json.loads((job / 'm0/training.json').read_bytes())
    assert m0['status'] == 'M0_PASS' and m0['training']['optimizer_steps'] == 8
    phase('train', run_directory)
    training = json.loads((run_directory / 'training.json').read_bytes())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and training['training']['epochs'] == 50
    assert training['initializer'] == m0['initializer']
    phase('evaluate', run_directory)
    result = json.loads((run_directory / 'official_metrics.json').read_bytes())
    assert result['status'] == 'COMPLETE' and result['baseline_f1_metrics_equal'] and result['independent_upstream_metrics_equal']
    targets = json.loads((ROOT / plan['paper_targets']).read_bytes())['datasets']['RGBNT100']
    metrics = result['outputs']['fused']['metrics']
    delta = {key: metrics[key] - value for key, value in targets.items()}
    state.update(status='COMPLETE', completed_at=datetime.now().astimezone().isoformat(),
                 metrics_path=str(run_directory / 'official_metrics.json'), metrics=metrics,
                 signal_paper_delta_pp=delta, all_paper_metrics_strictly_exceeded=all(x > 0 for x in delta.values()))
    verify_sources()
    save()


if __name__ == '__main__':
    main()
