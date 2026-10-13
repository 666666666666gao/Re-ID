"""Three fixed201 controls, one two-card process and fresh M0/full50 each."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_native_research as previous
from tools.run_joint_spectral_delta import SCHEMA, POLICY

base = previous.base
MODES = ('joint_delta', 'add_delta', 'joint_absolute')
SCOPE = ROOT / 'refine-logs/joint_spectral_training_20261013_943/SOURCE_SCOPE.json'
CONTROL = ROOT / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
STORAGE_BYTES = 2 * 1024**3 + 6 * 384 * 1024**2 + 256 * 1024**2


def source_map():
    value = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in value.items())
    return value


def command(mode, aggregation, campaign, output):
    value = previous.command('RGBNT201', 'semantic', mode, campaign, output)
    value[2] = str(ROOT / 'tools/run_joint_spectral_delta.py')
    return [*value, '--spectral-aggregation', aggregation]


def coordinate(campaign):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and not campaign.exists()
    assert shutil.disk_usage(ROOT).free >= STORAGE_BYTES
    base.SCHEMA = SCHEMA
    base.source_map = source_map
    base.output_dir = previous.output_dir
    control = json.loads(CONTROL.read_text())
    original = next(row['initializer'] for row in control['rows']
                    if (row['dataset'], row['variant']) == ('RGBNT201', 'semantic'))
    sources = source_map()
    campaign.mkdir()
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(),
                 arms=[], expected=3, accepted=0)
    base.queue.write(campaign / 'SOURCE_SEAL.json', sources)
    base.queue.write(campaign / 'campaign.json', state)
    matched_initial_state = None
    results = []
    for aggregation in MODES:
        child = campaign / (campaign.name + '_' + aggregation)
        child.mkdir()
        manifest = dict(schema=SCHEMA, seed=42, epochs=50, dataset='RGBNT201', variant='semantic',
                        spectral_aggregation=aggregation, source_sha256=sources,
                        initialization_sha256={}, physical_gpus=[0, 1], max_parallel_jobs=1,
                        poll_seconds=240, policy=POLICY, user_authorization='继续训练',
                        control_seal_sha256=base.sha(CONTROL), storage_budget_bytes=STORAGE_BYTES)
        arm = dict(status='RUNNING', spectral_aggregation=aggregation, controller_pid=os.getpid(),
                   steps=[], campaign=str(child), started_at=base.queue.stamp())
        state['arms'].append(arm)
        base.queue.write(child / 'manifest.json', manifest)
        base.queue.write(child / 'campaign.json', arm)
        base.queue.write(campaign / 'campaign.json', state)
        for mode in ('prepare', 'm0', 'train', 'evaluate'):
            phase = 'm0' if mode == 'm0' else 'full'
            output = previous.output_dir(child, phase, 'RGBNT201', 'semantic')
            assert mode == 'evaluate' or not output.exists()
            row = dict(mode=mode, command=command(mode, aggregation, child, output), output_dir=str(output))
            arm['steps'].append(row)
            code = previous.run_logged(child, arm, row, mode + '.log')
            if code:
                state.update(status='FAILED', failed_arm=aggregation, completed_at=base.queue.stamp())
                base.queue.write(campaign / 'campaign.json', state)
                return code
            if mode == 'prepare':
                binding = base.expected_binding(child, 'RGBNT201', 'semantic')
                assert binding['objective_gradient_policy'] == POLICY
                assert binding['spectral_aggregation'] == aggregation
                assert binding['trainable_parameters'] == original['trainable_parameters'] + 589836
                assert binding['trainable_parameter_tensors'] == original['trainable_parameter_tensors'] + 2
                for key in ('shared_initializer_sha256', 'visual_initial_sha256', 'camera_initial_sha256',
                            'author_source_commit', 'cfg_yaml', 'batch_size', 'num_instances', 'head_names'):
                    assert binding[key] == original[key], key
                if matched_initial_state is None:
                    matched_initial_state = binding['initial_model_state_sha256']
                assert binding['initial_model_state_sha256'] == matched_initial_state
                path = child / 'initialization/RGBNT201_semantic.json'
                manifest['initialization_sha256'][str(path)] = base.sha(path)
                base.queue.write(child / 'manifest.json', manifest)
            elif mode == 'm0':
                accepted = previous.verify_m0(child, 'RGBNT201', 'semantic')
                diagnostics = accepted['production_m0_diagnostics']
                assert diagnostics['role_shared_gradient_isolated']
                assert len(diagnostics['spectral_updates']) == 8
                base.queue.write(child / 'M0_ACCEPTED.json', accepted)
        accepted = base.verify(child, 'RGBNT201', 'semantic')
        accepted['spectral_aggregation'] = aggregation
        base.queue.write(child / 'ACCEPTED_RESULT.json', accepted)
        arm.update(status='COMPLETE', result=accepted, completed_at=base.queue.stamp())
        base.queue.write(child / 'campaign.json', arm)
        results.append(accepted)
        state['accepted'] = len(results)
        base.queue.write(campaign / 'campaign.json', state)
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), results=results)
    base.queue.write(campaign / 'campaign.json', state)
    base.queue.write(campaign / 'ACCEPTED_MATRIX.json', dict(expected=3, accepted=3, rows=results))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(coordinate(args.campaign.resolve()))
