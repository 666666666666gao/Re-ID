"""One fresh201/seed42 prepare, real M0, full50 and first strict evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_native_research as previous
from tools.run_normalized_role_classification import SCHEMA, POLICY

base = previous.base
SCOPE = ROOT / 'refine-logs/normalized_role_training_20261013_938/SOURCE_SCOPE.json'
CONTROL = ROOT / 'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
STORAGE_BYTES = 2 * 1024**3 + 2 * 384 * 1024**2 + 128 * 1024**2


def source_map():
    value = json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT / name) == digest for name, digest in value.items())
    return value


def command(mode, campaign, output):
    value = previous.command('RGBNT201', 'semantic', mode, campaign, output)
    value[2] = str(ROOT / 'tools/run_normalized_role_classification.py')
    return value


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
    campaign.mkdir()
    manifest = {'schema': SCHEMA, 'seed': 42, 'epochs': 50, 'dataset': 'RGBNT201', 'variant': 'semantic',
                'source_sha256': source_map(), 'initialization_sha256': {}, 'physical_gpus': [0, 1],
                'max_parallel_jobs': 1, 'poll_seconds': 240, 'storage_budget_bytes': STORAGE_BYTES,
                'control_seal_sha256': base.sha(CONTROL), 'user_authorization': '继续训练',
                'policy': POLICY, 'scope': 'One201 only; no old retrain, vehicles, seed/scale/margin/layer search.'}
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': base.queue.stamp(),
             'steps': [], 'dataset': 'RGBNT201', 'variant': 'semantic'}
    base.queue.write(campaign/'manifest.json', manifest)
    base.queue.write(campaign/'campaign.json', state)
    for mode in ('prepare', 'm0', 'train', 'evaluate'):
        phase = 'm0' if mode == 'm0' else 'full'
        output = previous.output_dir(campaign, phase, 'RGBNT201', 'semantic')
        assert mode == 'evaluate' or not output.exists()
        row = {'mode': mode, 'command': command(mode, campaign, output), 'output_dir': str(output)}
        state['steps'].append(row)
        if previous.run_logged(campaign, state, row, mode+'.log'):
            return row['exit_code']
        if mode == 'prepare':
            binding = base.expected_binding(campaign, 'RGBNT201', 'semantic')
            assert binding['objective_gradient_policy'] == POLICY
            assert binding['role_classification_feature'] == 'l2_joint_1536_before_existing_detached_author_bn'
            assert binding['role_metric_feature'] == 'raw_joint_1536'
            excluded = {'architecture', 'entry_sha256', 'scope', 'objective_gradient_policy',
                        'role_classification_feature', 'role_metric_feature'}
            assert {k:v for k,v in binding.items() if k not in excluded} == {
                k:v for k,v in original.items() if k not in excluded}
            witness = campaign/'initialization/RGBNT201_semantic.json'
            manifest['initialization_sha256'][str(witness)] = base.sha(witness)
            base.queue.write(campaign/'manifest.json', manifest)
        elif mode == 'm0':
            accepted = previous.verify_m0(campaign, 'RGBNT201', 'semantic')
            base.queue.write(campaign/'M0_ACCEPTED.json', accepted)
    accepted = base.verify(campaign, 'RGBNT201', 'semantic')
    base.queue.write(campaign/'ACCEPTED_RESULT.json', accepted)
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), result=accepted)
    base.queue.write(campaign/'campaign.json', state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(coordinate(args.campaign.resolve()))


if __name__ == '__main__':
    main()
