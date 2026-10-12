"""One unattempted original condition after a recorded storage-stop, not a retry."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_region_use_probe as panel


def coordinate(campaign):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not campaign.exists()
    original = ROOT/'logs/region_use_v1_20261012_906'
    exit_path = ROOT/'logs/region_use_resume906/EXIT.json'
    parent = json.loads((original/'campaign.json').read_text())
    parent_exit = json.loads(exit_path.read_text())
    assert parent_exit['exit_code'] == 1
    assert parent['schema'] == panel.SCHEMA
    remaining = next(j for j in parent['jobs'] if (j['dataset'], j['comparison']) == ('RGBNT100', 'ambiguity'))
    assert remaining['status'] == 'PENDING' and remaining['stages'] == []
    assert sum(j['status'] == 'COMPLETE' for j in parent['jobs']) == 5
    seal = json.loads(panel.SCOPE.read_text())
    assert parent['input_seal_sha256'] == panel.base.sha(panel.SCOPE)
    assert all(panel.base.sha(Path(p)) == h for p, h in seal['artifact_sha256'].items())
    sources = panel.source_map()
    assert shutil.disk_usage(ROOT).free >= panel.RESERVE_BYTES+panel.NEXT_ARTIFACT_BYTES
    campaign.mkdir(parents=True)
    (campaign/'acceptance').mkdir()
    job = dict(dataset='RGBNT100', comparison='ambiguity', status='PENDING', stages=[])
    state = dict(schema=panel.SCHEMA, status='RUNNING', controller_pid=os.getpid(),
        started_at=panel.base.queue.stamp(), jobs=[job], physical_gpus=[0, 1],
        optimizer_updates_per_m0=8, epochs_per_full=50, source_sha256=sources,
        input_seal_sha256=panel.base.sha(panel.SCOPE), initialization_sha256={},
        continuation_controller_sha256=panel.base.sha(Path(__file__)),
        original_campaign=str(original), original_campaign_sha256=panel.base.sha(original/'campaign.json'),
        original_exit_sha256=panel.base.sha(exit_path), original_failed_exit_code=1,
        report_status='PENDING_SEPARATE_TERMINAL_ANALYSIS',
        continuation_scope='Only originally unattempted100 ambiguity; no completed-endpoint replay or scientific change.')
    panel.base.source_map = panel.source_map
    panel.base.queue.write(campaign/'manifest.json', dict(state))
    panel.base.queue.write(campaign/'campaign.json', state)
    for mode in ('prepare', 'm0', 'train', 'evaluate'):
        if shutil.disk_usage(ROOT).free < panel.RESERVE_BYTES+panel.NEXT_ARTIFACT_BYTES:
            state.update(status='STOPPED_BEFORE_STAGE_INSUFFICIENT_STORAGE', stopped_before_mode=mode,
                         completed_at=panel.base.queue.stamp())
            panel.base.queue.write(campaign/'campaign.json', state)
            return 1
        stage = dict(mode=mode, command=panel.command(campaign, 'RGBNT100', 'ambiguity', mode))
        job['stages'].append(stage)
        if panel.previous.run_logged(campaign, state, stage, f'RGBNT100_ambiguity_{mode}.log'):
            job['status'] = 'FAILED'
            state.update(status='FAILED', completed_at=panel.base.queue.stamp())
            panel.base.queue.write(campaign/'campaign.json', state)
            return 1
        if mode == 'prepare':
            binding = panel.accept_initialization(campaign, 'RGBNT100', 'ambiguity', seal)
            path = campaign/'initialization/RGBNT100_ambiguity.json'
            state['initialization_sha256'][str(path)] = panel.base.sha(path)
            panel.base.queue.write(campaign/'manifest.json', dict(state))
        elif mode == 'm0':
            panel.accept_m0(campaign, 'RGBNT100', 'ambiguity', binding)
        elif mode == 'evaluate':
            result = json.loads((panel.output_dir(campaign, 'full', 'RGBNT100', 'ambiguity')/'official_metrics.json').read_text())
            assert result['schema'] == panel.SCHEMA and result['status'] == 'COMPLETE'
            assert result['training_epochs'] == 50 and result['condition']['region_use_comparison'] == 'ambiguity'
            job['result'] = result
        panel.base.queue.write(campaign/'campaign.json', state)
    job.update(status='COMPLETE', completed_at=panel.base.queue.stamp())
    state.update(status='TRAINING_AND_STRICT_EVALUATION_COMPLETE', completed_at=panel.base.queue.stamp())
    panel.base.queue.write(campaign/'campaign.json', state)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    raise SystemExit(coordinate(parser.parse_args().campaign.resolve()))
