"""Run the two planned visual-update M0s and preserve their actual execution rows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_visual_update_control as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-campaign', type=Path, required=True)
    parser.add_argument('--after-summary', type=Path, required=True)
    parser.add_argument('--after-summary-sha256', required=True)
    parser.add_argument('--gpus', type=int, nargs=2, choices=range(4), required=True)
    args = parser.parse_args()
    assert len(set(args.gpus)) == 2 and not args.output.exists()
    previous = json.loads((args.after_campaign / 'campaign.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['jobs']) == 6
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in previous['jobs'])
    assert panel.sha(args.after_summary) == args.after_summary_sha256
    summary = json.loads(args.after_summary.read_text())
    assert summary['schema'] == 'visual-start-six-end-complete-analysis-v1' and summary['accepted'] == 6
    assert all(summary['source_artifacts_sha256'][str(path.resolve())] == panel.sha(path)
               for path in (args.after_campaign / name for name in
                            ('campaign.json', 'manifest.json', 'accepted_matrix.json')))
    assert all(panel.sha(ROOT / path) == digest for path, digest in
               json.loads((args.after_campaign / 'manifest.json').read_text())['source_sha256'].items())
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in memory.splitlines()}
    assert all(used[gpu] < 500 for gpu in args.gpus)
    assert shutil.disk_usage(ROOT).free >= 10 * 1024**3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_at': panel.queue.stamp(),
             'source_sha256': panel.source_map(), 'after_summary_sha256': args.after_summary_sha256,
             'after_campaign': str(args.after_campaign.resolve()), 'jobs': []}
    processes = []
    for variant, gpu in zip(('low_lr_roles', 'low_lr_global_only'), args.gpus):
        output = ROOT / f'trained-model/{args.output.parent.name}_RGBNT201_{variant}_m0'
        assert not output.exists()
        row = {'dataset': 'RGBNT201', 'variant': variant, 'mode': 'm0', 'gpu': gpu,
               'status': 'RUNNING', 'started_at': panel.queue.stamp(), 'output_dir': str(output),
               'command': panel.command('RGBNT201', variant, 'm0', output)}
        with (args.output.parent / f'{variant}.log').open('x') as log:
            child = subprocess.Popen(row['command'], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = child.pid
        state['jobs'].append(row)
        processes.append((row, child))
        panel.queue.write(args.output, state)
    failed = False
    for row, child in processes:
        code = child.wait()
        row.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=panel.queue.stamp(),
                   completion_time_semantics='Parent-observed exit after wait; use child receipt for measured M0 runtime.')
        failed |= code != 0
        panel.queue.write(args.output, state)
    assert state['source_sha256'] == panel.source_map()
    state.update(status='FAILED' if failed else 'COMPLETE', completed_at=panel.queue.stamp())
    panel.queue.write(args.output, state)
    print(json.dumps(state), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
