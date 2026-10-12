"""Six fixed-best region-use diagnostics, sequential and inference-only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_region_use_best as consumer

RESERVE_BYTES = 2*1024**3
METADATA_BYTES = 64*1024**2


def coordinate(args, seal):
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert args.output_dir.is_relative_to(ROOT/'logs') and not args.output_dir.exists()
    consumer.require_inputs(seal)
    rows = seal['rows']
    payloads = [5*4*row['query_count']*row['gallery_count'] for row in rows]
    assert seal['storage_required_bytes'] == RESERVE_BYTES+METADATA_BYTES+sum(payloads)
    assert shutil.disk_usage(ROOT).free >= seal['storage_required_bytes']
    args.output_dir.mkdir()
    state = dict(schema=consumer.SCHEMA, status='RUNNING', pid=os.getpid(),
        started_at=consumer.frozen.stamp(), physical_gpus=[0, 1], optimizer_updates=0,
        input_seal_sha256=consumer.base.panel.base.sha(args.seal), jobs=[])
    consumer.frozen.write(args.output_dir/'campaign.json', state)
    stage = 'before_first_case'
    try:
        for index, row in enumerate(rows):
            name = row['dataset']+'_'+row['comparison']
            stage = name+':admission'
            # The prior training campaign stopped at a real storage boundary.
            assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES+METADATA_BYTES+sum(payloads[index:])
            used = subprocess.check_output(['nvidia-smi', '--id=0,1',
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            assert len(used.splitlines()) == 2
            assert all(int(line.split(',')[1]) < 500 for line in used.splitlines())
            output = args.output_dir/name
            job = dict(dataset=row['dataset'], comparison=row['comparison'], status='RUNNING',
                       started_at=consumer.frozen.stamp(), output_dir=str(output))
            state['jobs'].append(job)
            stage = name+':child_execution'
            with (args.output_dir/(name+'.log')).open('x') as log:
                process = subprocess.Popen([sys.executable, '-B', str(ROOT/'tools/diagnose_region_use_best.py'),
                    '--seal', str(args.seal), '--output-dir', str(output),
                    '--dataset', row['dataset'], '--comparison', row['comparison']],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = process.pid
                consumer.frozen.write(args.output_dir/'campaign.json', state)
                code = process.wait()
            job.update(exit_code=code, completed_at=consumer.frozen.stamp())
            if code:
                job['status'] = 'FAILED'
                state.update(status='FAILED', completed_at=consumer.frozen.stamp())
                consumer.frozen.write(args.output_dir/'campaign.json', state)
                return code
            stage = name+':result_validation'
            result = json.loads((output/'DIAGNOSIS.json').read_text())
            assert result['schema'] == consumer.SCHEMA and result['status'] == 'COMPLETE'
            assert (result['dataset'], result['comparison']) == (row['dataset'], row['comparison'])
            assert result['input_checkpoint_sha256'] == row['checkpoint_sha256']
            assert result['optimizer_updates'] == 0
            job.update(status='COMPLETE', diagnosis_sha256=consumer.base.panel.base.sha(output/'DIAGNOSIS.json'))
            consumer.frozen.write(args.output_dir/'campaign.json', state)
        stage = 'final_input_recheck'
        consumer.require_inputs(seal)
        state.update(status='COMPLETE', completed_at=consumer.frozen.stamp())
        consumer.frozen.write(args.output_dir/'campaign.json', state)
        return 0
    except Exception as error:
        # C1 reproduced the original campaign's stale RUNNING failure record.
        for job in state['jobs']:
            if job['status'] == 'RUNNING' and 'exit_code' in job:
                job['status'] = 'FAILED'
        state.update(status='FAILED', failure_stage=stage,
            exception_type=type(error).__name__, exception_message=str(error),
            completed_at=consumer.frozen.stamp())
        consumer.frozen.write(args.output_dir/'campaign.json', state)
        raise



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seal', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.seal, args.output_dir = args.seal.resolve(), args.output_dir.resolve()
    raise SystemExit(coordinate(args, json.loads(args.seal.read_text())))
