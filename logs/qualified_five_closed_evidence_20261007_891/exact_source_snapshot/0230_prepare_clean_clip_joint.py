"""Witness six clean initializations, then the two real RGBNT201 eight-batch M0s."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_clean_clip_joint as run
from tools import queue_clean_clip_joint as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=2, choices=range(4), required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    assert len(set(args.gpus)) == 2 and not args.output_dir.exists()
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) == args.gpus[0]
    assert shutil.disk_usage(ROOT).free >= 10*1024**3
    assert all(panel.free_gpus()[gpu] < 500 for gpu in args.gpus)
    sources = panel.source_map()
    args.output_dir.mkdir(parents=True)
    state = {'schema':run.SCHEMA, 'status':'INITIALIZING', 'pid':os.getpid(),
             'started_at':panel.queue.stamp(), 'source_sha256':sources, 'initializations':[], 'jobs':[]}
    panel.queue.write(args.output_dir/'preflight.json', state)
    for dataset in panel.DATASETS:
        protocol = run.control.runner.read_protocol(panel.PROTOCOLS/f'{dataset}.json', dataset)
        reference = None
        common_digest = None
        for variant in panel.CONDITIONS:
            options = panel.options(dataset, variant, 'prepare', args.output_dir, ROOT/'trained-model/clean_clip_prepare_unused')
            assert not options.initialization.exists()
            model, _cfg, _config, binding = run.build_core(options, protocol)
            common = {f'{module}.{name}':value.detach().cpu().clone()
                      for module in ('backbone','neck','classifier')
                      for name,value in getattr(model,module).state_dict().items()}
            if reference is None:
                reference, common_digest = common, binding['common_initializer_sha256']
            assert set(common) == set(reference)
            assert all(torch.equal(common[key], reference[key]) for key in common)
            assert binding['common_initializer_sha256'] == common_digest
            witness = {'schema':run.SCHEMA,'status':'FRESH_PUBLIC_INITIALIZATION_VERIFIED',
                       'prepared_at':panel.queue.stamp(),'binding':binding,
                       'boundary':'Fresh constructor and152public tensor parity; common states actually bitwise compared. No forward, update or retrieval.'}
            panel.queue.write(options.initialization, witness)
            state['initializations'].append({'dataset':dataset,'variant':variant,
                    'path':str(options.initialization),'sha256':panel.sha(options.initialization),
                    'binding':binding,'common_states_bitwise_equal':True})
            panel.queue.write(args.output_dir/'preflight.json',state)
            del common, model
            gc.collect()
            torch.cuda.empty_cache()
        del reference
    assert panel.source_map() == sources
    state['status'] = 'M0_RUNNING'
    panel.queue.write(args.output_dir/'preflight.json',state)
    processes = []
    for variant,gpu in zip(panel.CONDITIONS,args.gpus):
        output = ROOT/f'trained-model/{args.output_dir.name}_RGBNT201_{variant}_m0'
        assert not output.exists()
        command = panel.command('RGBNT201',variant,'m0',args.output_dir,output)
        row = {'dataset':'RGBNT201','variant':variant,'mode':'m0','gpu':gpu,'status':'RUNNING',
               'started_at':panel.queue.stamp(),'output_dir':str(output),'command':command}
        with (args.output_dir/f'{variant}.m0.log').open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdout=log,stderr=subprocess.STDOUT)
        row['pid'] = child.pid
        state['jobs'].append(row)
        processes.append((row,child))
        panel.queue.write(args.output_dir/'preflight.json',state)
    failed = False
    for row,child in processes:
        code = child.wait()
        row.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=panel.queue.stamp(),
                   completion_time_semantics='Parent-observed after wait; child receipt gives actual M0 interval.')
        failed |= code != 0
        panel.queue.write(args.output_dir/'preflight.json',state)
    assert panel.source_map() == sources
    if not failed:
        for row in state['jobs']:
            panel.verify_m0(Path(row['output_dir']),row['dataset'],row['variant'],args.output_dir)
    state.update(status='FAILED' if failed else 'COMPLETE',completed_at=panel.queue.stamp())
    panel.queue.write(args.output_dir/'preflight.json',state)
    print(json.dumps({'status':state['status'],'initializations':len(state['initializations']),
                      'm0_exits':[row['exit_code'] for row in state['jobs']],'output_dir':str(args.output_dir)}),flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
