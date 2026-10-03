"""Verify actual fresh initial states and run two real RGBNT201 eight-batch M0s."""
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
from tools import run_native_detail as old_run
from tools import queue_native_detail as old_panel
from tools import run_semantic_native_evidence as run
from tools import queue_semantic_native_evidence as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs=2,choices=range(4),required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    assert len(set(args.gpus)) == 2 and not args.output_dir.exists()
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) == args.gpus[0]
    assert shutil.disk_usage(ROOT).free >= 10*1024**3
    assert all(panel.free_gpus()[gpu] < 500 for gpu in args.gpus)
    panel.install()
    sources = panel.source_map()
    args.output_dir.mkdir(parents=True)
    state = {'schema':run.SCHEMA,'status':'INITIALIZING','pid':os.getpid(),
             'started_at':panel.queue.stamp(),'source_sha256':sources,'initializations':[],'jobs':[]}
    panel.queue.write(args.output_dir/'preflight.json',state)
    for dataset in panel.DATASETS:
        protocol = run.clean.control.runner.read_protocol(panel.panel.PROTOCOLS/f'{dataset}.json',dataset)
        options = old_panel.options(dataset,'high','prepare',args.output_dir,ROOT/'trained-model/evidence_prepare_unused')
        original, _cfg, _config, old_binding = old_run.build_core(options,protocol)
        previous = json.loads((ROOT/f'logs/native_detail_preflight_20261002_v1/{dataset}_high.json').read_text())
        assert old_binding == previous['binding']
        original_state = {name:value.detach().cpu().clone() for name,value in original.state_dict().items()}
        stem_keys = {'roles.detail_stem.'+name for name in original.roles.detail_stem.state_dict()}
        del original
        gc.collect()
        torch.cuda.empty_cache()
        common = None
        for variant in panel.CONDITIONS:
            options = panel.options(dataset,variant,'prepare',args.output_dir,ROOT/'trained-model/evidence_prepare_unused')
            model, _cfg, _config, binding = run.build_core(options,protocol)
            current = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            assert set(original_state) - set(current) == (stem_keys if variant == 'semantic' else set())
            assert set(current).issubset(original_state)
            assert all(torch.equal(value,original_state[name]) for name,value in current.items())
            assert binding['common_initializer_sha256'] == old_binding['common_initializer_sha256']
            assert binding['trainable_parameters'] == old_binding['trainable_parameters'] - (93248 if variant == 'semantic' else 0)
            if common is None:
                common = current
            assert all(torch.equal(current[name],value) for name,value in common.items())
            panel.queue.write(options.initialization,
                {'schema':run.SCHEMA,'status':'FRESH_PUBLIC_INITIALIZATION_VERIFIED',
                 'prepared_at':panel.queue.stamp(),'binding':binding,
                 'boundary':'Real common initial-state parity and combined/native-high full-state parity; no forward or training.'})
            state['initializations'].append({'dataset':dataset,'variant':variant,'path':str(options.initialization),
                'sha256':panel.sha(options.initialization),'binding':binding,
                'common_states_bitwise_equal':True,'original_high_states_equal':True,
                'original_high_scope':'all states' if variant == 'combined' else 'all non-stem states'})
            del model, current
            gc.collect()
            torch.cuda.empty_cache()
        panel.queue.write(args.output_dir/'preflight.json',state)
        del common, original_state
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
        row.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=panel.queue.stamp())
        failed |= code != 0
        panel.queue.write(args.output_dir/'preflight.json',state)
    assert panel.source_map() == sources
    if not failed:
        for row in state['jobs']:
            panel.panel.verify_m0(Path(row['output_dir']),row['dataset'],row['variant'],args.output_dir)
    state.update(status='FAILED' if failed else 'COMPLETE',completed_at=panel.queue.stamp())
    panel.queue.write(args.output_dir/'preflight.json',state)
    print(json.dumps({'status':state['status'],'initializations':len(state['initializations']),
                      'm0_exits':[row['exit_code'] for row in state['jobs']]}),flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
