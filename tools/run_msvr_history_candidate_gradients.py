"""Run fixed-state diagnostics once; no optimizer or retrieval stages."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.train_msvr310_signal_oof import sha256, write_json


def run(args):
    repo=Path.cwd().resolve();root=args.output_dir.resolve();config=args.config.resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    assert sha256(config)==args.config_sha256
    spec=json.loads(config.read_bytes())
    assert sha256(__file__)==spec['project_file_sha256']['tools/run_msvr_history_candidate_gradients.py']
    assert shutil.disk_usage(root.parent).free>=spec['minimum_free_bytes']
    root.mkdir()
    common=['--config',str(config)]
    stages=[('t0','tools.probe_msvr_history_candidate_gradients',common+['--mode','math','--output-dir',str(root/'t0.json')],False)]
    for mode in ('preflight','source'):
        stages.append((mode,'tools.probe_msvr_history_candidate_gradients',common+['--mode',mode,'--output-dir',str(root/mode)],True))
        stages.append((mode+'_cpu','tools.verify_msvr_history_candidate_gradients',common+['--input-dir',str(root/mode)],False))
    result=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=args.code_commit,config_sha256=args.config_sha256,
                started_at=datetime.now().astimezone().isoformat(),optimizer_updates=0,heldout_record_forwards=0,
                official_image_reads=0,stages=[])
    def save():write_json(root/'pipeline.json',result)
    save()
    for name,module,arguments,gpu in stages:
        command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m',module,*arguments]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',
                         MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(repo/'modeling')+':'+str(repo))
        start=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=name,original_pid=process.pid,command=command,started_at=datetime.now().astimezone().isoformat())
            result['stages'].append(row);save();code=process.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-start)
        save()
        if code:
            result.update(status='STOPPED_AT_'+name.upper(),ended_at=row['ended_at']);save();return code
    result.update(status='COMPLETE_VERIFIED_SOURCE_ONLY',ended_at=datetime.now().astimezone().isoformat(),
                  summary_sha256=sha256(root/'source/summary.json'),cpu_sha256=sha256(root/'source/cpu_verification.json'))
    save();return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--code-commit',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
