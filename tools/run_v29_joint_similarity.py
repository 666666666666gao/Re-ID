#!/usr/bin/env python3
"""Run one pinned CPU analysis with a durable child and terminal receipts."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from tools.analyze_v29_joint_similarity import sha


def main(args):
    repo=Path.cwd().resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    assert sha(args.contract)==args.contract_sha256
    contract=json.loads(args.contract.read_bytes())
    assert sha(__file__)==contract['pipeline_sha256']
    prefix=str(args.output_dir.resolve())
    assert args.output_dir.resolve().parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert not args.output_dir.exists()
    started=time.time()
    def write(suffix,row):
        with Path(prefix+suffix).open('x',encoding='utf-8') as stream:
            json.dump(row,stream,indent=2)
            stream.write('\n')
    command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.analyze_v29_joint_similarity',
             '--contract',str(args.contract),'--output-dir',str(args.output_dir)]
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(repo))
    with Path(prefix+'.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write('_launch.json',dict(wrapper_pid=os.getpid(),original_pid=child.pid,command=command,
                                  code_commit=args.code_commit,contract_sha256=args.contract_sha256,
                                  started_at=datetime.now().astimezone().isoformat()))
        code=child.wait()
    write('_exit.json',dict(wrapper_pid=os.getpid(),original_pid=child.pid,exit_code=code,
                           ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-started))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    raise SystemExit(main(parser.parse_args()))
