#!/usr/bin/env python3
"""Persistent registered CPU check -> remaining Q1 -> complete CPU verification."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.train_msvr310_signal_oof import sha256


def main(args):
    repo=Path.cwd().resolve();root=args.output_dir.resolve();contract=args.contract.resolve()
    assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    assert sha256(contract)==args.contract_sha256
    plan=json.loads(contract.read_bytes())
    assert sha256(__file__)==plan['project_file_sha256']['tools/run_msvr310_style_distance_resume.py']
    assert shutil.disk_usage(root.parent).free>=2*1024**3
    assert not root.exists();root.mkdir()
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=args.code_commit,
                 contract_sha256=args.contract_sha256,started_at=datetime.now().astimezone().isoformat(),stages=[])
    def save():(root/'pipeline.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for mode in ('check','q1','cpu'):
        command=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.resume_msvr310_style_exact_distance',
                 '--mode',mode,'--contract',str(contract),'--root',str(root)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if mode=='q1' else '',OMP_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(repo/'modeling')+':'+str(repo))
        started=time.time()
        with (root/(mode+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=mode,original_pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save();code=child.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-started)
        save()
        if code:
            receipt.update(status='STOPPED_AT_'+mode.upper(),ended_at=row['ended_at']);save();return code
    verification=json.loads((root/'resume_verification.json').read_bytes())
    assert verification['status']=='PASS_FIXED_ORIGINAL_END_REUSE_AND_EXACT_CPU_DISTANCE'
    summary=json.loads((root/'q1/summary.json').read_bytes())
    receipt.update(status='COMPLETE_VERIFIED_'+summary['status'],ended_at=datetime.now().astimezone().isoformat(),
                   summary_sha256=sha256(root/'q1/summary.json'),resume_verification_sha256=sha256(root/'resume_verification.json'))
    save();return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
