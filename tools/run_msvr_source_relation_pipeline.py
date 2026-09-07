"""Persistent ordered source-only check, extraction, and complete CPU verification."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.train_msvr310_signal_oof import write_json, sha256


def run(args):
    root=args.root
    root.mkdir()
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        contract_sha256=sha256(args.contract),stages=[])
    def save():write_json(root/'pipeline.json',receipt)
    save()
    for stage in ('check','extract','cpu'):
        environment['CUDA_VISIBLE_DEVICES']='0' if stage=='extract' else ''
        module='tools.verify_msvr_source_relations' if stage=='cpu' else 'tools.diagnose_msvr_source_relations'
        command=[sys.executable,'-u','-m',module,'--contract',str(args.contract),'--root',str(root)]
        if stage!='cpu':command+=['--mode',stage]
        started=time.monotonic()
        with (root/f'{stage}.log').open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
            row=dict(stage=stage,original_pid=process.pid,command=command,
                     started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save()
            code=process.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-started)
        save()
        if code:
            receipt.update(status='STOPPED_AT_'+stage.upper(),ended_at=row['ended_at']);save()
            return code
    verified=json.loads((root/'cpu_verification.json').read_bytes())
    assert verified['status']=='PASS_COMPLETE_SOURCE_RELATION_CENSUS'
    receipt.update(status='COMPLETE_VERIFIED_SOURCE_DIAGNOSTIC',ended_at=datetime.now().astimezone().isoformat(),
                   cpu_verification_sha256=sha256(root/'cpu_verification.json'))
    save()
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    sys.exit(run(parser.parse_args()))
