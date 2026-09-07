"""Persistent source-only preflight and complete freshness measurement."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.train_msvr310_signal_oof import sha256, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();repo=Path.cwd().resolve();root=a.output_dir.resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert sha256(a.config)==a.config_sha256
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==a.code_commit
    spec=json.loads(a.config.read_bytes())
    assert shutil.disk_usage(root.parent).free>=spec['minimum_free_bytes']
    root.mkdir()
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=a.code_commit,config_sha256=a.config_sha256,
                 started_at=datetime.now().astimezone().isoformat(),stages=[])
    def save(): write_json(root/'pipeline.json',receipt)
    save()
    for mode,folder,gpu in [('preflight','preflight',True),('verify','preflight',False),('source','source',True),('verify','source',False)]:
        if mode=='source':
            proof=json.loads((root/'preflight/cpu_verification.json').read_bytes())
            assert proof['status']=='PASS_COMPLETE_FRESHNESS_PREFLIGHT' and proof['all_steps']==72
            assert proof['summary_sha256']==sha256(root/'preflight/summary.json')
        stage=folder+('_cpu' if mode=='verify' else '')
        cmd=['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.msvr_freshness_probe',
             '--config',str(a.config.resolve()),'--output',str(root/folder),'--mode',mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(repo/'modeling')+':'+str(repo))
        started=time.monotonic()
        with (root/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=stage,original_pid=child.pid,command=cmd,started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save()
            code=child.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-started);save()
        if code:
            receipt.update(status='STOPPED_AT_'+stage.upper(),ended_at=row['ended_at']);save();return code
    final=json.loads((root/'source/cpu_verification.json').read_bytes())
    assert final['status']=='PASS_COMPLETE_FRESHNESS_SOURCE' and final['all_steps']==1560
    receipt.update(status='COMPLETE_VERIFIED_SOURCE_FRESHNESS',ended_at=datetime.now().astimezone().isoformat(),
                   summary_sha256=sha256(root/'source/summary.json'),cpu_sha256=sha256(root/'source/cpu_verification.json'))
    save();return 0


if __name__=='__main__':
    raise SystemExit(main())
