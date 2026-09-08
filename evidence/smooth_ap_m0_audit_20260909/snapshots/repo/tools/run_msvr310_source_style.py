#!/usr/bin/env python3
"""Persistent fixed T0 -> M0 -> CPU -> Q1 -> CPU pipeline; stop on original failure."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    repo=Path.cwd().resolve();root=args.output_dir.resolve();config=args.config.resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    assert sha(config)==args.config_sha256
    contract=json.loads(config.read_bytes())
    assert sha(__file__)==contract['project_source_file_sha256']['tools/run_msvr310_source_style.py']
    assert shutil.disk_usage(root.parent).free>=contract['STORAGE']['MINIMUM_FREE_BYTES']
    assert not root.exists();root.mkdir()
    python='/root/miniconda3/envs/tri_reid/bin/python'
    common=['--config',str(config)]
    m0=root/'m0';q1=root/'q1'
    stages=[('t0','tools.check_msvr310_source_style_contract',common+['--output',str(root/'t0.json')],False),
            ('m0','tools.train_msvr310_source_style',common+['--mode','m0','--output-dir',str(m0)],True),
            ('m0_cpu','tools.verify_msvr310_source_style',common+['--run-dir',str(m0),'--output',str(root/'m0_cpu.json')],False),
            ('q1','tools.train_msvr310_source_style',common+['--mode','q1','--output-dir',str(q1),
                '--m0-receipt',str(m0/'summary.json'),'--m0-verification',str(root/'m0_cpu.json')],True),
            ('q1_cpu','tools.verify_msvr310_source_style',common+['--run-dir',str(q1),'--output',str(root/'q1_cpu.json')],False)]
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=args.code_commit,
                 config_sha256=args.config_sha256,started_at=datetime.now().astimezone().isoformat(),stages=[])
    def save():
        (root/'pipeline.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for name,module,arguments,gpu in stages:
        command=[python,'-u','-m',module,*arguments]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 PYTHONPATH=str(repo/'modeling')+':'+str(repo))
        started=time.time()
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=name,original_pid=child.pid,command=command,gpu_enabled=gpu,
                     started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save()
            print(json.dumps(row),flush=True)
            code=child.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-started)
        save();print(json.dumps(row),flush=True)
        if code:
            receipt.update(status='STOPPED_AT_'+name.upper(),ended_at=row['ended_at']);save()
            return code
    result=json.loads((q1/'summary.json').read_bytes())
    proof=json.loads((root/'q1_cpu.json').read_bytes())
    assert proof['status']=='PASS_COMPLETE_MSVR_STYLE_Q1' and proof['summary_sha256']==sha(q1/'summary.json')
    receipt.update(status='COMPLETE_VERIFIED_'+result['status'],ended_at=datetime.now().astimezone().isoformat(),
                   terminal_summary_sha256=sha(q1/'summary.json'),terminal_verification_sha256=sha(root/'q1_cpu.json'))
    save();return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
