"""Bound, CPU-only execution of the role-set mathematical checks."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    repo=Path.cwd().resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    spec=json.loads(args.config.read_bytes())
    assert spec['schema']=='msvr310-role-set-math-v1' and spec['seed']==42
    for name,digest in spec['source_sha256'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
    root=args.output_dir.resolve()
    assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    root.mkdir()
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=args.code_commit,
                 config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                 started_at=datetime.now().astimezone().isoformat())
    def save():
        (root/'pipeline.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    save()
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-u','-m',
             'tools.check_msvr_role_set_relations','--output',str(root/'math.json')]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=str(repo/'modeling')+':'+str(repo),OMP_NUM_THREADS='4',
                     MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (root/'math.log').open('x') as log:
        child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(command=command,original_pid=child.pid);save()
        code=child.wait()
    receipt.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat(),
                   status='COMPLETE_MATH_CHECKS' if code==0 else 'FAILED_MATH_CHECKS')
    if code==0:
        assert json.loads((root/'math.json').read_bytes())['status']=='PASS_ROLE_SET_MATHEMATICS'
        receipt['math_sha256']=hashlib.sha256((root/'math.json').read_bytes()).hexdigest()
    save()
    raise SystemExit(code)
