"""Persistent source-only GPU probe followed by complete CPU arithmetic replay."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();repo=Path.cwd().resolve()
    assert repo==Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==args.code_commit
    spec=json.loads(args.config.read_bytes())
    assert spec['schema']=='msvr310-role-set-gradient-check-v1'
    for name,digest in spec['source_sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
    root=args.output_dir.resolve();assert root.parent==Path('/root/autodl-tmp/trifusion-v2/artifacts')
    free=shutil.disk_usage(root.parent).free;assert free>=spec['minimum_free_bytes']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    assert int(gpu.strip().split(',')[1])<500
    root.mkdir()
    receipt=dict(status='RUNNING',wrapper_pid=os.getpid(),code_commit=args.code_commit,
                 config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                 gpu_before=gpu,free_bytes_before=free,started_at=datetime.now().astimezone().isoformat(),stages=[])
    def save():
        (root/'pipeline.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    save()
    stages=[('probe','tools.probe_msvr_role_set_gradients',['--output-dir',str(root/'probe')],'0'),
            ('cpu','tools.verify_msvr_role_set_gradients',['--run-dir',str(root/'probe'),'--output',str(root/'cpu.json')],'')]
    for name,module,extra,device in stages:
        command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-u','-m',module,'--config',str(args.config.resolve()),*extra]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=device,PYTHONDONTWRITEBYTECODE='1',
                         PYTHONPATH=str(repo/'modeling')+':'+str(repo),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=name,original_pid=child.pid,command=command,started_at=datetime.now().astimezone().isoformat())
            receipt['stages'].append(row);save();code=child.wait()
        row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat());save()
        if code:
            receipt.update(status='STOPPED_AT_'+name.upper(),ended_at=row['ended_at']);save();raise SystemExit(code)
    result=json.loads((root/'cpu.json').read_bytes())
    assert result['status']=='PASS_COMPLETE_ROLE_SET_GRADIENT_PROBE_TEXT_AND_ARRAYS'
    receipt.update(status='COMPLETE_FIXED_STATE_ENGINEERING_CHECK',ended_at=datetime.now().astimezone().isoformat(),
                   cpu_sha256=hashlib.sha256((root/'cpu.json').read_bytes()).hexdigest(),free_bytes_after=shutil.disk_usage(root).free)
    save()
