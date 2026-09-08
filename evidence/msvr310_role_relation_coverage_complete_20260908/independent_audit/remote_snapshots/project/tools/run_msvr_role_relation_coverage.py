"""Durable CPU-only wrapper for the sealed source relation reanalysis."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    assert not args.root.exists()
    args.root.mkdir()
    repository = Path(__file__).resolve().parents[1]
    state = dict(status='RUNNING', started_at=datetime.now().astimezone().isoformat(),
                 wrapper_pid=os.getpid(), code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip(),
                 contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest())
    command = [sys.executable,'-u','-m','tools.analyze_msvr_role_relation_coverage','--contract',str(args.contract),'--output',str(args.root/'analysis')]
    state['command'] = command
    started=time.perf_counter()
    with (args.root/'analysis.log').open('xb') as log:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
        child=subprocess.Popen(command,cwd=repository,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['analysis_pid']=child.pid
        (args.root/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        code=child.wait()
    state.update(status='COMPLETE_UNAUDITED_SOURCE_REANALYSIS' if code==0 else 'ERROR_SOURCE_REANALYSIS',exit_code=code,elapsed_seconds=time.perf_counter()-started,completed_at=datetime.now().astimezone().isoformat())
    (args.root/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args())
