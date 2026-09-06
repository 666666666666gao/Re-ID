from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,time
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/autodl-tmp/trifusion-v2/artifacts/source_support_decomposition_20260907')
contract=repo/'configs/RGBNT201/source-support-decomposition.json'
expected='6bd3fc612ae3d826fa6a888daaa08410a1f8466182c2eb63aa7bb8803ce64e8b'
assert hashlib.sha256(contract.read_bytes()).hexdigest()==expected
assert not run.exists() and not Path(str(run)+'.launch.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
state={'started_at':datetime.now().astimezone().isoformat(),'wrapper_pid':os.getpid(),'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'contract_sha256':expected,'automatic_retries':0,'run_dir':str(run)}
started=time.time()
with Path(str(run)+'.log').open('xb') as log:
 child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-u','-m','tools.decompose_source_retrieval_support','--contract',str(contract),'--contract-sha256',expected,'--output-dir',str(run)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 state['original_pid']=child.pid
 Path(str(run)+'.launch.json').write_bytes((json.dumps(state,indent=2)+'\n').encode())
 code=child.wait()
Path(str(run)+'.exit').write_text(str(code)+'\n')
state.update(status='COMPLETE_SOURCE_SUPPORT_PROCESS',exit_code=code,completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-started)
Path(str(run)+'.terminal.json').write_bytes((json.dumps(state,indent=2)+'\n').encode())
raise SystemExit(code)
