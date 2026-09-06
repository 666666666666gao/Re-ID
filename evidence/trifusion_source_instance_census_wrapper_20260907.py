from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,time

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/autodl-tmp/trifusion-v2/artifacts/trifusion_source_instance_census_seed42_20260907')
contract=repo/'configs/RGBNT201/source-instance-census-v26.json'
expected='3139d941b3f3e2a2394b48438aefceb1e4caaba3b2108c4fbe4ad27b5d7a1e92'
assert hashlib.sha256(contract.read_bytes()).hexdigest()==expected
assert not run.exists()
assert not Path(str(run)+'.launch.json').exists()
env=dict(os.environ,PYTHONPATH=str(repo)+':'+str(repo/'modeling'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
python='/root/miniconda3/envs/tri_reid/bin/python'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
state={'status':'RUNNING_SOURCE_ONLY_CENSUS','started_at':datetime.now().astimezone().isoformat(),'wrapper_pid':os.getpid(),'execution_commit':commit,'contract_sha256':expected,'automatic_retries':0,'run_dir':str(run)}
started=time.time()
with Path(str(run)+'.log').open('xb') as log:
 child=subprocess.Popen([python,'-u','-m','tools.census_v26_source_instances','--contract',str(contract),'--contract-sha256',expected,'--output-dir',str(run)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 state['original_pid']=child.pid
 Path(str(run)+'.launch.json').write_bytes((json.dumps(state,indent=2)+'\n').encode())
 code=child.wait()
Path(str(run)+'.exit').write_text(str(code)+'\n')
state.update(status='SOURCE_CENSUS_PROCESS_COMPLETE',exit_code=code,census_completed_at=datetime.now().astimezone().isoformat())
if code==0:
 with (run/'verification.log').open('xb') as log:
  result=subprocess.run([python,'-u','-m','tools.verify_source_instance_census','--summary',str(run/'source_census.json'),'--contract',str(contract),'--output',str(run/'verification.json')],cwd=repo,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 state['verification_exit_code']=result.returncode
 (run/'verification.exit').write_text(str(result.returncode)+'\n')
 code=result.returncode
state.update(status='COMPLETE_SOURCE_CENSUS_WRAPPER',completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-started)
Path(str(run)+'.terminal.json').write_bytes((json.dumps(state,indent=2)+'\n').encode())
raise SystemExit(code)
