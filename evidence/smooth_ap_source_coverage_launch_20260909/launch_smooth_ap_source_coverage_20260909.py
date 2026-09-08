from pathlib import Path
import datetime,hashlib,json,os,subprocess
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
expected='f7a0590c891a8e3aa98be16dbb0f71adcf7dbd56'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==expected
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip())<500
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
wrapper=Path(str(root)+'_wrapper.py');status=Path(str(root)+'_pipeline.json')
assert not root.exists() and not wrapper.exists() and not status.exists()
body='''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys,time
root=Path(ROOT_VALUE);repo=Path(REPO_VALUE)
status=Path(str(root)+'_pipeline.json')
state=dict(status='RUNNING',pid=os.getpid(),execution_commit=COMMIT_VALUE,stages=[],started_at=datetime.now().astimezone().isoformat())
def save():status.write_text(json.dumps(state,indent=2)+'\\n')
save()
for mode in ('math','extract','analyze'):
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-m','tools.diagnose_msvr_smooth_ap_coverage','--contract','configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json','--root',str(root),'--mode',mode]
    with Path(str(root)+'_'+mode+'.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        row=dict(stage=mode,pid=child.pid,started_at=datetime.now().astimezone().isoformat());state['stages'].append(row);save()
        code=child.wait();row.update(exit_code=code,ended_at=datetime.now().astimezone().isoformat());save()
    if code:
        state['status']='FAILED';save();sys.exit(code)
state.update(status='COMPLETE',ended_at=datetime.now().astimezone().isoformat());save()
'''.replace('ROOT_VALUE',repr(str(root))).replace('REPO_VALUE',repr(str(repo))).replace('COMMIT_VALUE',repr(expected))
with wrapper.open('x') as f:f.write(body)
with Path(str(root)+'_wrapper.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(wrapper)],cwd=repo,
        start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(pid=child.pid,root=str(root),wrapper=str(wrapper),wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    execution_commit=expected,launched_at=datetime.datetime.now().astimezone().isoformat(),proc_exists=Path('/proc') .joinpath(str(child.pid)).exists())
Path(str(root)+'_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
