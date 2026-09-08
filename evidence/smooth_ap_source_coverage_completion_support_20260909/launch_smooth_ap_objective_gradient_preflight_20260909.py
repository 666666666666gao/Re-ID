from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');expected='a2dec7f535fe13b08222c219fc7e3678399fd98c'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==expected
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
assert shutil.disk_usage('/root/trifusion-storage').free>2*1024**3
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f')
wrapper=Path(str(root)+'_wrapper.py');assert not root.exists() and not wrapper.exists()
code='''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path(ROOT_VALUE);repo=Path(REPO_VALUE)
state=dict(status='RUNNING',pid=os.getpid(),execution_commit=COMMIT_VALUE,started_at=datetime.now().astimezone().isoformat())
target=Path(str(root)+'_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\\n')
save()
with Path(str(root)+'_preflight.log').open('xb') as log:
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-m','tools.probe_msvr_smooth_ap_objective_gradients','--contract','configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json','--root',str(root),'--mode','preflight']
    child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save();code=child.wait()
    state.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',ended_at=datetime.now().astimezone().isoformat());save()
sys.exit(code)
'''.replace('ROOT_VALUE',repr(str(root))).replace('REPO_VALUE',repr(str(repo))).replace('COMMIT_VALUE',repr(expected))
wrapper.write_text(code)
with Path(str(root)+'_wrapper.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(wrapper)],cwd=repo,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(pid=child.pid,proc_exists=Path('/proc',str(child.pid)).exists(),root=str(root),
    wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),execution_commit=expected,
    launched_at=datetime.datetime.now().astimezone().isoformat())
Path(str(root)+'_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
