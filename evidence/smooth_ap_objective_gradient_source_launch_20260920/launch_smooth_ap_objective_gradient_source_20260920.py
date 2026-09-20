from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');expected='c6fbfb468cf83920018c22d2ce8be53b2a92753e'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==expected
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
assert shutil.disk_usage('/root/trifusion-storage').free>2*1024**3
preflight=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f')
checks={preflight/'summary.json':'bc5aa928dc98f0694008d044753718c87877ad14fddc797b37acf8e6abcf98d0',preflight/'preflight_verification.json':'93fd2174c5084e565db65e8a0f09208132836c6a770c4b92726facf73e2d398a',repo/'configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json':'5ea1d53a427fa6621b8b138ed903eb2869c91a39e00bf55f04c9499f4f96d987'}
for p,digest in checks.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,str(p)
assert json.loads((preflight/'preflight_verification.json').read_bytes())['status']=='PASS_COMPLETE_OBJECTIVE_GRADIENT_PREFLIGHT'
assert json.loads((preflight/'summary.json').read_bytes())['status']=='PASS_PREFLIGHT'
assert subprocess.check_output(['git','show','a2dec7f:tools/probe_msvr_smooth_ap_objective_gradients.py'],cwd=repo)==(repo/'tools/probe_msvr_smooth_ap_objective_gradients.py').read_bytes()
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
wrapper=Path(str(root)+'_wrapper.py');assert not root.exists() and not wrapper.exists()
code='''from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path(ROOT_VALUE);repo=Path(REPO_VALUE)
state=dict(status='RUNNING',pid=os.getpid(),execution_commit=COMMIT_VALUE,started_at=datetime.now().astimezone().isoformat())
target=Path(str(root)+'_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\\n')
save()
with Path(str(root)+'_source.log').open('xb') as log:
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-m','tools.probe_msvr_smooth_ap_objective_gradients','--contract','configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json','--root',str(root),'--mode','source','--preflight',PREFLIGHT_VALUE]
    child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save();code=child.wait()
    state.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',ended_at=datetime.now().astimezone().isoformat());save()
sys.exit(code)
'''.replace('ROOT_VALUE',repr(str(root))).replace('REPO_VALUE',repr(str(repo))).replace('COMMIT_VALUE',repr(expected)).replace('PREFLIGHT_VALUE',repr(str(preflight/'summary.json')))
wrapper.write_text(code)
with Path(str(root)+'_wrapper.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(wrapper)],cwd=repo,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(pid=child.pid,proc_exists=Path('/proc',str(child.pid)).exists(),root=str(root),wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),execution_commit=expected,verified_gates={str(k):v for k,v in checks.items()},launched_at=datetime.datetime.now().astimezone().isoformat(),output_free_bytes=shutil.disk_usage('/root/trifusion-storage').free)
Path(str(root)+'_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
