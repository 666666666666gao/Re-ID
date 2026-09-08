from pathlib import Path
import hashlib,json,subprocess,datetime
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
script=repo/'evidence/smooth_ap_source_coverage_launch_20260909/verify_smooth_ap_source_coverage_20260909.py'
expected='efb1d04f70f758d5d88574297cf83bf8c51909ed'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==expected
assert subprocess.check_output(['git','show',expected+':'+script.relative_to(repo).as_posix()],cwd=repo)==script.read_bytes()
wrapper=Path(str(root)+'_verify_wrapper.py');assert not wrapper.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
root=Path(ROOT_VALUE);script=Path(SCRIPT_VALUE)
assert hashlib.sha256(script.read_bytes()).hexdigest()==SHA_VALUE
state=dict(status='WAITING_ORIGINAL_COMPLETE',pid=os.getpid(),created_at=datetime.now().astimezone().isoformat(),script_sha256=SHA_VALUE)
target=Path(str(root)+'_verify_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\\n')
save()
while True:
    original=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
    if original['status']!='RUNNING':break
    assert Path('/proc',str(original['pid'])).exists()
    active=[r for r in original['stages'] if 'exit_code' not in r]
    assert len(active)==1 and Path('/proc',str(active[0]['pid'])).exists()
    state['last_wait_at']=datetime.now().astimezone().isoformat();save();time.sleep(180)
assert original['status']=='COMPLETE' and all(r['exit_code']==0 for r in original['stages'])
with Path(str(root)+'_verify.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(script)],stdout=log,stderr=subprocess.STDOUT)
    state.update(status='RUNNING',child_pid=child.pid,started_at=datetime.now().astimezone().isoformat());save()
    code=child.wait();state.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',ended_at=datetime.now().astimezone().isoformat());save()
sys.exit(code)
'''.replace('ROOT_VALUE',repr(str(root))).replace('SCRIPT_VALUE',repr(str(script))).replace('SHA_VALUE',repr(hashlib.sha256(script.read_bytes()).hexdigest()))
wrapper.write_text(code)
with Path(str(root)+'_verify_wrapper.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(wrapper)],cwd=repo,
                           start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(pid=child.pid,proc_exists=Path('/proc',str(child.pid)).exists(),root=str(root),
    wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    launched_at=datetime.datetime.now().astimezone().isoformat(),verification_code_commit=expected)
Path(str(root)+'_verify_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
