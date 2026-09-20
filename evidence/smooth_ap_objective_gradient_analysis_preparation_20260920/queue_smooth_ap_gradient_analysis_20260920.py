from pathlib import Path
import datetime
import hashlib
import json
import subprocess

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
analyzer=repo/'evidence/smooth_ap_objective_gradient_analysis_preparation_20260920/analyze_smooth_ap_objective_gradients_20260920.py'
manifest=json.loads((analyzer.parent/'ANALYSIS_CONTRACT.json').read_bytes())
assert hashlib.sha256(analyzer.read_bytes()).hexdigest()==manifest['analyzer_sha256']
dependency=json.loads(Path(str(root)+'_verification_pipeline.json').read_bytes())
assert dependency['status']=='WAITING_FOR_SOURCE' and Path('/proc',str(dependency['pid'])).exists()
wrapper=Path(str(root)+'_analysis_wrapper.py');assert not wrapper.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
root=Path(ROOT_VALUE);analyzer=Path(ANALYZER_VALUE);repo=Path(REPO_VALUE)
state=dict(status='WAITING_FOR_VERIFICATION',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),poll_seconds=300,analyzer_sha256=HASH_VALUE)
target=Path(str(root)+'_analysis_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\\n')
save()
while True:
    dependency=json.loads(Path(str(root)+'_verification_pipeline.json').read_bytes())
    if dependency['status']=='COMPLETE':
        assert dependency['exit_code']==0
        break
    if dependency['status'] not in ('WAITING_FOR_SOURCE','RUNNING_VERIFICATION'):
        state.update(status='DEPENDENCY_NOT_PASSED',dependency_status=dependency['status'],ended_at=datetime.now().astimezone().isoformat());save();sys.exit(1)
    if not Path('/proc',str(dependency['pid'])).exists():
        state.update(status='DEPENDENCY_HANDLE_MISSING',ended_at=datetime.now().astimezone().isoformat());save();sys.exit(1)
    time.sleep(300)
assert hashlib.sha256(analyzer.read_bytes()).hexdigest()==state['analyzer_sha256']
with Path(str(root)+'_analysis.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(analyzer),str(root),str(root/'analysis')],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='RUNNING_ANALYSIS',child_pid=child.pid,analysis_started_at=datetime.now().astimezone().isoformat());save();status=child.wait()
state.update(status='COMPLETE' if status==0 else 'ANALYSIS_FAILED',exit_code=status,ended_at=datetime.now().astimezone().isoformat());save();sys.exit(status)
'''.replace('ROOT_VALUE',repr(str(root))).replace('ANALYZER_VALUE',repr(str(analyzer))).replace('REPO_VALUE',repr(str(repo))).replace('HASH_VALUE',repr(manifest['analyzer_sha256']))
wrapper.write_text(code)
with Path(str(root)+'_analysis_wrapper.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(wrapper)],cwd=repo,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(pid=child.pid,proc_exists=Path('/proc',str(child.pid)).exists(),launched_at=datetime.datetime.now().astimezone().isoformat(),wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),analyzer_sha256=manifest['analyzer_sha256'],dependency_pid=dependency['pid'],root=str(root))
Path(str(root)+'_analysis_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
