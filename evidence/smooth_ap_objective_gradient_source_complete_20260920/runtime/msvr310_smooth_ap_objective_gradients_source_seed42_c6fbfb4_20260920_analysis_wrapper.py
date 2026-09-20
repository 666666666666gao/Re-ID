from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920');analyzer=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID/evidence/smooth_ap_objective_gradient_analysis_preparation_20260920/analyze_smooth_ap_objective_gradients_20260920.py');repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
state=dict(status='WAITING_FOR_VERIFICATION',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),poll_seconds=300,analyzer_sha256='a4337f88e93200d7738018a4288022a3a98f10a34c724fdda13a33d6849a0653')
target=Path(str(root)+'_analysis_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\n')
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
