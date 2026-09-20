from pathlib import Path
from datetime import datetime
import json,os,subprocess,time,sys,hashlib
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920');checker=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID/evidence/smooth_ap_objective_gradient_source_launch_20260920/verify_smooth_ap_objective_gradient_source_20260920.py');repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
state=dict(status='WAITING_FOR_SOURCE',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),poll_seconds=300,checker_sha256='79802db8ef471e8fc495f625aa15321f2d9c1034b13afd82a8f08b1541fb2ef4')
target=Path(str(root)+'_verification_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\n')
save()
while True:
    original=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
    if original['status']=='COMPLETE':
        assert original['exit_code']==0
        break
    if original['status']=='FAILED':
        state.update(status='SOURCE_FAILED_NOT_VERIFIED',source_exit_code=original['exit_code'],ended_at=datetime.now().astimezone().isoformat());save();sys.exit(1)
    if not Path('/proc',str(original['pid'])).exists() or not Path('/proc',str(original['child_pid'])).exists():
        state.update(status='SOURCE_HANDLE_MISSING_NOT_VERIFIED',ended_at=datetime.now().astimezone().isoformat());save();sys.exit(1)
    time.sleep(300)
assert hashlib.sha256(checker.read_bytes()).hexdigest()==state['checker_sha256']
assert json.loads((root/'summary.json').read_bytes())['status']=='COMPLETE_SOURCE_OBJECTIVE_GRADIENTS'
with Path(str(root)+'_verification.log').open('xb') as log:
    child=subprocess.Popen(['/root/miniconda3/envs/tri_reid/bin/python','-B',str(checker),str(root)],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='RUNNING_VERIFICATION',child_pid=child.pid,verification_started_at=datetime.now().astimezone().isoformat());save();status=child.wait()
state.update(status='COMPLETE' if status==0 else 'VERIFICATION_FAILED',exit_code=status,ended_at=datetime.now().astimezone().isoformat());save();sys.exit(status)
