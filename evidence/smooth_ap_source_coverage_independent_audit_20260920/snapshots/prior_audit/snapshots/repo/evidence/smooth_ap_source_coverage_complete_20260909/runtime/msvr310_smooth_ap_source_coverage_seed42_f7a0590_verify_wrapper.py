from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,sys,time
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590');script=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID/evidence/smooth_ap_source_coverage_launch_20260909/verify_smooth_ap_source_coverage_20260909.py')
assert hashlib.sha256(script.read_bytes()).hexdigest()=='03765d14fb49031c86e282a8af38da5780cd92c21330e74d3857e8943aa19799'
state=dict(status='WAITING_ORIGINAL_COMPLETE',pid=os.getpid(),created_at=datetime.now().astimezone().isoformat(),script_sha256='03765d14fb49031c86e282a8af38da5780cd92c21330e74d3857e8943aa19799')
target=Path(str(root)+'_verify_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\n')
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
