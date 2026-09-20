from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920');repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
state=dict(status='RUNNING',pid=os.getpid(),execution_commit='c6fbfb468cf83920018c22d2ce8be53b2a92753e',started_at=datetime.now().astimezone().isoformat())
target=Path(str(root)+'_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\n')
save()
with Path(str(root)+'_source.log').open('xb') as log:
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-m','tools.probe_msvr_smooth_ap_objective_gradients','--contract','configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json','--root',str(root),'--mode','source','--preflight','/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f/summary.json']
    child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save();code=child.wait()
    state.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',ended_at=datetime.now().astimezone().isoformat());save()
sys.exit(code)
