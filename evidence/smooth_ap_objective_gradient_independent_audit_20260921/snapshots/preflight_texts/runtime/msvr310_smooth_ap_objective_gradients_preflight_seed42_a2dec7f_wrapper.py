from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f');repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
state=dict(status='RUNNING',pid=os.getpid(),execution_commit='a2dec7f535fe13b08222c219fc7e3678399fd98c',started_at=datetime.now().astimezone().isoformat())
target=Path(str(root)+'_pipeline.json')
def save():target.write_text(json.dumps(state,indent=2)+'\n')
save()
with Path(str(root)+'_preflight.log').open('xb') as log:
    command=['/root/miniconda3/envs/tri_reid/bin/python','-B','-m','tools.probe_msvr_smooth_ap_objective_gradients','--contract','configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json','--root',str(root),'--mode','preflight']
    child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save();code=child.wait()
    state.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED',ended_at=datetime.now().astimezone().isoformat());save()
sys.exit(code)
