from pathlib import Path
from datetime import datetime
import json,os,subprocess,sys,time
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590');repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
status=Path(str(root)+'_pipeline.json')
state=dict(status='RUNNING',pid=os.getpid(),execution_commit='f7a0590c891a8e3aa98be16dbb0f71adcf7dbd56',stages=[],started_at=datetime.now().astimezone().isoformat())
def save():status.write_text(json.dumps(state,indent=2)+'\n')
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
