from pathlib import Path
import json,subprocess
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3')
p=json.loads((root/'pipeline.json').read_bytes())
pids=[p['wrapper_pid']]+[s['original_pid'] for s in p['stages']]
processes=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,stat,etime,args'],capture_output=True,text=True)
stage=p['stages'][-1]['stage'];log=root/(stage+'.log')
print(json.dumps(dict(pipeline=p,processes=processes.stdout,log_tail=log.read_text()[-2500:]),indent=2))
