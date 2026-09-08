from pathlib import Path
import argparse,json,shlex
tmp=Path('C:/Users/gb/.codex_tmp')
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=tmp)
destination=parser.parse_args().output_dir
assert destination.is_dir()
launch=json.loads((tmp/'history_gradient_training_launch_20260908.json').read_bytes())
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
remote=r'''
from pathlib import Path
import json,subprocess,shutil
from datetime import datetime
root=Path(__RUN_ROOT__)
pipeline=json.loads((root/'pipeline.json').read_bytes())
assert pipeline['code_commit']==__RUN_HEAD__ and pipeline['config_sha256']==__CONFIG_SHA__
stage=pipeline['stages'][-1]
processes=[]
for pid in (pipeline['wrapper_pid'],stage['original_pid']):
 p=Path('/proc')/str(pid)
 processes.append(dict(pid=pid,present=p.exists(),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode() if p.exists() else None))
summaries={}
step_logs=[]
for name in ('m0','q1'):
 p=root/name/'summary.json'
 if p.exists():
  j=json.loads(p.read_bytes())
  summaries[name]=dict(status=j['status'],optimizer_steps=j.get('optimizer_steps'),
                       endpoints=[dict(fold=f['fold'],endpoint=e,steps=r.get('training',{}).get('optimizer_steps'),
                                       receipt_complete=(root/name/f"fold_{f['fold']}_{e}"/'receipt.json').exists(),
                                       retrieval_recorded='retrieval' in r) for f in j['folds'] for e,r in f['endpoints'].items()],
                       overfit={e:dict(steps=r['training']['optimizer_steps'],passed=r['gate']['passed']) for e,r in j['overfit'].items()})
 for log in sorted((root/name).glob('*/memory_steps.jsonl')):
  lines=log.read_bytes().split(b'\n')[:-1]
  if lines:
   last=json.loads(lines[-1])
   step_logs.append(dict(path=str(log.relative_to(root)),complete_lines=len(lines),last_step=last['step'],epoch=last.get('epoch'),modified_at=log.stat().st_mtime))
result=dict(checked_at=datetime.now().astimezone().isoformat(),pipeline=pipeline,processes=processes,summaries=summaries,
            step_logs=step_logs,
            log_tail=(root/(stage['stage']+'.log')).read_text()[-3500:],free_bytes=shutil.disk_usage(root).free,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(result))
'''
remote=remote.replace('__RUN_ROOT__',repr(launch['root'])).replace('__RUN_HEAD__',repr(launch['code_commit'])).replace('__CONFIG_SHA__',repr(launch['config_sha256']))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c '+shlex.quote(remote))
data=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
(destination/'history_gradient_training_latest_observation_20260908.json').write_bytes(data)
with (destination/'history_gradient_training_observations_20260908.jsonl').open('ab') as f:f.write(data.rstrip()+b'\n')
j=json.loads(data)
print(json.dumps(dict(checked_at=j['checked_at'],status=j['pipeline']['status'],
                     latest_stage=j['pipeline']['stages'][-1],
                     processes=[dict(pid=p['pid'],present=p['present']) for p in j['processes']],
                     summaries=j['summaries'],step_logs=j['step_logs'],
                     recent_epoch_events=[json.loads(line) for line in j['log_tail'].splitlines() if line.startswith('{"event":')][-3:],
                     free_bytes=j['free_bytes'],gpu=j['gpu'])))
c.close()
