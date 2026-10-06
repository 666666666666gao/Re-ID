settings={'root': '/data/gaob/Re-ID/Trifusion', 'python': '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', 'launch': '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_launch_20261006_862', 'diagnosis': '/data/gaob/Re-ID/Trifusion/logs/fixed_best_row_transport_diagnosis_20261006_862', 'report': '/data/gaob/Re-ID/Trifusion/results/fixed_best_row_transport_diagnosis_20261006_862'}

from datetime import datetime
import json,os
from pathlib import Path
import subprocess

launch=Path(settings['launch'])
def write(name,value):
 (launch/name).write_text(json.dumps(value,indent=2)+'\n')
steps=[]
for name,source,arguments,visible in (
 ('diagnose','diagnose_row_mass_fixed_best.py',
  ['--seal',str(launch/'INPUT_SEAL.json'),'--output',settings['diagnosis']],'0,1'),
 ('report','report_row_mass_fixed_best.py',
  ['--diagnosis',settings['diagnosis'],'--output',settings['report']],'')):
 command=[settings['python'],'-B',settings['root']+'/tools/'+source,*arguments]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible,PYTHONUNBUFFERED='1')
 with (launch/(name+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=settings['root'],env=env,stdout=log,stderr=subprocess.STDOUT)
  step=dict(stage=name,command=command,pid=child.pid,
    started_at=datetime.now().astimezone().isoformat(),status='RUNNING')
  steps.append(step)
  write('CHILD.json',dict(status='RUNNING',steps=steps))
  code=child.wait()
 step.update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,
  completed_at=datetime.now().astimezone().isoformat())
 write('CHILD.json',dict(status=step['status'],steps=steps))
 if code:
  write('EXIT.json',dict(exit_code=code,stage=name,steps=steps,
   completed_at=datetime.now().astimezone().isoformat()))
  raise SystemExit(code)
write('EXIT.json',dict(exit_code=0,steps=steps,completed_at=datetime.now().astimezone().isoformat()))
