"""Observe once near the initial/M0 milestone, then every240s; never restart."""
from pathlib import Path
from datetime import datetime,timedelta
import json
import os
import subprocess
import sys
import time

private=Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
launch=json.loads((private/'deploy/LAUNCH.json').read_bytes())
out=private/'m0_observer'
assert not out.exists();out.mkdir()
due=datetime.fromisoformat(launch['observed_at'])+timedelta(minutes=12)
record={'status':'WAITING_INITIAL_M0_MILESTONE','pid':os.getpid(),'due':due.isoformat(),
        'poll_seconds':240,'launch_controller_pid':launch['controller_pid'],
        'boundary':'Read-only observation, no automatic restart or new training.'}
def save():
    record['updated_at']=datetime.now().astimezone().isoformat()
    (out/'STATUS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
save()
while datetime.now().astimezone()<due:
    time.sleep(min(240,(due-datetime.now().astimezone()).total_seconds()))
for index in range(1,10000):
    label=f'm0_milestone_{index:04d}'
    result=subprocess.run([sys.executable,'-X','utf8','C:/Users/gb/.codex_tmp/observe_metric_scale755.py','--label',label],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/f'{label}.stdout.txt').write_bytes(result.stdout)
    (out/f'{label}.stderr.txt').write_bytes(result.stderr)
    record.update(last_label=label,last_observer_exit=result.returncode)
    if result.returncode:
        record['status']='OBSERVATION_FAILED_NO_RESTART';save();raise SystemExit(result.returncode)
    path=private/'observations'/label/'STATUS.json'
    actual=json.loads(path.read_bytes());state=actual['files'].get('campaign.json',{})
    record.update(last_receipt=str(path),actual_campaign_status=state.get('status'),actual_phase=state.get('phase'))
    if state.get('status')=='FAILED' or not actual['controller_live']:
        record['status']='CONTROLLER_TERMINAL_OBSERVED';save();break
    if state.get('phase')=='full':
        m0=[row for row in state['jobs'] if row['phase']=='m0']
        assert len(m0)==6 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in m0)
        record['status']='ALL_SIX_M0_COMPLETE_FULL_PHASE_OBSERVED';save();break
    record['status']='WAITING_INITIAL_OR_M0';save();time.sleep(240)
print(json.dumps(record,indent=2))
