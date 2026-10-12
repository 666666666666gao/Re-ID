"""One delayed observation of the original read-only controller."""
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

work=Path(__file__).resolve().parent
launch=json.loads((work/'launch918/LAUNCH.json').read_text(encoding='utf-8'))
due=datetime.fromisoformat(launch['started_at'])+timedelta(minutes=8)
print(json.dumps(dict(status='SOLE_PASSIVE_OBSERVER_SCHEDULED',due=due.isoformat(),
                     controller_pid=launch['controller_pid'],basis='First two cases actually completed in75/73sec; observe near estimated final cases.')),flush=True)
time.sleep(max((due-datetime.now().astimezone()).total_seconds(),0))
code='''import json
from pathlib import Path
from datetime import datetime
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/region_use_fixed_best_diagnosis_admin918';out=root/'logs/region_use_fixed_best_diagnosis_20261013_918'
launch=json.loads((admin/'LAUNCH.json').read_text())
state=json.loads((out/'campaign.json').read_text())
terminal=json.loads((admin/'EXIT.json').read_text()) if (admin/'EXIT.json').exists() else None
live={str(pid):Path('/proc/'+str(pid)).exists() for pid in (launch['supervisor_pid'],launch['controller_pid'])}
for job in state['jobs']:
 job['process_live']=Path('/proc/'+str(job['pid'])).exists()
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),process_live=live,campaign=state,exit=terminal,
 controller_log_tail=(admin/'controller.log').read_text()[-5000:],neural_calls_by_observer=0)))
'''
command=['ssh','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-i','C:/Users/gb/.ssh/id_ed25519',
         '-p','2026','gaob@172.19.12.138','/usr/bin/python3','-']
result=subprocess.run(command,input=code,text=True,capture_output=True)
(work/'OBSERVATION_918.stdout').write_text(result.stdout,encoding='utf-8')
(work/'OBSERVATION_918.stderr').write_text(result.stderr,encoding='utf-8')
(work/'OBSERVATION_COMMAND_918.json').write_text(json.dumps(dict(exit_code=result.returncode,
    observed_at=datetime.now().astimezone().isoformat(),due=due.isoformat(),command=command),indent=2)+'\n',encoding='utf-8')
print(result.stdout,flush=True)
print(result.stderr,flush=True)
raise SystemExit(result.returncode)
