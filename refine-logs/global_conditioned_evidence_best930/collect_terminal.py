"""One delayed observation and byte-preserving text collection, no model calls."""
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

w = Path(__file__).resolve().parent
assert not (w / 'TERMINAL_OBSERVATION.json').exists()
launch = json.loads((w / 'LAUNCH.json').read_text())
time.sleep(180)
request = {'supervisor_pid': launch['supervisor_pid']}
code = 'request=' + repr(request) + '\n' + r'''
import base64,hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
admin=root/'logs/global_conditioned_evidence_best930'
campaign=json.loads((admin/'campaign.json').read_text())
exit_path=admin/'CONTROLLER_EXIT.json'
exit_record=json.loads(exit_path.read_text()) if exit_path.exists() else None
worker=campaign.get('worker_pid')
names=['INPUT_SEAL.json','campaign.json','SUPERVISOR.json','CONTROLLER_EXIT.json',
       'gradient_EXIT.json','random_EXIT.json','gradient/DIAGNOSIS.json','random/DIAGNOSIS.json',
       'gradient.log','random.log','SUPERVISOR.log']
files={}
for n in names:
 p=admin/n
 if p.exists():
  data=p.read_bytes();files[n]={'remote_path':str(p),'bytes':len(data),
   'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'campaign':campaign,
 'controller_exit':exit_record,'supervisor_pid':request['supervisor_pid'],
 'supervisor_live':Path('/proc/'+str(request['supervisor_pid'])).exists(),
 'worker_pid':worker,'worker_live':Path('/proc/'+str(worker)).exists() if worker else None,
 'files':files}))
'''
opts = ['-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none', '-o', 'BatchMode=yes']
result = subprocess.run(['ssh', *opts, '-p', '2026', 'gaob@172.19.12.138', '/usr/bin/python3', '-'],
                        input=code, text=True, capture_output=True)
(w / 'COLLECT.stderr').write_text(result.stderr, encoding='utf-8')
(w / 'COLLECT_EXIT.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),
                                              'exit_code':result.returncode}, indent=2)+'\n')
assert result.returncode == 0, result.stderr
value = json.loads(result.stdout)
import base64
for name, row in value['files'].items():
    data = base64.b64decode(row.pop('base64'))
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    p = w / 'results' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
(w / 'TERMINAL_OBSERVATION.json').write_text(json.dumps(value, indent=2)+'\n')
print(json.dumps({k:value[k] for k in ('at','controller_exit','supervisor_live','worker_live')}, ensure_ascii=False))
print(json.dumps({'campaign_status':value['campaign']['status'],'files':len(value['files'])}))
