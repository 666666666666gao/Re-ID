import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

w=Path(__file__).resolve().parent
row=json.loads((w/'SEED44_RGBNT100_M0_OBSERVED.json').read_bytes())
assert row['status']=='M0_PASS_FRESH_TRAIN_CONTROLLER_LIVE'
due=datetime.fromisoformat(row['near_finish_due'])
assert not (w/'SEED44_RGBNT100_NEAR_FINISH_REQUEST.json').exists()
(w/'SEED44_RGBNT100_NEAR_FINISH_REQUEST.json').write_text(json.dumps(dict(dataset='RGBNT100',seed=44,due=due.isoformat(),
    registered_at=datetime.now().astimezone().isoformat(),remote_queries_before_due=0,
    source='Actual new fresh start plus qualified seed42 full50 wall including official epoch evaluations, less3min. One near-end read only.'))+'\n',encoding='utf-8')
print(json.dumps(dict(status='WAITING_UNTIL_REGISTERED_ESTIMATE',seed=44,due=due.isoformat())),flush=True)
time.sleep(max(0,(due-datetime.now().astimezone()).total_seconds()))
code=r'''
import base64,hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
directory=root/'logs/official_extra_seed44_RGBNT100_R2_near_signal_rgbnt100_seed44_20261014_982_bestmap_e50_20260926'
state=json.loads((directory/'campaign.json').read_text());assert state['seed']==44 and state['training_epochs']==50
admin=root/'logs/near_signal_rgbnt100_seed44_launch982'
files=[directory/'campaign.json'];files.extend(p for p in (admin/'RUNNING.json',admin/'EXIT.json',admin/'stderr') if p.is_file())
if state['status']=='COMPLETE':
    run=Path(state['jobs'][0]['metrics_path']).parent
    files.extend(run/name for name in ('training.json','official_metrics.json','epoch_official_metrics.jsonl'))
    receipt=json.loads((run/'official_metrics.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['seed']==44 and receipt['training_epochs']==50
    for name,key in (('roles_best_map.pth','role_checkpoint_sha256'),('official_distances.pt','distance_arrays_sha256')):
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==receipt[key]
result=dict(status=state['status'],state=state,observed_at=datetime.now().astimezone().isoformat(),
    files={str(p.relative_to(root)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=base64.b64encode(p.read_bytes()).decode()) for p in files})
print(json.dumps(result))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'SEED44_RGBNT100_NEAR_FINISH.stdout').write_bytes(run.stdout);(w/'SEED44_RGBNT100_NEAR_FINISH.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);dest=w/'seed44_near_finish_RGBNT100';dest.mkdir()
for name,item in value.pop('files').items():
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=dest/name;assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'SEED44_RGBNT100_NEAR_FINISH_OBSERVED.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(seed=44,status=value['status'],observed_at=value['observed_at'])),flush=True)
