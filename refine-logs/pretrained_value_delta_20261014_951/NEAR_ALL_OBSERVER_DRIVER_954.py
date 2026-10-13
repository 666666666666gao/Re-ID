"""Single near-finish observation of the original queue, no earlier queries."""
import base64,hashlib,json,subprocess,time
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
child=w/'value_transition955/logs/pretrained_value_training951/pretrained_value_training951_uniform_reference/campaign.json'
state=json.loads(child.read_bytes());train=next(row for row in state['steps'] if row['mode']=='train')
assert train['status']=='RUNNING' and not (w/'VALUE_NEAR_ALL_REQUEST_956.json').exists()
deadline=datetime.fromisoformat(train['started_at']).timestamp()+40*60
request=dict(deadline_epoch=deadline,deadline=datetime.fromtimestamp(deadline).astimezone().isoformat(),
    original_controller_pid=state['controller_pid'],original_train_pid=train['pid'],poll_seconds=240,
    remote_observations=1,reason='Second-full nearETA based on first measured42.197-minute stage; no current-epoch query beforedeadline')
(w/'VALUE_NEAR_ALL_REQUEST_956.json').write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps(dict(status='WAIT_ORIGINAL_FIRST_FULL_NEAR_ETA',**request)),flush=True)
while time.time()<deadline:time.sleep(min(240,deadline-time.time()))
code=r'''
import base64,hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/pretrained_value_training951';admin=root/'logs/pretrained_value_launch951'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((campaign/'campaign.json').read_text());launch=json.loads((admin/'RUNNING.json').read_text())
files=[campaign/'campaign.json',admin/'RUNNING.json',admin/'controller.log'];arms=[]
if (admin/'EXIT.json').exists():files.append(admin/'EXIT.json')
for old_arm in state['arms']:
    child=Path(old_arm['campaign']);arm=json.loads((child/'campaign.json').read_text())
    files.extend(child/name for name in ('campaign.json','manifest.json'))
    for name in ('initialization/RGBNT201_semantic.json','M0_ACCEPTED.json','ACCEPTED_RESULT.json'):
        if (child/name).exists():files.append(child/name)
    for step in arm['steps']:
        if 'pid' in step:step['process_live']=(Path('/proc')/str(step['pid'])).exists()
        path=Path(step['output_dir'])/'training.json'
        if path.exists() and step['mode'] in ('m0','train'):
            training=json.loads(path.read_text());step['recorded_history']=training['history'];files.append(path)
        if step['mode']=='evaluate':
            path=Path(step['output_dir'])/'official_metrics.json'
            if path.exists():files.append(path)
    arms.append(arm)
sources=json.loads((campaign/'SOURCE_SEAL.json').read_text());assert all(sha(root/name)==digest for name,digest in sources.items())
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),state=state,arms=arms,source_count=len(sources),
    original_controller_live=(Path('/proc')/str(launch['controller_pid'])).exists(),
    original_supervisor_live=(Path('/proc')/str(launch['supervisor_pid'])).exists(),
    files={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=sha(path),data=base64.b64encode(path.read_bytes()).decode()) for path in set(files)})))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'VALUE_NEAR_ALL_956.stdout').write_bytes(run.stdout.encode());(w/'VALUE_NEAR_ALL_956.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_NEAR_ALL_EXIT_956.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n');assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'value_all956';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
value['files']={name:{key:item[key] for key in ('bytes','sha256')} for name,item in value['files'].items()}
(w/'VALUE_NEAR_ALL_OBSERVED_956.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(observed_at=value['observed_at'],status=value['state']['status'],accepted=value['state']['accepted'],
    original_controller_live=value['original_controller_live'],arms=[dict(reference=arm['value_reference'],status=arm['status'],
        steps=[{key:row[key] for key in ('mode','status','pid','process_live','exit_code') if key in row} for row in arm['steps']]) for arm in value['arms']])))
