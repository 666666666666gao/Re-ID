"""One original queue observation after its estimated real M0 milestone."""
import base64,json,subprocess,time
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
launch=json.loads((w/'VALUE_LAUNCH_VERIFIED_951.json').read_bytes())
assert not (w/'VALUE_M0_OBSERVATION_REQUEST_952.json').exists()
deadline=datetime.fromisoformat(launch['started_at']).timestamp()+180
(w/'VALUE_M0_OBSERVATION_REQUEST_952.json').write_text(json.dumps(dict(deadline_epoch=deadline,
    deadline=datetime.fromtimestamp(deadline).astimezone().isoformat(),original_controller_pid=launch['controller_pid'],
    original_supervisor_pid=launch['supervisor_pid'],remote_observations=1,poll_seconds=240),indent=2)+'\n')
while time.time()<deadline:time.sleep(min(240,deadline-time.time()))
code=r'''
import base64,hashlib,json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/pretrained_value_training951';admin=root/'logs/pretrained_value_launch951'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
launch=json.loads((admin/'LAUNCH_VERIFIED.json').read_text());state=json.loads((campaign/'campaign.json').read_text())
files=[admin/'LAUNCH_VERIFIED.json',admin/'RUNNING.json',admin/'controller.log',campaign/'campaign.json',campaign/'SOURCE_SEAL.json']
if (admin/'EXIT.json').exists():files.append(admin/'EXIT.json')
arms=[]
for arm in state['arms']:
    child=Path(arm['campaign']);item=dict(arm)
    files.extend(child/name for name in ('campaign.json','manifest.json'))
    for name in ('initialization/RGBNT201_semantic.json','M0_ACCEPTED.json','ACCEPTED_RESULT.json'):
        if (child/name).exists():files.append(child/name)
    steps=[]
    for step in arm['steps']:
        row=dict(step);path=Path(step['output_dir'])/'training.json'
        if path.exists():
            training=json.loads(path.read_text());row['recorded_history']=training['history']
        if 'pid' in step:row['process_live']=(Path('/proc')/str(step['pid'])).exists()
        if step['mode'] in ('prepare','m0') and (child/(step['mode']+'.log')).exists():files.append(child/(step['mode']+'.log'))
        if step['mode']=='m0' and path.exists():files.append(path)
        steps.append(row)
    item['steps']=steps;arms.append(item)
sources=json.loads((campaign/'SOURCE_SEAL.json').read_text());assert all(sha(root/name)==digest for name,digest in sources.items())
result=dict(observed_at=datetime.now().astimezone().isoformat(),state=state,arms=arms,
    original_controller_live=(Path('/proc')/str(launch['controller_pid'])).exists(),
    original_supervisor_live=(Path('/proc')/str(launch['supervisor_pid'])).exists(),source_count=len(sources),
    files={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=sha(path),data=base64.b64encode(path.read_bytes()).decode()) for path in set(files)})
print(json.dumps(result))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'VALUE_M0_OBSERVATION_952.stdout').write_bytes(run.stdout.encode());(w/'VALUE_M0_OBSERVATION_952.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_M0_OBSERVATION_EXIT_952.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n');assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'value_m0952';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert len(data)==item['bytes'] and __import__('hashlib').sha256(data).hexdigest()==item['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
value['files']={name:{key:item[key] for key in ('bytes','sha256')} for name,item in value['files'].items()}
(w/'VALUE_M0_OBSERVED_952.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(observed_at=value['observed_at'],status=value['state']['status'],accepted=value['state']['accepted'],
    original_controller_live=value['original_controller_live'],arms=[dict(reference=arm['value_reference'],status=arm['status'],
    steps=[{key:step[key] for key in ('mode','status','pid','process_live') if key in step} for step in arm['steps']]) for arm in value['arms']])))
