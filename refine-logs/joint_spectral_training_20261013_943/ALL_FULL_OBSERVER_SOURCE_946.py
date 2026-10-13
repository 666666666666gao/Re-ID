"""One final-group ETA observation of the original three-arm controller."""
from datetime import datetime,timedelta
import json,os,subprocess,time
from pathlib import Path
w=Path(__file__).resolve().parent
first=json.loads((w/'FIRST_FULL_OBSERVATION.json').read_bytes())
second=first['arms'][1]['state']['active_command']
deadline=datetime.fromisoformat(second['started_at'])+timedelta(minutes=87)
assert not (w/'ALL_FULL_OBSERVER_946.json').exists()
timer=dict(deadline=deadline.isoformat(),owner_pid=os.getpid(),born_at=datetime.now().astimezone().isoformat(),
           original_controller=first['campaign']['controller_pid'],
           estimate='First full loop40m11s/first strict45s, second original start00:35:48; two remaining arms plus prepare/M0 and approximately3min allowance.',
           scope='One near entire-group ETA metadata observation; no NN/restart/progress query before deadline.')
(w/'ALL_FULL_OBSERVER_946.json').write_text(json.dumps(timer,indent=2)+'\n')
remaining=deadline.timestamp()-time.time()
if remaining>0:time.sleep(remaining)
code=r'''
import json
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');admin=root/'logs/joint_spectral_launch943';campaign=root/'logs/joint_spectral_training943'
launch=json.loads((admin/'LAUNCH_VERIFIED.json').read_text());state=json.loads((campaign/'campaign.json').read_text())
result=dict(observed_at=datetime.now().astimezone().isoformat(),campaign=state,
    original_controller_live=(Path('/proc')/str(launch['controller_pid'])).is_dir(),
    original_supervisor_live=(Path('/proc')/str(launch['supervisor_pid'])).is_dir(),
    supervisor_exit=json.loads((admin/'EXIT.json').read_text()) if (admin/'EXIT.json').exists() else None,
    arms=[],controller_log_tail=(admin/'controller.log').read_text()[-7000:])
for mode in ('joint_delta','add_delta','joint_absolute'):
    child=campaign/(campaign.name+'_'+mode)
    if not child.exists():continue
    arm=json.loads((child/'campaign.json').read_text());row=dict(mode=mode,state=arm)
    active=arm.get('active_command')
    if active:
        proc=Path('/proc')/str(active['pid']);row['active_child_live']=proc.is_dir()
        if proc.is_dir():
            row['active_child_cmdline']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            row['active_child_cuda']=next(x.split(b'=',1)[1].decode() for x in (proc/'environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES='))
    full=root/('trained-model/'+child.name+'_full_semantic_RGBNT201')
    path=full/'training.json'
    if path.exists():row['full_training_receipt']=json.loads(path.read_text())
    if (child/'ACCEPTED_RESULT.json').exists():row['accepted']=json.loads((child/'ACCEPTED_RESULT.json').read_text())
    if (child/'M0_ACCEPTED.json').exists():row['M0_accepted']=json.loads((child/'M0_ACCEPTED.json').read_text())
    for stage in ('prepare','m0','train','evaluate'):
        path=child/(stage+'.log')
        if path.exists() and any(step['mode']==stage and step.get('status')=='FAILED' for step in arm['steps']):
            row[stage+'_failure_log_tail']=path.read_text()[-7000:]
    result['arms'].append(row)
if (campaign/'ACCEPTED_MATRIX.json').exists():result['accepted_matrix']=json.loads((campaign/'ACCEPTED_MATRIX.json').read_text())
print(json.dumps(result))
'''
opt=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opt,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'ALL_FULL_OBSERVATION_946.stdout').write_bytes(run.stdout.encode());(w/'ALL_FULL_OBSERVATION_946.stderr').write_bytes(run.stderr.encode())
(w/'ALL_FULL_OBSERVATION_EXIT_946.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);(w/'ALL_FULL_OBSERVATION_946.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(controller_live=value['original_controller_live'],supervisor_exit=value['supervisor_exit'],
    status=value['campaign']['status'],arms=[dict(mode=row['mode'],status=row['state']['status'],
        full_epochs=len(row.get('full_training_receipt',{}).get('history',[])),accepted='accepted' in row) for row in value['arms']])))
