"""First CPU saved-distance report, only after original three arms terminate."""
import base64, hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
observed=json.loads((w/'ALL_FULL_OBSERVATION_946.json').read_bytes())
registered=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())
assert observed['campaign']['status']=='COMPLETE'
assert observed['campaign']['accepted']==observed['campaign']['expected']==3
assert observed['supervisor_exit']['exit_code']==0
assert not observed['original_controller_live'] and not observed['original_supervisor_live']
assert not (w/'REPORT_REQUEST_948.json').exists()
request=dict(at=datetime.now().astimezone().isoformat(),source_sha256=registered['source_sha256'],
    report_sha256=sha(r/'tools/report_joint_spectral_delta.py'),
    campaign='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_training943',
    original_admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_launch943',
    admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_report948',
    output='/data/gaob/Re-ID/Trifusion/results/joint_spectral_report948')
(w/'REPORT_REQUEST_948.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/n)==v for n,v in request['source_sha256'].items())
assert sha(root/'tools/report_joint_spectral_delta.py')==request['report_sha256']
camp=Path(request['campaign']);admin=Path(request['admin']);out=Path(request['output'])
state=json.loads((camp/'campaign.json').read_text())
ended=json.loads((Path(request['original_admin'])/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and state['accepted']==state['expected']==3
assert ended['exit_code']==0
assert not (Path('/proc')/str(state['controller_pid'])).exists()
assert not admin.exists() and not out.exists()
assert all(arm['status']=='COMPLETE' for arm in state['arms'])
assert len({arm['result']['initializer']['initial_model_state_sha256'] for arm in state['arms']})==1
admin.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/report_joint_spectral_delta.py'),
    '--campaign',str(camp),'--output-dir',str(out)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
receipt=dict(request=request,command=command,started_at=datetime.now().astimezone().isoformat(),
    owner_pid=os.getpid(),cuda_visible_devices='',report_invocations=1,neural_forwards=0)
(admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (admin/'stdout.log').open('wb') as stdout,(admin/'stderr.log').open('wb') as stderr:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=stdout,stderr=stderr)
    receipt['child_pid']=child.pid
    (admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status=child.wait()
receipt.update(exit_code=status,completed_at=datetime.now().astimezone().isoformat())
(admin/'EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[admin/'REQUEST.json',admin/'EXIT.json',admin/'stdout.log',admin/'stderr.log',camp/'campaign.json',camp/'ACCEPTED_MATRIX.json']
if status==0:files.extend([out/'SUMMARY.json',out/'REPORT.md'])
inventory=[]
for arm in state['arms']:
    directory=Path(arm['campaign']);accepted=arm['result']
    files.extend(directory/n for n in ('manifest.json','SOURCE_SEAL.json','campaign.json','ACCEPTED_RESULT.json','M0_ACCEPTED.json'))
    full=Path(accepted['run_dir']);files.extend(full/n for n in ('training.json','official_metrics.json'))
    for folder in (full,Path(accepted['m0_dir'])):
        inventory.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in folder.glob('*.pth'))
assert all(sha(root/n)==v for n,v in request['source_sha256'].items())
payload=dict(exit_code=status,weight_inventory=inventory,remote_free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode()) for p in files})
print(json.dumps(payload))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'REPORT_TRANSPORT_948.stdout').write_bytes(run.stdout.encode());(w/'REPORT_TRANSPORT_948.stderr').write_bytes(run.stderr.encode())
(w/'REPORT_TRANSPORT_EXIT_948.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'report948';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'REPORT_COLLECTED_948.json').write_text(json.dumps(dict(exit_code=value['exit_code'],weight_inventory=value['weight_inventory'],
    remote_free_bytes=value['remote_free_bytes'],files={name:{k:v for k,v in item.items() if k!='data'} for name,item in value['files'].items()}),indent=2)+'\n')
assert value['exit_code']==0,(destination/'logs/joint_spectral_report948/stderr.log').read_text()
print((destination/'results/joint_spectral_report948/REPORT.md').read_text())
