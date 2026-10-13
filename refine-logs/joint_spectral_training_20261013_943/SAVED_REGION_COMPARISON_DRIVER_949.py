"""First fixed-data local-region comparison; CPU only, no more extraction."""
import base64,hashlib,json,subprocess
from datetime import datetime
from pathlib import Path
w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
diagnosis=json.loads((w/'DIAGNOSIS_COLLECTED_948.json').read_bytes())
assert diagnosis['exit_code']==0
assert not (w/'REGION_COMPARISON_REQUEST_949.json').exists()
old=json.loads((w/'DIAGNOSIS_REQUEST_948.json').read_bytes())
source='tools/compare_saved_region_reads.py';assert sha(r/source)==old['source_sha256'][source]
request=dict(source_sha256=old['source_sha256'],diagnosis_admin=old['admin'],diagnosis_output=old['output'],
    original_campaign=old['parent']+'/campaign.json',original_exit=old['original_admin']+'/EXIT.json',
    admin='/data/gaob/Re-ID/Trifusion/logs/saved_region_comparison949',
    output='/data/gaob/Re-ID/Trifusion/results/saved_region_comparison949',at=datetime.now().astimezone().isoformat(),
    expected_conditions=old['expected_conditions'],comparison_invocations=1,neural_forwards=0)
(w/'REGION_COMPARISON_REQUEST_949.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import base64,hashlib,json,os,subprocess
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=Path(request['diagnosis_admin']);features=Path(request['diagnosis_output'])
receipt=json.loads((prior/'EXIT.json').read_text());assert receipt['status']=='COMPLETE' and receipt['exit_code']==0
assert [row['aggregation'] for row in receipt['arms']]==request['expected_conditions']
assert all(row['status']=='COMPLETE' and row['exit_code']==0 for row in receipt['arms'])
assert all(sha(root/n)==v for n,v in request['source_sha256'].items())
admin=Path(request['admin']);output=Path(request['output']);assert not admin.exists() and not output.exists()
rows=[];artifacts=[prior/'EXIT.json',prior/'INPUT_SEAL.json',Path(request['original_campaign']),Path(request['original_exit'])]
for mode in request['expected_conditions']:
    run=features/mode;accepted=json.loads((run/'DIAGNOSIS.json').read_text())
    assert accepted['status']=='COMPLETE' and accepted['optimizer_updates']==0
    assert accepted['model_state_before_sha256']==accepted['model_state_after_sha256']
    assert accepted['fused_distance_max_abs_difference']==0
    artifacts.append(run/'DIAGNOSIS.json')
    for name,item in accepted['artifacts'].items():
        path=run/name;assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'];artifacts.append(path)
    rows.append(dict(aggregation=mode,diagnosis_dir=str(run)))
admin.mkdir()
seal=dict(schema='trifusion-saved-region-comparison-v1',conditions=request['expected_conditions'],rows=rows,
    source_sha256=request['source_sha256'],artifact_sha256={str(p):sha(p) for p in artifacts},
    original_campaign=request['original_campaign'],original_controller_exit=request['original_exit'])
seal_path=admin/'INPUT_SEAL.json';seal_path.write_text(json.dumps(seal,indent=2)+'\n')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/compare_saved_region_reads.py'),
    '--seal',str(seal_path),'--output-dir',str(output)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
state=dict(request=request,command=command,owner_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
    neural_forwards=0,comparison_invocations=1,cuda_visible_devices='')
(admin/'REQUEST.json').write_text(json.dumps(state,indent=2)+'\n')
with (admin/'stdout.log').open('wb') as stdout,(admin/'stderr.log').open('wb') as stderr:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=stdout,stderr=stderr)
    state['child_pid']=child.pid;(admin/'REQUEST.json').write_text(json.dumps(state,indent=2)+'\n');status=child.wait()
state.update(exit_code=status,completed_at=datetime.now().astimezone().isoformat())
(admin/'EXIT.json').write_text(json.dumps(state,indent=2)+'\n')
files=[admin/'REQUEST.json',admin/'EXIT.json',admin/'INPUT_SEAL.json',admin/'stdout.log',admin/'stderr.log']
if status==0:files.extend([output/'SUMMARY.json',output/'REPORT.md'])
print(json.dumps(dict(exit_code=status,files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size,
    data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'REGION_COMPARISON_TRANSPORT_949.stdout').write_bytes(run.stdout.encode());(w/'REGION_COMPARISON_TRANSPORT_949.stderr').write_bytes(run.stderr.encode())
(w/'REGION_COMPARISON_TRANSPORT_EXIT_949.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'regions949';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'REGION_COMPARISON_COLLECTED_949.json').write_text(json.dumps(dict(exit_code=value['exit_code'],neural_forwards=0,
    files={name:{k:v for k,v in item.items() if k!='data'} for name,item in value['files'].items()}),indent=2)+'\n')
assert value['exit_code']==0,(destination/'logs/saved_region_comparison949/stderr.log').read_text()
print((destination/'results/saved_region_comparison949/REPORT.md').read_text())
