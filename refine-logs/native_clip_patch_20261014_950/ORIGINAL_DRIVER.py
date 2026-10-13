"""One frozen source diagnosis; original handle is the only execution owner."""
import base64, hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (w/'NATIVE_PATCH_REQUEST_950.json').exists()
reg=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())
cpu=json.loads((w/'NATIVE_PATCH_CPU_950.json').read_bytes())
assert cpu['status']=='CPU_QQ_MATH_MATCHES_PINNED_AUTHOR_FUNCTION'
assert all(sha(r/name)==digest for name,digest in cpu['source_sha256'].items())
old=json.loads((w/'diagnosis948/logs/joint_spectral_diagnosis948/INPUT_SEAL.json').read_bytes())
diag=json.loads((w/'diagnosis948/results/joint_spectral_diagnosis948/joint_delta/DIAGNOSIS.json').read_bytes())
assert diag['status']=='COMPLETE' and diag['model_state_before_sha256']==diag['model_state_after_sha256']
names=[*cpu['source_sha256'],'refine-logs/native_clip_patch_20261014_950/DIAGNOSIS_PLAN.md']
sources=dict(reg['source_sha256'],**{name:sha(r/name) for name in names})
request=dict(source_sha256=sources,files={name:base64.b64encode((r/name).read_bytes()).decode() for name in names},
    old_seal=old,global_metrics=diag['scores']['global']['metrics'],
    admin='/data/gaob/Re-ID/Trifusion/logs/native_clip_patch950',
    output='/data/gaob/Re-ID/Trifusion/results/native_clip_patch950')
(w/'NATIVE_PATCH_REQUEST_950.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import ast,base64,hashlib,json,os,shutil,subprocess
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
admin=Path(request['admin']);output=Path(request['output']);old=request['old_seal']
assert not admin.exists() and not output.exists()
assert json.loads(Path(old['campaign']).read_text())['status']=='COMPLETE'
assert json.loads(Path(old['controller_exit']).read_text())['exit_code']==0
assert shutil.disk_usage(root).free>=2*1024**3+32*1024**2
processes=subprocess.check_output(['nvidia-smi','--id=0,1','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
assert not processes.strip(),processes
for name,digest in request['source_sha256'].items():
    if name not in request['files']:assert sha(root/name)==digest,name
for name,encoded in request['files'].items():
    path=root/name;assert path.resolve().is_relative_to(root.resolve())
    data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==request['source_sha256'][name]
    assert not path.exists()
    if name.endswith('.py'):ast.parse(data.decode(),filename=name)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
assert all(sha(root/name)==digest for name,digest in request['source_sha256'].items())
row=old['rows'][0];assert row['spectral_aggregation']=='joint_delta'
run=Path(row['run_dir']);assert sha(run/'best_map.pth')==row['checkpoint_sha256']
prior=root/'results/joint_spectral_diagnosis948/joint_delta/diagnostic_distances.pt'
artifacts=[Path(old['campaign']),Path(old['controller_exit']),run/'best_map.pth',Path(row['initialization']),
    Path(old['protocol']),Path(old['clip_weight']),prior,
    root/'results/joint_spectral_diagnosis948/joint_delta/DIAGNOSIS.json',
    root/'results/joint_spectral_report948/SUMMARY.json']
seal=dict(schema='trifusion-frozen-native-clip-patch-v1',source_sha256=request['source_sha256'],
    artifact_sha256={str(p):sha(p) for p in artifacts},row=row,
    original_campaign=old['campaign'],original_exit=old['controller_exit'],
    protocol=old['protocol'],signal_source=old['signal_source'],clip_weight=old['clip_weight'],
    prior_global_distance=str(prior),global_metrics=request['global_metrics'])
admin.mkdir();seal_path=admin/'INPUT_SEAL.json';seal_path.write_text(json.dumps(seal,indent=2)+'\n')
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_native_clip_patch.py'),
    '--seal',str(seal_path),'--output-dir',str(output)]
receipt=dict(status='RUNNING',owner_pid=os.getpid(),command=command,started_at=datetime.now().astimezone().isoformat(),
    gpu_01_compute_process_query=processes,input_seal_sha256=sha(seal_path),optimizer_updates=0)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (admin/'diagnosis.log').open('wb') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt['pid']=child.pid;(admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status=child.wait()
receipt.update(status='COMPLETE' if status==0 else 'FAILED',exit_code=status,completed_at=datetime.now().astimezone().isoformat())
(admin/'EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[admin/'REQUEST.json',admin/'EXIT.json',seal_path,admin/'diagnosis.log']
if status==0:files.extend(output/name for name in ('SUMMARY.json','REPORT.md'))
print(json.dumps(dict(exit_code=status,remote_free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p),data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'NATIVE_PATCH_TRANSPORT_950.stdout').write_bytes(run.stdout.encode())
(w/'NATIVE_PATCH_TRANSPORT_950.stderr').write_bytes(run.stderr.encode())
(w/'NATIVE_PATCH_TRANSPORT_EXIT_950.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'native_patch950';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'NATIVE_PATCH_COLLECTED_950.json').write_text(json.dumps(dict(exit_code=value['exit_code'],remote_free_bytes=value['remote_free_bytes'],
    files={name:{key:item[key] for key in ('bytes','sha256')} for name,item in value['files'].items()}),indent=2)+'\n')
assert value['exit_code']==0
summary=json.loads((destination/'results/native_clip_patch950/SUMMARY.json').read_bytes())
print(json.dumps(dict(status=summary['status'],global_parity=summary['global_distance_max_abs_difference'],
    local_metrics={name:{key:value['metrics'] for key,value in row['scores'].items()} for name,row in summary['scores'].items()})))
