"""Run three once-only fixed-best extractions after accepted report/retention."""
import hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((w/'REPORT_COLLECTED_948.json').read_bytes())
retention=json.loads((w/'M0_RETENTION_948.json').read_bytes())
assert report['exit_code']==0 and retention['status']=='THREE_M0_D_ARCHIVES_VERIFIED_REMOTE_DUPLICATES_RETIRED'
assert all(sha(Path(row['path']))==row['sha256'] for row in retention['archives'].values())
assert not (w/'DIAGNOSIS_REQUEST_948.json').exists()
reg=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())
owned=json.loads((w/'REGION_CHECK_947_REQUEST.json').read_bytes())['sources']
assert all(sha(r/n)==v for n,v in owned.items())
plan='refine-logs/joint_spectral_training_20261013_943/SAVED_REGION_COMPARISON_PLAN_947.md'
request=dict(at=datetime.now().astimezone().isoformat(),source_sha256=dict(reg['source_sha256'],**owned,**{plan:sha(r/plan)}),
    parent='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_training943',
    original_admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_launch943',
    report_admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_report948',
    report='/data/gaob/Re-ID/Trifusion/results/joint_spectral_report948',
    admin='/data/gaob/Re-ID/Trifusion/logs/joint_spectral_diagnosis948',
    output='/data/gaob/Re-ID/Trifusion/results/joint_spectral_diagnosis948',
    driver_sha256=sha(Path(__file__)),expected_conditions=['joint_delta','add_delta','joint_absolute'],
    expected_approximate_seconds=300,physical_gpus=[0,1],optimizer_updates=0)
(w/'DIAGNOSIS_REQUEST_948.json').write_text(json.dumps(request,indent=2)+'\n')
code='request='+repr(request)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=Path(request['parent']);report=Path(request['report']);admin=Path(request['admin']);out=Path(request['output'])
state=json.loads((parent/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['accepted']==state['expected']==3
assert json.loads((Path(request['original_admin'])/'EXIT.json').read_text())['exit_code']==0
assert json.loads((Path(request['report_admin'])/'EXIT.json').read_text())['exit_code']==0
assert json.loads((report/'SUMMARY.json').read_text())['status']=='COMPLETE_SAVED_DISTANCE_DIAGNOSIS'
assert json.loads((Path(request['report_admin'])/'M0_RETENTION.json').read_text())['status']=='THREE_M0_D_ARCHIVES_VERIFIED_REMOTE_DUPLICATES_RETIRED'
assert all(sha(root/n)==v for n,v in request['source_sha256'].items())
assert not (Path('/proc')/str(state['controller_pid'])).exists()
assert not admin.exists() and not out.exists()
assert shutil.disk_usage(root).free>=2*1024**3+1096096320+256*1024**2
gpu_rows=subprocess.check_output(['nvidia-smi','--id=0,1','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
assert not gpu_rows.strip(),gpu_rows
controls_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
controls=json.loads(controls_path.read_text());global_row=next(row for row in controls['rows'] if row['dataset']=='RGBNT201' and row['variant']=='global_only')
protocol=root/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'
clip=root/'pertrained-model/ViT-B-16.pt';independent=Path(global_row['run_dir'])/'official_distances.pt'
assert sha(independent)==global_row['distance_sha256']
rows=state['results'];assert [row['spectral_aggregation'] for row in rows]==request['expected_conditions']
artifacts=[parent/'campaign.json',parent/'SOURCE_SEAL.json',parent/'ACCEPTED_MATRIX.json',
    Path(request['original_admin'])/'EXIT.json',Path(request['report_admin'])/'EXIT.json',
    Path(request['report_admin'])/'M0_RETENTION.json',report/'SUMMARY.json',report/'REPORT.md',controls_path,protocol,clip,independent]
for row,arm in zip(rows,state['arms']):
    child=Path(arm['campaign']);row['initialization']=str(child/'initialization/RGBNT201_semantic.json')
    artifacts.extend(child/n for n in ('manifest.json','campaign.json','ACCEPTED_RESULT.json','M0_ACCEPTED.json'))
    artifacts.append(Path(row['initialization']))
    run=Path(row['run_dir']);artifacts.extend(run/n for n in ('training.json','official_metrics.json','best_map.pth','official_distances.pt'))
    assert sha(run/'best_map.pth')==row['checkpoint_sha256'] and sha(run/'official_distances.pt')==row['distance_sha256']
    assert sha(clip)==row['initializer']['public_clip_sha256'] and sha(protocol)==row['initializer']['protocol_sha256']
    assert not (Path(row['m0_dir'])/'m0_reload_probe.pth').exists()
admin.mkdir();out.mkdir()
seal=dict(schema='trifusion-joint-spectral-fixed-best-diagnosis-v1',conditions=request['expected_conditions'],
    source_sha256=request['source_sha256'],artifact_sha256={str(p):sha(p) for p in artifacts},
    campaign=str(parent/'campaign.json'),controller_exit=str(Path(request['original_admin'])/'EXIT.json'),
    rows=rows,protocol=str(protocol),clip_weight=str(clip),signal_source=str(root/'comparators/Signal-cd1b0a6'),
    independent_global_distance=str(independent),independent_global_metrics=global_row['metrics'],
    physical_gpus=[0,1],optimizer_updates=0,
    boundary='First strict and CPU report completed before exact M0 archive retirement; this extractor reads receipts/witness/best, not retired probe binaries. Restore probes before old verifier reuse.')
seal_path=admin/'INPUT_SEAL.json';seal_path.write_text(json.dumps(seal,indent=2)+'\n')
receipt=dict(request=request,status='RUNNING',owner_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
    seal_sha256=sha(seal_path),gpu_01_compute_process_query=gpu_rows,arms=[])
(admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
status=0
for mode in request['expected_conditions']:
    command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/diagnose_joint_spectral_delta_best.py'),
        '--aggregation',mode,'--seal',str(seal_path),'--output-dir',str(out/mode)]
    arm=dict(aggregation=mode,command=command,started_at=datetime.now().astimezone().isoformat(),status='RUNNING')
    receipt['arms'].append(arm)
    with (admin/(mode+'.log')).open('wb') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        arm['pid']=child.pid;(admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
        status=child.wait()
    arm.update(exit_code=status,status='COMPLETE' if status==0 else 'FAILED',completed_at=datetime.now().astimezone().isoformat())
    (admin/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if status:break
receipt.update(status='COMPLETE' if status==0 else 'FAILED',exit_code=status,completed_at=datetime.now().astimezone().isoformat())
(admin/'EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[admin/'REQUEST.json',admin/'EXIT.json',seal_path,*[admin/(arm['aggregation']+'.log') for arm in receipt['arms']]]
files.extend(out/arm['aggregation']/'DIAGNOSIS.json' for arm in receipt['arms'] if arm['exit_code']==0)
print(json.dumps(dict(exit_code=status,remote_free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'DIAGNOSIS_TRANSPORT_948.stdout').write_bytes(run.stdout.encode());(w/'DIAGNOSIS_TRANSPORT_948.stderr').write_bytes(run.stderr.encode())
(w/'DIAGNOSIS_TRANSPORT_EXIT_948.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'diagnosis948';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'DIAGNOSIS_COLLECTED_948.json').write_text(json.dumps(dict(exit_code=value['exit_code'],remote_free_bytes=value['remote_free_bytes'],
    files={name:{k:v for k,v in item.items() if k!='data'} for name,item in value['files'].items()}),indent=2)+'\n')
assert value['exit_code']==0
print(json.dumps(dict(status='THREE_FIXED_BEST_DIAGNOSES_COLLECTED',conditions=3,optimizer_updates=0)))
